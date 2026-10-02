# ============================================================
# IMPORTS AND DEPENDENCIES
# ============================================================
# math: mathematical functions used for body and knee angle calculations.
# time: timing utilities used for FPS calculation and periodic terminal output.
# typing: type hints used to make function inputs and outputs easier to understand.
import math
import time
from typing import Dict, Optional, Tuple, List

# OpenCV is used to draw bounding boxes, pose skeletons, labels,
# display the camera image, and handle keyboard input.
# NumPy is used for vector mathematics, keypoint validation, and depth processing.
# pyzed.sl is the Python API of the Stereolabs ZED SDK.
# Ultralytics provides the YOLO model interface used for human pose estimation.
import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

# Try to import PyTorch and check whether a CUDA-capable GPU is available.
# If PyTorch cannot be imported or CUDA initialization fails, the program
# automatically falls back to CPU mode instead of stopping immediately.
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


# ============================================================
# PERFORMANCE / SPEED CONFIGURATION
# ============================================================
# These constants control the compromise between inference speed,
# detection quality, number of people processed, and display responsiveness.
# Speed parameters

# Path to the YOLO11 Nano Pose model.
# YOLO_IMGSZ defines the inference image size.
# Confidence and IoU thresholds control detection filtering.
# MAX_DET limits the maximum number of detected persons.
# PROCESS_EVERY_N_FRAMES reduces inference frequency to improve speed.
# PRINT_INTERVAL limits how often results are printed in the terminal.
# DISPLAY_SCALE can reduce the visualization window size if needed.
MODEL_PATH = "yolo11n-pose.pt"     # fastest pose model
YOLO_IMGSZ = 416                  # 416 faster than 640, try 320 if still slow
YOLO_CONF = 0.35
YOLO_IOU = 0.50
MAX_DET = 3                       # only detect max 3 persons
PROCESS_EVERY_N_FRAMES = 2        # 1 = every frame, 2 = every 2 frames, 3 = faster but less smooth
PRINT_INTERVAL = 1.0              # terminal print interval in seconds
DISPLAY_SCALE = 1.0               # set 0.75 or 0.5 if the window itself is slow

# Select the inference device automatically.
# CUDA is used when available; otherwise the CPU is used.
# Half precision is enabled only with CUDA to improve GPU inference speed.
DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
HALF = True if CUDA_AVAILABLE else False


# ============================================================
# COCO HUMAN POSE KEYPOINT DEFINITIONS
# ============================================================
# YOLO pose uses the standard COCO 17-keypoint human skeleton.
# The constants below give readable names to the keypoint indices.
# COCO 17 keypoint indices

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Define the connections used to draw the human skeleton.
# Each tuple connects two COCO keypoints with a line.
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# ============================================================
# FPS COUNTER
# ============================================================
# This helper class estimates the current processing frame rate.
# The FPS value is refreshed approximately once per second.
class FPSCounter:
    def __init__(self):
        self.t0 = time.time()
        self.frames = 0
        self.fps = 0.0

    def update(self) -> float:
        self.frames += 1
        now = time.time()
        dt = now - self.t0
        if dt >= 1.0:
            self.fps = self.frames / dt
            self.frames = 0
            self.t0 = now
        return self.fps


# ============================================================
# RULE-BASED HUMAN POSTURE ANALYZER
# ============================================================
# This class converts YOLO pose keypoints into simple posture labels.
# The classification is based on geometric rules rather than another
# neural-network classifier.
class PoseAnalyzer:
    """Simple rule-based posture analyzer using YOLO pose keypoints."""

    def __init__(self, keypoint_conf_threshold: float = 0.4):
        self.kpt_conf_th = keypoint_conf_threshold

# Check whether a keypoint can safely be used.
# A valid keypoint must exist, have sufficient confidence,
# contain finite coordinates, and have positive image coordinates.
    def valid_point(self, kpts: np.ndarray, confs: np.ndarray, idx: int) -> bool:
        if kpts is None or confs is None or idx >= len(kpts) or idx >= len(confs):
            return False
        x, y = kpts[idx]
        return confs[idx] >= self.kpt_conf_th and np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0

# Calculate the midpoint between two body keypoints.
# If only one of the two points is valid, use that point as a fallback.
# If neither point is valid, return None.
    def midpoint(self, kpts: np.ndarray, confs: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
        if self.valid_point(kpts, confs, a) and self.valid_point(kpts, confs, b):
            return (kpts[a] + kpts[b]) / 2.0
        if self.valid_point(kpts, confs, a):
            return kpts[a]
        if self.valid_point(kpts, confs, b):
            return kpts[b]
        return None

# Calculate the inclination of the torso relative to the vertical direction.
# A small angle indicates a more vertical torso; a large angle indicates
# that the torso is closer to horizontal.
    @staticmethod
    def angle_from_vertical(p_top: np.ndarray, p_bottom: np.ndarray) -> float:
        dx = float(p_top[0] - p_bottom[0])
        dy = float(p_top[1] - p_bottom[1])
        return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

# Calculate the angle at a knee from hip-knee and ankle-knee vectors.
# The result is expressed in degrees.
    @staticmethod
    def knee_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> Optional[float]:
        v1 = hip - knee
        v2 = ankle - knee
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cosang = np.dot(v1, v2) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        return float(math.degrees(math.acos(cosang)))

# Analyze one detected person's pose.
# The returned dictionary contains the posture classification,
# danger state, body angle, and a textual explanation of the decision.
    def analyze(self, kpts: np.ndarray, confs: np.ndarray, bbox_xyxy: np.ndarray) -> Dict:
        result = {
            "posture": "unknown",
            "danger": False,
            "body_angle": None,
            "reason": "not enough keypoints",
        }

# Estimate representative points for the shoulders, hips, and knees
# by combining the corresponding left and right keypoints.
        shoulder_mid = self.midpoint(kpts, confs, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_mid = self.midpoint(kpts, confs, LEFT_HIP, RIGHT_HIP)
        knee_mid = self.midpoint(kpts, confs, LEFT_KNEE, RIGHT_KNEE)

# Obtain the bounding-box dimensions and width/height ratio.
# A wide bounding box can provide additional evidence that a person is lying.
        x1, y1, x2, y2 = bbox_xyxy.astype(float)
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        box_ratio = box_w / box_h

# If shoulder and hip information is available, estimate torso orientation.
# The following rules distinguish lying, sitting, and standing postures.
        if shoulder_mid is not None and hip_mid is not None:
            body_angle = self.angle_from_vertical(shoulder_mid, hip_mid)
            result["body_angle"] = body_angle

            # Lying: body trunk is almost horizontal or bbox is very wide.
            if body_angle > 60 or box_ratio > 1.45:
                result["posture"] = "lying"
                result["danger"] = True
                result["reason"] = f"body_angle={body_angle:.1f}, bbox_ratio={box_ratio:.2f}"
                return result

# Build a simple sitting score.
# Relative hip-to-knee distance and knee bending are used as evidence
# that the detected person may be sitting.
            sitting_score = 0
            if knee_mid is not None:
                hip_to_knee = np.linalg.norm(knee_mid - hip_mid)
                shoulder_to_hip = np.linalg.norm(shoulder_mid - hip_mid)
                if shoulder_to_hip > 1e-6 and hip_to_knee / shoulder_to_hip < 1.15:
                    sitting_score += 1

# Calculate available left and right knee angles.
# Only complete hip-knee-ankle chains with valid keypoints are considered.
            knee_angles = []
            if self.valid_point(kpts, confs, LEFT_HIP) and self.valid_point(kpts, confs, LEFT_KNEE) and self.valid_point(kpts, confs, LEFT_ANKLE):
                a = self.knee_angle(kpts[LEFT_HIP], kpts[LEFT_KNEE], kpts[LEFT_ANKLE])
                if a is not None:
                    knee_angles.append(a)
            if self.valid_point(kpts, confs, RIGHT_HIP) and self.valid_point(kpts, confs, RIGHT_KNEE) and self.valid_point(kpts, confs, RIGHT_ANKLE):
                a = self.knee_angle(kpts[RIGHT_HIP], kpts[RIGHT_KNEE], kpts[RIGHT_ANKLE])
                if a is not None:
                    knee_angles.append(a)
            if knee_angles and min(knee_angles) < 135:
                sitting_score += 1

# Use torso angle together with the sitting evidence to classify sitting.
# A sufficiently vertical torso without sitting evidence is classified standing.
            if body_angle < 40 and sitting_score >= 1:
                result["posture"] = "sitting"
                result["reason"] = f"body_angle={body_angle:.1f}, sitting_score={sitting_score}"
                return result

            if body_angle < 40:
                result["posture"] = "standing"
                result["reason"] = f"body_angle={body_angle:.1f}"
                return result

            result["reason"] = f"body_angle={body_angle:.1f}, bbox_ratio={box_ratio:.2f}"
            return result

# If important pose keypoints are missing, fall back to the bounding-box
# aspect ratio to make a limited lying/standing estimate.
        # Fallback if keypoints are partially missing.
        if box_ratio > 1.55:
            result["posture"] = "lying"
            result["danger"] = True
            result["reason"] = f"fallback bbox_ratio={box_ratio:.2f}"
        elif box_ratio < 0.75:
            result["posture"] = "standing"
            result["reason"] = f"fallback bbox_ratio={box_ratio:.2f}"

        return result


# ============================================================
# ZED DEPTH AND HORIZONTAL POSITION ESTIMATION
# ============================================================
# Estimate a person's distance D and horizontal position X using the ZED depth map.
# The function samples a central region of the person's bounding box,
# rejects invalid depth values, and uses the median valid depth.
def get_depth_distance(depth_map: np.ndarray, bbox_xyxy: np.ndarray, fx: float, cx: float) -> Tuple[Optional[float], Optional[float]]:
    img_h, img_w = depth_map.shape
    x1, y1, x2, y2 = bbox_xyxy.astype(int)

# Clamp the YOLO bounding-box coordinates to the valid depth-image limits.
# This prevents invalid NumPy indexing when a detection touches an image border.
    x1 = max(0, min(img_w - 1, x1))
    x2 = max(0, min(img_w - 1, x2))
    y1 = max(0, min(img_h - 1, y1))
    y2 = max(0, min(img_h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None, None

# Select an internal Region Of Interest (ROI) inside the person bounding box.
# The ROI avoids using the complete box and focuses on its central body region.
    h = y2 - y1
    roi_y1 = int(y1 + 0.45 * h)
    roi_y2 = int(y1 + 0.90 * h)
    roi_x1 = int(x1 + 0.25 * (x2 - x1))
    roi_x2 = int(x1 + 0.75 * (x2 - x1))

# Extract the selected region from the ZED depth map.
# If the region is empty, no reliable distance can be returned.
    roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None, None

# Keep only finite depth measurements between 0.2 m and 8.0 m.
# If no valid depth sample remains, return no position or distance.
    valid = np.isfinite(roi) & (roi > 0.2) & (roi < 8.0)
    if not np.any(valid):
        return None, None

# Use the median valid depth as the person's distance.
# The horizontal image center of the bounding box is then projected using
# the camera intrinsic parameters:
#
#     X = (u - cx) * D / fx
#
# X is the horizontal metric position relative to the camera,
# and D is the measured depth in meters.
    D = float(np.median(roi[valid]))
    u = (x1 + x2) / 2.0
    X = float((u - cx) * D / fx)
    return X, D


# ============================================================
# POSE SKELETON VISUALIZATION
# ============================================================
# Draw valid pose keypoints as circles and connect valid keypoint pairs
# according to the COCO skeleton definition.
def draw_skeleton(image: np.ndarray, kpts: np.ndarray, confs: np.ndarray, conf_th: float = 0.4) -> None:
    for i, (x, y) in enumerate(kpts):
        if confs[i] >= conf_th and np.isfinite(x) and np.isfinite(y):
            cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

    for a, b in SKELETON:
        if confs[a] >= conf_th and confs[b] >= conf_th:
            xa, ya = kpts[a]
            xb, yb = kpts[b]
            if np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb):
                cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)


# ============================================================
# DETECTION RESULT VISUALIZATION
# ============================================================
# Draw each detected person, its skeleton, posture label, distance,
# and danger indication on a copy of the current camera frame.
def draw_results(frame: np.ndarray, humans: List[Dict]) -> np.ndarray:
    vis = frame.copy()
    for h in humans:
        bbox = h["bbox"].astype(int)
        kpts = h["kpts"]
        confs = h["confs"]
        x1, y1, x2, y2 = bbox
# Choose the visualization color according to the danger state,
# draw the person bounding box and skeleton, and prepare the text label.
        color = (0, 255, 255) if h["danger"] else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        draw_skeleton(vis, kpts, confs, 0.4)
        d_text = "NA" if h["d"] is None else f"{h['d']:.2f}m"
        label = f"Person {h['id']}: {h['posture']} D={d_text}"
        if h["danger"]:
            label += " DANGER"
        cv2.putText(vis, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


# ============================================================
# MAIN PROGRAM
# ============================================================
# Initialize the ZED camera, YOLO pose model, posture analyzer,
# depth processing, FPS monitoring, and visualization loop.
def main():
    print("\n====================================")
    print("ZED + YOLO11 Pose Analyzer V2 Fast")
    print(f"Model: {MODEL_PATH}, device: {DEVICE}, half: {HALF}")
    print(f"YOLO imgsz={YOLO_IMGSZ}, process every {PROCESS_EVERY_N_FRAMES} frame(s)")
    print("====================================\n")

# ------------------------------------------------------------
# ZED camera configuration
# ------------------------------------------------------------
# HD720 and 15 FPS reduce computational load compared with higher settings.
# PERFORMANCE depth mode favors speed, while metric units ensure that
# depth values returned by the SDK are expressed in meters.
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720   # faster than HD1080
    init.camera_fps = 15                           # reduce camera load
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER

# Open the ZED camera.
# If initialization fails, print the ZED SDK error and stop main().
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Cannot open ZED:", status)
        return

# Allocate runtime parameters and ZED Mat objects used repeatedly
# for image and depth acquisition inside the processing loop.
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

# Read the calibrated horizontal focal length and principal point.
# These values are later used to convert pixel position and depth into X.
    cam_info = zed.get_camera_information()
    fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
    cx = cam_info.camera_configuration.calibration_parameters.left_cam.cx

# Load the YOLO11 pose model and attempt to move it to the selected device.
# If moving the model fails, a warning is printed.
    model = YOLO(MODEL_PATH)
    try:
        model.to(DEVICE)
    except Exception as e:
        print("Warning: cannot move model to device:", e)

# Create the rule-based posture analyzer and FPS counter.
# The remaining variables store frame state and the most recent results.
    analyzer = PoseAnalyzer(keypoint_conf_threshold=0.4)
    fps_counter = FPSCounter()

    frame_id = 0
    last_print_time = 0.0
    last_humans: List[Dict] = []
    last_vis = None

# ============================================================
# REAL-TIME PROCESSING LOOP
# ============================================================
    try:
# Continuously acquire frames until the user presses 'q' or Ctrl+C.
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

# Retrieve the synchronized left image and depth map from the ZED camera.
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

# Convert ZED data into NumPy arrays.
# The RGB frame is copied because OpenCV drawing operations modify image data.
            frame = image.get_data()[:, :, :3].copy()
            depth_map = depth.get_data()
            frame_id += 1

# YOLO pose inference is intentionally performed only every N frames.
# This lowers GPU/CPU workload while keeping the display loop responsive.
            # Run YOLO only every N frames for speed.
            if frame_id % PROCESS_EVERY_N_FRAMES == 0:
# Run pose inference for the person class only.
# The configured confidence, IoU, image size, maximum detections,
# device, and half-precision settings are passed directly to Ultralytics.
                results = model.predict(
                    source=frame,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    classes=[0],
                    imgsz=YOLO_IMGSZ,
                    max_det=MAX_DET,
                    device=DEVICE,
                    half=HALF,
                    verbose=False,
                )
# Use the first inference result and create a new list for people
# detected in the current processed frame.
                result = results[0]
                humans: List[Dict] = []

# Continue only when bounding boxes and pose keypoints are available.
# Convert PyTorch tensors to CPU NumPy arrays for subsequent processing.
                if result.boxes is not None and result.keypoints is not None and result.keypoints.conf is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    keypoints_xy = result.keypoints.xy.cpu().numpy()
                    keypoints_conf = result.keypoints.conf.cpu().numpy()

# Process every detected person independently:
# 1. read the bounding box and keypoints,
# 2. classify the posture,
# 3. estimate horizontal position and depth,
# 4. store all information in a dictionary.
                    for i in range(len(boxes)):
                        bbox = boxes[i]
                        kpts = keypoints_xy[i]
                        confs = keypoints_conf[i]
                        pose_info = analyzer.analyze(kpts, confs, bbox)
                        X, D = get_depth_distance(depth_map, bbox, fx, cx)

                        humans.append({
                            "id": i + 1,
                            "score": float(scores[i]),
                            "posture": pose_info["posture"],
                            "danger": pose_info["danger"],
                            "body_angle": pose_info["body_angle"],
                            "reason": pose_info["reason"],
                            "x": X,
                            "d": D,
                            "bbox": bbox,
                            "kpts": kpts,
                            "confs": confs,
                        })

# Save the newest detection results and render their visualization.
# These values are reused between YOLO inference frames.
                last_humans = humans
                last_vis = draw_results(frame, last_humans)
# Frames skipped by YOLO inference keep the most recently rendered result.
# This avoids running the neural network on every camera frame.
            else:
                # On skipped frames, only show raw frame with last results not redrawn.
                # This keeps the UI responsive. If you prefer constant overlay, set PROCESS_EVERY_N_FRAMES=1.
                if last_vis is None:
                    last_vis = frame

# Update the FPS estimate and prepare the frame displayed to the user.
            fps = fps_counter.update()
            vis = last_vis.copy() if last_vis is not None else frame.copy()
# Overlay runtime information including FPS, inference device,
# and YOLO input image size.
            cv2.putText(vis, f"FPS: {fps:.1f}  device:{DEVICE}  imgsz:{YOLO_IMGSZ}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# Print pose results at the configured time interval instead of every frame.
# This keeps terminal output readable and reduces unnecessary printing.
            now = time.time()
            if now - last_print_time > PRINT_INTERVAL:
                last_print_time = now
# Print either a no-human message or detailed information for each person:
# confidence score, posture, danger state, X position, depth, body angle,
# and the rule-based reason for the posture classification.
                print("\n====================================")
                if not last_humans:
                    print("No human detected")
                else:
                    print("Human Posture Results")
                    for h in last_humans:
                        x_str = "NA" if h["x"] is None else f"{h['x']:.2f} m"
                        d_str = "NA" if h["d"] is None else f"{h['d']:.2f} m"
                        angle_str = "NA" if h["body_angle"] is None else f"{h['body_angle']:.1f} deg"
                        print("------------------------------")
                        print(f"Human  : {h['id']}")
                        print(f"Score  : {h['score']:.2f}")
                        print(f"Pose   : {h['posture']}")
                        print(f"Danger : {h['danger']}")
                        print(f"X      : {x_str}")
                        print(f"D      : {d_str}")
                        print(f"Angle  : {angle_str}")
                        print(f"Reason : {h['reason']}")
                        print("------------------------------")

# Optionally resize the visualization window without changing inference data.
            if DISPLAY_SCALE != 1.0:
                vis = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

# Display the processed frame.
# Press 'q' in the OpenCV window to leave the main loop.
            cv2.imshow("ZED YOLO Pose Analyzer V2 Fast", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

# Allow Ctrl+C to stop the program cleanly from the terminal.
    except KeyboardInterrupt:
        print("\nStopping...")

# Always release the ZED camera and destroy OpenCV windows,
# including when the program exits because of an interruption.
    finally:
        zed.close()
        cv2.destroyAllWindows()


# Standard Python entry point.
# main() runs only when this file is executed directly.
if __name__ == "__main__":
    main()