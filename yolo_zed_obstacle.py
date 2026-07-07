import pyzed.sl as sl
from ultralytics import YOLO
import numpy as np

# ZED 
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER

status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED:", status)
    exit(1)

runtime = sl.RuntimeParameters()

image = sl.Mat()
depth = sl.Mat()

cam_info = zed.get_camera_information()

fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
fy = cam_info.camera_configuration.calibration_parameters.left_cam.fy
cx0 = cam_info.camera_configuration.calibration_parameters.left_cam.cx

print("ZED ready")
print(f"fx={fx:.2f}")
print(f"fy={fy:.2f}")
print()

# YOLO


model = YOLO("yolo11n.pt")

print("Running")
print()

try:
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        frame = image.get_data()[:, :, :3]
        depth_map = depth.get_data()
        results = model(frame, verbose=False)
        lines = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h = y2 - y1
                roi_y1 = int(y1 + h * 0.6)
                roi_y2 = y2
                roi = depth_map[roi_y1:roi_y2, x1:x2]
                if roi.size == 0:
                    continue

                valid = (
                    np.isfinite(roi)
                    & (roi > 0.2)
                    & (roi < 5.0)
                )

                ys, xs = np.where(valid)

                if len(xs) < 3:
                    continue

                depths = roi[ys, xs]
                nearest_idx = np.argsort(depths)[:3]
                for idx in nearest_idx:
                    u = x1 + xs[idx]
                    D = float(depths[idx])
                    X = (u - cx0) * D / fx
                    lines.append(
                        f"{class_name} | X={X:.2f}m | D={D:.2f}m"
                    )

        if lines:
            print("-" * 60)
            for line in lines:
                print(line)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    zed.close()
