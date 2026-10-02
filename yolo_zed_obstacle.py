# Import the ZED SDK Python API.
# pyzed.sl provides access to the ZED camera, depth sensing,
# calibration parameters, images, and other ZED functionalities.
import pyzed.sl as sl

# Import the Ultralytics YOLO API.
# This library is used to load the YOLO11 model and perform
# object detection on images captured by the ZED camera.
from ultralytics import YOLO

# NumPy is used here mainly for processing the depth map,
# filtering invalid depth values, and finding the nearest points.
import numpy as np


# ============================================================
# ZED CAMERA INITIALIZATION
# ============================================================

# Create a ZED Camera object.
# This object will be used to open the camera, grab frames,
# retrieve images, and retrieve depth information.
zed = sl.Camera()


# Create the initialization parameters for the ZED camera.
init_params = sl.InitParameters()

# Use ULTRA depth mode.
# ULTRA provides high-quality depth estimation and is useful
# when accurate obstacle distance measurements are required.
init_params.depth_mode = sl.DEPTH_MODE.ULTRA

# Configure the ZED SDK to return depth measurements in meters.
# Therefore, all depth values used later in the program
# will be expressed in meters.
init_params.coordinate_units = sl.UNIT.METER


# Open the ZED camera using the parameters defined above.
status = zed.open(init_params)


# Check whether the camera was successfully opened.
# If the ZED camera cannot be initialized, print the error
# returned by the ZED SDK and stop the program.
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED:", status)
    exit(1)


# Create runtime parameters.
# These parameters are used by zed.grab() each time a new
# camera frame is captured.
runtime = sl.RuntimeParameters()


# Create ZED Mat objects.
#
# "image" will contain the RGB image captured by the left camera.
# "depth" will contain the depth map calculated by the ZED SDK.
image = sl.Mat()
depth = sl.Mat()


# ============================================================
# CAMERA CALIBRATION PARAMETERS
# ============================================================

# Get information about the connected ZED camera.
# This includes resolution, camera model, calibration parameters, etc.
cam_info = zed.get_camera_information()


# Extract the intrinsic calibration parameters of the left camera.
#
# fx = focal length along the horizontal image axis
# fy = focal length along the vertical image axis
# cx0 = horizontal coordinate of the principal point
#
# These intrinsic parameters are useful for converting image
# coordinates into metric coordinates.
fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
fy = cam_info.camera_configuration.calibration_parameters.left_cam.fy
cx0 = cam_info.camera_configuration.calibration_parameters.left_cam.cx


# Display camera initialization information.
print("ZED ready")
print(f"fx={fx:.2f}")
print(f"fy={fy:.2f}")
print()


# ============================================================
# YOLO OBJECT DETECTION
# ============================================================

# Load the YOLO11 nano object detection model.
#
# "yolo11n.pt" is the PyTorch model file used by Ultralytics.
# The model will be used to detect objects in each RGB frame
# captured by the ZED camera.
model = YOLO("yolo11n.pt")


print("Running")
print()


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

try:
    while True:

        # Grab a new frame from the ZED camera.
        #
        # If grabbing the frame fails, skip the current iteration
        # and try again with the next frame.
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue


        # Retrieve the image from the LEFT camera.
        #
        # This image will later be passed to YOLO for object detection.
        zed.retrieve_image(image, sl.VIEW.LEFT)


        # Retrieve the depth map corresponding to the camera image.
        #
        # Each pixel in the depth map represents the estimated
        # distance between the camera and the observed scene.
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)


        # Convert the ZED image into a NumPy array.
        #
        # ZED images contain four channels.
        # [:, :, :3] keeps only the first three channels
        # for YOLO object detection.
        frame = image.get_data()[:, :, :3]


        # Convert the ZED depth Mat into a NumPy array.
        #
        # depth_map[y, x] gives the depth value corresponding
        # to a pixel in the camera image.
        depth_map = depth.get_data()


        # Run YOLO object detection on the current camera frame.
        #
        # verbose=False prevents Ultralytics from printing
        # detection information for every frame.
        results = model(frame, verbose=False)


        # Create an empty list that will store the text information
        # for all valid detected objects in the current frame.
        lines = []


        # ========================================================
        # PROCESS YOLO DETECTIONS
        # ========================================================

        # Iterate through the YOLO detection results.
        for r in results:

            # Iterate through every detected bounding box.
            for box in r.boxes:

                # Get the detected object's class ID.
                # Example:
                # person -> class ID 0
                # car    -> another class ID
                cls_id = int(box.cls[0])


                # Convert the YOLO class ID into its readable name.
                # For example:
                # 0 -> "person"
                class_name = model.names[cls_id]


                # Get the bounding box coordinates.
                #
                # x1, y1 = top-left corner
                # x2, y2 = bottom-right corner
                #
                # The coordinates are converted to integers
                # because they will be used as image indices.
                x1, y1, x2, y2 = map(int, box.xyxy[0])


                # Calculate the height of the detected bounding box.
                h = y2 - y1


                # ====================================================
                # SELECT THE LOWER PART OF THE DETECTED OBJECT
                # ====================================================

                # Instead of using the entire bounding box for depth
                # estimation, only the lower 40% of the detected object
                # is selected.
                #
                # roi_y1 starts at 60% of the bounding box height.
                # roi_y2 corresponds to the bottom of the bounding box.
                #
                # This region is used as the depth Region Of Interest (ROI).
                roi_y1 = int(y1 + h * 0.6)
                roi_y2 = y2


                # Extract the corresponding region from the ZED depth map.
                #
                # The ROI uses the same bounding box coordinates obtained
                # from the YOLO detection.
                roi = depth_map[roi_y1:roi_y2, x1:x2]


                # If the ROI contains no pixels, ignore this detection.
                if roi.size == 0:
                    continue


                # ====================================================
                # FILTER INVALID DEPTH VALUES
                # ====================================================

                # Create a Boolean mask containing only valid depth values.
                #
                # A depth value is considered valid when:
                #
                # 1. It is finite (not NaN or infinity).
                # 2. It is greater than 0.2 meters.
                # 3. It is smaller than 5.0 meters.
                #
                # Therefore, this program only considers detected
                # obstacle points between 0.2 m and 5.0 m.
                valid = (
                    np.isfinite(roi)
                    & (roi > 0.2)
                    & (roi < 5.0)
                )


                # Find the pixel coordinates of all valid depth values
                # inside the ROI.
                #
                # ys = vertical coordinates inside the ROI
                # xs = horizontal coordinates inside the ROI
                ys, xs = np.where(valid)


                # At least three valid depth points are required because
                # the program later selects the three nearest points.
                if len(xs) < 3:
                    continue


                # Extract the depth values corresponding to all
                # valid pixels inside the ROI.
                depths = roi[ys, xs]


                # ====================================================
                # FIND THE THREE NEAREST DEPTH POINTS
                # ====================================================

                # Sort the valid depth values from nearest to farthest
                # and keep the indices of the three nearest points.
                nearest_idx = np.argsort(depths)[:3]


                # Process each of the three nearest depth points.
                for idx in nearest_idx:

                    # Convert the horizontal coordinate from ROI coordinates
                    # back to the original image coordinate system.
                    u = x1 + xs[idx]


                    # Get the depth value of this point.
                    #
                    # Because coordinate_units was configured as METER,
                    # D is expressed in meters.
                    D = float(depths[idx])


                    # ====================================================
                    # CONVERT IMAGE POSITION TO HORIZONTAL METRIC POSITION
                    # ====================================================

                    # Calculate the horizontal X coordinate of the point
                    # relative to the camera using the pinhole camera model:
                    #
                    #               (u - cx) * D
                    #       X = -------------------
                    #                     fx
                    #
                    # where:
                    #
                    # u   = horizontal pixel coordinate
                    # cx0 = camera principal point
                    # D   = depth of the point
                    # fx  = horizontal focal length
                    #
                    # X therefore represents the horizontal position
                    # of the detected point relative to the camera.
                    X = (u - cx0) * D / fx


                    # Store the detection information.
                    #
                    # Example output:
                    #
                    # person | X=-0.35m | D=1.82m
                    #
                    # class_name = detected object category
                    # X          = horizontal position relative to camera
                    # D          = depth/distance measured by ZED
                    lines.append(
                        f"{class_name} | X={X:.2f}m | D={D:.2f}m"
                    )


        # ========================================================
        # DISPLAY DETECTION RESULTS
        # ========================================================

        # Print results only when at least one valid object/depth
        # measurement was found in the current frame.
        if lines:

            # Print a separator to make each group of results
            # easier to read in the terminal.
            print("-" * 60)

            # Print every detected point.
            for line in lines:
                print(line)


# ============================================================
# PROGRAM TERMINATION
# ============================================================

# Allow the user to stop the program safely with Ctrl+C.
except KeyboardInterrupt:
    print("\nStopping...")


finally:

    # Always close the ZED camera before exiting the program.
    # This releases the camera and associated ZED SDK resources.
    zed.close()
