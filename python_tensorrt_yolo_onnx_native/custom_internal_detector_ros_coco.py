########################################################################
#
# ZED + YOLO11n + ROS2 MarkerArray Publisher
# Based on Stereolabs custom_internal_detector.py
#
########################################################################

import argparse
import cv2
import numpy as np

import pyzed.sl as sl

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer


# COCO 80 classes. Ultralytics YOLO11n official model uses this order.
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


class YoloMarkerPublisher(Node):
    def __init__(self, frame_id: str = "zed_camera_link"):
        super().__init__("yolo_marker_publisher")
        self.frame_id = frame_id
        self.publisher = self.create_publisher(MarkerArray, "/yolo/markers", 10)

    def publish_objects(self, objects: sl.Objects):
        marker_array = MarkerArray()

        # Clear old markers at every frame, otherwise disappeared objects remain in RViz.
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        now = self.get_clock().now().to_msg()

        for i, obj in enumerate(objects.object_list):
            try:
                raw_label = int(obj.raw_label)
            except Exception:
                raw_label = -1

            label_name = COCO_CLASSES.get(raw_label, f"class_{raw_label}")
            confidence = float(getattr(obj, "confidence", 0.0))

            try:
                x = float(obj.position[0])
                y = float(obj.position[1])
                z = float(obj.position[2])
            except Exception:
                continue

            # Ignore invalid ZED positions.
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue

            # Text marker
            text = Marker()
            text.header.frame_id = self.frame_id
            text.header.stamp = now
            text.ns = "yolo_text"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = z + 0.35
            text.pose.orientation.w = 1.0
            text.scale.z = 0.30
            text.text = f"{label_name}\n{confidence:.1f}%"
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            marker_array.markers.append(text)

            # Small sphere marker at the object 3D position
            sphere = Marker()
            sphere.header.frame_id = self.frame_id
            sphere.header.stamp = now
            sphere.ns = "yolo_points"
            sphere.id = i + 1000
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.18
            sphere.scale.y = 0.18
            sphere.scale.z = 0.18
            sphere.color.r = 1.0
            sphere.color.g = 0.0
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            marker_array.markers.append(sphere)

        self.publisher.publish(marker_array)


def __main(opt: argparse.Namespace):
    rclpy.init()
    ros_node = YoloMarkerPublisher(frame_id=opt.ros_frame)

    print("Initializing Camera...")
    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50
    is_playback = opt.svo is not None and len(opt.svo) > 0

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open : {repr(status)}. Exit program.")
        ros_node.destroy_node()
        rclpy.shutdown()
        exit()

    camera_configuration = zed.get_camera_information().camera_configuration
    print("Initializing Camera... DONE")

    print("Enabling Positional Tracking...")
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    status = zed.enable_positional_tracking(positional_tracking_parameters)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Positional Tracking enable : {repr(status)}. Exit program.")
        zed.close()
        ros_node.destroy_node()
        rclpy.shutdown()
        exit()
    print("Enabling Positional Tracking... DONE")

    print("Enabling Object Detection...")
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS
    obj_param.custom_onnx_file = opt.custom_onnx
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False

    status = zed.enable_object_detection(obj_param)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Object Detection enable : {repr(status)}. Exit program.")
        zed.close()
        ros_node.destroy_node()
        rclpy.shutdown()
        exit()
    print("Enabling Object Detection... DONE")

    detection_parameters_rt = sl.CustomObjectDetectionRuntimeParameters()
    detection_parameters_rt.object_detection_properties.detection_confidence_threshold = 30

    props_dict = {
        1: sl.CustomObjectDetectionProperties(),
        2: sl.CustomObjectDetectionProperties(),
    }
    props_dict[1].native_mapped_class = sl.OBJECT_SUBCLASS.PERSON
    props_dict[1].object_acceleration_preset = sl.OBJECT_ACCELERATION_PRESET.MEDIUM
    props_dict[1].detection_confidence_threshold = 40
    props_dict[2].detection_confidence_threshold = 50
    props_dict[2].max_allowed_acceleration = 10 * 10
    detection_parameters_rt.object_class_detection_properties = props_dict

    quit_bool = False
    gl_viewer_available = True

    if not opt.disable_gui:
        image_aspect_ratio = camera_configuration.resolution.width / camera_configuration.resolution.height
        requested_low_res_w = min(1280, camera_configuration.resolution.width)

        display_resolution = sl.Resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio)
        image_left_ocv = np.full(
            (display_resolution.height, display_resolution.width, 4),
            [245, 239, 239, 255],
            np.uint8,
        )

        camera_config = zed.get_camera_information().camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = cv_viewer.TrackingViewer(
            tracks_resolution,
            camera_config.fps,
            init_params.depth_maximum_distance * 1000,
            2,
        )
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

        viewer = gl.GLViewer()
        pc_resolution = sl.Resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio)
        viewer.init(zed.get_camera_information().camera_model, pc_resolution, obj_param.enable_tracking)
        point_cloud = sl.Mat(pc_resolution.width, pc_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        image_left = sl.Mat()
        cam_w_pose = sl.Pose()
        image_scale = (
            display_resolution.width / camera_config.resolution.width,
            display_resolution.height / camera_config.resolution.height,
        )

    objects = sl.Objects()
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    window_name = "ZED | YOLO11n ROS2 Marker"

    __printHelp()
    while rclpy.ok():
        grab_status = zed.grab(runtime_parameters)
        if grab_status != sl.ERROR_CODE.SUCCESS or quit_bool:
            break
        if opt.disable_gui and not gl_viewer_available:
            break

        status = zed.retrieve_custom_objects(objects, detection_parameters_rt)
        if status == sl.ERROR_CODE.SUCCESS:
            ros_node.publish_objects(objects)
            rclpy.spin_once(ros_node, timeout_sec=0.001)

            if not opt.disable_gui:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, pc_resolution)
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)

                image_render_left = image_left.get_data()
                np.copyto(image_left_ocv, image_render_left)

                track_view_generator.generate_view(
                    objects,
                    image_left_ocv,
                    image_scale,
                    cam_w_pose,
                    image_track_ocv,
                    objects.is_tracked,
                )
                global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                viewer.updateData(point_cloud, objects)
                gl_viewer_available = viewer.is_available()

                cv2.imshow(window_name, global_image)
                key = cv2.waitKey(10)
                if key == ord("q"):
                    quit_bool = True
                if key == ord("i"):
                    track_view_generator.zoomIn()
                if key == ord("o"):
                    track_view_generator.zoomOut()

        if is_playback and zed.get_svo_position() == zed.get_svo_number_of_frames() - 1:
            quit_bool = True

    if not opt.disable_gui:
        viewer.exit()
        point_cloud.free()
        image_left.free()

    zed.disable_object_detection()
    zed.close()
    ros_node.destroy_node()
    rclpy.shutdown()


def __printHelp():
    print("\n\n Birds eye view hotkeys:")
    print("* Zoom in tracking view            'i'")
    print("* Zoom out tracking view           'o'")
    print("* Exit:                            'q'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_onnx", type=str, required=True, help="Path to custom ONNX model to use")
    parser.add_argument("--svo", type=str, default=None, help="Optional SVO file, if not passed, use the plugged camera instead")
    parser.add_argument("--disable_gui", action="store_true", help="Disable GUI to improve performance")
    parser.add_argument("--ros_frame", type=str, default="zed_camera_link", help="RViz frame_id for YOLO markers")
    opt = parser.parse_args()
    __main(opt)
