########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
########################################################################

"""
Detect objects, draw 3D bounding boxes in an OpenGL window,
print object distances, and save a camera image every 5 seconds
into the ./image folder.

This version uses OpenCV instead of Pillow.
Click the OpenCV window X or press q to exit normally.
"""

import os
import time
import math
import argparse

import cv2
import ogl_viewer.viewer as gl
import pyzed.sl as sl


def parse_args(init, opt):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input:", opt.input_svo_file)

    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address

        if ip_str.replace(':', '').replace('.', '').isdigit() and \
           len(ip_str.split('.')) == 4 and \
           len(ip_str.split(':')) == 2:

            ip = ip_str.split(':')[0]
            port = int(ip_str.split(':')[1])

            init.set_from_stream(ip, port)
            print("[Sample] Using Stream input, IP:", ip_str)

        elif ip_str.replace(':', '').replace('.', '').isdigit() and \
             len(ip_str.split('.')) == 4:

            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP:", ip_str)

        else:
            print("Invalid IP format. Using live camera")

    if "HD2K" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif "HD1200" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif "HD1080" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif "HD720" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif "SVGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif "VGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution) > 0:
        print("[Sample] No valid resolution entered. Using default")
    else:
        print("[Sample] Using default resolution")


def main(opt):
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    parse_args(init_params, opt)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera open error:", err)
        exit(1)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

    if obj_param.enable_tracking:
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        err = zed.enable_positional_tracking(positional_tracking_parameters)

        if err != sl.ERROR_CODE.SUCCESS:
            print("Positional Tracking Error:", err)
            zed.close()
            exit(1)

    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Object detection error:", err)
        zed.close()
        exit(1)

    camera_info = zed.get_camera_information()

    viewer = gl.GLViewer()
    viewer.init(
        camera_info.camera_configuration.calibration_parameters.left_cam,
        obj_param.enable_tracking
    )

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 30

    # 如果只想检测人，取消下面这一行注释：
    # obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.PERSON]

    objects = sl.Objects()
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    capture_dir = os.path.join(script_dir, "image")
    os.makedirs(capture_dir, exist_ok=True)

    capture_interval = 5.0
    last_capture_time = 0.0
    capture_count = 0

    window_name = "ZED Left Image - OpenCV"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while viewer.is_available():
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)

            frame_rgba = image.get_data()

            # ZED LEFT image 是 RGBA，OpenCV 显示/保存通常使用 BGR
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

            current_time = time.time()

            if current_time - last_capture_time >= capture_interval:
                filename = os.path.join(
                    capture_dir,
                    time.strftime("capture_%Y%m%d_%H%M%S.jpg")
                )

                ok = cv2.imwrite(filename, frame_bgr)

                if ok:
                    capture_count += 1
                    print(f"[Capture] Saved {filename} ({capture_count})")
                else:
                    print(f"[Capture] Failed to save {filename}")

                last_capture_time = current_time

            zed.retrieve_objects(objects, obj_runtime_param)

            print("Detected:", len(objects.object_list))

            for obj in objects.object_list:
                x = obj.position[0]
                y = obj.position[1]
                z = obj.position[2]

                distance = math.sqrt(x * x + y * y + z * z)

                print(
                    "ID:", obj.id,
                    "Class:", obj.label,
                    "Z:", round(z, 2), "m",
                    "Distance:", round(distance, 2), "m"
                )

            # OpenGL 窗口：显示 ZED 官方 3D 检测框
            viewer.update_view(image, objects)

            # OpenCV 窗口：显示普通左目画面
            cv2.imshow(window_name, frame_bgr)

            key = cv2.waitKey(1) & 0xFF

            # 按 q 正常退出
            if key == ord('q'):
                break

            # 点击 OpenCV 窗口右上角 X 正常退出
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    viewer.exit()
    cv2.destroyAllWindows()

    image.free(memory_type=sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_svo_file',
        type=str,
        help='Path to an .svo file, if you want to replay it',
        default=''
    )

    parser.add_argument(
        '--ip_address',
        type=str,
        help='IP Address, in format a.b.c.d:port or a.b.c.d',
        default=''
    )

    parser.add_argument(
        '--resolution',
        type=str,
        help='Resolution: HD2K, HD1200, HD1080, HD720, SVGA or VGA',
        default=''
    )

    opt = parser.parse_args()

    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, not both. Exit program")
        exit()

    main(opt)
