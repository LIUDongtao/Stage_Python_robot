########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Detect objects, draw 3D bounding boxes in an OpenGL window,
    print object distances, and save a camera image every 5 seconds
    into the ./image folder.

    This version does NOT use cv2, so it avoids the OpenCV / NumPy 2.x issue.
"""

import sys
import os
import time
import math
import argparse

from PIL import Image
import ogl_viewer.viewer as gl
import pyzed.sl as sl


def parse_args(init, opt):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ip_str.replace(':', '').replace('.', '').isdigit() and len(ip_str.split('.')) == 4 and len(ip_str.split(':')) == 2:
            init.set_from_stream(ip_str.split(':')[0], int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ", ip_str)
        elif ip_str.replace(':', '').replace('.', '').isdigit() and len(ip_str.split('.')) == 4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ", ip_str)
        else:
            print("Unvalid IP format. Using live stream")

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

    # Save images into ./image beside this script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    capture_dir = os.path.join(script_dir, "image")
    os.makedirs(capture_dir, exist_ok=True)

    capture_interval = 5.0
    last_capture_time = 0.0
    capture_count = 0

    while viewer.is_available():
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)

            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                frame_rgba = image.get_data()

                filename = os.path.join(
                    capture_dir,
                    time.strftime("capture_%Y%m%d_%H%M%S.jpg")
                )

                try:
                    # ZED LEFT image is RGBA. Save RGB channels only.
                    img = Image.fromarray(frame_rgba[:, :, :3])
                    img.save(filename, quality=95)
                    capture_count += 1
                    print(f"[Capture] Saved {filename} ({capture_count})")
                except Exception as e:
                    print(f"[Capture] Failed to save {filename}: {e}")

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

            viewer.update_view(image, objects)

    viewer.exit()

    image.free(memory_type=sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it', default='')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default='')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default='')
    opt = parser.parse_args()

    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()

    main(opt)
