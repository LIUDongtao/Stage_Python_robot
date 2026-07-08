


import pyzed.sl as sl
import numpy as np


class ZEDCamera:

    def __init__(
            self,
            resolution=sl.RESOLUTION.HD720,
            fps=15,
            depth_mode=sl.DEPTH_MODE.PERFORMANCE
    ):

        self.zed = sl.Camera()

        init = sl.InitParameters()
        init.camera_resolution = resolution
        init.camera_fps = fps
        init.depth_mode = depth_mode
        init.coordinate_units = sl.UNIT.METER

        status = self.zed.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Cannot open ZED camera : {status}")

        self.runtime = sl.RuntimeParameters()

        self.image = sl.Mat()
        self.depth = sl.Mat()

        cam_info = self.zed.get_camera_information()

        self.fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
        self.fy = cam_info.camera_configuration.calibration_parameters.left_cam.fy
        self.cx = cam_info.camera_configuration.calibration_parameters.left_cam.cx
        self.cy = cam_info.camera_configuration.calibration_parameters.left_cam.cy

    def grab(self):

        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return False, None, None

        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

        rgb = self.image.get_data()[:, :, :3].copy()
        depth = self.depth.get_data().copy()

        return True, rgb, depth

    def get_intrinsics(self):

        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy
        }

    def close(self):
        self.zed.close()
