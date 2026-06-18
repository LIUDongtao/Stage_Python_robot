#!/usr/bin/env python3

import argparse
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException


class DepthObstacleMapper(Node):
    def __init__(self, args):
        super().__init__('depth_obstacle_mapper')
        self.bridge = CvBridge()

        self.target_frame = args.frame_id
        self.camera_frame = args.camera_frame
        self.min_depth = float(args.min_depth)
        self.max_depth = float(args.max_depth)
        self.stride = int(args.stride)
        self.max_markers = int(args.max_markers)
        self.marker_lifetime = float(args.marker_lifetime)
        self.text_scale = float(args.text_scale)
        self.min_height = float(args.min_height)
        self.max_height = float(args.max_height)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(MarkerArray, args.marker_topic, 10)

        self.create_subscription(Image, args.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self.camera_info_callback, 10)

        self.get_logger().info(f"Depth topic: {args.depth_topic}")
        self.get_logger().info(f"CameraInfo topic: {args.camera_info_topic}")
        self.get_logger().info(f"Publishing obstacle markers: {args.marker_topic}")
        self.get_logger().info(f"Target frame: {self.target_frame}")
        self.get_logger().info(f"Camera frame fallback: {self.camera_frame}")

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    @staticmethod
    def rotate_vector_by_quaternion(v, q):
        x, y, z = v
        qx, qy, qz, qw = q

        uvx = qy * z - qz * y
        uvy = qz * x - qx * z
        uvz = qx * y - qy * x

        uuvx = qy * uvz - qz * uvy
        uuvy = qz * uvx - qx * uvz
        uuvz = qx * uvy - qy * uvx

        rx = x + 2.0 * (qw * uvx + uuvx)
        ry = y + 2.0 * (qw * uvy + uuvy)
        rz = z + 2.0 * (qw * uvz + uuvz)

        return rx, ry, rz

    def transform_point(self, x, y, z, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time()
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF not ready: {self.target_frame} <- {source_frame}: {e}",
                throttle_duration_sec=2.0
            )
            return None

        t = trans.transform.translation
        r = trans.transform.rotation

        rx, ry, rz = self.rotate_vector_by_quaternion(
            (float(x), float(y), float(z)),
            (r.x, r.y, r.z, r.w)
        )

        return rx + t.x, ry + t.y, rz + t.z

    def make_lifetime(self):
        if self.marker_lifetime <= 0.0:
            return None
        sec = int(self.marker_lifetime)
        nanosec = int((self.marker_lifetime - sec) * 1e9)
        return Duration(sec=sec, nanosec=nanosec)

    def depth_callback(self, msg: Image):
        if self.fx is None:
            return

        source_frame = msg.header.frame_id if msg.header.frame_id else self.camera_frame
        if not source_frame:
            source_frame = self.camera_frame

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")
            return

        depth = depth.astype(np.float32)
        finite_depth = depth[np.isfinite(depth)]
        if finite_depth.size == 0:
            return

        if np.nanmax(finite_depth) > 100.0:
            depth = depth / 1000.0

        h, w = depth.shape[:2]

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        lifetime = self.make_lifetime()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0

        v_start = int(h * 0.35)
        v_end = int(h * 0.90)

        for v in range(v_start, v_end, self.stride):
            for u in range(0, w, self.stride):
                d = float(depth[v, u])

                if not math.isfinite(d):
                    continue
                if d < self.min_depth or d > self.max_depth:
                    continue

                cam_x = (u - self.cx) * d / self.fx
                cam_y = (v - self.cy) * d / self.fy
                cam_z = d

                map_pos = self.transform_point(cam_x, cam_y, cam_z, source_frame)
                if map_pos is None:
                    return

                mx, my, mz = map_pos

                if mz < self.min_height or mz > self.max_height:
                    continue

                cube = Marker()
                cube.header.frame_id = self.target_frame
                cube.header.stamp = stamp
                cube.ns = 'depth_obstacles'
                cube.id = marker_id
                cube.type = Marker.CUBE
                cube.action = Marker.ADD
                cube.pose.position.x = float(mx)
                cube.pose.position.y = float(my)
                cube.pose.position.z = float(mz)
                cube.pose.orientation.w = 1.0
                cube.scale.x = 0.12
                cube.scale.y = 0.12
                cube.scale.z = 0.12
                cube.color.r = 0.0
                cube.color.g = 0.3
                cube.color.b = 1.0
                cube.color.a = 0.75
                if lifetime is not None:
                    cube.lifetime = lifetime
                marker_array.markers.append(cube)

                if marker_id % 25 == 0:
                    text = Marker()
                    text.header.frame_id = self.target_frame
                    text.header.stamp = stamp
                    text.ns = 'depth_obstacle_labels'
                    text.id = marker_id + 10000
                    text.type = Marker.TEXT_VIEW_FACING
                    text.action = Marker.ADD
                    text.pose.position.x = float(mx)
                    text.pose.position.y = float(my)
                    text.pose.position.z = float(mz + 0.25)
                    text.pose.orientation.w = 1.0
                    text.scale.z = self.text_scale
                    text.color.r = 0.0
                    text.color.g = 0.6
                    text.color.b = 1.0
                    text.color.a = 1.0
                    text.text = "obstacle"
                    if lifetime is not None:
                        text.lifetime = lifetime
                    marker_array.markers.append(text)

                marker_id += 1
                if marker_id >= self.max_markers:
                    break

            if marker_id >= self.max_markers:
                break

        self.marker_pub.publish(marker_array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_topic', default='/zed/zed_node/depth/depth_registered')
    parser.add_argument('--camera_info_topic', default='/zed/zed_node/rgb/camera_info')
    parser.add_argument('--marker_topic', default='/depth_obstacles/markers')
    parser.add_argument('--frame_id', default='map')
    parser.add_argument('--camera_frame', default='zed_left_camera_frame')

    parser.add_argument('--min_depth', type=float, default=0.4)
    parser.add_argument('--max_depth', type=float, default=3.0)
    parser.add_argument('--stride', type=int, default=35)
    parser.add_argument('--max_markers', type=int, default=120)

    parser.add_argument('--min_height', type=float, default=-0.2)
    parser.add_argument('--max_height', type=float, default=1.5)

    parser.add_argument('--marker_lifetime', type=float, default=0.5)
    parser.add_argument('--text_scale', type=float, default=0.18)

    args = parser.parse_args()

    rclpy.init()
    node = DepthObstacleMapper(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
