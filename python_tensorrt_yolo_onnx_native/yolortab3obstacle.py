#!/usr/bin/env python3

import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}


class YoloNearest3Rtabmap(Node):
    def __init__(self, args):
        super().__init__('yolo_nearest3_rtabmap')

        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Run: pip3 install ultralytics")

        self.model = YOLO(args.model)
        self.bridge = CvBridge()

        self.conf = args.conf
        self.max_obstacles = args.max_obstacles
        self.target_frame = args.frame_id
        self.default_camera_frame = args.camera_frame

        self.latest_depth = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.marker_pub = self.create_publisher(MarkerArray, args.marker_topic, 10)
        self.cloud_pub = self.create_publisher(PointCloud2, args.cloud_topic, 10)
        self.pose_pub = self.create_publisher(PoseArray, args.pose_topic, 10)

        self.create_subscription(Image, args.image_topic, self.image_callback, 10)
        self.create_subscription(Image, args.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self.camera_info_callback, 10)

        self.get_logger().info(f"YOLO model: {args.model}")
        self.get_logger().info(f"Sub image: {args.image_topic}")
        self.get_logger().info(f"Sub depth: {args.depth_topic}")
        self.get_logger().info(f"Sub camera_info: {args.camera_info_topic}")
        self.get_logger().info(f"Pub markers: {args.marker_topic}")
        self.get_logger().info(f"Pub cloud: {args.cloud_topic}")
        self.get_logger().info(f"Pub poses: {args.pose_topic}")
        self.get_logger().info(f"Target frame: {self.target_frame}")

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    def get_depth_m(self, u, v):
        if self.latest_depth is None:
            return None

        h, w = self.latest_depth.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))

        r = 4
        patch = self.latest_depth[max(0, v-r):min(h, v+r+1),
                                  max(0, u-r):min(w, u+r+1)]
        patch = patch.astype(np.float32)
        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0]

        if patch.size == 0:
            return None

        d = float(np.median(patch))

        if d > 100.0:   # 16UC1 mm -> m
            d /= 1000.0

        if d <= 0.05 or d > 20.0:
            return None

        return d

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

    def transform_camera_to_target(self, x, y, z, source_frame):
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

    @staticmethod
    def make_cloud(points, frame_id, stamp):
        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        arr = np.asarray(points, dtype=np.float32)
        if arr.size == 0:
            data = b''
            width = 0
        else:
            arr = arr.reshape(-1, 3)
            data = arr.tobytes()
            width = arr.shape[0]

        return PointCloud2(
            header=header,
            height=1,
            width=width,
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=12 * width,
            data=data,
            is_dense=False
        )

    def image_callback(self, msg: Image):
        if self.fx is None or self.latest_depth is None:
            return

        source_frame = msg.header.frame_id if msg.header.frame_id else self.default_camera_frame
        if not source_frame:
            source_frame = self.default_camera_frame

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        results = self.model(img, verbose=False, conf=self.conf)
        candidates = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                u = int((x1 + x2) / 2.0)
                v = int((y1 + y2) / 2.0)

                depth = self.get_depth_m(u, v)
                if depth is None:
                    continue

                # Camera optical convention: X right, Y down, Z forward.
                cam_x = (u - self.cx) * depth / self.fx
                cam_y = (v - self.cy) * depth / self.fy
                cam_z = depth

                map_pos = self.transform_camera_to_target(cam_x, cam_y, cam_z, source_frame)
                if map_pos is None:
                    continue

                name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
                candidates.append({
                    'name': name,
                    'conf': conf,
                    'depth': depth,
                    'camera_xyz': (cam_x, cam_y, cam_z),
                    'target_xyz': map_pos,
                })

        candidates.sort(key=lambda o: o['depth'])
        nearest = candidates[:self.max_obstacles]

        stamp = self.get_clock().now().to_msg()

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        pose_array = PoseArray()
        pose_array.header.frame_id = self.target_frame
        pose_array.header.stamp = stamp

        cloud_points = []

        for marker_id, obj in enumerate(nearest):
            mx, my, mz = obj['target_xyz']
            cloud_points.append((float(mx), float(my), float(mz)))

            pose = Pose()
            pose.position.x = float(mx)
            pose.position.y = float(my)
            pose.position.z = float(mz)
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

            sphere = Marker()
            sphere.header.frame_id = self.target_frame
            sphere.header.stamp = stamp
            sphere.ns = 'yolo_points'
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(mx)
            sphere.pose.position.y = float(my)
            sphere.pose.position.z = float(mz)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.15
            sphere.scale.y = 0.15
            sphere.scale.z = 0.15
            sphere.color.r = 1.0
            sphere.color.g = 0.0
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            marker_array.markers.append(sphere)

            text = Marker()
            text.header.frame_id = self.target_frame
            text.header.stamp = stamp
            text.ns = 'yolo_labels'
            text.id = marker_id + 1000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(mx)
            text.pose.position.y = float(my)
            text.pose.position.z = float(mz + 0.35)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"{obj['name']}\n{obj['conf'] * 100:.1f}%\n{obj['depth']:.2f}m"
            marker_array.markers.append(text)

        cloud_msg = self.make_cloud(cloud_points, self.target_frame, stamp)

        self.marker_pub.publish(marker_array)
        self.pose_pub.publish(pose_array)
        self.cloud_pub.publish(cloud_msg)

        if nearest:
            txt = " | ".join([
                f"{o['name']}: d={o['depth']:.2f}m map=({o['target_xyz'][0]:.2f},{o['target_xyz'][1]:.2f},{o['target_xyz'][2]:.2f})"
                for o in nearest
            ])
            self.get_logger().info(f"Nearest obstacles: {txt}", throttle_duration_sec=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11n.pt', help='YOLO model path, e.g. yolo11n.pt')
    parser.add_argument('--image_topic', default='/zed/zed_node/rgb/image_rect_color')
    parser.add_argument('--depth_topic', default='/zed/zed_node/depth/depth_registered')
    parser.add_argument('--camera_info_topic', default='/zed/zed_node/rgb/camera_info')
    parser.add_argument('--marker_topic', default='/yolo/obstacle_markers')
    parser.add_argument('--cloud_topic', default='/rtabmap/cloud_obstacles')
    parser.add_argument('--pose_topic', default='/yolo/obstacle_poses')
    parser.add_argument('--frame_id', default='map', help='Target frame for markers/cloud/poses, same as the successful old script')
    parser.add_argument('--camera_frame', default='zed_left_camera_frame', help='Fallback source camera frame')
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--max_obstacles', type=int, default=3)

    args = parser.parse_args()

    rclpy.init()
    node = YoloNearest3Rtabmap(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
