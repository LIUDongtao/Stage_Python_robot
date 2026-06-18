#!/usr/bin/env python3

import argparse
import math
from collections import defaultdict

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None


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


class YoloSemanticDbscanForced(Node):
    def __init__(self, args):
        super().__init__('yolo_semantic_dbscan')

        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Run: pip3 install ultralytics")

        self.model = YOLO(args.model)
        self.bridge = CvBridge()

        self.conf = float(args.conf)
        self.target_frame = args.frame_id
        self.camera_frame = args.camera_frame

        self.dbscan_eps = float(args.dbscan_eps)
        self.dbscan_min_samples = int(args.dbscan_min_samples)
        self.max_points_per_class = int(args.max_points_per_class)
        self.text_scale = float(args.text_scale)
        self.marker_lifetime = float(args.marker_lifetime)

        self.allowed_classes = set()
        if args.classes.strip():
            for item in args.classes.split(','):
                item = item.strip()
                if not item:
                    continue
                if item.isdigit():
                    self.allowed_classes.add(int(item))
                else:
                    for cid, cname in COCO_CLASSES.items():
                        if cname == item:
                            self.allowed_classes.add(cid)

        self.latest_depth = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.raw_points = defaultdict(list)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.marker_pub = self.create_publisher(MarkerArray, args.marker_topic, 10)

        self.create_subscription(Image, args.image_topic, self.image_callback, 10)
        self.create_subscription(Image, args.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self.camera_info_callback, 10)

        self.timer = self.create_timer(1.0 / float(args.publish_rate), self.publish_semantic_map)

        self.get_logger().info(f"YOLO model: {args.model}")
        self.get_logger().info(f"Publishing semantic markers: {args.marker_topic}")
        self.get_logger().info(f"Target frame: {self.target_frame}")
        self.get_logger().info(f"FORCED OPTICAL camera frame: {self.camera_frame}")
        self.get_logger().info(f"DBSCAN eps: {self.dbscan_eps} m, min_samples: {self.dbscan_min_samples}")
        if DBSCAN is None:
            self.get_logger().warn("sklearn not installed. Fallback radius clustering will be used. For DBSCAN: pip3 install scikit-learn")

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

        r = 6
        patch = self.latest_depth[max(0, v-r):min(h, v+r+1),
                                  max(0, u-r):min(w, u+r+1)]
        patch = patch.astype(np.float32)
        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0]

        if patch.size == 0:
            return None

        d = float(np.median(patch))
        if d > 100.0:
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

    def transform_camera_to_target(self, x, y, z):
        # IMPORTANT: use latest TF + forced camera frame.
        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time()
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF not ready: {self.target_frame} <- {self.camera_frame}: {e}",
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

    def add_raw_point(self, name, pos):
        points = self.raw_points[name]
        points.append((float(pos[0]), float(pos[1]), float(pos[2])))
        if len(points) > self.max_points_per_class:
            del points[:len(points) - self.max_points_per_class]

    def image_callback(self, msg: Image):
        if self.fx is None or self.latest_depth is None:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        results = self.model(img, verbose=False, conf=self.conf)
        added = 0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if self.allowed_classes and cls_id not in self.allowed_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                u = int((x1 + x2) / 2.0)
                v = int((y1 + y2) / 2.0)

                depth = self.get_depth_m(u, v)
                if depth is None:
                    continue

                # Camera optical convention: X right, Y down, Z forward.
                # This version forces zed_left_camera_optical_frame and uses latest TF.
                cam_x = (u - self.cx) * depth / self.fx
                cam_y = (v - self.cy) * depth / self.fy
                cam_z = depth

                map_pos = self.transform_camera_to_target(cam_x, cam_y, cam_z)
                if map_pos is None:
                    continue

                name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
                self.add_raw_point(name, map_pos)
                added += 1

        if added > 0:
            total = sum(len(v) for v in self.raw_points.values())
            self.get_logger().info(f"Added {added} detections, total raw points: {total}", throttle_duration_sec=2.0)

    def cluster_points(self, points):
        if len(points) == 0:
            return []

        X = np.array(points, dtype=np.float32)
        X2 = X[:, :2]

        if DBSCAN is not None:
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(X2)
            labels = db.labels_
            centers = []
            for label in sorted(set(labels)):
                if label == -1:
                    continue
                mask = labels == label
                cluster_xyz = X[mask]
                mean = cluster_xyz.mean(axis=0)
                centers.append((float(mean[0]), float(mean[1]), float(mean[2]), int(cluster_xyz.shape[0])))
            return centers

        clusters = []
        for p in X:
            assigned = False
            for c in clusters:
                cx, cy, cz, count = c
                dist = math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                if dist <= self.dbscan_eps:
                    new_count = count + 1
                    c[0] = (cx * count + p[0]) / new_count
                    c[1] = (cy * count + p[1]) / new_count
                    c[2] = (cz * count + p[2]) / new_count
                    c[3] = new_count
                    assigned = True
                    break
            if not assigned:
                clusters.append([float(p[0]), float(p[1]), float(p[2]), 1])

        return [(c[0], c[1], c[2], c[3]) for c in clusters if c[3] >= self.dbscan_min_samples]

    def make_lifetime(self):
        if self.marker_lifetime <= 0.0:
            return None
        sec = int(self.marker_lifetime)
        nanosec = int((self.marker_lifetime - sec) * 1e9)
        return Duration(sec=sec, nanosec=nanosec)

    def publish_semantic_map(self):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        lifetime = self.make_lifetime()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0
        for name, points in sorted(self.raw_points.items()):
            for x, y, z, count in self.cluster_points(points):
                sphere = Marker()
                sphere.header.frame_id = self.target_frame
                sphere.header.stamp = stamp
                sphere.ns = 'semantic_points'
                sphere.id = marker_id
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose.position.x = float(x)
                sphere.pose.position.y = float(y)
                sphere.pose.position.z = float(z)
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = 0.20
                sphere.scale.y = 0.20
                sphere.scale.z = 0.20
                sphere.color.r = 1.0
                sphere.color.g = 0.0
                sphere.color.b = 0.0
                sphere.color.a = 1.0
                if lifetime is not None:
                    sphere.lifetime = lifetime
                marker_array.markers.append(sphere)

                text = Marker()
                text.header.frame_id = self.target_frame
                text.header.stamp = stamp
                text.ns = 'semantic_labels'
                text.id = marker_id + 10000
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = float(x)
                text.pose.position.y = float(y)
                text.pose.position.z = float(z + 0.35)
                text.pose.orientation.w = 1.0
                text.scale.z = self.text_scale
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 0.0
                text.color.a = 1.0
                text.text = name
                if lifetime is not None:
                    text.lifetime = lifetime
                marker_array.markers.append(text)

                marker_id += 1

        self.marker_pub.publish(marker_array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--image_topic', default='/zed/zed_node/rgb/image_rect_color')
    parser.add_argument('--depth_topic', default='/zed/zed_node/depth/depth_registered')
    parser.add_argument('--camera_info_topic', default='/zed/zed_node/rgb/camera_info')
    parser.add_argument('--marker_topic', default='/semantic/markers')
    parser.add_argument('--frame_id', default='map')
    parser.add_argument('--camera_frame', default='zed_left_camera_optical_frame')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--classes', default='')
    parser.add_argument('--dbscan_eps', type=float, default=0.6)
    parser.add_argument('--dbscan_min_samples', type=int, default=1)
    parser.add_argument('--max_points_per_class', type=int, default=1000)
    parser.add_argument('--publish_rate', type=float, default=2.0)
    parser.add_argument('--marker_lifetime', type=float, default=0.0)
    parser.add_argument('--text_scale', type=float, default=0.25)

    args = parser.parse_args()

    rclpy.init()
    node = YoloSemanticDbscanForced(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
