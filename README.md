# Stage_Python_robot

Projet de stage ESIGELEC visant à développer une plateforme d’apprentissage et d’expérimentation en intelligence artificielle, Python et robotique autonome.

## document lien https://www.stereolabs.com/docs/embedded/zed-box
## demo lien https://github.com/stereolabs/zed-sdk/tree/master
## SDK download: https://www.stereolabs.com/en-fr/developers/release
## YOLO document:https://docs.ultralytics.com/#where-to-start
## YOLO-ZED:https://github.com/stereolabs/zed-yolo

# Stage_Python_robot

> **ESIGELEC Internship Project**  
> An educational robotics platform integrating **ZED2i**, **ROS2**, **RTAB-Map**, **YOLO11**, and **AI** for autonomous robot perception.

---

# Table of Contents

- Project Overview
- Project Objectives
- Repository Structure
- Hardware Platform
- Software Environment
- System Architecture
- Installation
- GPU Configuration
- ROS2 Integration
- RTAB-Map SLAM
- YOLO11 Detection
- Launch Instructions
- ROS2 Topics
- Applications
- Performance
- Troubleshooting
- Future Work
- References

---

# Project Overview

Stage_Python_robot is an internship project developed at ESIGELEC.

The objective is to build a modular robotic perception platform based on:

- ZED2i Stereo Camera
- NVIDIA Jetson Orin NX
- ROS2 Humble
- RTAB-Map SLAM
- YOLO11 Object Detection
- Human Pose Estimation

The project is intended for education, experimentation, and future autonomous navigation research.

---

# Project Objectives

- Learn Python for robotics
- Learn Artificial Intelligence algorithms
- Build a reusable robotics platform
- Integrate computer vision with ROS2
- Develop semantic mapping capabilities
- Prepare for autonomous navigation

---

# Repository Structure

```text
Stage_Python_robot
│
├── README.md
├── launch/
├── scripts/
├── models/
├── docs/
├── images/
└── ros2_ws/
```

---

# Hardware Platform

## Embedded Platform

- NVIDIA Jetson Orin NX
- ARM64 (aarch64)
- Ubuntu 22.04 LTS
- JetPack 6.0
- CUDA 12.2

## Camera

- Stereolabs ZED2i
- Stereo RGB Camera
- Depth Camera
- Visual Odometry
- IMU

## Default Login

Username

```text
user
```

Password

```text
admin
```

---

# Software Environment

| Component | Version |
|------------|---------|
| Ubuntu | 22.04 |
| ROS2 | Humble |
| JetPack | 6.0 |
| CUDA | 12.2 |
| Python | 3.10 |
| ZED SDK | 4.2.x |
| OpenCV | Installed |
| PyTorch | Jetson Version |
| Ultralytics | YOLO11 |

---

# System Architecture

```text
                    ZED2i Camera
                         │
              RGB Image + Depth Image
                         │
                     ZED SDK
                         │
          ┌──────────────┴──────────────┐
          │                             │
      RTAB-Map                     YOLO11
          │                             │
      2D / 3D Map               Object Detection
          │                             │
          └────────────TF───────────────┘
                         │
                 Semantic Mapping
                         │
                       RViz2
                         │
                 Autonomous Robot
```

---

# Installation

## Install ROS2

```bash
sudo apt update
sudo apt install ros-humble-desktop
```

## Create Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

## Clone ZED Wrapper

```bash
git clone https://github.com/stereolabs/zed-ros2-wrapper.git
```

```bash
cd zed-ros2-wrapper
git checkout humble-v4.2.5
```

## Build

```bash
cd ~/ros2_ws

source /opt/ros/humble/setup.bash

rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release
```

---

# GPU Configuration

Pipeline

```text
ZED2i
   │
RGB Image
   │
PyTorch
   │
CUDA 12.2
   │
Jetson GPU
   │
YOLO11
```

Check CUDA

```bash
python3 -c "import torch;print(torch.cuda.is_available())"
```

Expected

```text
True
```

---

# ROS2 Integration

Modules

- ZED ROS2 Wrapper
- RTAB-Map
- RViz2
- TF
- YOLO ROS Node

Data Flow

```text
Camera
 ↓
ROS2 Topics
 ↓
YOLO
 ↓
Markers
 ↓
RViz
```

---

# RTAB-Map SLAM

RTAB-Map is responsible for

- Visual SLAM
- Loop Closure
- Occupancy Grid
- Point Cloud Mapping

Launch

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
rgb_topic:=/zed/zed_node/rgb/image_rect_color \
depth_topic:=/zed/zed_node/depth/depth_registered \
camera_info_topic:=/zed/zed_node/rgb/camera_info \
odom_topic:=/zed/zed_node/odom \
frame_id:=zed_camera_link \
approx_sync:=true \
subscribe_odom_info:=false
```

---

# YOLO11 Detection

Current implementation

- Object Detection
- Human Detection
- Pose Estimation
- Distance Estimation (with ZED depth)

Pipeline

```text
RGB Image
 ↓
YOLO11
 ↓
Bounding Boxes
 ↓
Depth Query
 ↓
3D Position
```

---

# Launch Instructions

## Terminal 1

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

---

## Terminal 2

```bash
source /opt/ros/humble/setup.bash

ros2 launch rtabmap_launch rtabmap.launch.py \
frame_id:=zed_left_camera_frame \
subscribe_odom:=true \
visual_odometry:=false \
odom_topic:=/zed/zed_node/odom
```

---

## Terminal 3

```bash
cd ~/Desktop/python_tensorrt_yolo_onnx_native

python3 yolo_ros_subscriber.py \
--model yolo11n.pt
```

---

## Terminal 4

```bash
rviz2
```

---

# ROS2 Topics

| Topic | Description |
|--------|-------------|
| /zed/zed_node/rgb/image_rect_color | RGB Image |
| /zed/zed_node/depth/depth_registered | Depth Image |
| /zed/zed_node/odom | Visual Odometry |
| /tf | Coordinate Transform |
| /yolo/markers | MarkerArray |
| /rtabmap/cloud_map | Point Cloud |
| /rtabmap/grid_prob_map | Occupancy Grid |

---

# Applications

- Object Detection
- Human Pose Estimation
- Semantic Mapping
- Visual SLAM
- Obstacle Detection
- Robot Localization

---

# Performance

Coming Soon

Future benchmarks

- FPS
- GPU Usage
- CPU Usage
- Memory Usage
- RTAB-Map Performance
- Detection Accuracy

---

# Troubleshooting

## torch.cuda.is_available() == False

Install the correct NVIDIA PyTorch wheel.

---

## Camera Stream Failed

- Close ZED Explorer
- Check USB connection
- Restart camera

---

## RTAB-Map does not update

Check

```bash
ros2 topic echo /zed/zed_node/odom
```

---

## Marker not displayed

Check

- TF
- RViz Fixed Frame
- Marker Topic

---

## GPU usage is 0%

The GPU is only used during neural network inference.
Idle periods are normal.

---

# Future Work

Completed

- ZED2i Integration
- ROS2 Communication
- RTAB-Map
- YOLO11 Detection
- Human Pose Estimation

Planned

- TensorRT Optimization
- Navigation2
- Obstacle Avoidance
- Face Recognition
- Facial Expression Recognition
- Human Following
- Voice Interaction

---

# References

- ZED SDK Documentation
- ZED ROS2 Wrapper
- RTAB-Map
- ROS2 Humble Documentation
- Ultralytics YOLO
