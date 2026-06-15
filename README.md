# Stage_Python_robot

Projet de stage ESIGELEC visant à développer une plateforme d’apprentissage et d’expérimentation en intelligence artificielle, Python et robotique autonome.

## document lien https://www.stereolabs.com/docs/embedded/zed-box
## demo lien https://github.com/stereolabs/zed-sdk/tree/master
## SDK download: https://www.stereolabs.com/en-fr/developers/release
## YOLO document:https://docs.ultralytics.com/#where-to-start
## YOLO-ZED:https://github.com/stereolabs/zed-yolo
## Membres

- Ran Pan – Détection des situations d'urgence
- Zhen Yang – Navigation autonome entre A et B
- Liu Dongtao – Détection d'obstacles en temps réel

## Objectifs

1. Développer les compétences en IA et Python.
2. Construire une plateforme pédagogique pour les TP de l'ESIGELEC.
3. Optimiser les modules et favoriser leur interopérabilité.




-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
## Configuration du matériel : ZED Box
Démarrage du ZED Box

1. Alimentation
Connecter l’alimentation du ZED Box.
Après la mise sous tension, le voyant vert (Power Status) doit s’allumer, indiquant que le système est alimenté correctement.

2. Connexion de l’écran
Connecter un écran au port HDMI du ZED Box afin d’accéder à l’interface Ubuntu.

3. Connexion réseau
Connecter un câble Ethernet entre le ZED Box et l’ordinateur (ou le réseau local) pour permettre la communication et l’accès à distance via SSH.

4. Clavier et souris
Brancher un clavier et une souris sur les ports USB du ZED Box.

5. Démarrage
Une fois toutes les connexions effectuées, démarrer le système et se connecter à Ubuntu.


## Configuration matérielle

### Plateforme embarquée

Le projet est développé sur une plateforme embarquée **Stereolabs ZED Box** intégrant :

- NVIDIA Jetson Orin NX
- Architecture ARM64 (aarch64)
- NVIDIA JetPack 6.0
- Ubuntu 22.04.4 LTS
- CUDA 12.2
- ZED SDK préinstallé

### Caméra utilisée

- Stereolabs ZED 2i
- Caméra stéréoscopique (vision binoculaire)
- Acquisition RGB et profondeur (Depth)
- Compatible avec le SDK ZED pour la détection d'obstacles en temps réel

### Logiciels installés

Le système dispose des outils suivants :

- ZED Explorer
- ZED Depth Viewer
- ZED Calibration
- ZED Diagnostic
- ZED Sensor Viewer
- ZEDfu
- ZED Media Server

### Informations système

| Élément | Valeur |
|----------|----------|
| Plateforme | Stereolabs ZED Box |
| Module GPU | NVIDIA Jetson Orin NX |
| Architecture | ARM64 (aarch64) |
| Système | Ubuntu 22.04.4 LTS |
| JetPack | 6.0 |
| L4T | R36.3 |
| Python | 3.10.12 |
| CUDA | 12.2 |
| SDK Vision | ZED SDK |

### Informations de connexion par défaut

Pour la configuration initiale du ZED Box :

Nom d'utilisateur : user
Mot de passe : admin

## IDE Configuration (VS Code)

### Why VS Code

The project is mainly developed in Python and will later integrate:

* ZED SDK
* OpenCV
* YOLO
* ROS2

Visual Studio Code is recommended because it provides:

* Python development support
* Integrated terminal
* Git integration
* Remote development capabilities
* ROS2 extensions
* YOLO/OpenCV development support

---

### Install VS Code on Jetson (ARM64)

#### 1. Install required tools

```bash
sudo apt install wget gpg
```

#### 2. Import Microsoft's signing key

```bash
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg
```

#### 3. Add the Microsoft repository

```bash
echo "Types: deb
URIs: https://packages.microsoft.com/repos/code
Suites: stable
Components: main
Architectures: amd64,arm64,armhf
Signed-By: /usr/share/keyrings/microsoft.gpg" | sudo tee /etc/apt/sources.list.d/vscode.sources
```

#### 4. Update package list

```bash
sudo apt update
```

#### 5. Install VS Code

```bash
sudo apt install code
```

#### 6. Launch VS Code

```bash
code
```

or from the Ubuntu application menu.

---

### Notes

The Jetson platform uses the ARM64 architecture (`aarch64`).

Verify the architecture using:
## VS Code – Création du trousseau de clés (Keyring)

Lors de la première connexion à GitHub ou GitHub Copilot dans VS Code sous Linux, une fenêtre peut apparaître pour créer un **Keyring** (« Default Keyring »).

### Solution

Saisissez votre **mot de passe de session Linux**, puis cliquez sur **Continue**.

### Pourquoi ?

Le Keyring permet de stocker de manière sécurisée les identifiants et jetons d’authentification utilisés par VS Code, GitHub et GitHub Copilot.

### Remarque

Si vous cliquez sur **Cancel**, VS Code risque de ne pas enregistrer vos informations de connexion.




-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------

## ZED 2i Camera Processing Pipeline

This project uses the ZED 2i stereo camera for real-time image acquisition, object detection, distance measurement, and robot control.

### 1. Hardware Connection

Current testing setup:

```text
ZED 2i
  ↓ USB 3.0
ZED Box
  ↓ Ethernet
Network / Remote PC
```

Future robot setup:

```text
ZED 2i
  ↓ USB 3.0
Jetson on the vehicle
  ↓ WiFi
Network / Remote PC
```

The ZED 2i itself does not have an IP address. The IP address belongs to the ZED Box or the Jetson computer connected to the camera.

---

### 2. ZED SDK Data Flow

The ZED SDK is responsible for opening the camera and retrieving image, depth, point cloud, IMU, and object detection data.

For live camera input:

```text
ZED 2i
  ↓
ZED SDK
  ↓
sl.Mat
  ↓ get_data()
NumPy array
```

For SVO replay:

```text
SVO file
  ↓
ZED SDK
  ↓
sl.Mat
  ↓ get_data()
NumPy array
```

SVO is mainly used as a recorded ZED video format for testing and replay. In real-time robot operation, the camera is normally accessed directly through USB.

---

### 3. OpenCV Processing

OpenCV is optional in this pipeline.

The ZED SDK stores image data in `sl.Mat`. By calling:

```python
frame = image.get_data()
```

the `sl.Mat` image is converted into a NumPy array.

OpenCV can then be used for optional image processing:

```text
NumPy array
  ↓
OpenCV
  ↓
resize / color conversion / drawing boxes / saving images
```

Example uses of OpenCV:

```python
frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
cv2.imshow("ZED Image", frame_bgr)
cv2.imwrite("capture.jpg", frame_bgr)
```

However, OpenCV is not required just to send the image to YOLO. YOLO can directly process the NumPy array.

---

### 4. YOLO Object Detection

YOLO receives the image as a NumPy array:

```text
ZED SDK
  ↓
sl.Mat
  ↓ get_data()
NumPy array
  ↓
YOLO
  ↓
Detection results
```

YOLO is responsible for recognizing objects such as people, vehicles, traffic signs, or obstacles.

Example:

```python
results = model(frame)
```

The ZED depth data can also be used together with YOLO bounding boxes to estimate the distance between the robot and detected objects.

---

### 5. ROS2 Robot Integration

In the future robot system, ROS2 can be used to connect all modules together:

```text
ZED 2i
  ↓
ZED SDK
  ↓
ZED ROS2 Wrapper
  ↓
ROS2 Topics
  ↓
YOLO Node
  ↓
Decision Node
  ↓
Control Module
  ↓
Vehicle Motion
```

ROS2 allows the Jetson on the vehicle to publish camera images, depth maps, point clouds, odometry, and detection results over WiFi.

A remote computer can view the camera and robot status using:

```bash
rviz2
```

or:

```bash
ros2 run rqt_image_view rqt_image_view
```

---

### 6. Control Logic

After YOLO detects an object, the detection result can be published to a ROS2 topic.

Example logic:

```text
YOLO detects obstacle
  ↓
Get distance from ZED depth
  ↓
Decision node checks safety distance
  ↓
Control module sends velocity command
  ↓
Vehicle stops, slows down, or avoids obstacle
```

Example behavior:

```text
If person detected within 2 meters:
    stop the vehicle

If obstacle detected ahead:
    slow down or avoid

If path is clear:
    continue moving
```

---

### 7. Summary

The full system can be understood as:

```text
ZED SDK:
    Get image, depth, point cloud, and camera data

OpenCV:
    Optional image processing and visualization

YOLO:
    Object detection from NumPy image data

ROS2:
    Communication between camera, detection, navigation, and control modules

Control Module:
    Sends movement commands to the vehicle
```

Final target architecture:

```text
ZED 2i → Jetson → ZED SDK → ROS2 → YOLO → Decision Node → Vehicle Control
```
# ROS2 Integration with ZED 2i

## Environment

* Platform: NVIDIA Jetson AGX Orin
* Operating System: Ubuntu 22.04
* ROS2 Distribution: Humble
* Camera: ZED 2i
* ZED SDK Version: 4.2.x

---

## Verify ROS2 Installation

Check whether ROS2 is installed:

```bash
ls /opt/ros
```

Expected output:

```text
humble
```

Load ROS2 environment:

```bash
source /opt/ros/humble/setup.bash
```

Verify ROS2:

```bash
printenv ROS_DISTRO
```

Expected output:

```text
humble
```

---

## Install Required Tools

```bash
sudo apt update
sudo apt install python3-rosdep python3-colcon-common-extensions -ysource /opt/ros/humble/setup.bash
source ~/ros2_ws/install/local_setup.bash

ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

Initialize rosdep:

```bash
sudo rosdep init
rosdep update
```

---

## Create ROS2 Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

Clone the ZED ROS2 Wrapper:

```bash
git clone https://github.com/stereolabs/zed-ros2-wrapper.git
```

---

## Select Compatible Wrapper Version

The latest wrapper is not compatible with ZED SDK 4.2.x.

Switch to the compatible version:

```bash
cd ~/ros2_ws/src/zed-ros2-wrapper
git checkout humble-v4.2.5
```

---

## Build the Workspace

```bash
cd ~/ros2_ws

rm -rf build install log

source /opt/ros/humble/setup.bash

rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release
```

Successful build:

```text
Summary: 3 packages finished

## Source the Workspace

```bash
source ~/ros2_ws/install/local_setup.bash
```

To automatically load ROS2 at startup:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```


----------------------------------------------------------------------------------------------------------------------------------------
## Launch ZED 2i ROS2 Wrapper
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/local_setup.bash

ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
The first launch may take a few minutes while initializing internal components.

## Launch RTAB-Map with ZED Camera

First source the ROS 2 and workspace environments:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/local_setup.bash
```

Then launch RTAB-Map:

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
  frame_id:=zed_camera_link \
  subscribe_odom:=true \
  odom_topic:=/zed/zed_node/odom \
  visual_odometry:=false \
  rgb_topic:=/zed/zed_node/rgb/image_rect_color \
  depth_topic:=/zed/zed_node/depth/depth_registered \
  camera_info_topic:=/zed/zed_node/rgb/camera_info \
  approx_sync:=true \
  topic_queue_size:=30 \
  sync_queue_size:=30 \
  wait_for_transform:=1.0 \
  grid:=true \
  Grid/FromDepth:=true \
  Grid/3D:=false
```
--------------------------------------------------------------------------------------------------------------
## Check Available Topics

Open a new terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/local_setup.bash
```

List topics:

```bash
ros2 topic list
```

1. 确认图像和深度有频率

新开终端：

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/local_setup.bash

ros2 topic hz /zed/zed_node/rgb/image_rect_color

再执行：

ros2 topic hz /zed/zed_node/depth/depth_registered

如果有 average rate，说明图像正常。

2. 确认里程计在动
ros2 topic echo /zed/zed_node/odom

拿起 ZED 2i 轻轻移动，看 position 数值会不会变化。

## Applications

The ROS2 integration provides:

* RGB image streaming
* Depth perception
* Point cloud generation
* Visual odometry
* Pose tracking
* 3D object localization
* SLAM support
* Navigation support


### Zed2i + RTABmap +rviz2 + yolo11n
# Launch Instructions

## Terminal 1 - Start ZED2i Camera

```bash
source /opt/ros/humble/setup.bash

ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

Verify ZED topics:

```bash
ros2 topic list | grep zed
```

---

## Terminal 2 - Start RTAB-Map

```bash
source /opt/ros/humble/setup.bash

ros2 launch rtabmap_launch rtabmap.launch.py \
frame_id:=zed_left_camera_frame \
subscribe_odom:=true \
odom_topic:=/zed/zed_node/odom \
visual_odometry:=false \
rgb_topic:=/zed/zed_node/rgb/image_rect_color \
depth_topic:=/zed/zed_node/depth/depth_registered \
camera_info_topic:=/zed/zed_node/rgb/camera_info \
approx_sync:=true \
grid:=true \
Grid/FromDepth:=true \
Grid/3D:=false \
delete_db_on_start:=true
```

Verify RTAB-Map topics:

```bash
ros2 topic list | grep rtabmap
```

---

## Terminal 3 - Start YOLO11n Semantic Detection

Enter the project directory:

```bash
cd ~/Desktop/python_tensorrt_yolo_onnx_native
```

Run YOLO11n:

```bash
python3 yolo_ros_subscriber.py \
--model yolo11n.pt \
--image_topic /zed/zed_node/rgb/image_rect_color \
--depth_topic /zed/zed_node/depth/depth_registered \
--camera_info_topic /zed/zed_node/rgb/camera_info \
--frame_id map
```

Verify marker publishing:

```bash
ros2 topic hz /yolo/markers
```

---

## Terminal 4 - Start RViz2

```bash
source /opt/ros/humble/setup.bash

rviz2
```

### RViz Configuration

Set **Fixed Frame**:

```text
map
```

Add the following displays:

| Display Type | Topic |
|-------------|--------|
| Map | /rtabmap/grid_prob_map |
| PointCloud2 | /rtabmap/cloud_map |
| MarkerArray | /yolo/markers |

---

## System Overview

```text
ZED2i
  │
  ▼
RGB + Depth
  │
  ▼
RTAB-Map
  │
  ▼
2D / 3D Map
  │
  ▼
YOLO11n
  │
  ▼
Semantic Object Detection
  │
  ▼
TF Transform (Camera → Map)
  │
  ▼
MarkerArray
  │
  ▼
RViz Semantic Map
```


### 降低建图压力

ros2 launch zed_wrapper zed_camera.launch.py \
camera_model:=zed2i \
resolution:=VGA




ros2 launch rtabmap_launch rtabmap.launch.py \
  frame_id:=zed_left_camera_frame \
  subscribe_odom:=true \
  odom_topic:=/zed/zed_node/odom \
  visual_odometry:=false \
  rgb_topic:=/zed/zed_node/rgb/image_rect_color \
  depth_topic:=/zed/zed_node/depth/depth_registered \
  camera_info_topic:=/zed/zed_node/rgb/camera_info \
  approx_sync:=true \
  wait_imu_to_init:=false \
  rtabmap_viz:=true \
  rviz:=false \
  database_path:=~/.ros/rtabmap.db \
  delete_db_on_start:=true \
  Rtabmap/DetectionRate:=2 \
  RGBD/LinearUpdate:=0.05 \
  RGBD/AngularUpdate:=0.05
