## Models

The project uses the following pretrained models:

| Module | Model | Description |
|--------|-------|-------------|
| Obstacle Detection | **YOLO11n** | Used for real-time obstacle detection. The model is pretrained on the **COCO dataset**, enabling the detection of 80 common object categories. |
| Human Pose Estimation | **YOLO11n-Pose** | Used for real-time human pose estimation. |
| Emotion Recognition | **FER-PyTorch** | Used for real-time facial emotion recognition. The corresponding project link is available on the repository homepage. |

---

## Usage

### 1. Obstacle Detection with ZED Camera

Run:

```bash
python yolo_zed_obstacle.py
```

This script detects the **three closest obstacles** using the ZED camera and the **YOLO11n** object detection model.

---

### 2. Human Pose Estimation

Run:

```bash
python zed_yolo_pose_v2_fast.py
```

This script performs **real-time human pose estimation** using the ZED camera and the **YOLO11n-Pose** model.

---

### 3. Mapping YOLO Detections to RTAB-Map

Run:

```bash
python python_tensorrt_yolo_onnx_native/yolo_ros_subscriber_final.py
```

This script subscribes to the YOLO detection results and **projects the detected obstacles directly onto the RTAB-Map**, allowing the detected objects to be visualized in the generated map.

---

### 4. Emotion Recognition

The `emotion-recognition` module performs **real-time facial emotion recognition** using the **FER-PyTorch** model.

Please refer to the documentation in the `emotion-recognition` directory for setup and execution instructions.
