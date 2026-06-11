## YOLO11n + ZED 2i 3D Object Detection

### Objective

This part explains how to run a YOLO11n ONNX model with the ZED 2i camera using the ZED SDK internal inference engine.
The goal is to perform real-time object detection, distance estimation, 3D tracking and Bird's Eye View visualization.

---

### 1. Go to the project directory

```bash
cd ~/Desktop/python_tensorrt_yolo_onnx_native
```

---

### 2. Check the ZED Python API

```bash
python3 -c "import pyzed.sl as sl; print('ZED OK')"
```

Expected output:

```bash
ZED OK
```

---

### 3. Install Ultralytics YOLO

```bash
pip install ultralytics
```

If ONNX export dependencies are missing, install them:

```bash
pip install onnx onnxruntime onnxslim
```

---

### 4. Check YOLO installation

```bash
python3 -c "from ultralytics import YOLO; print('YOLO OK')"
```

Expected output:

```bash
YOLO OK
```

---

### 5. Export YOLO11n to ONNX

The ZED SDK internal YOLO detector requires an ONNX model.
Therefore, the PyTorch model `yolo11n.pt` must be exported to `yolo11n.onnx`.

```bash
yolo export model=yolo11n.pt format=onnx opset=12 simplify=True
```

After export, check that the ONNX file has been created:

```bash
ls -lh *.onnx
```

Expected file:

```bash
yolo11n.onnx
```

---

### 6. Run YOLO11n with ZED SDK

```bash
python3 custom_internal_detector.py --custom_onnx yolo11n.onnx
```

During the first execution, the ZED SDK optimizes the ONNX model for the GPU:

```bash
Optimizing yolo11n ...
```

This operation is performed only once and may take several minutes.

---

###  Run YOLO11n with Ros2 RTABMap + rviz2

```bash
cd ~/Desktop/python_tensorrt_yolo_onnx_native

python3 custom_internal_detector_ros_coco.py \
--custom_onnx yolo11n.onnx \
--ros_frame zed_camera_link
```


### 7. Expected result

The program displays:

* RGB camera image
* YOLO object detection boxes
* Object class names
* Distance estimation in meters
* Object tracking ID
* Bird's Eye View visualization

Example output on the image:

```text
ID 73
chair
2.8M
```

---

### 8. Processing pipeline

```text
ZED 2i Camera
      ↓
RGB Image
      ↓
YOLO11n ONNX Model
      ↓
Object Detection
      ↓
ZED Depth Estimation
      ↓
3D Position Estimation
      ↓
Object Tracking
      ↓
Bird's Eye View
```

---

### 9. Notes

YOLO is used to identify known object categories such as person, chair, table, bottle, laptop, etc.

However, some obstacles such as walls, doors or large flat surfaces may not be detected by YOLO as specific object classes.
For safe robot navigation, obstacle avoidance should also rely on ZED depth information and spatial perception.

In this project:

* YOLO provides semantic object detection.
* ZED SDK provides depth, distance estimation and 3D localization.
* Bird's Eye View provides a 2D top-view representation useful for robot navigation.
