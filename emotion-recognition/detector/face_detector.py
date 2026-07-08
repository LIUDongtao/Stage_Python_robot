
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
detector/face_detector.py

Face detector for emotion recognition.

Input:
    - BGR image from OpenCV / ZED

Output:
    - list of face dictionaries:
        {
            "bbox": (x1, y1, x2, y2),
            "face_roi": face image
        }
"""

import cv2
from typing import List, Dict, Tuple


class FaceDetector:
    def __init__(
        self,
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (40, 40)
    ):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError(f"Cannot load Haar cascade: {cascade_path}")

    def detect(self, frame) -> List[Dict]:
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        results = []

        h, w = frame.shape[:2]

        for (x, y, fw, fh) in faces:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + fw)
            y2 = min(h, y + fh)

            face_roi = frame[y1:y2, x1:x2].copy()

            if face_roi.size == 0:
                continue

            results.append({
                "bbox": (x1, y1, x2, y2),
                "face_roi": face_roi
            })

        return results

    @staticmethod
    def draw(frame, faces, color=(0, 255, 0)):
        vis = frame.copy()

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face["bbox"]

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                vis,
                f"Face {i + 1}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return vis
