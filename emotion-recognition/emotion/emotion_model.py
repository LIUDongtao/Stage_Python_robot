

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emotion/emotion_model.py

FER2013 PyTorch Emotion Recognition Model

Input:
    - face ROI from OpenCV image

Output:
    - emotion label
    - confidence
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from emotion.FERModel import FERModel, classes
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


class FER(nn.Module):
    """
    Simple CNN for FER2013.
    Input: 1x48x48 grayscale face image.
    Output: 7 emotion classes.
    """

    def __init__(self):
        super(FER, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 12 * 12, 1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)

        return x



class EmotionRecognizer:
    def __init__(self, model_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.model = FERModel(1, 7).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def preprocess(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))

        gray = gray.astype(np.float32) / 255.0

        tensor = torch.from_numpy(gray)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        return tensor

    @torch.no_grad()
    def predict(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return "Unknown", 0.0

        tensor = self.preprocess(face_roi)

        outputs = self.model(tensor)
        probs = F.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, dim=1)

        label = classes[pred.item()]
        score = conf.item()

        return label, score
