
"""
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
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = FER().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def preprocess(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return None

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))

        gray = gray.astype(np.float32) / 255.0
        gray = (gray - 0.5) / 0.5

        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        return tensor

    @torch.no_grad()
    def predict(self, face_roi):
        tensor = self.preprocess(face_roi)
        if tensor is None:
            return "Unknown", 0.0
            
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        emotion = EMOTION_LABELS[pred.item()]
        confidence = conf.item()

        return emotion, confidence
