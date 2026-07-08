
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2

from camera.zed_camera import ZEDCamera
from detector.face_detector import FaceDetector
from emotion.emotion_model import EmotionRecognizer


def main():

    print("====================================")
    print("Emotion Recognition")
    print("Press q to quit")
    print("====================================")

    camera = ZEDCamera()

    detector = FaceDetector()

    emotion_model = EmotionRecognizer(
        model_path="models/FER2013-Resnet9.pth"
    )

    try:

        while True:

            ret, frame, depth = camera.grab()

            if not ret:
                continue

            faces = detector.detect(frame)

            for face in faces:

                x1, y1, x2, y2 = face["bbox"]

                roi = face["face_roi"]

                emotion, score = emotion_model.predict(roi)

                color = (0,255,0)

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    color,
                    2
                )

                label = f"{emotion} {score:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            cv2.imshow("Emotion Recognition", frame)

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:

        camera.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
