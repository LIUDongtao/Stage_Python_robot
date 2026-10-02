import cv2
import time

from camera.zed_camera import ZEDCamera
from detector.face_detector import FaceDetector
from emotion.emotion_model import EmotionRecognizer


# Terminal output interval in seconds.
# The detection still runs continuously, but results are printed
# only once per second to avoid flooding the terminal.
PRINT_INTERVAL = 1.0


def main():
    print("====================================")
    print("ZED Emotion Recognition")
    print("====================================")
    print("Starting camera and models...")
    print()

    camera = ZEDCamera()
    detector = FaceDetector()

    emotion_model = EmotionRecognizer(
        model_path="models/FER2013-Resnet9.pth"
    )

    print("System ready")
    print("Press 'q' to quit")
    print()

    # Store the time of the previous terminal output.
    last_print_time = 0.0

    try:
        while True:
            ret, frame, depth = camera.grab()

            if not ret:
                continue

            # Detect faces in the current frame.
            faces = detector.detect(frame)

            # Determine whether it is time to print new results.
            now = time.time()
            should_print = now - last_print_time >= PRINT_INTERVAL

            if should_print:
                last_print_time = now
                print("\n====================================")

                if not faces:
                    print("No face detected")
                else:
                    print(f"Detected faces: {len(faces)}")

            # Process every detected face.
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face["bbox"]
                roi = face["face_roi"]

                # Predict facial emotion.
                emotion, score = emotion_model.predict(roi)

                # Print the emotion result in the terminal.
                if should_print:
                    print("------------------------------------")
                    print(f"Face       : {i + 1}")
                    print(f"Emotion    : {emotion}")
                    print(f"Confidence : {score:.2f}")

                # Green bounding box.
                color = (0, 255, 0)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Display emotion and confidence on the image.
                label = f"{emotion} {score:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            # Display the processed camera image.
            cv2.imshow("Emotion Recognition", frame)

            # Press q to stop the program.
            key = cv2.waitKey(1)

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Camera closed")
        print("Program stopped")


if __name__ == "__main__":
    main()
