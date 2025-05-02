import cv2
from .config_HumanDetector import HumanDetectorConfig
from . import load_model


def camera_show(config=HumanDetectorConfig()):
    mdl = load_model(config)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        results = mdl(frame, classes=[0], verbose=False)
        annotated_frame = results[0].plot()
        cv2.imshow("Live", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()