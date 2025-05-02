import cv2
from .config_HumanDetector import HumanDetectorConfig
from . import load_model


def detect_human(img=None, config=HumanDetectorConfig()):
    if img is None:
        img = config.img_path
    mdl = load_model(config)
    results = mdl(img, classes=[0], verbose=False)
    return results[0]


def img_show_human(config=HumanDetectorConfig()):
    img = config.img_path
    results = detect_human(config)
    if isinstance(img, str):
        img = cv2.imread(img)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Person: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


