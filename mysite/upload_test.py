import requests
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(
        "http://127.0.0.1:8000/upload/",
        files={"image": img_encoded.tobytes()}
    )
