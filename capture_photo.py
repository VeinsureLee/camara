import cv2


def capture_photo(save_path="photo.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        cv2.imshow("Press 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite(save_path, frame)
            print(f"照片已保存: {save_path}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_photo()
