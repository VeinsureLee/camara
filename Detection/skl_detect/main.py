from skl_detect.skl_detect import img_read
from skl_detect.skl_detect import test_photo_root_path, camera_detect_skl, photo_skl
import cv2

if __name__ == '__main__':
    img = cv2.imread("./photo/test01.png")
    cv2.imshow('img', img)
    cv2.waitKey(0)

    img_pre = photo_skl(img)
    cv2.imshow('img_pre', img_pre)
    cv2.waitKey(0)
    cv2.imwrite("./img_pre.png", img_pre)
    cv2.destroyAllWindows()
