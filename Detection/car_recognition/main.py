from Detection.car_recognition.car_recognition.PlateDetector import plate_detector_with_overlay
from Detection.img_show import show_auto_scaled_image
from Detection.car_recognition.car_recognition.config_CarDetector import CarDetectorConfig
import cv2


def main(config=CarDetectorConfig()):
    config.img = cv2.imread('./CarIdentityData/car_test/bupt_2.jpg')
    show_auto_scaled_image(config.img)
    draw_img, plates_info = plate_detector_with_overlay(config.img, config.plate_model_path,
                                                        config.char_model_path, False, True)
    show_auto_scaled_image(draw_img)


if __name__ == '__main__':
    main()
