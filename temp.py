from Detection.car_recognition.car_recognition.PlateDetector import plate_detector, plate_detector_with_overlay
from Detection.detect_draw import draw_detection, run_camera
import cv2
from Detection.model.model.config_ActionDetector import ActionDetectorConfig
from Detection.human_detect.human_detect.config_HumanDetector import HumanDetectorConfig
from Detection.human_face_recognition.human_face_recognition.face_detect import load_face_database
# if __name__ == '__main__':
#     img_path = './Detection/car_recognition/CarIdentityData/car_test/1.jpg'
#     plate_model_path = './Detection/car_recognition/CarIdentityData/model/plate_recognize/plate_cnn_model.pth'
#     char_model_path = './Detection/car_recognition/CarIdentityData/model/char_recognize/char_cnn_model.pth'
#     img = cv2.imread(img_path)
#     cv2.imshow('img', plate_detector_with_overlay(img, plate_model_path, char_model_path))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
action_model_config = ActionDetectorConfig(
    labels=None,
    base_path=None,  # test database save path
    model_path='Detection/model/model_pretrained/best_pose_cnn_model.pth',  # pretrained model path
    save_path=None,
    num_epochs=None,
    root_total_input_folder=None,
    root_total_output_folder=None,
    root_test_input_folder=None,
    root_test_output_folder=None,
    train_log_path=None,
)


human_model_config = HumanDetectorConfig(
    img_path=None,
    model_path='Detection/human_detect/model/yolov8n.pt'
)
if __name__ == '__main__':
    encodings, names = load_face_database('Detection/human_face_recognition/facebase')
    run_camera(encodings, names,
               True, True, True,
               action_model_config, human_model_config)