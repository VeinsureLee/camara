import cv2


class CarDetectorConfig:
    def __init__(
            self,
            # 模型训练与测试部分：
            data_char_dir='CarIdentityData/cnn_char_train',
            test_char_dir='CarIdentityData/cnn_char_test/1-6.jpg',
            train_model_char_path='CarIdentityData/model/char_recognize/plate_cnn_model.pth',
            model_char_path='CarIdentityData/model/char_recognize/plate_cnn_model.pth',

            img_test_dir='CarIdentityData/car_test/1.jpg',

            data_plate_dir='CarIdentityData/cnn_plate_train',
            test_plate_dir='CarIdentityData/cnn_plate_test/test01_has.jpg',
            train_model_plate_path='CarIdentityData/model/plate_recognize/plate_cnn_model.pth',
            model_plate_path='CarIdentityData/model/plate_recognize/plate_cnn_model.pth',

            # Detector配置部分：
            img=cv2.imread('CarIdentityData/car_test/bupt_2.jpg'),
            plate_model_path='CarIdentityData/model/plate_recognize/plate_cnn_model.pth',
            char_model_path='CarIdentityData/model/char_recognize/char_cnn_model.pth'
    ):
        # 模型训练与测试部分：
        self.data_char_dir = data_char_dir
        self.test_char_dir = test_char_dir
        self.train_model_char_path = train_model_char_path
        self.model_char_path = model_char_path

        self.img_test_dir = img_test_dir

        self.data_plate_dir = data_plate_dir
        self.test_plate_dir = test_plate_dir
        self.train_model_plate_path = train_model_plate_path
        self.model_plate_path = model_plate_path

        # Detector配置部分：
        self.img = img
        self.plate_model_path = plate_model_path
        self.char_model_path = char_model_path


if __name__ == '__main__':
    # 默认配置
    default_config = CarDetectorConfig()
