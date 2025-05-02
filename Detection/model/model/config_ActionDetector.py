import logging


class ActionDetectorConfig:
    def __init__(
            self,
            labels=None,
            base_path='data/coordinates/total',  # test data base save path
            model_path='model_pretrained/best_pose_cnn_model.pth',  # pretrained model path
            save_path='model_pretrained/best_pose_cnn_model.pth',
            num_epochs=500,
            root_total_input_folder='data/photo/total',
            root_total_output_folder='data/coordinates/total',
            root_test_input_folder='data/photo/test',
            root_test_output_folder='data/coordinates/test',
            train_log_path='logs/train.log',
    ):
        if labels is None:
            labels = ['climb', 'people fall down', 'people lay down', 'people stand', 'people sit down']
        self.labels = labels
        self.base_path = base_path
        self.model_path = model_path
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.root_total_input_folder = root_total_input_folder
        self.root_total_output_folder = root_total_output_folder
        self.root_test_input_folder = root_test_input_folder
        self.root_test_output_folder = root_test_output_folder
        self.train_log_path = train_log_path


if __name__ == '__main__':
    # 默认配置
    default_config = ActionDetectorConfig()
