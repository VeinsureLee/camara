import mediapipe as mp

# 初始化 Mediapipe Pose 模块
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# 创建 Pose 对象
pose = mp_pose.Pose(static_image_mode=True)

test_photo_root_path = 'photo'
