import os
import random

# 设置目标文件夹
root_folder = "./downloaded_images"


def photo_processing(path):
    # 获取所有文件的完整路径
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # 打乱文件顺序
    random.shuffle(files)

    # 重命名为 00001.xxx，保留扩展名
    for idx, filename in enumerate(files, start=1):
        # 获取扩展名
        ext = os.path.splitext(filename)[-1].lower()
        new_name = f"{idx:05d}{ext}"  # 00001.jpg 这种格式
        src_path = os.path.join(path, filename)
        dst_path = os.path.join(path, new_name)

        os.rename(src_path, dst_path)
        print(f"重命名：{filename} -> {new_name}")

    print("重命名完成。")


labels = ['climb', 'people fall down', 'people lay down', 'people stand', 'people sit down']
for label in labels:

    print(root_folder+'/'+label)
    photo_processing(root_folder+'/'+label)
