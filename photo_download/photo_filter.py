from PIL import Image
import os

# 设置图片保存路径
root_folder = "./downloaded_images"


# 记录删除的数量
def filter_photos(folder):
    deleted = 0
    # 遍历文件夹下所有文件
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            # 尝试用 PIL 打开图片
            with Image.open(filepath) as img:
                img.verify()  # 验证图片是否完整
        except Exception as e:
            print(f"删除损坏图片：{filename}，原因：{e}")
            os.remove(filepath)
            deleted += 1
    print(f"已删除 {deleted} 张损坏或无法识别的图片。")


labels = ['climb', 'people fall down', 'people lay down', 'people stand', 'people sit down']
for label in labels:

    print(root_folder+'/'+label)
    filter_photos(folder=root_folder+'/'+label)
