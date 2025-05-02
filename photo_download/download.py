from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
import requests
from urllib.parse import urlparse, quote
import uuid


def download_images(keyword, save_root="./downloaded_images", max_count=200):
    # 编码搜索关键词并生成搜索链接
    encoded_keyword = quote(keyword)
    search_url = f"https://www.bing.com/images/search?q={encoded_keyword}&form=HDRSC2&first=1"

    # 创建保存路径
    save_folder = os.path.join(save_root, keyword)
    os.makedirs(save_folder, exist_ok=True)

    # 配置 Selenium
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # 无头模式
    chrome_options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=chrome_options)

    print(f"正在搜索关键词: {keyword}")
    driver.get(search_url)
    time.sleep(2)

    # 滚动页面以加载更多图片
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # 抓取图片元素
    img_elements = driver.find_elements("tag name", "img")
    print(f"找到 {len(img_elements)} 个图片元素，开始下载...")

    count = 0
    for img in img_elements:
        src = img.get_attribute("src")
        if src and src.startswith("http"):
            try:
                # 获取文件扩展名
                path = urlparse(src).path
                ext = os.path.splitext(path)[-1]
                if ext.lower() not in ['.jpg', '.png', '.jpeg', '.webp']:
                    ext = '.jpg'
                filename = f"{uuid.uuid4().hex}{ext}"
                filepath = os.path.join(save_folder, filename)

                # 下载图片
                r = requests.get(src, timeout=5)
                with open(filepath, "wb") as f:
                    f.write(r.content)
                count += 1
                print(f"下载第 {count} 张: {filename}")
                if count >= max_count:
                    break
            except Exception as e:
                print(f"下载失败：{e}")

    driver.quit()
    print(f"下载完成，共下载 {count} 张图片，保存于: {save_folder}")


# 示例：使用关键词“攀爬”下载图片
# download_images("climb")

labels = ['climb', 'people fall down', 'people lay down', 'people stand', 'people sit down']
for label in labels:
    print(label)
    download_images(label)
