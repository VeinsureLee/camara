import cv2


def show_auto_scaled_image(img, window_name='ResizableImage'):
    if img is None:
        raise ValueError("输入图像为空")

    original_h, original_w = img.shape[:2]

    # 创建可缩放窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, original_w, original_h)

    while True:
        # 获取当前窗口尺寸
        try:
            x, y, win_w, win_h = cv2.getWindowImageRect(window_name)
        except cv2.error:
            break  # 窗口已关闭，退出循环

        # 防止宽高为0
        if win_w < 1 or win_h < 1:
            continue

        # 缩放图像以适应窗口
        resized_img = cv2.resize(img, (win_w, win_h), interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, resized_img)

        key = cv2.waitKey(30)
        if key == 27:  # ESC 退出
            cv2.destroyWindow(window_name)
            break