import threading
import camera
import fpr
import test
# 创建线程
t1 = threading.Thread(target=camara.main)
t2 = threading.Thread(target=fpr.main)
t3 = threading.Thread(target=test.main)

# 启动线程
t1.start()
t2.start()
# t3.start()

# 等待线程结束（若需要）
t1.join()
t2.join()
# t3.join()
