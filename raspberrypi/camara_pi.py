'''
******************************************************************

    Project Description: camera vedio send to Server

    Hardware information:
    ---------------------------------------------------------
        camera                   |              OV5640
    ---------------------------------------------------------


    Author: Veinsure Lee
    Edit Date: 2025.04.08

******************************************************************
'''
icamera2
import Picamera2
from libcamera import controls
import numpy as np

# camera initialize
picam2 = Picamera2()

# camera config
config = picam2.create_still_configuration(
    main={"size": (800, 600)},
    transform=libcamera.Transform(hflip=1, vflip=1)
)
picam2.configure(config)

# set camera pram
picam2.set_controls({
    "AwbEnable": True,
    "AwbMode": controls.AwbModeEnum.Auto,
    "Brightness": 0.0,
    "Contrast": 1.0,
    "Saturation": 1.0,
})

picam2.start()

# socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("115.159.216.201", 9000))  # connect to Server

try:
    while True:
        # capture image
        image_data = picam2.capture_array("main")

        # img2JPEG
        from PIL import Image

        img = Image.fromarray(image_data)
        buf = img.tobytes('jpeg', quality=20)

        buf_len = len(buf)
        print("len in theo:", buf_len)
        s.sendall(str.encode("%-16s" % buf_len))

        ret = s.sendall(buf)
        time.sleep(0.05)

except Exception as ret:
    print("error:", ret)
finally:
    s.close()
    picam2.stop()