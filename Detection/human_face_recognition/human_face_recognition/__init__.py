import os
import cv2
import numpy as np
import face_recognition
from PIL import ImageFont, ImageDraw, Image

print('face_recognition initialized')
facebase_path = "facebase"
print('facebase_path is', facebase_path)