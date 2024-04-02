import cv2
import numpy as np
from PIL import Image
import os

# 读取输入图像
image = cv2.imread('input_image.jpg')

# 中值滤波去噪
image = cv2.medianBlur(image, 5)

# 打开图像
img = Image.open('input_image.jpg')
size_img = img.size

# 获取图像大小
size_x, size_y = size_img

x = 0
y = 0

# 定义裁剪区域大小
w = 608
h = 608

# 计算裁剪数量
x_num = int(size_x / w)
y_num = int(size_y / h)

print(x_num)
print(y_num)

for k in range(x_num):
    for v in range(y_num):
        # 裁剪图像
        region = img.crop((x + k * w, y + v * h, x + w * (k + 1), y + h * (v + 1)))
        
        # 将PIL图像转换为OpenCV格式
        region_cv = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
        
        # 将裁剪后的图像进行处理
        hsv_image = cv2.cvtColor(region_cv, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        v = cv2.add(v, 30)
        s = cv2.add(s, 30)
        hsv_image = cv2.merge((h, s, v))
        result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # 保存处理后的图像
        cv2.imwrite(f'output_image_{k}_{v}.jpg', result_image)

