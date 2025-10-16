import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取 iPhone 视频
cap = cv2.VideoCapture("finger.mp4")

red_vals = []
green_vals = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 缩小图像，加快处理
    frame = cv2.resize(frame, (320, 240))

    # 选取中心 100x100 的 ROI
    h, w, _ = frame.shape
    roi_size = 100
    x1, y1 = w//2 - roi_size//2, h//2 - roi_size//2
    x2, y2 = x1 + roi_size, y1 + roi_size
    roi = frame[y1:y2, x1:x2]

    # OpenCV 默认是 BGR 顺序
    b, g, r = cv2.split(roi)

    # 计算均值
    red_vals.append(np.mean(r))
    green_vals.append(np.mean(g))

cap.release()

# 绘制红绿通道随时间的变化曲线
plt.plot(red_vals, 'r', label='Red channel mean')
plt.plot(green_vals, 'g', label='Green channel mean')
plt.legend()
plt.show()