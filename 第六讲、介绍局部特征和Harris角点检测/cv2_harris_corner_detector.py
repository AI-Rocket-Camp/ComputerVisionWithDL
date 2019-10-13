#!-*-coding:UTF8-*-


import cv2
import numpy as np

img = cv2.imread('chess.png')
img = cv2.resize(img, (720, 540))

# 1. Harris角点检测基于灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Harris角点检测
dst = 10*cv2.cornerHarris(gray, 2, 3, 0.04)


# 膨胀，便于标记
dst = cv2.dilate(dst, None)
cv2.imshow('dst', dst)
cv2.waitKey(0)

# 3. 角点标记为红色
img[dst > 0.005 * dst.max()] = [0, 0, 255]

cv2.imshow('dst', img)


cv2.waitKey()
