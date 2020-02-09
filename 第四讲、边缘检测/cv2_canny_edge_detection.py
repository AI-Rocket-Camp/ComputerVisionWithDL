#!-*-coding:UTF8-*-

import cv2
import numpy as np

# 1.Canny边缘检测
img = cv2.imread('lena.png', 0)
edges = cv2.Canny(img, 30, 70)
cv2.namedWindow("lena and it's Canny edges",0)
cv2.imshow("lena and it's Canny edges", np.hstack((img, edges)))
cv2.waitKey(0)


# 2.先阈值，后边缘检测R
# 阈值分割（使用到了Otsu自动阈值）
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(thresh, 30, 70)
cv2.namedWindow("lena image,thresh and edges",0)

cv2.imshow("lena image,thresh and edges", np.hstack((img, thresh, edges)))
cv2.waitKey(0)
