#!-*-coding:UTF8-*-
import cv2
import numpy as np

#读入图片
imgA = cv2.imread('gakki101.png')
imgB = cv2.imread('gakki102.png')

# imgA=cv2.resize(imgA,(400,300))
# imgB = cv2.resize(imgB, (640, 480))


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 第一步：构造sift，求解出特征点和sift特征向量
sift = cv2.xfeatures2d.SIFT_create()
kpsA, dpA = sift.detectAndCompute(imgA, None)
kpsB, dpB = sift.detectAndCompute(imgB, None)

# 第二步：构造BFMatcher()蛮力匹配，匹配sift特征向量距离最近对应组分
bf = cv2.BFMatcher()
# 获得匹配的结果
matches = bf.match(dpA, dpB)

#第三步：对匹配的结果按照距离进行排序操作
matches = sorted(matches, key=lambda x: x.distance)

# 第四步：使用cv2.drawMacthes进行画图操作
ret = cv2.drawMatches(imgA, kpsA, imgB, kpsB, matches[:40], None,flags=2)

cv2.imshow('ret', ret)
cv2.waitKey(0)
cv2.destroyAllWindows()