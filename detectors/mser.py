import cv2 as cv
import numpy as np

scale = 1
win_name = 'MSER'

def Callback(val):
    blur = cv.getTrackbarPos('blur', win_name)
    img_blur = cv.GaussianBlur(img_eh, (blur, blur), 0)
    regions, _ = mser.detectRegions(img_blur)
    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    dst = src_resized.copy()
    cv.polylines(dst, hulls, 1, (0, 255, 0), thickness=3)
    cv.imshow(win_name, dst)


src = cv.imread('data/ID17320.B_Fa_002.png')
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_resized = cv.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)))
img_resized = cv.resize(src_gray, (int(src_gray.shape[1] * scale), int(src_gray.shape[0] * scale)))
img_eh = cv.equalizeHist(img_resized)

mser = cv.MSER_create()
mser.setMinArea(500)
mser.setMaxArea(10000)

cv.namedWindow(win_name, cv.WINDOW_NORMAL)
cv.createTrackbar('blur', win_name , 3, 49, Callback)
cv.resizeWindow(win_name, 1920, 1025)
Callback(0)
cv.waitKey()