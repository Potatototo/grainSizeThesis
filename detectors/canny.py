"""
Canny Edge detection
"""
import cv2 as cv
import numpy as np

WIN_NAME = 'Edge Map'
KERNEL_SIZE = 3
SCALE = 0.5

def canny_thresh(_):
    """
    does canny thresholding
    """
    low_threshold = cv.getTrackbarPos('Min', WIN_NAME)
    high_threshold = cv.getTrackbarPos('Max', WIN_NAME)
    blur = cv.getTrackbarPos('Blur', WIN_NAME)
    img_resized = cv.resize(src_gray, (int(src_gray.shape[1] * SCALE),
        int(src_gray.shape[0] * SCALE)))
    img_eh = cv.equalizeHist(img_resized)
    img_blur = cv.GaussianBlur(img_eh, (blur, blur), 0)
    detected_edges = cv.Canny(img_blur, low_threshold, high_threshold, KERNEL_SIZE)
    mask = detected_edges != 0
    dst = np.maximum(src_resized, (mask[:,:,None].astype(src_resized.dtype) * 255))
    # dst = src_resized * (mask[:,:,None].astype(src_resized.dtype))
    cv.imshow(WIN_NAME, dst)

src = cv.imread('data/ID17320.B_Fa_002.png')
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_resized = cv.resize(src, (int(src.shape[1] * SCALE), int(src.shape[0] * SCALE)))
cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
cv.createTrackbar('Min', WIN_NAME , 100, 255, canny_thresh)
cv.createTrackbar('Max', WIN_NAME , 200, 255, canny_thresh)
cv.createTrackbar('Blur', WIN_NAME , 3, 49, canny_thresh)
canny_thresh(0)
cv.waitKey()
