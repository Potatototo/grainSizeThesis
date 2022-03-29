import cv2 as cv
import numpy as np

scale = 1
win_name = 'Sobel'
#thrsh = 4
src = cv.imread('data/ID17320.B_Fa_002.png')
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_resized = cv.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)))
img_resized = cv.resize(src_gray, (int(src_gray.shape[1] * scale), int(src_gray.shape[0] * scale)))
img_eh = cv.equalizeHist(img_resized)
img_blur = cv.GaussianBlur(src_gray, (49, 49), 0)

def Callback(val):
    thrsh = cv.getTrackbarPos('thrsh', win_name)
    sobelx = cv.Sobel(img_blur, cv.CV_8U, 1, 0)
    sobely = cv.Sobel(img_blur, cv.CV_8U, 0, 1)

    mask = np.maximum(sobelx, sobely)
    # mask = mask > thrsh
    mask = cv.adaptiveThreshold(mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 251, 0)
    mask = mask != 0

    dst = np.maximum(src_resized, (mask[:,:,None].astype(src_resized.dtype) * 255))
    cv.imshow(win_name, dst)

def detect(filename):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.createTrackbar('thrsh', win_name , 0, 255, Callback)
    cv.resizeWindow(win_name, 1920, 1025)
    Callback(0)
    cv.waitKey()

if __name__ == "__main__":
    detect('')

#cv.imshow(win_name, sobelxy)
#cv.waitKey()
#cv.imshow(win_name, sobelx)
#cv.waitKey()
#cv.imshow(win_name, sobely)
#cv.waitKey()
# cv.imshow(win_name, dstx)
# cv.waitKey()
# cv.imshow(win_name, dsty)
# cv.waitKey()
# cv.imshow(win_name, dst)
# cv.waitKey()
