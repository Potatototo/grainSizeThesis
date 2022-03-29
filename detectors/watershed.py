import cv2 as cv
import numpy as np
# from skimage.feature import peak_local_max
# from skimage.morphology import watershed
# from scipy import ndimage
# import imutils

win_name = 'Watershed'

def detect(filename):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, 1920, 1025)

    src = cv.imread(filename)
    shifted = cv.pyrMeanShiftFiltering(src, 21, 20)
    cv.imshow(win_name, shifted)
    cv.waitKey()
    gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    cv.imshow(win_name, thresh)
    cv.waitKey()
    
    # D = ndimage.distance_transform_edt(thresh)
    # localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
    # markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    # labels = watershed(-D, markers, mask=thresh)
    # print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))