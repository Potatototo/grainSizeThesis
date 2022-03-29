"""
Metal grain detector using using the Laplacian over Gaussian operator.
"""
import math
import cv2 as cv
import numpy as np

WIN_NAME = 'LoG'

# contour size filter


def check_area(cnt):
    """
    Filter function to check for area size. Filters out areas that are too small.
    """
    return cv.contourArea(cnt) > 500


def generate_mask(filename, blur, dilation_iter, erosion_iter):
    """Generates a binary mask of image filename using the given parameters.

    Args:
        filename (String): Filename of image
        blur (int): Scaling factor to determine blur kernel size
        close_iter (int): How often to perform the closing operations
        open_iter (int): How often to perform the open operation

    Returns:
        np.array: Binary mask of grain edges
    """
    src = cv.imread(filename)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # apply LoG
    img_gauss = cv.GaussianBlur(src_gray, (blur, blur), 0)
    img_lap = cv.Laplacian(img_gauss, cv.CV_8U)
    dst = img_lap / img_lap.max()

    # connect edges
    kernel_3 = np.ones((3, 3), np.uint8)
    dst_dilated = cv.dilate(dst, kernel_3, iterations=dilation_iter)
    dst_eroded = cv.erode(dst_dilated, kernel_3, iterations=erosion_iter)
    # dst_closed = cv.morphologyEx(dst, cv.MORPH_CLOSE, kernel_3, iterations=close_iter)
    # dst_opened = cv.morphologyEx(dst_closed, cv.MORPH_OPEN, kernel_3, iterations=open_iter)

    # turn mask into uint8 and binarize
    dst = np.zeros(src.shape, np.uint8)
    dst = dst_eroded.copy()
    dst = dst - dst.min()
    dst = dst / dst_eroded.max() * 255
    dst = np.uint8(dst)
    _, mask = cv.threshold(dst, 1, 255, cv.THRESH_BINARY)
    out = src.copy()
    out[mask>0] = (0, 255, 0)
    cv.imwrite("small_log.png", out)
    return mask

def detect(filename):
    """
    Gets a filename and returns masked out metal grain borders.
    """
    img = cv.imread(filename)
    img_height, img_width, _ = img.shape
    # 9163872
    factor = img_height * img_width / 9163872
    blur = 61
    # blur = round(63 * factor)
    # if blur % 2 == 0:
    #     blur += 1
    dilation_iter = math.floor(4 * factor)
    erosion_iter = math.ceil(10 * factor)

    mask = generate_mask(filename, blur, dilation_iter, erosion_iter)
    return mask


def detect_interactive(filename):
    """
    Interactive environment to detect metal grain edges using the LoG operator.
    Just for demonstrative purpose. Not using up to date parameters.
    """
    close_iter = 4
    open_iter = 10
    src = cv.imread(filename)
    mask = np.zeros(src.shape, np.uint8)

    def trackbar_cb(_):
        blur = cv.getTrackbarPos('Blur', WIN_NAME)
        # iterations = cv.getTrackbarPos('Iterations', WIN_NAME)
        mask = generate_mask(blur, blur, close_iter, open_iter)
        mask_as_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        img_out = np.hstack((src, mask_as_bgr))
        cv.imshow(WIN_NAME, img_out)

    cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
    cv.createTrackbar('Blur', WIN_NAME, 51, 99, trackbar_cb)
    cv.createTrackbar('Iterations', WIN_NAME, 1, 10, trackbar_cb)
    cv.resizeWindow(WIN_NAME, 1920, 1025)

    trackbar_cb(3)
    while True:
        key = cv.waitKey()
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv.destroyAllWindows()
            return mask
    cv.destroyAllWindows()


def detect_tune(filename, blur, dilation_iter, erosion_iter):
    """
    Detection function with parameters for tuning. Returns binary mask
    """
    img = cv.imread(filename)
    img_height, img_width, _ = img.shape
    factor = img_height * img_width / blur
    blur_kernel = round(72 * factor)
    # blur_kernel = 61
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    dilation_iter = round(4 * factor)
    erosion_iter = round(10 * factor)

    mask = generate_mask(filename, blur_kernel, dilation_iter, erosion_iter)

    return mask
