"""
Scanline evaluator
Counts how many blobs are in a horizontal line on average.
"""
import math
import statistics as s
import cv2 as cv
import numpy as np

WIN_NAME = 'Scanline'


class Lines:
    """
    Class to hold current line count and intersections.
    Used to avoid global variables in interactive.
    """
    count = 4
    intersections: list[tuple[int, int]] = []


def remove_outliers(blobs, m=2.):
    """Removes outliers that deviate too far from median. Unused

    Args:
        blobs ([int]): Array of detected blob counts
        m (float, optional): Max standard deviation. Defaults to 2.0.

    Returns:
        [int]: Array with outliers removed
    """
    d = np.abs(blobs - np.median(blobs))
    mdev = np.median(d)
    sd = d / mdev if mdev else 0.
    return blobs[sd < m]


def scan(mask):
    """Performs scanline algorithm on given binary mask.

    Args:
        mask ([[uint8]]): binary grain boundary mask

    Returns:
        float, [(int, int)]: detected average grain count, array containing detected edge intersections (x, y)
    """
    step = int(mask.shape[0] / (Lines.count + 1))
    y = step
    detected_blob_count = []
    intersections = []
    while y <= mask.shape[0] - mask.shape[0] / (Lines.count + 1):
        d_i = 1 if mask[y][0] > 0 else 0
        if d_i:
            intersections.append((0, y))
        for x in range(mask.shape[1] - 1):
            if mask[y][x] != mask[y][x + 1]:
                d_i += 1
                if d_i % 2 == 0 and d_i > 0:
                    l = intersections.pop()
                    intersections.append((int((x + l[0]) / 2), y))
                else:
                    intersections.append((x, y))
        detected_blob_count.append(int(d_i / 2 - 1))
        y += step
    return detected_blob_count, intersections


def evaluate(filename, mask, ground_truths=None):
    """Evaluate binary mask to given image filename using the scanline method.

    Args:
        filename (str): filename of image to be evaluated
        mask ([[uint8]]): binary grain boundary mask
        ground_truths (dict(str, float), optional): Dictionary containing ground truth data. Unused here. Defaults to None.
    """
    mask = np.array(mask)
    detected_blob_count, intersections = scan(mask)

    filename_out = filename.replace('./data/', '')
    eval_out = (filename_out, s.mean(detected_blob_count), intersections)
    out = f'\n###Scanline Evaluation: {filename}###\nDetected:\n{detected_blob_count}\n\
Average: {eval_out[1]}\n'
    print(out)
    return eval_out


def evaluate_compare(filename, mask, ground_truths):
    """Evaluates given binary mask to image filename and compares it to ground truth values.

    Args:
        filename (str): filename to image
        mask ([[uint8]]): binary grain boundary mask
        ground_truths (dict(str, float), optional): Dictionary containing ground truth data.
    """
    mask = np.array(mask)
    filename = filename.replace('./data/', '')
    ground_truths = dict(ground_truths)

    detected_blob_count, _ = scan(mask)
    em_blob_count = ground_truths[filename]

    eval_out = (filename, s.mean(detected_blob_count), em_blob_count)

    out = f'\n###Scanline Evaluation: {filename}###\nDetected:\n{detected_blob_count}\n\
Average: {eval_out[1]}\nActual: {eval_out[2]}\n'
    print(out)
    return eval_out


def evaluate_interactive(filename, mask, ground_truths = None):
    """Interactively evaluate binary mask to given image filename using the scanline method.

    Args:
        filename (str): filename of image to be evaluated
        mask ([[uint8]]): binary grain boundary mask
        ground_truths (dict(str, float), optional): Dictionary containing ground truth data. Unused here. Defaults to None.
    """
    image = cv.imread(filename)
    mask = np.array(mask)

    def draw_image():
        img_out = image.copy()
        cv.putText(img_out,
                   f'Average Grain Count / Line:\
{round((len(Lines.intersections) - Lines.count) / Lines.count, 2)}',
                   (20, img_out.shape[0] - 20),
                   cv.FONT_HERSHEY_SIMPLEX,
                   2,
                   (0, 0, 0),
                   thickness=3)
        for i in range(1, Lines.count + 1):
            cv.line(img_out, (0, int(i * img_out.shape[0] / (Lines.count + 1))), (img_out.shape[1],
                    int(i * img_out.shape[0] / (Lines.count + 1))), (0, 255, 0), thickness=2)
        for p in Lines.intersections:
            cv.circle(img_out, p, 8, (0, 0, 255), 2)
        cv.imshow(WIN_NAME, img_out)

    def mouse_callback(event, x, y, flags, param):
        _ = flags, param
        if event == cv.EVENT_LBUTTONDOWN:
            max_dist = 20
            found = False
            for p in Lines.intersections:
                if math.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2) <= max_dist:
                    found = True
                    Lines.intersections.remove(p)
            if not found:
                Lines.intersections.append((x, y))
            draw_image()

    def tb_callback(x):
        Lines.count = x
        _, intersections = scan(mask)
        Lines.intersections = intersections
        draw_image()

    cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
    cv.createTrackbar('Lines', WIN_NAME, Lines.count, 100, tb_callback)
    cv.setMouseCallback(WIN_NAME, mouse_callback)
    cv.resizeWindow(WIN_NAME, 1600, 900)

    tb_callback(Lines.count)
    while True:
        if cv.waitKey() == ord('q'):
            cv.destroyAllWindows()
            return (filename, (len(Lines.intersections) - Lines.count) / Lines.count, Lines.intersections)


def evaluate_tune(filename, mask, ground_truths):
    """Evaluation using the scanline method used for parameter tuning

    Args:
        filename (str): filename of image
        mask ([[uint8]]): binary grain boundary mask
        ground_truths (dict(str, float)): dictionary containing ground truth data to compare to

    Returns:
        float: absolute mean error of detected grain count
    """
    mask = np.array(mask)

    detected_blob_count, _ = scan(mask)
    actual_blob_count = ground_truths[filename.replace('/src/data/', '')]

    return abs((s.mean(detected_blob_count) - 1) - actual_blob_count)
