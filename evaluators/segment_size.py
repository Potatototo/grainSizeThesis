import cv2 as cv
import numpy as np
import statistics as s

def evaluate(filename, mask):
    mask = np.array(mask)
    em_filename = filename.replace('.png', '_EM.png')
    em = cv.imread(em_filename)
    em_mask = cv.inRange(em, (250, 0, 0), (255, 0, 0))

    mask_inverted = 255 - mask
    em_mask_inverted = 255 - em_mask

    # cv.namedWindow('masks', cv.WINDOW_NORMAL)
    # cv.resizeWindow('masks', 1920, 1025)
    # cv.imshow('masks', mask_inverted)
    # cv.waitKey()
    # cv.imshow('masks', em_mask_inverted)
    # cv.waitKey()

    mask_contours, _ = cv.findContours(mask_inverted, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    em_mask_contours, _ = cv.findContours(em_mask_inverted, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    mask_areas = []
    em_mask_areas = []
    for i in range(len(mask_contours)):
        mask_areas.append(cv.contourArea(mask_contours[i]))
    for i in range(len(em_mask_contours)):
        em_mask_areas.append(cv.contourArea(em_mask_contours[i]))
    
    mask_areas.sort()
    em_mask_areas.sort()
    # print(str(mask_areas))
    # print(str(em_mask_areas))

    # mask_areas_sliced = mask_areas[int(len(mask_areas) * .3) : int(len(mask_areas) * .95)]
    # em_mask_areas_sliced = em_mask_areas[int(len(em_mask_areas) * .05) : len(em_mask_areas)]
    lower_thresh = 2000
    upper_thresh = 20000
    mask_areas_sliced = list(filter(lambda a: a > lower_thresh and a < upper_thresh, mask_areas))
    em_mask_areas_sliced = list(filter(lambda a: a > lower_thresh and a < upper_thresh, em_mask_areas))
    # print(str(mask_areas_sliced))
    print(str(em_mask_areas_sliced))

    # fcontours = list(filter(lambda c: cv.contourArea(c) > lower_thresh and cv.contourArea(c) < upper_thresh, mask_contours))
    # cv.drawContours(em_mask, fcontours, -1, 127, thickness=3)
    # cv.imshow('masks', em_mask)
    # cv.waitKey()

    print('\n###Segment Size Evaluation###')
    print('Detected:\n' + str(len(mask_contours)) + 
        ' Blobs with Min Area: ' + str(min(mask_areas)) + 
        ', Max Area: ' + str(max(mask_areas)) + 
        ', Average Area: ' + str(s.mean(mask_areas)) + 
        ', Median Area: ' + str(s.median(mask_areas)))
    print('Annotated:\n' + str(len(em_mask_contours)) + 
        ' Blobs with Min Area: ' + str(min(em_mask_areas)) + 
        ', Max Area: ' + str(max(em_mask_areas)) + 
        ', Average Area: ' + str(s.mean(em_mask_areas)) + 
        ', Median Area: ' + str(s.median(em_mask_areas)))

    print('\nDetected Sliced:\n' + str(len(mask_areas_sliced)) + 
        ' Blobs with Min Area: ' + str(min(mask_areas_sliced)) + 
        ', Max Area: ' + str(max(mask_areas_sliced)) + 
        ', Average Area: ' + str(s.mean(mask_areas_sliced)) + 
        ', Median Area: ' + str(s.median(mask_areas_sliced)))
    print('Annotated Sliced:\n' + str(len(em_mask_areas_sliced)) + 
        ' Blobs with Min Area: ' + str(min(em_mask_areas_sliced)) + 
        ', Max Area: ' + str(max(em_mask_areas_sliced)) + 
        ', Average Area: ' + str(s.mean(em_mask_areas_sliced)) + 
        ', Median Area: ' + str(s.median(em_mask_areas_sliced)))
