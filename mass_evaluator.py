"""
Program to mass evaluate all metal grain pictures using the given detector and
evaluator, saving the output to a file for better evaluation of how effective
the given operations were.
"""
import math
import os
import csv
from sys import maxsize
from time import time
import numpy as np

import grain_size


def log_intersections(filename, intersections):
    """
    Function to store detected intersection data in csv files.

    Args:
        filename (str): Filename of analyzed image
        intersections ([(int, int)]): Array containing (x, y) coordinates of detected intersections.
    """
    filename = filename.replace('./data/', '').replace('.png', '.csv')
    with open(filename, 'w', encoding='utf-8') as f:
        for i in intersections:
            f.write(f'{i[0]},{i[1]}\n')


def round_partial (value, resolution):
    """Rounds value to specified resolution

    Args:
        value (float): value to round
        resolution (float): decimal to round to

    Returns:
        float: rounded value
    """
    return round (value / resolution) * resolution


def eval_all():
    """
    Evaluates all images stored in ./data/
    """
    d = 1
    e = 1
    filename_eval = f'eval_{d}_{e}_{int(time())}.csv'

    # empty file
    f = open(filename_eval, 'w', encoding='utf-8')
    f.write('')
    f.close()

    count = 0
    mean_detected = 0
    processed = []
    f = open(filename_eval, 'a', encoding='utf-8', buffering=1)
    start_path = './data'
    for _, _, files in os.walk(start_path):
        for file in files:
            if 'png' in file and file not in processed:
                eval_out = grain_size.process('./data/' + file, str(d), str(e))
                f.write(f'{eval_out[0]},{eval_out[1]}\n')
                # log_intersections(eval_out[0], eval_out[2])
                count += 1
                mean_detected += eval_out[1]
                processed.append(eval_out[0])
    f.close()
    mean_detected /= count
    print(f'Average grain count per line: {mean_detected}')


def eval_compare():
    """
    Compares detected results to gound truth values stored in ./data/truths/ground_truths.csv
    """
    d = 1
    e = 3
    filename_eval = f'eval_comp_{d}_{e}.csv'
    truth_path = 'data/truths/'

    # empty file
    f = open(filename_eval, 'w', encoding='utf-8')
    f.write('')
    f.close()

    # read ground_truth data
    ground_truths = {}
    with open(f'{truth_path}ground_truths.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            line_count += 1
            ground_truths[row[0]] = float(row[1])
    print(f'Read ground truths for {line_count} images.')

    count = 0
    mean_detect_err = 0
    means_errs = []

    f = open(filename_eval, 'a', encoding='utf-8')
    ground_truth_file = f'{truth_path}ground_truths_validation.csv'
    with open(ground_truth_file, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            eval_out = grain_size.process(f'./data/{row[0]}',
                                          str(d),
                                          str(e),
                                          ground_truths)
            f.write(','.join([str(int) for int in eval_out]) + '\n')
            count += 1
            mean_detect_err += abs(eval_out[1] - eval_out[2])
            means_errs.append(round_partial(eval_out[1] - eval_out[2], 0.25))
    f.close()
    mean_detect_err /= count
    print(' '.join(str(e) + ' ' + str(means_errs.count(e)) for e in means_errs))
    print(f'Mean Error: {mean_detect_err}\nσ={np.std(means_errs)}')


def eval_accuracy():
    """
    Evaluates how accurate the detected grain boundaries are.
    Compares detection with intersection data stored in ./data/truths/intersections/
    """
    d = 1
    e = 3
    filename_eval = f'eval_acc_{int(time())}.csv'
    start_path = 'data/truths/intersections/'

    processed = []
    averages = []
    averages_clean = []
    with open(filename_eval, 'w', encoding='utf-8', buffering=1) as eval_file:
        eval_file.write('filename,grain_count,avg_distance\n')
        for _, _, files in os.walk(start_path):
            for file in files:
                if file not in processed:
                    with open(f'{start_path}{file}', 'r', encoding='utf-8') as csv_file:
                        processed.append(file)
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        intersections = []
                        for x, y in csv_reader:
                            intersections.append((int(x), int(y)))
                        img_filename = file.replace('.csv', '.png')
                        eval_out = grain_size.process(f'data/{img_filename}', d, e)
                        # print(eval_out)
                        avg_distance = 0
                        avg_clean = 0
                        clean_count = 0
                        for x, y in eval_out[2]:
                            min_distance = maxsize
                            min_distance_clean = maxsize
                            for xt, yt in intersections:
                                distance = math.sqrt((x - xt) * (x - xt) + (y - yt) * (y - yt))
                                if distance < min_distance:
                                    min_distance = distance
                                    if distance < 50:
                                        min_distance_clean = distance
                            if min_distance_clean < 50:
                                avg_clean += min_distance_clean
                                clean_count += 1
                            avg_distance += min_distance
                        avg_distance /= len(eval_out[2])
                        avg_distance = round(avg_distance, 2)
                        avg_clean /= clean_count
                        avg_clean = round(avg_clean, 2)
                        averages.append(avg_distance)
                        averages_clean.append(avg_clean)
                        eval_file.write(f'{eval_out[0]},{eval_out[1]},{avg_distance},{avg_clean}\n')
            print(f'Average: {np.mean(averages)}\nAverage clean: {np.mean(averages_clean)} {np.std(averages)} {np.std(averages_clean)}')


# eval_all()
# eval_compare()
eval_accuracy()
