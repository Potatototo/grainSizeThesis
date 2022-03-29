"""Runs parameter tuning for the parameters defined inside.
"""
import csv

from ray import tune
from ray.tune.utils.log import Verbosity

import detectors.log
import evaluators.scanline


def objective(blur: int, close_iter: int, open_iter: int):
    """Funtion to evaluate all images in the /src/data/ directory using the given set of parameters.

    Args:
        blur (int): Kernel size of for Gaussian blur
        close_iter (int): How often the closing operation will be done
        open_iter (int): How often the opening operation will be done

    Returns:
        float: Absolute mean error for the given set of parameters
    """
    error = 0
    count = 0
    start_path = '/src/data'
    for i in range(len(list(ground_truths.keys()))):
        filename = list(ground_truths.keys())[i]
        count += 1
        mask = detectors.log.detect_tune(
            f'{start_path}/{filename}', blur, close_iter, open_iter)
        error = evaluators.scanline.evaluate_tune(
            f'{start_path}/{filename}', mask, ground_truths)
        error += error
    mean_error = error / count
    return mean_error


def training_function(config):
    """Training function for Ray Tune

    Args:
        config (Dict[str, ray_function]): Config for Ray Tune defined in tune.run()
    """
    blur, close_iter, open_iter = config['blur'], config[
        'close_iter'], config['open_iter']
    intermediate_score = objective(blur, close_iter, open_iter)
    tune.report(mean_loss=intermediate_score)


ground_truths = {}
with open('ground_truths.csv', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        line_count += 1
        ground_truths[row[0]] = float(row[1])
print(f'Added {line_count} images to training data pool.')

# analysis = tune.run(
#     training_function,
#     name='optimal_quared_old',
#     local_dir='./ray_results',
#     config={
#         "blur": tune.grid_search(list(range(30, 120))),
#         "close_iter": tune.grid_search(list(range(2, 6))),
#         "open_iter": tune.grid_search(list(range(4, 12))),
#     },
#     # resume=True,
#     verbose=Verbosity.V1_EXPERIMENT)

analysis = tune.run(
    training_function,
    name='factor_8_10_1000',
    local_dir='./ray_results',
    config={
        "blur": tune.grid_search(list(range(8000000, 10000000, 1000))),
        "close_iter": 1,
        "open_iter": 1,
    },
    # resume=True,
    verbose=Verbosity.V1_EXPERIMENT)

print("Best config: ", analysis.get_best_config(
    metric="mean_loss", mode="min"))
