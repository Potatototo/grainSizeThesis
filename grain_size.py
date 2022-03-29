"""
Interactive program to use given detectors and evaluators to determine metal grain size.
"""
import sys

import cv2

# import detectors.blob
# import detectors.canny
# import detectors.mser
# import detectors.sobel
# import detectors.watershed
import detectors.log

# import evaluators.segment_size
import evaluators.scanline

d = {'1': detectors.log.detect,
     '2': detectors.log.detect_interactive}
e = {'1': evaluators.scanline.evaluate,
     '2': evaluators.scanline.evaluate_interactive,
     '3': evaluators.scanline.evaluate_compare}

def process(filename: str, d_i: str, e_i: str, ground_truths: dict[str, float] = None) -> tuple[str, float, list[tuple[int, int]]]:
    """
    Processes the given filename with given detector and evaluator ID.
    """
    mask = d.get(d_i, d['1'])(filename)
    eval_out = e.get(e_i, e['1'])(filename, mask, ground_truths)
    return eval_out

def main():
    """
    Main function to read filename and user input for detector and evaluator selection.
    """
    if '-h' in sys.argv:
        print('Usage: grain_size.py [filename] [detector id] [evaluator id]')
        return

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print('Please provide a file to analyze.')
        return

    d_i, e_i = 0, 0
    if len(sys.argv) > 3:
        d_i = sys.argv[2]
        e_i = sys.argv[3]
    else:
        d_i = input('Choose a detector: (default = LoG)\n\
[1] LoG\n\
[2] LoG Interactive\n')
        e_i = input('Choose an evaluator: (default = Scanline)\n\
[1] Scanline\n\
[2] Interactive Scanline\n')

    process(filename, d_i, e_i)

if __name__ == "__main__":
    main()
