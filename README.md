# Usage
Developed using the following libraries and versions.
• Python - 3.9.7
• OpenCV - 4.5.3
• NumPy - 1.20.3
• Ray - 1.8.0
• Ray Tune - 1.8.0
• TensorBoard - 2.7.0

pip3 install -r requirements.txt

# Documentation
## Grain Size Determination Tool
The main program that will be used by the end user can be started with:
$ python3 grain_size.py [options] [file]
Adding −h as an option will bring up the help text where all other options are explained in detail.
When running the program, the user will be asked to select a detection and evaluation module to
process the given image. The default selections are shown and can just be accepted with ENTER.
They are generally the best option to quickly evaluate an image. The user can also select interactive
versions of some detection and evaluation modules to check step by step results and manually correct
them if needed.
## Image Processing Modules
The image processing modules, or detectors, are in the detectors subdirectory. Each detector module
provides a function detect ( filename ) that takes the path to an image as input and returns a binary
mask where detected edges of metal grains have the value 1 and everything else 0. These masks can
then be used to determine the average grain size present in the image.
Some detectors also provide a function detect_interactive ( filename ) that allows the user to adjust
algorithm specific settings in an interactive graphical environment to change the result until satisfied.
Image Evaluation Modules
The image evaluation modules are stored in the evaluators subdirectory. They all provide an
evaluate ( filename , mask) function that takes evaluates the binary masks that are generated in the
detector modules. The function takes the file name of the image that is currently getting processed
and the accompanying generated mask containing all detected edges and returns a tuple containing
algorithm specific metrics.
Some evaluator modules also provide an evaluate_interactive ( filename , mask) function that, similar to the detector modules, provide a graphical interactive environment to check and correct results,
and an evaluate_tune ( filename , mask) function that is only used in parameter tuning, comparing
detected results to validation data.
## Parameter Tuning Tool
The script used for parameter tuning can be started with:
$ python3 parameter_tuning.py
It uses the Ray Tune framework[9] using parameters defined directly in the script. Its functionality is
further explained in chapter 3. It was used during development to optimize default parameters for best
results without the user having to correct them using the interactive mode.
## Data Evaluation Tool
The data used in chapter 4 was generated by this script. It can be started using:
$ python3 mass_evaluator.py [ o p t i o n s ]
Starting with the −h option will bring up a help text to provide more information about its functionality.
The tool can either be used to process every image stored in the data subdirectory, storing evaluation results in a csv file, or to compare detected results to pre-processed data to evaluate
performance, robustness and accuracy.
