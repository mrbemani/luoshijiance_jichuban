# -*- encoding: utf-8 -*-

from typing import Tuple
import zipfile
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

import logging
from logging.handlers import TimedRotatingFileHandler
import re
import sys

# Define a pattern to exclude from logs and stdout
EXCLUDE_PATTERN = re.compile(r'RKNN', re.IGNORECASE)
PRINT_PATTERN = re.compile(r'PRINT:')

# Function to create a handler for a specific log level and file
def create_handler(filename, level):
    handler = TimedRotatingFileHandler(filename, when='midnight', interval=1, backupCount=10)
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    return handler

# Create the base logger
tslogger = logging.getLogger('ts_app_logger')
tslogger.setLevel(logging.DEBUG)  # Capture all levels of log messages

# Create handlers for different log levels
handlers = {
    logging.DEBUG: create_handler('debug_daily.log', logging.DEBUG),
    logging.INFO: create_handler('info_daily.log', logging.INFO),
    logging.WARNING: create_handler('warning_daily.log', logging.WARNING),
    logging.ERROR: create_handler('error_daily.log', logging.ERROR),
    logging.CRITICAL: create_handler('critical_daily.log', logging.CRITICAL),
}

# Add handlers to the logger
for handler in handlers.values():
   tslogger.addHandler(handler)

# Override the built-in print function
def print(*args, **kwargs):
    # Convert all arguments to string and concatenate them
    message = ' '.join(map(str, args))
    # Check if the message contains the excluded pattern
    if not EXCLUDE_PATTERN.search(message):
        # If it doesn't contain the pattern, log the message and print it
        if PRINT_PATTERN.search(message):
            sys.stdout.write(message, **kwargs)
        else:
            tslogger.info(message)  # Adjust the logging level if necessary
    # If the pattern is found, neither log nor print the message



# pad image to square
def square_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError('The input image is None.')
    if len(image.shape) != 3 and len(image.shape) != 2:
        raise ValueError('The input image must be a (h, w, c) or (h, w) image.')
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[0], image.shape[1]
    if height == width:
        return image
    elif height > width:
        pad = (height - width) // 2
        image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return image
    else:
        pad = (width - height) // 2
        image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return image


def pad_cvimg_to_square(cvimg: np.ndarray, dest_size: Tuple[int, int]) -> np.ndarray:
    h, w = cvimg.shape[:2]
    if h == w:
        return cv2.reisze(cvimg, dest_size, interpolation=cv2.INTER_AREA)
    elif h > w:
        pad = h - w
        _im = cv2.copyMakeBorder(cvimg, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return cv2.resize(_im, dest_size, interpolation=cv2.INTER_AREA)
    else:
        pad = w - h
        _im = cv2.copyMakeBorder(cvimg, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return cv2.resize(_im, dest_size, interpolation=cv2.INTER_AREA)


def zip_dir(dirname, zipfilename, clear_dir=False):
    logging.debug(f"zipping dir: {dirname}")
    filelist = []
    for root, dirs, files in os.walk(dirname):
        for dir in dirs:
            filelist.append(os.path.join(root,dir))
        for name in files:
            filelist.append(os.path.join(root, name))
    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:
        arcname = tar[len(dirname):]
        logging.debug(f"zipping {arcname} to {zipfilename}")
        zf.write(tar, arcname)
    zf.close()
    logging.debug(f"done zipping dir.")


def cv_put_text_zh(img, text, position, textColor=(0, 255, 255), textSize=20, font_file: str = 'assets/simhei.ttf'):
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    pilImg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(font_file, textSize, encoding="utf-8")
    ImageDraw.Draw(pilImg).text(position, text, textColor, font=font)
    return cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)


def draw_translucent_box(image, box, color=(255, 255, 255), alpha=0.5):
    overlay = image.copy()
    output = image.copy()
    cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def select_files_by_timestamp_range(folder_path: str, ts_start: int, ts_end: int):
    selected_files = []
    if ts_start >= ts_end:
        return []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Get the timestamp from the filename
            timestamp = int(os.path.splitext(filename)[0])
            
            # Check if the timestamp falls within the specified range
            if ts_start <= timestamp <= ts_end:
                selected_files.append(file_path)
        except ValueError:
            # Skip files with invalid timestamp format
            continue
    
    return selected_files


def is_mostly_green(image, threshold=0.5):
    if image is None:
        return True
    
    if image.shape[0] * image.shape[1] < 1:
        return True
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Create a mask to filter out green pixels
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Count the number of green pixels
    green_pixels = cv2.countNonZero(mask)

    # Calculate the percentage of green pixels
    total_pixels = image.shape[0] * image.shape[1]
    green_percentage = green_pixels / total_pixels

    # Compare the green percentage with the threshold
    if green_percentage >= threshold:
        return True
    else:
        return False


def is_mostly_red(image, threshold=0.5):

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    lower_red = np.array([0, 40, 40])
    upper_red = np.array([10, 255, 255])

    # Create a mask to filter out red pixels
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Count the number of red pixels
    red_pixels = cv2.countNonZero(mask)

    # Calculate the percentage of red pixels
    total_pixels = image.shape[0] * image.shape[1]
    red_percentage = red_pixels / total_pixels

    # Compare the red percentage with the threshold
    if red_percentage >= threshold:
        return True
    else:
        return False


def is_mostly_blue(image, threshold=0.5):

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for blue color in HSV
    lower_blue = np.array([110, 40, 40])
    upper_blue = np.array([130, 255, 255])

    # Create a mask to filter out blue pixels
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Count the number of blue pixels
    blue_pixels = cv2.countNonZero(mask)

    # Calculate the percentage of blue pixels
    total_pixels = image.shape[0] * image.shape[1]
    blue_percentage = blue_pixels / total_pixels

    # Compare the blue percentage with the threshold
    if blue_percentage >= threshold:
        return True
    else:
        return False
    

def is_mostly_yellow(image_path, threshold=0.5):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for yellow color in HSV
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([40, 255, 255])

    # Create a mask to filter out yellow pixels
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Count the number of yellow pixels
    yellow_pixels = cv2.countNonZero(mask)

    # Calculate the percentage of yellow pixels
    total_pixels = image.shape[0] * image.shape[1]
    yellow_percentage = yellow_pixels / total_pixels

    # Compare the yellow percentage with the threshold
    if yellow_percentage >= threshold:
        return True
    else:
        return False

# thermal_zone0: soc-thermal
# thermal_zone1: bigcore0-thermal
# thermal_zone2: bigcore1-thermal
# thermal_zone3: littlecore-thermal
# thermal_zone4: center-thermal
# thermal_zone5: gpu-thermal
# thermal_zone6: npu-thermal

def soc_temperature():
    return float(os.popen("cat /sys/class/thermal/thermal_zone0/temp").read().strip())

def center_temperature():
    return float(os.popen("cat /sys/class/thermal/thermal_zone4/temp").read().strip())

def cpu_temperature():
    bigcore0 = float(os.popen("cat /sys/class/thermal/thermal_zone1/temp").read().strip())
    bigcore1 = float(os.popen("cat /sys/class/thermal/thermal_zone2/temp").read().strip())
    littlecore = float(os.popen("cat /sys/class/thermal/thermal_zone3/temp").read().strip())
    return (bigcore0, bigcore1, littlecore)

def gpu_temperature():
    return float(os.popen("cat /sys/class/thermal/thermal_zone5/temp").read().strip())

def npu_temperature():
    return float(os.popen("cat /sys/class/thermal/thermal_zone6/temp").read().strip())

