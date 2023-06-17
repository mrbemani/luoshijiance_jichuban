# -*- coding: utf-8 -*-

__author__ = 'Mr.Bemani'

import cv2 as cv
import numpy as np
from rknnlite.api import RKNNLite


# Load the RKNN model
def load_rknn_model(model_path):
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print('Failed to load the RKNN model.')
        exit(-1)
    rknn.init_runtime()
    return rknn


# padd image to square
def pad_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError('The input image is None.')
    if len(image.shape) != 3:
        raise ValueError('The input image must be a 3-channel image.')
    if image.shape[2] != 3:
        raise ValueError('The input image must be a 3-channel image.')
    height, width = image.shape[0], image.shape[1]
    if height == width:
        return image
    elif height > width:
        pad = (height - width) // 2
        image = cv.copyMakeBorder(image, 0, 0, pad, pad, cv.BORDER_CONSTANT, value=(0, 0, 0))
        return image
    else:
        pad = (width - height) // 2
        image = cv.copyMakeBorder(image, pad, pad, 0, 0, cv.BORDER_CONSTANT, value=(0, 0, 0))
        return image


# Preprocess the input image
def preprocess_image(image: np.ndarray, input_size: tuple = (64, 64)) -> np.ndarray:
    if image is None:
        raise ValueError('The input image is None.')
    if len(image.shape) != 3:
        raise ValueError('The input image must be a 3-channel image.')
    h, w, c = image.shape
    if c != 3:
        raise ValueError('The input image must be a 3-channel image.')
    if h / w < 0.9 or h / w > 1.1:
        image = pad_image(image)
    if image.shape[0] != input_size[1] or image.shape[1] != input_size[0]:
        image = cv.resize(image, input_size)  # Resize the image according to your model input size
    image = np.float32(image)
    #image = (image - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]  # Apply normalization
    image = np.transpose(image, (2, 0, 1))  # Change from HWC to CHW format
    #image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Postprocess the output results
def postprocess(output):
    # Implement your custom postprocessing logic here
    # softmax
    output = np.exp(output) / np.sum(np.exp(output))
    return output


# Run inference
def run_inference(rknn_model, image):
    image = preprocess_image(image)
    outputs = rknn_model.inference(inputs=[image])[0]
    #print (outputs)
    outputs = postprocess(outputs)
    return outputs


