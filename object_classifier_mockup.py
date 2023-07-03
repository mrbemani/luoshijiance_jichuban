# -*- coding: utf-8 -*-

__author__ = 'Mr.Bemani'

import numpy as np
import cv2 as cv


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


def load_rknn_model(fpath: str):
    def release():
        print ("fake model.release()")
    # create a fake model with release() method
    model = type('model', (), {'release': release})
    return model


def run_inference(model, sqr_patch):
    return [[0., 0., 0., 0., 0.]]

