# -*- coding: utf-8 -*-

__author__ = 'Mr.Bemani'

import cv2 as cv
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:
    raise ImportError('Please install RKNNToolkit first.')


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
    if len(image.shape) != 3 and len(image.shape) != 2:
        raise ValueError('The input image must be a (h, w, c) or (h, w) image.')
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if image.shape[2] != 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
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
def preprocess_image(image: np.ndarray, input_size: tuple = (224, 224)) -> np.ndarray:
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
    image = np.transpose(image, (2, 0, 1))  # Change from HWC to CHW format
    image = np.expand_dims(image, axis=0)  # Add one dimension
    return image


# Postprocess the output results
def postprocess(output):
    # Implement your custom postprocessing logic here
    # softmax
    output = np.exp(output) / np.sum(np.exp(output))
    return output


# Run inference
def run_inference(rknn_model, image, input_size=(224, 224)):
    image = preprocess_image(image, input_size)
    outputs = rknn_model.inference(inputs=[image])[0]
    outputs = postprocess(outputs)
    return outputs

if __name__ == '__main__':
    import sys
    # Load the model
    rknn_model = load_rknn_model(sys.argv[1])

    # Read the image
    image = cv.imread(sys.argv[2])

    # Run inference
    outputs = run_inference(rknn_model, image)[0]
    top1 = np.argmax(outputs)
    conf1 = outputs[top1]

    # Print the result
    print (outputs)
    print (top1+1, conf1)

    # Release the model
    rknn_model.release()

