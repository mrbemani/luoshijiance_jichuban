# -*- coding: utf-8 -*-

__author__ = 'Mr.Bemani'

import cv2 as cv
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
 
device = "cuda" if torch.cuda.is_available() else "cpu"
print ("device: " + device)
input_size = (224, 224)
 
# Data transformations
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load the RKNN model
def load_rknn_model(model_path):
    rknn = torch.load(model_path)
    print (rknn)
    #rknn.to(device)
    # add release method to rknn object
    def release():
        print ("fake model.release() for compatibility")
    if not hasattr(rknn, 'release'):
        rknn.release = release
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
def preprocess_image(image: np.ndarray, input_size: tuple = (224, 224)) -> torch.Tensor:
    if image is None:
        raise ValueError('The input image is None.')
    if len(image.shape) != 3:
        raise ValueError('The input image must be a 3-channel image.')
    h, w, c = image.shape
    if c != 3:
        raise ValueError('The input image must be a 3-channel image.')
    sqr_image = None
    if h / w < 0.9 or h / w > 1.1:
        sqr_image = pad_image(image)
    else:
        sqr_image = image
    if sqr_image.shape[0] != input_size[1] or sqr_image.shape[1] != input_size[0]:
        sqr_image = cv.resize(sqr_image, input_size)  # Resize the image according to your model input size
    rgb_sqr_image = cv.cvtColor(sqr_image, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_sqr_image)
    torch_image = transform(pil_image)
    return torch_image


# Postprocess the output results
def postprocess(output):
    # Implement your custom postprocessing logic here
    # softmax
    output = np.exp(output) / np.sum(np.exp(output))
    return output


# Run inference
def run_inference(rknn_model, image, input_size=(224, 224)):
    torch_image = preprocess_image(image, input_size)
    images = torch_image.unsqueeze(0)#.to(device)
    # evaluate model
    rknn_model.eval()
    with torch.no_grad():
        outputs = rknn_model(images).cpu().numpy()
        return postprocess(outputs)

if __name__ == '__main__':
    import sys
    # Load the model
    rknn_model = load_rknn_model(sys.argv[1])

    # Read the image
    image = cv.imread(sys.argv[2])

    # Run inference
    outputs = run_inference(rknn_model, image)

    print (outputs)

    # Release the model
    rknn_model.release()

