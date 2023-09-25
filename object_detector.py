# -*- encoding: utf-8 -*-
# yolo util functions
#

__author__ = 'Mr.Bemani'


import sys
import os
import cv2 as cv
import numpy as np
import logging

rknnsim = False
try:
    from rknnlite.api import RKNNLite
except ImportError:
    try:
        from rknn.api import RKNN as RKNNLite
        rknnsim = True
    except ImportError:
        raise ImportError('Please install RKNN-Toolkit first.')


def load_rknn_model(model_path: str):
    if model_path is None:
        raise Exception('Model path is None')
    
    if not os.path.exists(model_path):
        raise Exception('Model path is not exists')
    
    if not os.path.isfile(model_path):
        raise Exception('Model path is not a file')
    
    if not model_path.endswith('.rknn'):
        raise Exception('Model path is not a rknn file')

    # Create RKNN object
    rknn_lite = RKNNLite()

    # check RKNN initialized
    if rknn_lite is None:
        logging.error('RKNN not initialized.')
        raise Exception('RKNN not initialized.')
    
    # load RKNN model
    logging.info('--> Load RKNN model')
    if rknnsim:
        ret = rknn_lite.load_rknn(model_path, target='rk3588')
    else:
        ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        logging.error('Load RKNN model failed with: ', ret)
        raise Exception('Load RKNN model failed with: ', ret)

    # init runtime environment
    logging.info('--> Init runtime environment')
    # run on RK356x/RK3588 with Debian OS, do not need specify target.
    ret = rknn_lite.init_runtime()
    if ret != 0:
        logging.error('Init runtime environment failed with: ', ret)
        raise Exception('Init runtime environment failed with: ', ret)

    return rknn_lite


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


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def yolo_process(input, mask, anchors, in_size=416):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (in_size, in_size)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh=0.5):
    """Filter boxes with object threshold.

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= obj_thresh)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores, nms_thresh=0.5):
    """Suppress non-maximum boxes.

    # Arguments
        boxes: ndarray, boxes of objects. shape=(N, 4), (x1, y1, x2, y2), (0, 0) is top left. (x2, y2) is bottom right. x1 < x2, y1 < y2. x1, y1, x2, y2 are all in [0, 1]. 
        scores: ndarray, scores of objects. values are in [0, 1].
        nms_thresh: float, threshold for nms. values are in [0, 1].

    # Returns
        keep: ndarray, indexes of boxes to keep.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def yolov3_post_process(input_data, is_tiny=False, in_size=416, nms_thresh=0.5, obj_thresh=0.5):
    masks = []
    anchors = []
    
    if is_tiny:
        # yolov3-tiny
        masks = [[3, 4, 5], [0, 1, 2]]
        anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]
    else:
        # yolov3
        masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                  [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = yolo_process(input, mask, anchors, in_size)
        b, c, s = filter_boxes(b, c, s, obj_thresh)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, nms_thresh)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


# check if box1 in box2
def box_in_box(box1, box2):
    return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]


def get_intersection_box(box1, box2):
    """
    Returns the intersection box if box1 and box2 intersect, otherwise returns None.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 < x2 and y1 < y2:
        return [x1, y1, x2, y2]
    else:
        return None


def remove_inside_boxes(boxes):
    """
    Removes boxes that are inside another box.
    """
    # Convert numpy array to list of boxes
    if isinstance(boxes, np.ndarray):
        boxes_list = boxes.tolist()
    else:
        boxes_list = boxes
    
    # Check for boxes that are inside another box
    inside_boxes = []
    for i, box1 in enumerate(boxes_list):
        for j, box2 in enumerate(boxes_list):
            if i != j:
                x1, y1, w1, h1 = box1[0], box1[1], box1[2] - box1[0], box1[3] - box1[1]
                x2, y2, w2, h2 = box2[0], box2[1], box2[2] - box2[0], box2[3] - box2[1]
                if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                    inside_boxes.append(i)
                    break
    
    # Remove inside boxes
    boxes_list = [box for i, box in enumerate(boxes_list) if i not in inside_boxes]
    
    # Convert list of boxes back to numpy array
    boxes_array = np.array(boxes_list)
    
    return boxes_array


def combine_boxes(box1, box2):
    """
    Returns the smallest box that contains both box1 and box2.
    """
    if box1[4] == box2[4]:
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return [x1, y1, x2, y2, box1[4], max(box1[5], box2[5])]
    else:
        return None
    

def combine_overlapping_boxes(boxes, iou_threshold=0.9):
    """
    Combines overlapping boxes into a single box.
    """
    # Convert numpy array to list of boxes
    if isinstance(boxes, np.ndarray):
        boxes_list = boxes.tolist()
    else:
        boxes_list = boxes
    
    # Combine overlapping boxes
    combined_boxes = []
    for i, box1 in enumerate(boxes_list):
        for j, box2 in enumerate(boxes_list):
            if i != j:
                intersection_box = get_intersection_box(box1, box2)
                if intersection_box:
                    intersection_area = (intersection_box[2] - intersection_box[0]) * (intersection_box[3] - intersection_box[1])
                    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    iou = intersection_area / (box1_area + box2_area - intersection_area)
                    if iou >= iou_threshold:
                        combined_box = combine_boxes(box1, box2)
                        if combined_box is not None:
                            combined_boxes.append(combined_box)
    
    # Remove combined boxes
    boxes_list = [box for box in boxes_list if box not in combined_boxes]
    
    # Add combined boxes
    boxes_list.extend(combined_boxes)
    
    # Convert list of boxes back to numpy array
    boxes_array = np.array(boxes_list)
    
    return boxes_array
