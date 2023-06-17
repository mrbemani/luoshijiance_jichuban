# -*- encoding: utf-8 -*-


__author__ = "Shi Qi"

import sys
import cv2
import numpy as np
import time
import copy
import yaml
from addict import Dict

selected_shape_idx = -1
objs = []
canvas_state = 0
drawing_shape = []
mouse_pt = (0, 0)
EDITOR_WIN_NAME = "Rock Boundaries Editor - [ ENTER when done / Esc to discard ]"

def findShapeAtPoint(pt):
    global objs, drawing_shape
    idx = 0
    for poly in iter(objs):
        if cv2.pointPolygonTest(np.array(poly), pt, measureDist=False) > 0:
            return idx
        idx = idx + 1
    return -1

def mouse_evt(event, x, y, flags, param):
    global objs, canvas_state, drawing_shape, mouse_pt
    if canvas_state == 0:
        if event == cv2.EVENT_LBUTTONUP:
            selected_shape_idx = findShapeAtPoint((x, y))
            if selected_shape_idx >= 0:
                drawing_shape = objs[selected_shape_idx]
                del objs[selected_shape_idx]
            else:
                drawing_shape = [[x, y]]
            canvas_state = 1
    elif canvas_state == 1:
        if event == cv2.EVENT_MBUTTONUP:
            if len(drawing_shape) > 2:
                objs.append(drawing_shape)
            drawing_shape = []
            canvas_state = 0
        elif event == cv2.EVENT_LBUTTONUP:
            drawing_shape.append([x, y])
        elif event == cv2.EVENT_RBUTTONUP:
            if len(drawing_shape) > 0:
                del drawing_shape[-1]
            else:
                drawing_shape = []
            if len(drawing_shape) == 0:
                canvas_state = 0
    mouse_pt = (x, y)
    

def showEditWin(bg, polys):
    global objs, canvas_state, drawing_shape, mouse_pt
    objs = []
    objs.extend(polys)
    cv2.namedWindow(EDITOR_WIN_NAME)
    is_editing = True
    ret = False
    while is_editing:
        cv2.setMouseCallback(EDITOR_WIN_NAME, mouse_evt)
        frame = cv2.drawContours(bg.copy(), [np.array(x) for x in objs], -1, (0, 255, 255), 2)
        if canvas_state == 1 and len(drawing_shape) > 0:
            if len(drawing_shape) > 1:
                i = 0
                while i + 1 < len(drawing_shape):
                    frame = cv2.line(frame, tuple(drawing_shape[i]), tuple(drawing_shape[i+1]), (255, 255, 255), 2)
                    i = i + 1
            frame = cv2.line(frame, tuple(drawing_shape[-1]), tuple(mouse_pt), (255, 255, 255), 2)
            frame = cv2.line(frame, tuple(mouse_pt), tuple(drawing_shape[0]), (255, 255, 255), 2)
        k = cv2.waitKey(5)
        cv2.imshow(EDITOR_WIN_NAME, frame)
        if k == 27:
            is_editing = False
            ret = False
        elif k == 13:
            is_editing = False
            ret = True
    cv2.destroyAllWindows()
    return ret, objs