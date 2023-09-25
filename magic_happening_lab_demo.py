# -*- encoding: utf-8 -*-

__author__ = "Mr.Bemani"

import sys
import os
import json
import shutil
import random
from typing import Callable, Union, List
import multiprocessing as mp
import queue
from addict import Dict
import cv2
import numpy as np
import logging
from datetime import datetime
import math
import tsutil

cv2.ocl.setUseOpenCL(True)

logging.basicConfig(level=logging.DEBUG)

if "PC_TEST" in os.environ and os.environ["PC_TEST"] == "1":
    USE_RKNN = False
    logging.info("PC_TEST is set, object classification not available")
    import object_classifier_mockup as objcls
    import ultralytics as ulyt
else:
    try:
        USE_RKNN = True
        import object_classifier as objcls
    except ImportError:
        USE_RKNN = False
        logging.error("Failed to import object_classifier. Object classifier not available")
        import object_classifier_torch as objcls
from tracker import Tracker

if "LAB_DEMO" in os.environ and os.environ["LAB_DEMO"] == "1":
    LAB_DEMO = True
else:
    LAB_DEMO = False

from event_utils import create_falling_rock_event, store_event
#from sms import send_sms

PF_W = 1920
PF_H = 1080

MAX_MOVE_UP = 10
MAX_GAP_SECONDS = 3

original_frame = None
video_src_ended = False


def draw_object_tracks(frame: np.ndarray, objtracks: Dict):
    for track_id, track in objtracks.items():
        if len(track[1]) < 3:
            continue
        color = track[0]
        trace = track[1]
        for i in range(len(trace)-1):
            cv2.line(frame, trace[i][1], trace[i+1][1], color, 4)
        if len(trace) > 0:
            cv2.putText(frame, str(track_id), trace[-1][1], cv2.FONT_HERSHEY_PLAIN, 2, color, 2, cv2.LINE_AA)
    return frame


def format_info_text(rock_count=0, max_vol=0, max_speed=0, max_count=0):
    info_text = "   --- 统计数据 ---"
    info_text += "\n\n当前落石个数: {}".format(rock_count)
    info_text += "\n\n最大体积: {} 立方米".format(round(max_vol, 2))
    info_text += "\n\n最大速度: {} 米/秒".format(round(max_speed, 2))
    info_text += "\n\n最大个数: {}".format(max_count)
    return info_text


rock_count_history = queue.Queue(maxsize=400)
def draw_plot_rock_count_change(frame: np.ndarray, rock_count: int):
    if rock_count_history.full():
        rock_count_history.get()
    rock_count_history.put(rock_count)
    if rock_count_history.qsize() < 2:
        return frame
    color = (0, 255, 255)
    cv2.line(frame, (0, 400), (400, 400), color, 2)

    for i in range(rock_count_history.qsize()-1):
        pt1 = (i, 400 - rock_count_history.queue[i]*5)
        pt2 = (i+1, 400 - rock_count_history.queue[i+1]*5)
        cv2.line(frame, pt1, pt2, color, 2)
    return frame


def select_white_balls_in_frame(frame):
    # define range of white color in RGB
    # change it according to your need !
    lower_white = np.array([200,200,200], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(frame, lower_white, upper_white)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask= mask)

    # gray image
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # erode
    kernel = np.ones((3,3),np.uint8)
    res = cv2.erode(res,kernel,iterations = 3)

    # dilate
    kernel = np.ones((1,1),np.uint8)
    res = cv2.dilate(res,kernel,iterations = 3)

    # make a black 1920x300 rectangle
    res[0:300, 0:1920, :] = 0

    # morph
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # apply connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=8, ltype=cv2.CV_32S)

    boxes = []
    for c in stats:
        if c[0] < 400:
            continue
        if c[4] < 20 or c[4] > 700:
            continue
        boxes.append([c[0]-10, c[1]-10, c[2]+20, c[3]+20])

    return frame, boxes


def process_frame_loop(config: dict, main_loop_running_cb: Callable, frame_queue: Union[queue.Queue, mp.Queue], current_frame: np.ndarray, extra_info: Dict):
    global original_frame, video_src_ended

    human_mask = None
    yolo_model = None
    if not USE_RKNN:
        yolo_model = ulyt.YOLO("models/yolov8n.pt", task="detect")

    model = objcls.load_rknn_model(config.rknn_model_path)

    if LAB_DEMO:
        roi_mask = None
        if config.roi_mask is not None and os.path.isfile(config.roi_mask) and config.roi_mask != "None":
            roi_mask = cv2.imread(config.roi_mask, cv2.IMREAD_GRAYSCALE)
            roi_mask = cv2.resize(roi_mask, (PF_W, PF_H), interpolation=cv2.INTER_NEAREST)
    else:
        # load ROI Mask
        roi_mask = None
        if config.roi_mask is not None and os.path.isfile(config.roi_mask) and config.roi_mask != "None":
            roi_mask = cv2.imread(config.roi_mask, cv2.IMREAD_GRAYSCALE)
            roi_mask = cv2.resize(roi_mask, (config.tracking.det_w, config.tracking.det_h), interpolation=cv2.INTER_NEAREST)
            

    # Initial background subtractor and text font
    fgbg = cv2.createBackgroundSubtractorMOG2()
    font = cv2.FONT_HERSHEY_PLAIN

    blob_min_width_far = config.tracking.min_rock_pix
    blob_min_height_far = config.tracking.min_rock_pix

    blob_max_width_near = config.tracking.max_rock_pix
    blob_max_height_near = config.tracking.max_rock_pix

    frame_start_time = None

    # Create object tracker
    if LAB_DEMO:
        tracker = Tracker(
            150, 
            config.tracking.max_skip_frame, 
            config.tracking.max_trace_length, 1)
    else:
        tracker = Tracker(
            config.tracking.dist_thresh, 
            config.tracking.max_skip_frame, 
            config.tracking.max_trace_length, 1)

    rock_evt = None
    last_frame_ts = 0
    objtracks = dict()
    extra_info.objtracks = objtracks
    while main_loop_running_cb():
        max_cnt = 0
        max_vol = 0
        max_speed = 0
        obj_cnt = 0
        info_text = format_info_text()
        fps_f = cv2.getTickCount()

        ts_now = datetime.now().timestamp()
        if rock_evt is not None and ts_now - rock_evt.ts_end > MAX_GAP_SECONDS:
            # too slow, discard event
            if rock_evt.max_speed < 0.1:
                logging.debug(f"Discarding event, too slow: {rock_evt.max_speed}")
            else:#if True: # always save event
                pass
                #store_event("events.csv", rock_evt)
                #### to-do select ffmpeg recorded file-range and copy to a distinct folder with timestamp as folder name.
                #selected_videos = tsutil.select_files_by_timestamp_range(config.vcr_path, rock_evt.ts_start - 4, rock_evt.ts_end + 4)
                #for sel_idx, sel_vid in enumerate(selected_videos):
                #    shutil.copy(sel_vid, rock_evt.frame_dir)
                # save tracks
                json.dump(objtracks, open(os.path.join(rock_evt.frame_dir, "trace.json"), "w"), indent=2)
                #send_sms(config, rock_evt)
            objtracks.clear()
            rock_evt = None
        

        frame_start_time = datetime.utcnow()
        
        frame = frame_queue.get()
        if frame is None:
            print ("frame is empty")
            continue

        #frame = cv2.medianBlur(frame, 5)

        original_frame = frame

        if False:
            human_detected = False
            if not USE_RKNN:
                yolo_out = yolo_model(frame)
                if yolo_out is not None:
                    yolo_out = yolo_out[0]
                    bboxes = yolo_out.boxes
                    classes, scores = bboxes.cls, bboxes.conf
                    for i in range(len(classes)):
                        if classes[i] == 0 and scores[i] > 0.5:
                            human_detected = True
                            break
            
            if human_detected:
                #logging.debug("Human detected, skipping frame")
                _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
                current_frame[:] = _frame[:]
                continue

        original_w, original_h = original_frame.shape[1], original_frame.shape[0]
        det_w = config.tracking.det_w
        det_h = config.tracking.det_h
        pf_ratio_w = original_w / det_w
        pf_ratio_h = original_h / det_h
        

        
        centers = []
        obj_rects = []
        if LAB_DEMO:
            # use select_white_balls_in_frame
            frame, boxes = select_white_balls_in_frame(frame)
            for b in boxes:
                c = b[0] + b[2] // 2, b[1] + b[3] // 2
                if roi_mask is not None and roi_mask[c[1], c[0]] == 0:
                    continue
                centers.append(np.array([[c[0]], [c[1]]]))
                obj_rects.append([b[0], b[1], b[2], b[3], b[2]*b[3]])
        else:

            # Resize frame to fit the screen
            det_frame = cv2.resize(frame, (det_w, det_h))

            # Convert frame to grayscale and perform background subtraction
            gray = cv2.cvtColor (det_frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply (gray)

            # Perform some Morphological operations to remove noise
            #thresh = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            morph = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)

            extra_info.fgmask = fgmask
            extra_info.morph = morph
            extra_info.gray = gray
            extra_info.det_frame = det_frame
            
            avg_color = np.average(morph)
            extra_info.avg_color = avg_color
            if avg_color > 255//80:
                #logging.debug(f"Too much noise detected, skipping frame: {avg_color}")
                _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
                current_frame[:] = _frame[:]
                continue

            # apply connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8, ltype=cv2.CV_32S)
            extra_info.centroids = centroids
            extra_info.labels = labels
            extra_info.stats = stats
            extra_info.num_labels = num_labels
            
            # Find centers of all detected objects
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]

                if w / h > 3 or h / w > 3:
                    continue

                if roi_mask is not None and roi_mask[y+h//2, x+w//2] == 0:
                    continue

                x = int(x * pf_ratio_w)
                y = int(y * pf_ratio_h)
                w = int(w * pf_ratio_w)
                h = int(h * pf_ratio_h)

                if blob_max_width_near >= w >= blob_min_width_far and \
                    blob_max_height_near >= h >= blob_min_height_far:
                    max_wh = np.max([w, h])
                    max_wh = math.floor(max_wh * 1.2)
                    if max_wh < 224:
                        max_wh = 224
                    """
                    old_max_wh = max_wh
                    if max_wh < 224:
                        max_wh = 224
                    delta_xy = (max_wh - old_max_wh) // 2
                    _x1 = x - delta_xy
                    _y1 = y - delta_xy
                    """
                    _x1 = int(x - (w - max_wh) // 2)
                    _y1 = int(y - (h - max_wh) // 2)
                    _x1 = max(_x1, 0)
                    _y1 = max(_y1, 0)
                    max_x = _x1 + max_wh
                    max_y = _y1 + max_wh
                    if max_x >= original_w:
                        max_x = original_w - 1
                    if max_y >= original_h:
                        max_y = original_h - 1
                    #print (_x1, _y1, max_x, max_y, original_w, original_h)
                    patch = original_frame[_y1:max_y, _x1:max_x].copy()
                    if patch is None:
                        print ("---invalid patch---")
                    else:
                        #sqr_patch = objcls.pad_image(patch)
                        #if sqr_patch.shape[0] != 224 or sqr_patch.shape[1] != 224:
                        #    sqr_patch = cv2.resize(sqr_patch, (224, 224))
                        #outputs = objcls.run_inference(model, sqr_patch)[0]
                        #obj_class_index = int(np.argmax(outputs) + 1)
                        #obj_class_confidence = float(outputs[obj_class_index-1])
                        #print (f"obj_class_index: {obj_class_index}, obj_class_confidence: {obj_class_confidence}")
                        if True:#if obj_class_index == 1 and obj_class_confidence > 0.8:
                            center = np.array ([[x+w/2], [y+h/2]])
                            centers.append(np.round(center))
                            obj_rects.append([x, y, w, h, area])

            obj_cnt = len(centers)

        
        # if too many objects detected, skip this frame
        if obj_cnt > config.max_detection:
            #logging.debug(f"Too many movements detected, skipping frame: {num_labels}")
            _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            current_frame[:] = _frame[:]
            continue

        if config.debug:
            if obj_cnt > 0:
                print ("object_count: ", obj_cnt)

        #if LAB_DEMO:
        #    if centers:
        #        tracker.update(centers, allow_moving_up=True, move_up_thresh=200, dist_min_thresh=3)

        if True:
            if centers:
                if rock_evt is None:
                    rock_evt = create_falling_rock_event()
                    rock_evt.frame_dir = os.path.join(config.output_dir, str(int(rock_evt.ts_start*1000)))
                    rock_evt.obj_total = 0
                    rock_evt.max_vol = 0
                    rock_evt.max_count = 0
                    os.makedirs(rock_evt.frame_dir, exist_ok=True)
                    objtracks.clear()
                volumes = []
                for rt in obj_rects:
                    # draw rectangle
                    x, y, w, h, area = rt
                    # rotate area 180 degree to get volume
                    depth_esti = (w+h) // 2
                    obj_vol = w * h * depth_esti * config.frame_dist_cm / 100.0
                    volumes.append(obj_vol)
                if not LAB_DEMO:
                    max_vol = np.max(volumes)
                    max_cnt = len(volumes)
                tracker.update(centers, allow_moving_up=True, move_up_thresh=500)
                if not LAB_DEMO:
                    obj_cnt = max(len(tracker.tracks), len(centers))
                obj_path_length = 0
                if len(tracker.tracks) < config.tracking.max_object_count:
                    for tracked_object in tracker.tracks:
                        if (tracked_object.track_id not in objtracks) and len(tracked_object.trace) > 2:
                            dx = tracked_object.trace[-1][0][0] - tracked_object.trace[-2][0][0]
                            dy = tracked_object.trace[-1][1][0] - tracked_object.trace[-2][1][0]
                            distance = math.sqrt(dx**2 + dy**2)
                            if distance > 4:
                                objtracks[tracked_object.track_id] = [(),[]]
                                randcolor = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                                objtracks[tracked_object.track_id][0] = randcolor
                                obj_cnt += 1
                        elif (tracked_object.track_id in objtracks) and len(tracked_object.trace) > 2:
                            trace_x = tracked_object.trace[-1][0][0]
                            trace_y = tracked_object.trace[-1][1][0]

                            trace_x0 = tracked_object.trace[1][0][0]
                            trace_y0 = tracked_object.trace[1][1][0]
                            
                            dx = tracked_object.trace[-1][0][0] - tracked_object.trace[-2][0][0]
                            dy = tracked_object.trace[-1][1][0] - tracked_object.trace[-2][1][0]
                            distance = math.sqrt(dx**2 + dy**2)
                            if distance > 4:
                                obj_path_length = math.sqrt((trace_x-trace_x0)**2 + (trace_y-trace_y0)**2)
                                # Check if tracked object has reached the speed detection line
                                load_lag = (datetime.utcnow() - frame_start_time).total_seconds()
                                time_dur = (datetime.utcnow() - tracked_object.start_time).total_seconds() - load_lag
                                tracked_object.speed = obj_path_length / time_dur
                                rock_evt.max_speed = max(rock_evt.max_speed, tracked_object.speed * config.frame_dist_cm / 100)
                                obj_cnt += 1
                                # Display speed if available
                                cv2.putText(frame, "SPD: {} CM/s".format(round(tracked_object.speed, 2)), (int(trace_x), int(trace_y)), font, 3, (0, 255, 255), 2, cv2.LINE_AA)
                        if len(tracked_object.trace) > 2 and (tracked_object.track_id in objtracks):
                            this_pt = [int(tracked_object.trace[-1][0][0]), int(tracked_object.trace[-1][1][0])]
                            last_pt = [int(tracked_object.trace[-2][0][0]), int(tracked_object.trace[-2][1][0])]
                            distance = math.sqrt((this_pt[0]-last_pt[0])**2 + (this_pt[1]-last_pt[1])**2)
                            if distance > 4:
                                ts_this_pt = datetime.now().timestamp() - rock_evt.ts_start
                                objtracks[tracked_object.track_id][1].append((ts_this_pt, this_pt))

            if LAB_DEMO:
                max_cnt = max(obj_cnt, max_cnt)
                if max_cnt > 0:
                    max_vol = max(volumes)
                    max_speed = max(obj_path_length, max_speed)
            
            if rock_evt is not None:
                rock_evt.max_vol = max(rock_evt.max_vol, max_vol / 1_000_000)
                rock_evt.max_count = max(rock_evt.max_count, max_cnt)
                if obj_cnt > 0:
                    rock_evt.ts_end = datetime.now().timestamp()

        # Display all images
        if rock_evt is not None:
            info_text = format_info_text(obj_cnt, 
                                         rock_evt.max_vol, 
                                         rock_evt.max_speed, 
                                         rock_evt.max_count)
        else:
            info_text = format_info_text(obj_cnt, 
                                         0, 
                                         0, 
                                         0)

        
        frame = tsutil.draw_translucent_box(frame, (0, 0, 400, 400), (0, 0, 0), 0.618)

        frame = tsutil.cv_put_text_zh(frame, info_text, (16, 30), (255, 255, 255), 30)

        frame = draw_object_tracks(frame, objtracks)

        frame = draw_plot_rock_count_change(frame, obj_cnt)

        fps_e = (1.0 / ((cv2.getTickCount() - fps_f) / cv2.getTickFrequency()))
        fps_e = round(fps_e)

        #cv2.putText(frame, "FPS: {}".format(fps_e), (89, 124), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
        #cv2.putText(frame, "FPS: {}".format(fps_e), (87, 122), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        current_frame[:] = _frame[:]

    # Clean up
    if model is not None:
        model.release()
    video_src_ended = True
    if config.debug:
        print ("main_loop aborted.")


