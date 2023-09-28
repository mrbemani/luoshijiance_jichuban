# -*- encoding: utf-8 -*-

__author__ = "Mr.Bemani"

import sys
import os
import traceback
import json
import shutil
import random
from typing import Callable, List
import queue
from addict import Dict
import cv2
import numpy as np
import logging
from datetime import datetime
import math
import tsutil
from fuse_output import make_fuse_output


cv2.ocl.setUseOpenCL(True)

logging.basicConfig(level=logging.DEBUG)

if "PC_TEST" in os.environ and os.environ["PC_TEST"] == "1":
    logging.info("PC_TEST is set, object classification not available")
    #import object_classifier_mockup as objcls
else:
    try:
        USE_RKNN = True
        import object_detector as odet
        #import object_classifier as objcls
    except ImportError:
        USE_RKNN = False
        logging.error("Failed to import object_classifier. Object classifier not available")
        import object_detector_pc as odet
from tracker import Tracker


from event_utils import create_falling_rock_event, store_event
try:
    from sms import send_sms
except:
    logging.error("Failed to import sms. SMS not available")
    # print traceback
    # traceback.print_exc()
    def send_sms(config, rock_evt):
        logging.info("fake send_sms: " + repr(rock_evt))

PF_W = 1920
PF_H = 1080

MAX_MOVE_UP = 10
MAX_GAP_SECONDS = 5

original_frame = None
video_src_ended = False
frame_update_time = datetime.utcnow().timestamp()
bad_video_src = False


def draw_object_tracks(frame: np.ndarray, objtracks: Dict):
    for track_id, track in objtracks.items():
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



def process_frame_loop(config: dict, main_loop_running_cb: Callable, frame_queue: queue.Queue, current_frame: np.ndarray, extra_info: Dict):
    global original_frame, video_src_ended, frame_update_time, bad_video_src

    YOLO_SPAN = 3
    YOLO_NUM_CLASS = 80
    YOLO_IN_SIZE = 416
    rknn_det = None
    if USE_RKNN:
        rknn_det = odet.load_rknn_model("models/yolo.rknn")
    model = None
    last_patch = None

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
    tracker = Tracker(
        config.tracking.dist_thresh, 
        config.tracking.max_skip_frame, 
        config.tracking.max_trace_length, 1)

    rock_evt = None
    last_frame_ts = 0
    objtrace = []
    objtracks = dict()
    extra_info.objtracks = objtracks
    while main_loop_running_cb():
        info_text = format_info_text()
        fps_f = cv2.getTickCount()

        ts_now = datetime.now().timestamp()
        if rock_evt is not None and ts_now - rock_evt.ts_end > MAX_GAP_SECONDS:
            # too slow, discard event
            if rock_evt.max_speed < config.tracking.min_speed_threshold:
                logging.debug(f"Discarding event, too slow: {rock_evt.max_speed}")
                # remove record_id folder
                shutil.rmtree(rock_evt.frame_dir)
            else:#if True: # always save event
                valid_tracks = dict()
                for obj_id in objtracks:
                    if len(objtracks[obj_id][1]) < config.tracking.min_trace_length: # if trace length is less than config.min_trace_length, discard
                        continue
                    track_start_x = objtracks[obj_id][1][0][1][0]
                    track_start_y = objtracks[obj_id][1][0][1][1]
                    track_end_x = objtracks[obj_id][1][-1][1][0]
                    track_end_y = objtracks[obj_id][1][-1][1][1]
                    delta_x = max(0.00001, abs(track_end_x - track_start_x))
                    delta_y = track_end_y - track_start_y
                    # if y motion is less than downward 2 pixels, discard
                    if delta_y < config.tracking.min_y_motion * PF_H:
                        continue
                    # if y motion is less than config.min_y_x_ratio times x motion, discard
                    if delta_y / delta_x < config.tracking.min_y_x_ratio:
                        continue
                    # if not discard, add to valid_tracks
                    valid_tracks[obj_id] = objtracks[obj_id]
                if len(valid_tracks) > 0:
                    store_event("events.csv", rock_evt)
                    # select ffmpeg recorded file-range and copy to a distinct folder with timestamp as folder name.
                    selected_videos = tsutil.select_files_by_timestamp_range(config.vcr_path, rock_evt.ts_start - 4, rock_evt.ts_end + 4)
                    for sel_idx, sel_vid in enumerate(selected_videos):
                        shutil.copy(sel_vid, rock_evt.frame_dir)
                    # save tracks
                    json.dump(valid_tracks, open(os.path.join(rock_evt.frame_dir, "trace.json"), "w"), indent=2)
                    if USE_RKNN:
                        send_sms(config, rock_evt)
                    make_fuse_output(rock_evt.record)
                else:
                    # remove record_id folder
                    shutil.rmtree(rock_evt.frame_dir)
            objtracks.clear()
            rock_evt = None
        

        frame_start_time = datetime.utcnow()
        
        frame = None
        try:
            frame = frame_queue.get(timeout=10.0)
        except queue.Empty:
            logging.debug("frame_queue empty")
            bad_video_src = True
            logging.error("!!! Bad Video Source !!!")
            logging.error("Restarting Program...")
            os._exit(0)

        if frame is None:
            print ("frame is empty")
            continue

        #frame = cv2.medianBlur(frame, 5)

        original_frame = frame

        original_w, original_h = original_frame.shape[1], original_frame.shape[0]
        det_w = config.tracking.det_w
        det_h = config.tracking.det_h
        pf_ratio_w = original_w / det_w
        pf_ratio_h = original_h / det_h


        yl_dets = []
        if USE_RKNN:
            img = tsutil.pad_cvimg_to_square(frame, (YOLO_IN_SIZE, YOLO_IN_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            outputs = rknn_det.inference(inputs=[img])
            reshaped_outputs = []
            for i in range(len(outputs)):
                reshaped_outputs.append(outputs[i].reshape(YOLO_SPAN, (YOLO_NUM_CLASS + 5), outputs[i].shape[2], outputs[i].shape[3]))
    
            input_data = []
            for i in range(len(reshaped_outputs)):
                input_data.append(np.transpose(reshaped_outputs[i], (2, 3, 0, 1)))

            yl_boxes, yl_classes, yl_scores = odet.yolov3_post_process(input_data, is_tiny=True, in_size=YOLO_IN_SIZE, nms_thresh=0.5, obj_thresh=0.7)
            # remove boxes with class_id = 0 or 2.
            yl_boxes = np.array(yl_boxes)
            yl_classes = np.array(yl_classes)
            yl_scores = np.array(yl_scores)
            yl_boxes_0 = yl_boxes[yl_classes == 0]
            #yl_scores_0 = yl_scores[yl_classes == 0]
            yl_boxes_2 = yl_boxes[yl_classes == 2]
            #yl_scores_2 = yl_scores[yl_classes == 2]
            # concatenate
            yl_boxes = np.concatenate((yl_boxes_0, yl_boxes_2))
            yl_boxes = odet.remove_inside_boxes(yl_boxes)
            yl_dets = odet.combine_overlapping_boxes(yl_boxes, 0.5)
        if yl_dets is None:
            yl_dets = []

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
            frame_update_time = datetime.utcnow().timestamp()
            continue

        # apply connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8, ltype=cv2.CV_32S)
        extra_info.centroids = centroids
        extra_info.labels = labels
        extra_info.stats = stats
        extra_info.num_labels = num_labels
        
        centers = []
        obj_rects = []
        
        # Find centers of all detected objects
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if w / h > 3 or h / w > 3:
                continue

            if roi_mask is not None and roi_mask[y+h//2, x+w//2] == 0:
                continue
            
            center_x = x+w//2
            center_y = y+h//2

            in_yolo = False
            for yl_det in yl_dets:
                # if center_x, center_y in yl_det, skip
                yl_x, yl_y, yl_w, yl_h = yl_det
                if yl_x <= center_x <= yl_x+yl_w and yl_y <= center_y <= yl_y+yl_h:
                    in_yolo = True
                    break
            if in_yolo:
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
                    sqr_patch = tsutil.square_image(patch)
                    # get average color
                    is_green = tsutil.is_mostly_green(sqr_patch, config.green_max_ratio)
                    # if not green, skip
                    if not is_green:
                        center = np.array ([[x+w/2], [y+h/2]])
                        centers.append(np.round(center))
                        obj_rects.append([x, y, w, h, area])

        obj_cnt = len(centers)
        objtrace.append(centers)

        
        # if too many objects detected, skip this frame
        if obj_cnt > config.max_detection:
            #logging.debug(f"Too many movements detected, skipping frame: {num_labels}")
            _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            current_frame[:] = _frame[:]
            frame_update_time = datetime.utcnow().timestamp()
            continue

        if config.debug:
            if obj_cnt > 0:
                print ("object_count: ", obj_cnt)

        if centers:
            if rock_evt is None:
                rock_evt = create_falling_rock_event()
                rock_evt.frame_dir = os.path.join(config.output_dir, str(int(rock_evt.ts_start*1000)))
                rock_evt.obj_total = 0
                rock_evt.max_offset = 0
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
            max_vol = np.max(volumes)
            max_cnt = len(volumes)
            tracker.update(centers, allow_moving_up=True, move_up_thresh=10)
            obj_cnt = max(len(tracker.tracks), len(centers))
            obj_path_length = 0
            if len(tracker.tracks) < config.tracking.max_object_count:
                for tracked_object in tracker.tracks:
                    if tracked_object.track_id not in objtracks:
                        objtracks[tracked_object.track_id] = [(),[]]
                        randcolor = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                        objtracks[tracked_object.track_id][0] = randcolor
                    #if len(tracked_object.trace) <= 1:
                    #    print (tracked_object.track_id, tracked_object.trace)
                    if len(tracked_object.trace) > 1:
                        for j in range(len(tracked_object.trace)-1):
                            #if j < 2:
                            #    continue
                            # Draw trace line
                            x1 = tracked_object.trace[j][0][0]
                            y1 = tracked_object.trace[j][1][0]

                        trace_x = tracked_object.trace[-1][0][0]
                        trace_y = tracked_object.trace[-1][1][0]

                        trace_x0 = tracked_object.trace[1][0][0]
                        trace_y0 = tracked_object.trace[1][1][0]

                        
                        if len(tracked_object.trace) > 2:
                            obj_path_length = math.sqrt((trace_x-trace_x0)**2 + (trace_y-trace_y0)**2)
                            # Check if tracked object has reached the speed detection line
                            load_lag = (datetime.utcnow() - frame_start_time).total_seconds()
                            time_dur = (datetime.utcnow() - tracked_object.start_time).total_seconds() - load_lag
                            if time_dur == 0:
                                tracked_object.speed = 0
                            else:
                                tracked_object.speed = obj_path_length / time_dur / 100.0
                            rock_evt.max_speed = max(rock_evt.max_speed, tracked_object.speed * config.frame_dist_cm / 100)

                            # Display speed if available
                            cv2.putText(frame, "SPD: {} M/s".format(round(tracked_object.speed, 2)), (int(trace_x), int(trace_y)), font, 3, (0, 255, 255), 2, cv2.LINE_AA)
                            #cv2.putText(frame, 'ID: '+ str(tracked_object.track_id), (int(trace_x), int(trace_y)), font, 3, (255, 255, 255), 2, cv2.LINE_AA)
                    if len(tracked_object.trace) > 0:
                        this_pt = [int(tracked_object.trace[-1][0][0]), int(tracked_object.trace[-1][1][0])]
                        ts_this_pt = datetime.now().timestamp() - rock_evt.ts_start
                        objtracks[tracked_object.track_id][1].append((ts_this_pt, this_pt))
                    
            rock_evt.max_vol = max(rock_evt.max_vol, max_vol / 1_000_000)
            rock_evt.max_count = max(rock_evt.max_count, max_cnt)
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
        frame_update_time = datetime.utcnow().timestamp()

    # Clean up
    if rknn_det is not None:
        rknn_det.release()
    if model is not None:
        model.release()
    video_src_ended = True
    if config.debug:
        print ("main_loop aborted.")


