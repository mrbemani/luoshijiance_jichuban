# -*- encoding: utf-8 -*-

__author__ = "Mr.Bemani"

import sys
import os
import shutil
import random
from typing import Callable, Union, List
import multiprocessing as mp
import queue
from addict import Dict
import cv2
import numpy as np
import time
import logging
import copy
from datetime import datetime
from PIL import Image
import math
import threading

logging.basicConfig(level=logging.DEBUG)

if "PC_TEST" in os.environ and os.environ["PC_TEST"] == "1":
    logging.info("PC_TEST is set, object classification not available")
    import object_classifier_mockup as objcls
else:
    try:
        import object_classifier as objcls
    except ImportError:
        logging.error("Failed to import object_classifier. Object classifier not available")
        import object_classifier_torch as objcls
from tracker import Tracker


from event_utils import create_falling_rock_event, store_event
#from sms import send_sms
from video import make_video_ffmpeg

PF_W = 1920
PF_H = 1080

MAX_MOVE_UP = 10
MAX_GAP_SECONDS = 5

original_frame = None
video_src_ended = False


def process_frame_loop(config: dict, main_loop_running_cb: Callable, frame_queue: Union[queue.Queue, mp.Queue], out_queue: queue.Queue, current_frame: np.ndarray, extra_info: Dict):
    global original_frame, video_src_ended

    model = objcls.load_rknn_model(config.rknn_model_path)

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
        fps_f = cv2.getTickCount()

        ts_now = datetime.now().timestamp()
        if rock_evt is not None and ts_now - rock_evt.ts_end > MAX_GAP_SECONDS:
            # too slow, discard event
            if rock_evt.max_speed < 0.1:
                logging.debug(f"Discarding event, too slow: {rock_evt.max_speed}")
                shutil.rmtree(rock_evt.frame_dir)
            else:
                store_event("events.csv", rock_evt)
                frame_files = [os.path.join(rock_evt.frame_dir, f) for f in os.listdir(rock_evt.frame_dir) if f.endswith(".jpg")]
                try:
                    if len(frame_files) > 1:
                        make_video_thread = threading.Thread(target=make_video_ffmpeg, args=(str(rock_evt.record),))
                        make_video_thread.start()
                except:
                    logging.error("Failed to run make_video thread")
                    logging.error(f"{sys.exc_info()[0]}")
                #send_sms(config, rock_evt)
            rock_evt = None

        if rock_evt is not None:
            if not out_queue.empty():
                try:
                    wfrm = out_queue.get(timeout=0.005)
                    if wfrm is not None:
                        wfrm_fname = f"{int(datetime.now().timestamp()*1000)}.jpg"
                        threading.Thread(target=cv2.imwrite, args=(os.path.join(rock_evt.frame_dir, wfrm_fname), wfrm, [cv2.IMWRITE_JPEG_QUALITY, 95])).start()
                except queue.Empty:
                    pass    

        frame_start_time = datetime.utcnow()
        
        frame = frame_queue.get()
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
            # draw objtracks
            for oidx, ot in objtracks.items():
                if len(ot) < 2 or len(ot[1]) < 1:
                    continue
                cr, xy = ot
                for j in range(len(xy)-1):
                    if j < 2:
                        cv2.circle(frame, (int(xy[j][0]), int(xy[j][1])), 5, cr, -1)
                    cv2.line(frame, (int(xy[j][0]), int(xy[j][1])), (int(xy[j+1][0]), int(xy[j+1][1])), cr, 2, cv2.LINE_AA)
            _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            current_frame[:] = _frame[:]
            try:
                if out_queue.full():
                    out_queue.get()
                out_queue.put(_frame)
            except:
                logging.error("out_queue failed to put frame")
            continue

        # apply connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8, ltype=cv2.CV_32S)
        extra_info.centroids = centroids
        extra_info.labels = labels
        extra_info.stats = stats
        extra_info.num_labels = num_labels

        #print (f"num_labels: {num_labels}")
        #print (f"centroids: {centroids}")
        #print (f"stats: {stats}")
        #print (f"num_labels: {num_labels}")


        #morph = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        
        centers = []
        obj_rects = []
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
                    sqr_patch = objcls.pad_image(patch)
                    if sqr_patch.shape[0] != 224 or sqr_patch.shape[1] != 224:
                        sqr_patch = cv2.resize(sqr_patch, (224, 224))
                    #outputs = objcls.run_inference(model, sqr_patch)[0]
                    #obj_class_index = int(np.argmax(outputs) + 1)
                    #obj_class_confidence = float(outputs[obj_class_index-1])
                    #print (type(obj_class_index), type(obj_class_confidence))
                    #print (f"obj_class_index: {obj_class_index}, obj_class_confidence: {obj_class_confidence}")
                    if True:#obj_class_index == 2 and obj_class_confidence > 0.80:
                        center = np.array ([[x+w/2], [y+h/2]])
                        centers.append(np.round(center))
                        obj_rects.append([x, y, w, h, area])
                #if os.path.exists("/ssd_disk/dets"):
                    #cv2.imwrite(f"/ssd_disk/dets/patch_{int(time.time()*1000)}_{str(i).zfill(4)}_{obj_class_index}_{int(obj_class_confidence * 100)}.jpg", sqr_patch)
                #else:
                #    pass
                    #if obj_class_index == 2:
                    #    cv2.imwrite(f"./tmp/patch_{int(time.time()*1000)}_{str(i).zfill(4)}_{obj_class_index}_{int(obj_class_confidence * 100)}.jpg", sqr_patch)
                    
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        obj_cnt = len(centers)
        objtrace.append(centers)

        # draw objtracks
        for oidx, ot in objtracks.items():
            if len(ot) < 2 or len(ot[1]) < 1:
                continue
            cr, xy = ot
            for j in range(len(xy)-1):
                if j < 2:
                    cv2.circle(frame, (int(xy[j][0]), int(xy[j][1])), 10, cr, -1)
                else:
                    cv2.circle(frame, (int(xy[j][0]), int(xy[j][1])), 6, cr, -1)
                cv2.line(frame, (int(xy[j][0]), int(xy[j][1])), (int(xy[j+1][0]), int(xy[j+1][1])), cr, 2, cv2.LINE_AA)

        # if too many objects detected, skip this frame
        if obj_cnt > config.max_detection:
            #logging.debug(f"Too many movements detected, skipping frame: {num_labels}")
            _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            current_frame[:] = _frame[:]
            try:
                if out_queue.full():
                    out_queue.get()
                out_queue.put(_frame)
            except:
                logging.error("out_queue failed to put frame")
            continue
        

        if config.debug:
            if obj_cnt > 0:
                print ("object_count: ", obj_cnt)

        if centers:
            if rock_evt is None:
                rock_evt = create_falling_rock_event()
                rock_evt.frame_dir = os.path.join(config.output_dir, str(int(rock_evt.ts_start*1000)))
                os.makedirs(rock_evt.frame_dir, exist_ok=True)
            volumes = []
            for rt in obj_rects:
                # draw rectangle
                x, y, w, h, area = rt
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                # rotate area 180 degree to get volume
                radius = (w+h) // 2
                obj_vol = w * h * radius * config.frame_dist_cm / 100.0
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
                            tracked_object.speed = obj_path_length / time_dur
                            rock_evt.max_speed = max(rock_evt.max_speed, tracked_object.speed * config.frame_dist_cm / 100)

                            # Display speed if available
                            cv2.putText(frame, "SPD: {} CM/s".format(round(tracked_object.speed, 2)), (int(trace_x), int(trace_y)), font, 3, (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(frame, 'ID: '+ str(tracked_object.track_id), (int(trace_x), int(trace_y)), font, 3, (255, 255, 255), 2, cv2.LINE_AA)
                    if len(tracked_object.trace) > 0:
                        this_pt = [int(tracked_object.trace[-1][0][0]), int(tracked_object.trace[-1][1][0])]
                        objtracks[tracked_object.track_id][1].append(this_pt)
                    
            rock_evt.max_vol = max(rock_evt.max_vol, max_vol / 1_000_000)
            rock_evt.max_count = max(rock_evt.max_count, max_cnt)
            rock_evt.ts_end = datetime.now().timestamp()


        # draw rock boundaries
        #cts = [np.array(p) for p in config.rock_boundaries if len(p) > 2]
        #cv2.drawContours(frame, cts, -1, (0, 0, 255), 2)

        # draw objtraces
        #for oidx, ot in enumerate(objtrace):
        #    randcolor = (0, 255, 0)
        #    for xy in ot:
        #        cv2.circle(frame, (int(xy[0]), int(xy[1])), 5, randcolor, -1)


        # Display all images
        #info_text = "Landslides: 0"
        #if obj_cnt > 0 and obj_cnt < config.max_detection and len(tracked_object.trace) > 0:
        #    info_text = "Landslides: {}".format(obj_cnt)
        
        #cv2.putText(frame, info_text, (87, 62), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
        #cv2.putText(frame, info_text, (85, 60), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        fps_e = (1.0 / ((cv2.getTickCount() - fps_f) / cv2.getTickFrequency()))
        fps_e = round(fps_e)

        cv2.putText(frame, "FPS: {}".format(fps_e), (89, 124), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "FPS: {}".format(fps_e), (87, 122), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        current_frame[:] = _frame[:]
        try:
            if out_queue.full():
                out_queue.get()
            out_queue.put(_frame)
        except:
            logging.error("out_queue failed to put frame")

    # Clean up
    if model is not None:
        model.release()
    video_src_ended = True
    if config.debug:
        print ("main_loop aborted.")


