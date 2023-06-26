# -*- encoding: utf-8 -*-

__author__ = "Mr.Bemani"

import os

import queue
import cv2
import numpy as np
import time
import copy
from datetime import datetime
from PIL import Image

import object_classifier as objcls
from tracker import Tracker


PF_W = 1280
PF_H = 720

frame_queue = queue.Queue(1)
frame_is_ready = False
original_frame = None
pushed_frame = None
video_src_ended = False


def process_frame_loop(config: dict, alert_image_fname: str, main_loop_running_cb: function):
    global original_frame, pushed_frame, frame_is_ready, video_src_ended

    model = objcls.load_rknn_model(config.rknn_model_path)

    # load ROI Mask
    roi_mask = None
    if config.roi_mask is not None and os.path.isfile(config.roi_mask) and config.roi_mask != "None":
        roi_mask = cv2.imread(config.roi_mask, cv2.IMREAD_GRAYSCALE)

    # Initial background subtractor and text font
    fgbg = cv2.createBackgroundSubtractorMOG2()
    font = cv2.FONT_HERSHEY_PLAIN

    centers = [] 

    alert_im = Image.open(alert_image_fname)

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


    while main_loop_running_cb():
        fps_f = cv2.getTickCount()

        centers = []
        frame_start_time = datetime.utcnow()
        
        frame = frame_queue.get()
        if frame is None:
            print ("frame is empty")
            continue

        original_frame = copy.copy(frame)

        
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
        morph = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)

        avg_color = np.average(morph)
        if avg_color > 255//10:
            frame_is_ready = False
            _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            pushed_frame = cv2.imencode('.png', _frame)[1].tobytes()
            frame_is_ready = True
            continue

        # apply connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8, ltype=cv2.CV_32S)


        # if too many objects detected, skip this frame
        if num_labels > config.tracking.max_object_count:
            frame_is_ready = False
            _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            pushed_frame = cv2.imencode('.png', _frame)[1].tobytes()
            frame_is_ready = True
            continue
        
        
        debug_rects = []
        # Find centers of all detected objects
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if x < 10 or y < 10 or area < 100: # skip small objects (10x10 pixels)
                continue

            if w / h > 2 or h / w > 2:
                continue

            x = int(x * pf_ratio_w)
            y = int(y * pf_ratio_h)
            w = int(w * pf_ratio_w)
            h = int(h * pf_ratio_h)

            if roi_mask is not None:
                if not roi_mask[y+h//2, x+w//2] > 0:
                    continue

            if w >= blob_min_width_far and h >= blob_min_height_far:
                max_wh = np.max([w, h])
                max_x = x + max_wh
                max_y = y + max_wh
                if max_x >= original_w:
                    max_x = original_w - 1
                if max_y >= original_h:
                    max_y = original_h - 1
                patch = original_frame[y:max_y, x:max_x].copy()
                if patch is None:
                    print ("---invalid patch---")
                else:
                    sqr_patch = objcls.pad_image(patch)
                    if sqr_patch.shape[0] != 224 or sqr_patch.shape[1] != 224:
                        sqr_patch = cv2.resize(sqr_patch, (224, 224))
                    outputs = objcls.run_inference(model, sqr_patch)[0]
                    obj_class_index = np.argmax(outputs) + 1
                    obj_class_confidence = outputs[obj_class_index-1]
                    cv2.imwrite(f"./tmp/patch_{str(i).zfill(4)}_{obj_class_index}_{int(obj_class_confidence * 100)}_{int(time.time()*1000)}.jpg", sqr_patch)
                    if obj_class_confidence < 0.5:
                        center = np.array ([[x+w/2], [y+h/2]])
                        centers.append(np.round(center))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if config.debug:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        obj_cnt = len(centers)
        if config.debug:
            if obj_cnt > 0:
                print ("object_count: ", obj_cnt)
        
        if centers:
            tracker.update(centers)
            obj_cnt = max(len(tracker.tracks), len(centers))
            if len(tracker.tracks) < config.max_detection:
                for vehicle in tracker.tracks:
                    if len(vehicle.trace) > 1:
                        for j in range(len(vehicle.trace)-1):
                            # Draw trace line
                            x1 = vehicle.trace[j][0][0]
                            y1 = vehicle.trace[j][1][0]
                            x2 = vehicle.trace[j+1][0][0]
                            y2 = vehicle.trace[j+1][1][0]

                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

                        trace_i = len(vehicle.trace) - 1

                        trace_x = vehicle.trace[trace_i][0][0]
                        trace_y = vehicle.trace[trace_i][1][0]

                        # Check if tracked object has reached the speed detection line
                        load_lag = (datetime.utcnow() - frame_start_time).total_seconds()
                        time_dur = (datetime.utcnow() - vehicle.start_time).total_seconds() - load_lag
                        
                        vehicle.speed = config.frame_dist_cm / time_dur

                        # If calculated speed exceeds speed limit, save an image of speeding car
                        #if vehicle.speed > HIGHWAY_SPEED_LIMIT:
                        #	pass
                    
                        # Display speed if available
                        cv2.putText(frame, "SPD: {} CM/s".format(round(vehicle.speed, 2)), (int(trace_x), int(trace_y)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                        # cv2.putText(frame, 'ID: '+ str(vehicle.track_id), (int(trace_x), int(trace_y)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    

        # draw rock boundaries
        cts = [np.array(p) for p in config.rock_boundaries if len(p) > 2]
        cv2.drawContours(frame, cts, -1, (0, 0, 255), 2)


        # Display all images
        info_text = "Landslides: 0"
        if obj_cnt > 0 and obj_cnt < config.max_detection and len(vehicle.trace) > 0:
            info_text = "Landslides: {}".format(obj_cnt)
        
        cv2.putText(frame, info_text, (87, 62), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info_text, (85, 60), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # display alert !
        if obj_cnt > 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            f_pim = Image.fromarray(frame)
            
            f_pim.paste(alert_im, (0, 0), alert_im)

            frame = np.asarray(f_pim)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps_e = (1.0 / ((cv2.getTickCount() - fps_f) / cv2.getTickFrequency()))
        fps_e = round(fps_e)

        frame_is_ready = False
        _frame = cv2.resize(frame, (PF_W, PF_H), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        pushed_frame = cv2.imencode('.png', _frame)[1].tobytes()
        frame_is_ready = True
        #cv2.imshow("test", pushed_frame)
        cv2.waitKey(1)

    # Clean up
    model.release()
    video_src_ended = True
    if config.debug:
        print ("main_loop aborted.")


