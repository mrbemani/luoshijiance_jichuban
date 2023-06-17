# -*- encoding: utf-8 -*-


__author__ = "Shi Qi"

import sys
import os
import argparse
from datetime import datetime, timedelta
import cv2
import numpy as np
import time
import copy
import yaml
from addict import Dict
import threading
import multiprocessing as mpr
from datetime import datetime
from PIL import Image
import queue

import object_classifier as objcls

from flask import Flask, render_template, Response, request, jsonify, send_from_directory

from kalman_filter import KalmanFilter
from tracker import Tracker

#cv2.ocl.setUseOpenCL(True)


import boundaryeditor


if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
elif __file__:
    APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


#######################################################################################################################
# fix win10 scaling issue
if os.name == 'nt':
    import ctypes
    # Query DPI Awareness (Windows 10 and 8)
    awareness = ctypes.c_int()
    errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
    #print(awareness.value)

    # Set DPI Awareness  (Windows 10 and 8)
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
    # the argument is the awareness level, which can be 0, 1 or 2:
    # for 1-to-1 pixel control I seem to need it to be non-zero (I'm using level 2)

    # Set DPI Awareness  (Windows 7 and Vista)
    success = ctypes.windll.user32.SetProcessDPIAware()
    # behaviour on later OSes is undefined, although when I run it on my Windows 10 machine, it seems to work with 
    # effects identical to SetProcessDpiAwareness(1)
#######################################################################################################################

PF_W = 1280
PF_H = 720

frame_queue = queue.Queue(1)
main_loop_running = True
frame_is_ready = False
original_frame = None
pushed_frame = bytes()
blank_img = np.zeros((PF_H, PF_W, 3), np.uint8)
video_src_ended = False


config = Dict()

config.debug = False

config.frame_dist_cm = 1.0
config.max_detection = 240

config.video_src = 0
config.rock_boundaries = []

config.tracking.min_rock_pix = 8
config.tracking.max_rock_pix = 300
config.tracking.max_rock_ratio = 2.0
config.tracking.dist_thresh = 80
config.tracking.max_skip_frame = 3
config.tracking.max_trace_length = 2
config.tracking.max_object_count = 20

default_cfgfile = os.path.join(APP_BASE_DIR, "settings.yml")


# write out video file as mp4
def create_video_writer(fps=30, resolution=(PF_W, PF_H)):
    dtnow = datetime.now()
    datetime_str = dtnow.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(APP_BASE_DIR, "videos", datetime_str + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
    return video_writer, dtnow.timestamp()


def loadConfig(cfgfile=None):
    global config
    if cfgfile is None:
        cfgfile = default_cfgfile
    if not os.path.isfile(cfgfile):
        return False
    try:
        config.update(yaml.load(open(cfgfile, 'r'), Loader=yaml.FullLoader))
        return True
    except Exception as e:
        print (e)
    return False	

def saveConfig(cfgfile=None):
    global config
    if cfgfile is None:
        cfgfile = default_cfgfile
    if not os.path.isfile(cfgfile):
        return False
    try:
        yaml.dump(config.to_dict(), open(cfgfile, 'w+'))
        return True
    except Exception as e:
        print (e)
    return False	


def show_editBoundaries_window():
    global config
    ret, polys = boundaryeditor.showEditWin(original_frame, config.rock_boundaries)
    if ret:
        config.rock_boundaries = [list(list(p) for p in x) for x in polys]
        saveConfig()


def fetch_frame_loop():
    # Capture livestream
    cap = cv2.VideoCapture (config.video_src)
    ret = True
    while main_loop_running:
        ret, frame = cap.read()
        while not ret or frame is None: # try to reconnect forever
            cap.release()
            cap = cv2.VideoCapture (config.video_src)
            ret, frame = cap.read()
            if ret is None or frame is None:
                time.sleep(0.5)
                continue
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            frame_queue.get()
            frame_queue.put_nowait(frame)
    cap.release()


def main_loop():
    global original_frame, pushed_frame, frame_is_ready, video_src_ended
    
    # Create video writer
    video_writer, video_start_time = create_video_writer()

    # Initial background subtractor and text font
    fgbg = cv2.createBackgroundSubtractorMOG2()
    font = cv2.FONT_HERSHEY_PLAIN

    centers = [] 

    alert_im = Image.open(os.path.join(APP_BASE_DIR, "assets", "alert_image.png"))

    blob_min_width_far = config.tracking.min_rock_pix
    blob_min_height_far = config.tracking.min_rock_pix

    blob_min_width_near = blob_min_width_far * 4
    blob_min_height_near = blob_min_height_far * 4

    frame_start_time = None

    # Create object tracker
    tracker = Tracker(
        config.tracking.dist_thresh, 
        config.tracking.max_skip_frame, 
        config.tracking.max_trace_length, 1)

    
    while main_loop_running:
        fps_f = cv2.getTickCount()    

        centers = []
        frame_start_time = datetime.utcnow()

        dtnow = datetime.now().timestamp()
        if dtnow - video_start_time > 60 * 2: # 2 minutes
            video_writer.release()
            video_writer, video_start_time = create_video_writer()

        frame = frame_queue.get()

        # Save original frame
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

        # Find centers of all detected objects
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            if area < 25:
                continue
            
            x = int(x * pf_ratio_w)
            y = int(y * pf_ratio_h)
            w = int(w * pf_ratio_w)
            h = int(h * pf_ratio_h)

            if w >= blob_min_width_far and h >= blob_min_height_far:
                center = np.array ([[x+w/2], [y+h/2]])
                centers.append(np.round(center))

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        obj_cnt = len(centers)
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
        fps_e = int(round(fps_e))

        if frame is not None:
            if fps_e <= 0: fps_e = 30
            n_frame_repeat = 30 // fps_e
            for i in range(n_frame_repeat):
                video_writer.write(frame)
        

        frame_is_ready = False
        _frame = cv2.resize(frame, (720, 480), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        pushed_frame = cv2.imencode('.jpg', _frame)[1].tobytes()
        frame_is_ready = True
        #cv2.imshow("test", pushed_frame)
        cv2.waitKey(1)


    # Clean up
    video_src_ended = True
    video_writer.release()
    if config.debug:
        print ("main_loop aborted.")


def push_mjpeg_to_http():
    global frame_is_ready, pushed_frame
    while True:
        if frame_is_ready:
            out_img = pushed_frame
            frame_is_ready = False
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + out_img + b'\r\n')

webapp = Flask(__name__)

@webapp.route('/video')
def video_feed():
    #return push_mjpeg_to_http()
    return Response(push_mjpeg_to_http(), mimetype='multipart/x-mixed-replace; boundary=frame')


@webapp.route('/')
def index():
    return "<html><head><title>Video Streaming</title></head><body><img src='/video'></body></html>"


if __name__ == "__main__":
    if not os.path.exists(os.path.join(APP_BASE_DIR, "videos")):
        print ("videos directory not found. Creating...")
        os.mkdir(os.path.join(APP_BASE_DIR, "videos"))

    # load config
    loadConfig()
    
    # start video capture thread
    video_capture_thread = threading.Thread(target=fetch_frame_loop, args=())
    video_capture_thread.start()

    # start main_loop thread
    main_loop_thread = threading.Thread(target=main_loop, args=())
    main_loop_thread.start()

    # start bottle server
    try:
        webapp.run(host="0.0.0.0", port=8080)
    except KeyboardInterrupt as kbi:
        print ("Keyboard interrupt received.")

    main_loop_running = False
    print ("OK")
    if config.debug:
        print ("App closed.")
