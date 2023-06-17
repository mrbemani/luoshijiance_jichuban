# -*- encoding: utf-8 -*-


__author__ = "Shi Qi"

import sys
import os
import argparse
import queue
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

import object_classifier as objcls

# pysimplegui library
import PySimpleGUI as sg

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
pushed_frame = None
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


def fetch_frame_loop():
    # Capture livestream
    print (config.video_src)
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
            frame_queue.put(frame, timeout=0.03)
        except queue.Full:
            try:
                frame_queue.get(timeout=0.1)
            except:
                pass
            frame_queue.put_nowait(frame)
    cap.release()


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


def main_loop(args):
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

    alert_im = Image.open(os.path.join(APP_BASE_DIR, "assets", "alert_image.png"))

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


    while main_loop_running:
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
                patch = frame[y:y+h, x:x+w].copy()
                if patch is None:
                    print ("---invalid patch---")
                else:
                    sqr_patch = objcls.pad_image(patch)
                    if sqr_patch.shape[0] != 64 or sqr_patch.shape[1] != 64:
                        sqr_patch = cv2.resize(sqr_patch, (64, 64))
                    cv2.imwrite(f"./tmp/patch_{str(i).zfill(4)}_{int(time.time()*1000)}.jpg", sqr_patch)
                    outputs = objcls.run_inference(model, sqr_patch)
                    obj_class_index = np.argmax(outputs) + 1
                    print (obj_class_index)
                    if obj_class_index == 7:
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




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Rock detection and tracking')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--cfg', type=str, help='Config file path', default="settings.yml")
    parser.add_argument('-i', '--video_src', type=str, help='Video source')
    args = parser.parse_args()

    # load config
    loadConfig(args.cfg)
    if args.debug is True:
        config.debug = args.debug
    
    if args.video_src is not None and args.video_src != "":
        config.video_src = args.video_src

    # start video capture thread
    video_capture_thread = threading.Thread(target=fetch_frame_loop, args=())
    video_capture_thread.start()

    # start main_loop thread
    main_loop_thread = threading.Thread(target=main_loop, args=(args,))
    main_loop_thread.start()

    sg.theme('Dark Brown 3')

    frame_layout = [
        [sg.Button('摄像机设置', key="-CAMERA_SET-", button_color=(sg.YELLOWS[0], sg.BLUES[0]), size=(16, 3), pad=(16, 34))],
        [sg.Button('报警设置', key="-ALARM_SET-", button_color=(sg.YELLOWS[0], sg.BLUES[0]), size=(16, 3), pad=(16, 34))],
        [sg.Button('接口设置', key="-INTERFACE_SET-", button_color=(sg.YELLOWS[0], sg.BLUES[0]), size=(16, 3), pad=(16, 34))],
        [sg.Button('绘制边界', key="-BOUNDARY_SET-", button_color=(sg.YELLOWS[0], sg.BLUES[0]), size=(16, 3), pad=(16, 34))]
    ]

    layout = [
        [
            sg.Image(filename='', key='_FRAME_'), 
            sg.VerticalSeparator(pad=None),
            sg.Frame("sidebar", frame_layout, size=(200, 700))
        ]
    ]

    #sg.SetOptions(element_padding=(40,40))

    window = sg.Window('Landslide Visualization', layout, location=(0, 0))
    event, values = 0, 0
    while event != sg.WIN_CLOSED:
        event, values = window.read(timeout=5)
        #if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        if video_src_ended:
            break
        if frame_is_ready and pushed_frame:
            window['_FRAME_'](data=pushed_frame)
        if event == '-CAMERA_SET-':
            camera_set_win_layout = [
                [sg.Text("视频源地址:"), sg.InputText(config.video_src, key="-VIDEO_SRC-")],
                [sg.Button("确定", key="-BTN_OK-"), sg.Button("取消", key="-BTN_CANCEL-")]
            ]
            camera_set_win = sg.Window('摄像机设置 (重启后生效)', camera_set_win_layout)
            csw_event, csw_values = camera_set_win.read()
            if csw_event == '-BTN_OK-' and len(csw_values["-VIDEO_SRC-"]) > 0:
                if csw_values["-VIDEO_SRC-"] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    config.video_src = int(csw_values["-VIDEO_SRC-"])
                else:
                    config.video_src = csw_values["-VIDEO_SRC-"]
                saveConfig()
                sg.popup("设置成功!")
            camera_set_win.close()
        elif event == '-ALARM_SET-':
            alarm_set_win_layout = [
                [
                    sg.Text("偏移距域:\t"), sg.InputText(config.tracking.dist_thresh, key="-DIST_THRESH-", size=(5,1)), 
                    sg.Text("丢失帧域:\t"), sg.InputText(config.tracking.max_skip_frame, key="-MAX_SKIP_FRAME-", size=(5,1))
                ],
                [
                    sg.Text("跟踪帧域:\t"), sg.InputText(config.tracking.max_trace_length, key="-MAX_TRACE_LENGTH-", size=(5,1)),
                    sg.Text("最小岩体:\t"), sg.InputText(config.tracking.min_rock_pix, key="-MIN_ROCK_PIX-", size=(5,1))
                ],
                [sg.Button("确定", key="-BTN_OK-"), sg.Button("取消", key="-BTN_CANCEL-")]
            ]
            alarm_set_win = sg.Window('报警设置 (重启后生效)', alarm_set_win_layout)
            asw_event, asw_values = alarm_set_win.read()
            if asw_event == '-BTN_OK-':
                try:
                    config.tracking.min_rock_pix = int(csw_values["-MIN_ROCK_PIX-"])
                    config.tracking.dist_thresh = int(csw_values["-DIST_THRESH-"])
                    config.tracking.max_skip_frame = int(csw_values["-MAX_SKIP_FRAME-"])
                    config.tracking.max_trace_length = int(csw_values["-MAX_TRACE_LENGTH-"])
                    saveConfig()
                except:
                    sg.popup_no_wait("参数无效!")
            alarm_set_win.close()
        elif event == "-INTERFACE_SET-":
            sg.popup_no_wait("暂不支持")
        elif event == "-BOUNDARY_SET-":
            show_editBoundaries_window()

    print ("Application is Deinitializing...", end='', flush=True)
    main_loop_running = False
    time.sleep(0.5)
    main_loop_thread.join()	
    
    window.close()
    print ("OK")
    if config.debug:
        print ("App closed.")
