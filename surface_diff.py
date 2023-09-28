# -*- encoding: utf-8 -*-

import sys
import os
import time
import math
import threading
import queue
from typing import Callable
import cv2
import numpy as np

from addict import Dict

from flask import Flask, render_template, Response, request, jsonify, send_from_directory


def restart_program():
    """Restarts the current program, with file objects and descriptors
       cleanup
    """
    python = sys.executable
    os.execl(python, python, * sys.argv)

def exit_program():
    """Restarts the current program, with file objects and descriptors
       cleanup
    """
    os._exit(0)


PF_W = 1920
PF_H = 1080

frame_queue = queue.Queue(maxsize=1)
keep_running = True
current_frame = np.zeros((PF_H, PF_W, 3), dtype=np.uint8)
base_frame = np.zeros((PF_H, PF_W, 3), dtype=np.uint8)
target_frame = np.zeros((PF_H, PF_W, 3), dtype=np.uint8)
diff_map = np.zeros((PF_H, PF_W, 3), dtype=np.uint8)


# fetch frame from video source and put it into frame_queue
def fetch_frame_loop(video_src: str, keep_running: Callable, frame_put_queue: queue.Queue, fps: int = 25):
    # Capture livestream
    cap = cv2.VideoCapture(video_src)
    ret = True
    frame_gap = 1.0 / fps
    while keep_running():
        if video_src[-4:] in [".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg"]:
            time.sleep(frame_gap) # 20 fps for video
        ret, frame = cap.read()
        while not ret or frame is None: # try to reconnect forever
            cap.release()
            cap = cv2.VideoCapture (video_src)
            ret, frame = cap.read()
            if ret is None or frame is None:
                time.sleep(0.5)
                continue
        try:
            frame_put_queue.put(frame, timeout=0.05)
        except queue.Full:
            try:
                frame_put_queue.get(timeout=0.1)
            except:
                pass
            frame_put_queue.put_nowait(frame)
    cap.release()

# load video source
video_src = "rtsp://admin:nd12345678@192.168.0.117/cam/realmonitor?channel=1&subtype=0"

# start fetch frame loop
fetch_frame_thread = threading.Thread(target=fetch_frame_loop, args=(video_src, 
                                                                     lambda: keep_running, 
                                                                     frame_queue))
fetch_frame_loop_start_time = time.time()
fetch_frame_thread.start()


def select_base_frame():
    try:
        frame = frame_queue.get(timeout=0.5)
    except queue.Empty:
        return False
    if frame is None:
        return False
    # resize frame to PF_W x PF_H
    frame = cv2.resize(frame, (PF_W, PF_H))
    base_frame[:] = frame[:]
    return True


def select_target_frame():
    try:
        frame = frame_queue.get(timeout=0.5)
    except queue.Empty:
        return False
    if frame is None:
        return False
    # resize frame to PF_W x PF_H
    frame = cv2.resize(frame, (PF_W, PF_H))
    target_frame[:] = frame[:]
    return True

    
def show_diff_map():
    # compute the optical flow from base_frame to target_frame
    base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    # gaussian blur
    base_gray = cv2.GaussianBlur(base_gray, (3, 3), 0)
    target_gray = cv2.GaussianBlur(target_gray, (3, 3), 0)
    flow = cv2.calcOpticalFlowFarneback(base_gray, target_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # translate magnitude to range [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # translate angle to range [0, 255]
    angle = ((angle + np.pi) / (2 * np.pi)) * 255
    # draw vector field overlay onto target_frame, to indicate the flow direction
    augmented_target = target_frame.copy()
    step = 8
    for y in range(PF_H//10, PF_H, step):
        for x in range(0, PF_W, step):
            fx, fy = flow[y, x]
            # do not draw small vectors
            if abs(fx) < 1.5 and abs(fy) < 1.5:
                continue
            # magnify the flow vectors
            fx *= 2
            fy *= 2
            cv2.arrowedLine(augmented_target, (x, y), (int(x + fx), int(y + fy)), (200, 225, 255), 1)
    # build hsv image
    hsv = np.zeros((PF_H, PF_W, 3), dtype=np.uint8)
    hsv[..., 0] = angle
    hsv[..., 1] = 255
    hsv[..., 2] = magnitude
    # filter out magnitude < 5
    hsv[magnitude < 1.5] = 0
    # convert hsv to bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    zero_mask = np.zeros((PF_H//10, PF_W, 3), dtype=np.uint8)
    bgr[0:PF_H//10, :, :] = zero_mask
    # paste bgr with 50% opacity onto augmented_target, ignore black pixels
    diff_map[:] = cv2.addWeighted(augmented_target, 0.5, bgr, 0.8, 0)
    return True


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/base_frame')
def req_base_frame():
    ret = select_base_frame()
    ret, jpeg = cv2.imencode('.png', base_frame)
    return Response(jpeg.tobytes(), mimetype='image/png')


@app.route('/target_frame')
def req_target_frame():
    select_target_frame()
    ret, jpeg = cv2.imencode('.png', target_frame)
    return Response(jpeg.tobytes(), mimetype='image/png')


@app.route('/diff_map')
def req_diff_map():
    gray_base = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    if np.sum(gray_base) == 0 or np.sum(gray_target) == 0:
        return "NA"
    show_diff_map()
    # return as png image
    ret, jpeg = cv2.imencode('.png', diff_map)
    return Response(jpeg.tobytes(), mimetype='image/png')


@app.route('/wait/<int:seconds>')
def web_wait(seconds):
    # wait n seconds and href to querystring url
    target_url = request.args.get('url', '/')
    return render_template('wait.tpl.html', seconds=seconds, target_url=target_url)


@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('assets', path)


@app.route('/reboot')
def web_reboot():
    yield "<script>window.location.href='/wait/5';</script>"
    time.sleep(2)
    webserver.shutdown()
    webserver.server_close()
    restart_program()


if __name__ == '__main__':
    from werkzeug.serving import make_server
    ############################################################
    # start webserver
    try:
        print ('Starting webserver...')
        webserver = make_server("0.0.0.0", 15117, app, threaded=True)
        print ('Webserver started.')
        webserver.serve_forever()
    except KeyboardInterrupt:
        keep_running = False
        print("KeyboardInterrupt")
        sys.exit(0)
