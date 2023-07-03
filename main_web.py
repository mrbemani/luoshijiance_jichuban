# -*- encoding: utf-8 -*-

__author__ = 'Mr.Bemani'

import sys
import os
import time
import shutil
import threading
from datetime import datetime
import logging
import multiprocessing as mp
import queue
import subprocess as subp

import numpy as np
import cv2 as cv

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s %(levelname)s %(message)s') 
    #filename='main_web.log', filemode='w')


def restart_program():
    """Restarts the current program, with file objects and descriptors
       cleanup
    """
    python = sys.executable
    os.execl(python, python, * sys.argv)


############################################################
# set environment variables
os.environ["PC_TEST"] = "1"
############################################################

from flask import Flask, render_template, Response, request, jsonify, send_from_directory

from configure import config, loadConfig, saveConfig

from video import fetch_frame_loop, create_video_writer
from magic_happening import process_frame_loop, PF_W, PF_H

from event_utils import load_event_log

if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
elif __file__:
    APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

current_frame = None

main_loop_running = False
def main_loop_running_cb():
    return main_loop_running

app = Flask(__name__)

frame_queue = queue.Queue(2)
out_queue = queue.Queue(50)


@app.route('/')
def index():
    return "<script>window.location.href='/webui/';</script>"


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('assets', path)


@app.route('/webui/')
def webui():
    last_n_alerts = load_event_log("events.csv", None)[:8]
    last_n_alerts = [(datetime.fromtimestamp(int(float(x[5]))), int(float(x[4])), int(float(x[1])), int(round(float(x[3]))), str(x[7])) for x in last_n_alerts]
    video_preview_url = "/video_preview"
    return render_template('index.tpl.html', alerts=last_n_alerts, video_preview_url=video_preview_url)


@app.route('/webui/live-adjust')
def live_adjust():
    return render_template('live_adjust.tpl.html', config=config.to_dict(), video_preview_url="/video_preview")


@app.route('/api/live-adjust/update', methods=['POST'])
def live_adjust_update():
    global config
    if not request.json:
        return jsonify({'status': 'error', 'message': 'Invalid request'}), 400

    config.update(request.json)
    print (config)
    saveConfig("settings.yml")
    restart_program()
    return jsonify({'status': 'ok'}), 200


def gather_img():
    while True:
        time.sleep(1.0 / config.preview_fps)
        if current_frame is not None:
            minifrm = cv.resize(current_frame, (PF_W, PF_H), interpolation=cv.INTER_NEAREST)
            _, frame = cv.imencode('.jpg', minifrm)
        else:
            img = np.zeros((PF_H, PF_W, 3), dtype=np.uint8)
            _, frame = cv.imencode('.jpg', img)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')


@app.route('/video_preview')
def video_preview():
    minifrm = gather_img()
    return Response(minifrm, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_record_video/<record_id>')
def get_record_video(record_id):
    record_base = os.path.join(config.output_dir, record_id)
    output_mp4 = os.path.join(record_base, "output.mp4")
    record_frames = os.path.join(record_base, r"*.jpg")
    # combine jpegs into mp4 use ffmpeg
    if not os.path.exists(output_mp4):
        ffmpeg_subp = subp.Popen(f"ffmpeg -r 12 -f image2 -s 1280x720 -pattern_type glob -i '{record_frames}' -vcodec libx264 -crf 12 -pix_fmt yuv420p {output_mp4}", shell=True)
        ffmpeg_subp.wait()
    if os.path.exists(output_mp4):
        return send_from_directory(record_base, "output.mp4")
    else:
        return jsonify({'status': 'error', 'message': 'Record not found'}), 404


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Rock detection and tracking')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--cfg', type=str, help='Config file path', default="settings.yml")
    parser.add_argument('-i', '--video_src', type=str, help='Video source')
    parser.add_argument('-o', '--output_dir', type=str, help='Output video file basepath', default="outputs")
    args = parser.parse_args()

    # load config
    loadConfig(args.cfg)
    if args.debug is True:
        config.debug = args.debug
    
    if args.video_src is not None and args.video_src != "":
        config.video_src = args.video_src

    if args.output_dir is not None and args.output_dir != "":
        config.output_dir = args.output_dir

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    if not os.path.exists("events.csv"):
        with open("events.csv", "w") as f:
            pass

    current_frame = np.zeros((PF_H, PF_W, 3), dtype=np.uint8)

    # clear tmp dir
    if os.path.exists("./tmp"):
        shutil.rmtree("./tmp")
    os.mkdir("./tmp")

    main_loop_running = True
    
    # start video fetch loop
    video_fetch_thread = threading.Thread(target=fetch_frame_loop, args=(config, main_loop_running_cb, frame_queue))
    video_fetch_thread.setDaemon(True)
    video_fetch_thread.start()

    # start video processing loop
    video_process_thread = threading.Thread(target=process_frame_loop, args=(config, main_loop_running_cb, frame_queue, out_queue, current_frame))
    video_process_thread.setDaemon(True)
    video_process_thread.start()

    # start web server
    try:
        app.run(host='0.0.0.0', port=8080, debug=True)
    except KeyboardInterrupt:
        pass    

    print ("Application is Deinitializing...", end='', flush=True)
    main_loop_running = False
    time.sleep(0.5)
    video_fetch_thread.join(timeout=3)
    video_process_thread.join(timeout=3)

    print ("OK")
    if config.debug:
        print ("Server closed.")

