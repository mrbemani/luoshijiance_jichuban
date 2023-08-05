# -*- encoding: utf-8 -*-

__author__ = 'Mr.Bemani'

import sys
import os
import glob

if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
elif __file__:
    APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(APP_BASE_DIR)
webserver = None

import time
import shutil
import threading
from datetime import datetime
import logging
import multiprocessing as mp
from typing import Union, Callable
import queue
import subprocess as subp

import json
from addict import Dict
from tsutil import zip_dir

import numpy as np
import cv2 as cv

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s %(levelname)s %(message)s',
    filename='main_web.log', filemode='a')


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


############################################################
# set environment variables
os.environ["PC_TEST"] = "0"
############################################################

from flask import Flask, render_template, Response, request, jsonify, send_from_directory, make_response
from werkzeug.serving import make_server

from configure import config, loadConfig, saveConfig

from video import fetch_frame_loop
from magic_happening import process_frame_loop, PF_W, PF_H

from event_utils import load_event_log

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


@app.route('/sys/solar')
@app.route('/sys/solar/')
def sys_solar():
    return send_from_directory('/home/firefly/solar_energy_watcher', 'solar.html')


@app.route('/sys/temperature')
@app.route('/sys/temperature/')
def sys_temperature():
    return send_from_directory('/home/firefly/temperature_watcher', 'temperature.html')


@app.route('/webui/')
def webui():
    last_n_alerts = load_event_log("events.csv", None)[:20]
    last_n_alerts = [(datetime.fromtimestamp(int(float(x[5]))), 
                      int(float(x[4])), 
                      round(float(x[1]), 3), 
                      round(float(x[3]), 2), 
                      str(x[7])) for x in last_n_alerts]
    video_preview_url = "/video_preview"
    live_url = config.camera_web_url
    return render_template('index.tpl.html', live_url=live_url, alerts=last_n_alerts, video_preview_url=video_preview_url)


@app.route('/wait/<int:seconds>')
def web_wait(seconds):
    # wait n seconds and href to querystring url
    target_url = request.args.get('url', '/')
    return render_template('wait.tpl.html', seconds=seconds, target_url=target_url)


@app.route('/api/terminate')
def web_terminate():
    yield "<script>window.location.href='/wait/5';</script>"
    time.sleep(2)
    webserver.shutdown()
    webserver.server_close()
    exit_program()


@app.route('/api/get_dets')
def api_get_dets():
    try:
        os.system("rm -rf /tmp/dets.zip")
        if os.path.exists("/ssd_disk/dets"):
            zip_dir("/ssd_disk/dets/", "/tmp/dets.zip", clear_dir=False)
        else:
            zip_dir("./tmp/", "/tmp/dets.zip", clear_dir=False)
        return send_from_directory("/tmp/", "dets.zip", as_attachment=True)
    except:
        return jsonify({'status': 'error', 'message': 'failed to get dets'}), 500
   

@app.route('/api/manual_cutoff')
def api_manual_cutoff():
    config.manual_cutoff = not config.manual_cutoff
    return ("1" if config.manual_cutoff else "0")


@app.route('/webui/settings')
def webui_settings():
    _down_dict = config.to_dict()
    return render_template('settings.tpl.html', **_down_dict)


@app.route('/api/settings/update')
def webapi_settings_update():
    global config
    if not request.json:
        return jsonify({'status': 'error', 'message': 'Invalid request'}), 400

    try:
        _up_dict = Dict()
        _up_dict.sms = Dict()
        _up_dict.sms.enable = request.form.get('sms__enable', False)
        _up_dict.sms.phone = request.form.get('sms__phone', '')
        _up_dict.sms.sender = request.form.get('sms__sender', '')
        
        _up_dict.tracking = Dict()
        _up_dict.tracking.det_w = request.form.get('det_w', 720)
        _up_dict.tracking.det_h = request.form.get('det_h', 720)
        _up_dict.tracking.dist_thresh = request.form.get('dist_thresh', 5)
        _up_dict.tracking.max_object_count = request.form.get('max_object_count', 30)
        _up_dict.tracking.max_skip_frame = request.form.get('max_skip_frame', 3)
        _up_dict.tracking.max_trace_length = request.form.get('max_trace_length', 5)
        _up_dict.tracking.min_rock_pix = request.form.get('min_rock_pix', 16)
        _up_dict.tracking.max_rock_pix = request.form.get('max_rock_pix', 300)

        _up_dict.debug = request.form.get('debug', False)
        _up_dict.video_src = request.form.get('video_src', '')
        _up_dict.camera_web_url = request.form.get('camera_web_url', '')
        _up_dict.preview_fps = request.form.get('preview_fps', 10)
        _up_dict.preview_width = request.form.get('preview_width', 320)
        _up_dict.preview_height = request.form.get('preview_height', 180)
        _up_dict.max_detection = request.form.get('max_detection', 240)
    
        config.update(_up_dict)
        saveConfig("settings.yml")
        yield "<script>window.location.href='/wait/5';</script>"
        time.sleep(2)
        webserver.shutdown()
        webserver.server_close()
        exit_program()
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

def gather_img():
    while True:
        time.sleep(1.0 / config.preview_fps)
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), config.preview_quality]
        pw, ph = config.preview_width, config.preview_height
        if current_frame is not None:
            minifrm = cv.resize(current_frame, (pw, ph), interpolation=cv.INTER_NEAREST)
            _, frame = cv.imencode('.jpg', minifrm, encode_param)
        else:
            img = np.zeros((ph, pw, 3), dtype=np.uint8)
            _, frame = cv.imencode('.jpg', img, encode_param)
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
    record_frame_count = len(glob.glob(record_frames))
    if not os.path.exists(output_mp4) and record_frame_count > 0:
        time.sleep(3)
    if os.path.exists(output_mp4):
        file_resp = send_from_directory(record_base, "output.mp4")
        resp = make_response(file_resp)
        dt = datetime.fromtimestamp(int(record_id)//1000).strftime("%Y%m%d_%H%M%S")
        resp.headers["Content-Disposition"] = f"attachment; filename={dt}.mp4"
        return resp
    elif not os.path.exists(output_mp4):
        return jsonify({'status': 'error', 'message': 'Video not ready'}), 404


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

    # manual cutoff for detection and tracking
    config.manual_cutoff = False

    current_frame = np.zeros((PF_H, PF_W, 3), dtype=np.uint8)

    # clear tmp dir
    if not os.path.exists("./tmp"):
        #shutil.rmtree("./tmp")
        os.mkdir("./tmp")

    if os.path.exists("/ssd_disk") and not os.path.exists("/ssd_disk/dets"):
        os.mkdir("/ssd_disk/dets")

    main_loop_running = True
    
    # start video fetch loop
    video_fetch_thread = threading.Thread(target=fetch_frame_loop, args=(config.video_src, main_loop_running_cb, frame_queue))
    video_fetch_thread.setDaemon(True)
    video_fetch_thread.start()

    # start video processing loop
    video_process_thread = threading.Thread(target=process_frame_loop, args=(config, main_loop_running_cb, frame_queue, out_queue, current_frame))
    video_process_thread.setDaemon(True)
    video_process_thread.start()

    ############################################################
    # start webserver
    try:
        logging.info('Starting webserver...')
        #app.run(host="0.0.0.0", port=8080, debug=config.debug, threaded=True)
        webserver = make_server("0.0.0.0", 8080, app, threaded=True)
        logging.info('Webserver started.')
        webserver.serve_forever()
    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt, stopping...')

    print ("Application is Deinitializing...", end='', flush=True)
    main_loop_running = False
    time.sleep(0.5)
    video_fetch_thread.join(timeout=3)
    video_process_thread.join(timeout=3)

    print ("OK")
    if config.debug:
        print ("Server closed.")

