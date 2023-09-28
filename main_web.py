# -*- encoding: utf-8 -*-

__author__ = 'Mr.Bemani'

import sys
import os
import glob
import traceback

if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
elif __file__:
    APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(APP_BASE_DIR)
webserver = None
app_state = "running"

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
from magic_happening import process_frame_loop, PF_W, PF_H, frame_update_time

from event_utils import load_event_log

current_frame = None

main_loop_running = False
def main_loop_running_cb():
    return main_loop_running

app = Flask(__name__)

frame_queue = queue.Queue(2)


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
    return send_from_directory('/opt/sys/solar_energy_watcher', 'solar.html')


@app.route('/sys/temperature')
@app.route('/sys/temperature/')
def sys_temperature():
    return send_from_directory('/opt/sys/temperature_watcher', 'temperature.html')


@app.route('/webui/setup', methods=['GET'])
@app.route('/webui/setup/', methods=['GET'])
def webui_setup():
    settings_obj = dict(debug=False, 
                        video_src="",
                        camera_web_url="javascript:void(0);",
                        preview_fps=1,
                        preview_quality=80,
                        preview_width=720,
                        preview_height=405,
                        max_detection=500,
                        output_dir="./outputs",
                        tracking=dict(det_w=1280,
                                      det_h=720,
                                      dist_thresh=100,
                                      max_object_count=100,
                                      max_skip_frame=2,
                                      max_trace_length=64,
                                      min_rock_pix=24,
                                      max_rock_pix=300,
                                      min_speed_threshold=1.0,
                                      min_y_motion=0.1,
                                      min_y_x_ratio=1.618,
                                      min_trace_length=4,
                                      allow_move_up=False,
                                      move_up_thresh=10),
                        camera_id="",
                        location="",
                        sms=dict(enable=False,
                                 phone="",
                                 sender=""),
                        vcr_path="/opt/vcr",
                        roi_mask="./mask.png",
                        green_max_ratio=0.5,
                        frame_dist_cm=10.0)
    if app_state == "running":
        settings_obj = config.to_dict()
    return render_template('setup.tpl.html', 
                           video_src=settings_obj['video_src'],
                           camera_web_url=settings_obj['camera_web_url'],
                           preview_fps=settings_obj['preview_fps'],
                           preview_quality=settings_obj['preview_quality'],
                           preview_width=settings_obj['preview_width'],
                           preview_height=settings_obj['preview_height'],
                           max_detection=settings_obj['max_detection'],
                           output_dir=settings_obj['output_dir'],
                           tracking__det_w=settings_obj['tracking']['det_w'],
                           tracking__det_h=settings_obj['tracking']['det_h'],
                           tracking__dist_thresh=settings_obj['tracking']['dist_thresh'],
                           tracking__max_object_count=settings_obj['tracking']['max_object_count'],
                           tracking__max_skip_frame=settings_obj['tracking']['max_skip_frame'],
                           tracking__max_trace_length=settings_obj['tracking']['max_trace_length'],
                           tracking__min_rock_pix=settings_obj['tracking']['min_rock_pix'],
                           tracking__max_rock_pix=settings_obj['tracking']['max_rock_pix'],
                           tracking__min_speed_threshold=settings_obj['tracking']['min_speed_threshold'],
                           tracking__min_y_motion=settings_obj['tracking']['min_y_motion'],
                           tracking__min_y_x_ratio=settings_obj['tracking']['min_y_x_ratio'],
                           tracking__allow_move_up=settings_obj['tracking']['allow_move_up'],
                           tracking__move_up_thresh=settings_obj['tracking']['move_up_thresh'],
                           sms__enable=settings_obj['sms']['enable'],
                           sms__phone=settings_obj['sms']['phone'], 
                           sms__sender=settings_obj['sms']['sender'],
                           camera_id=settings_obj['camera_id'],
                           description=settings_obj['location'],
                           vcr_path=settings_obj['vcr_path'],
                           roi_mask=settings_obj['roi_mask'],
                           green_max_ratio=settings_obj['green_max_ratio'],
                           frame_dist_cm=settings_obj['frame_dist_cm'])


@app.route('/api/setup', methods=['POST'])
def api_setup():
    global config
    if not request.json:
        return jsonify({'status': 'error', 'message': 'Invalid request'}), 400
    # update config, then saveConfig to settings.yml
    try:
        config.debug = request.json.get('debug', False)
        config.video_src = request.json.get('video_src', '')
        config.camera_web_url = request.json.get('camera_web_url', '')
        config.preview_fps = request.json.get('preview_fps', 1)
        config.preview_quality = request.json.get('preview_quality', 80)
        config.preview_width = request.json.get('preview_width', 854)
        config.preview_height = request.json.get('preview_height', 480)
        config.max_detection = request.json.get('max_detection', 240)
        config.output_dir = request.json.get('output_dir', './outputs')
        config.tracking.det_w = request.json.get('tracking__det_w', 720)
        config.tracking.det_h = request.json.get('tracking__det_h', 720)
        config.tracking.dist_thresh = request.json.get('tracking__dist_thresh', 100)
        config.tracking.max_object_count = request.json.get('tracking__max_object_count', 30)
        config.tracking.max_skip_frame = request.json.get('tracking__max_skip_frame', 3)
        config.tracking.max_trace_length = request.json.get('tracking__max_trace_length', 4)
        config.tracking.min_rock_pix = request.json.get('tracking__min_rock_pix', 16)
        config.tracking.max_rock_pix = request.json.get('tracking__max_rock_pix', 300)
        config.tracking.min_speed_threshold = request.json.get('tracking__min_speed_threshold', 1.0)
        config.tracking.min_y_motion = request.json.get('tracking__min_y_motion', 0.1)
        config.tracking.min_y_x_ratio = request.json.get('tracking__min_y_x_ratio', 1.618)

        config.tracking.allow_move_up = True
        config.tracking.move_up_thresh = 0
        config.camera_id = request.json.get('camera_id', '')
        config.location = request.json.get('description', '')
        config.sms.enable = request.json.get('sms__enable', False)
        config.sms.phone = request.json.get('sms__phone', '')
        config.sms.sender = request.json.get('sms__sender', '')
        config.vcr_path = request.json.get('vcr_path', '/opt/vcr')
        config.roi_mask = request.json.get('roi_mask', './mask.png')
        config.green_max_ratio = request.json.get('green_max_ratio', 0.5)
        config.frame_dist_cm = request.json.get('frame_dist_cm', 10.0)
        ret = saveConfig("settings.yml")
        yield "<script>window.location.href='/wait/5';</script>"
        time.sleep(2)
        webserver.shutdown()
        webserver.server_close()
        exit_program()
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/webui')
@app.route('/webui/')
def webui():
    if app_state == "setup":
        return "<script>window.location.href='/webui/setup';</script>"
    
    last_n_alerts = []
    try:
        last_n_alerts = load_event_log("events.csv", None)[:20]
        last_n_alerts = [(datetime.fromtimestamp(int(float(x[5]))), 
                          int(float(x[4])), 
                          round(float(x[1]), 3), 
                          round(float(x[3]), 2), 
                          str(x[7])) for x in last_n_alerts]
    except:
        # log traceback
        logging.error(traceback.format_exc())
    video_preview_url = "/video_preview"
    live_url = config.camera_web_url
    location = config.location
    camera_id = config.camera_id
    target_name = config.target_name
    target_tags = config.target_tags
    target_desc = config.target_desc
    target_range = config.target_range
    target_range_unit = config.target_range_unit
    return render_template('index.tpl.html', 
                           live_url=live_url, 
                           alerts=last_n_alerts, 
                           video_preview_url=video_preview_url,
                           location=location, camera_id=camera_id,
                           target_name=target_name, target_tags=target_tags,
                           target_desc=target_desc, target_range=target_range,
                           target_range_unit=target_range_unit)


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
        if os.path.exists("/opt/dets"):
            zip_dir("/opt/dets/", "/tmp/dets.zip", clear_dir=False)
        else:
            zip_dir("./tmp/", "/tmp/dets.zip", clear_dir=False)
        return send_from_directory("/tmp/", "dets.zip", as_attachment=True)
    except:
        return jsonify({'status': 'error', 'message': 'failed to get dets'}), 500
   

@app.route('/api/manual_cutoff')
def api_manual_cutoff():
    config.manual_cutoff = not config.manual_cutoff
    return ("1" if config.manual_cutoff else "0")


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
        _up_dict.tracking.min_speed_threshold = request.form.get('min_speed_threshold', 1.0)
        _up_dict.tracking.min_y_motion = request.form.get('min_y_motion', 0.1)
        _up_dict.tracking.min_y_x_ratio = request.form.get('min_y_x_ratio', 1.618)
        _up_dict.tracking.allow_move_up = request.form.get('allow_move_up', False)
        _up_dict.tracking.move_up_thresh = request.form.get('move_up_thresh', 10)
        

        _up_dict.debug = request.form.get('debug', False)
        _up_dict.video_src = request.form.get('video_src', '')
        _up_dict.camera_web_url = request.form.get('camera_web_url', '')
        _up_dict.preview_fps = request.form.get('preview_fps', 10)
        _up_dict.preview_width = request.form.get('preview_width', 320)
        _up_dict.preview_height = request.form.get('preview_height', 180)
        _up_dict.max_detection = request.form.get('max_detection', 240)

        _up_dict.output_dir = request.form.get('output_dir', './outputs')
        _up_dict.vcr_path = request.form.get('vcr_path', '/opt/vcr')
        _up_dict.roi_mask = request.form.get('roi_mask', './mask.png')
        _up_dict.green_max_ratio = request.form.get('green_max_ratio', 0.5)
        _up_dict.frame_dist_cm = request.form.get('frame_dist_cm', 10.0)
        
        _up_dict.camera_id = request.form.get('camera_id', '')
        _up_dict.location = request.form.get('location', '')
        


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
    output_mp4 = os.path.join(record_base, "augmented.mkv")
    if os.path.exists(output_mp4):
        file_resp = send_from_directory(record_base, "augmented.mkv")
        resp = make_response(file_resp)
        dt = datetime.fromtimestamp(int(record_id)//1000).strftime("%Y%m%d_%H%M%S")
        resp.headers["Content-Disposition"] = f"attachment; filename={dt}.mkv"
        return resp
    elif not os.path.exists(output_mp4):
        return jsonify({'status': 'error', 'message': 'Video not ready'}), 404


def run_setup_loop():
    pass # to-do



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Rock detection and tracking')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--cfg', type=str, help='Config file path', default="settings.yml")
    parser.add_argument('-i', '--video_src', type=str, help='Video source')
    parser.add_argument('-o', '--output_dir', type=str, help='Output video file basepath', default="outputs")
    parser.add_argument('-v', '--vcr_dir', type=str, help='VCR directory', default="vcr")
    parser.add_argument('--port', type=int, help='Webserver port', default=8080)
    args = parser.parse_args()

    # load config
    has_config = loadConfig(args.cfg)
    if not has_config:
        app_state = "setup"
    
    if app_state == "setup":
        run_setup_loop()
        exit_program()
    

    if args.debug is True:
        config.debug = args.debug
    
    if args.video_src is not None and args.video_src != "":
        config.video_src = args.video_src

    if args.output_dir is not None and args.output_dir != "":
        config.output_dir = args.output_dir

    if args.vcr_dir is not None and args.vcr_dir != "":
        config.vcr_path = args.vcr_dir


    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    if not os.path.exists("events.csv"):
        with open("events.csv", "w") as f:
            f.write("")
            f.flush()
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
    video_fetch_thread = threading.Thread(target=fetch_frame_loop, 
                                          args=(config.video_src, main_loop_running_cb, frame_queue), 
                                          daemon=True)
    video_fetch_thread.start()

    # start video processing loop
    extra_info = Dict()
    video_process_thread = threading.Thread(target=process_frame_loop, 
                                            args=(config, main_loop_running_cb, frame_queue, current_frame, extra_info), 
                                            daemon=True)
    video_process_thread.start()

    ############################################################
    # start webserver
    try:
        print ("Start webserver...")
        logging.info('Starting Webserver...')
        #app.run(host="0.0.0.0", port=8080, debug=config.debug, threaded=True)
        webserver = make_server("0.0.0.0", args.port, app, threaded=True)
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

