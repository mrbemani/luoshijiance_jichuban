# -*- encoding: utf-8 -*-

__author__ = 'Mr.Bemani'

import sys
import os
import time
import shutil
import threading
import logging

from flask import Flask, render_template, Response, jsonify, send_from_directory

from .configure import config, loadConfig, saveConfig

from .video import fetch_frame_loop, create_video_writer
from .magic_happening import process_frame_loop, frame_queue

if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
elif __file__:
    APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


main_loop_running = False
def main_loop_running_cb():
    return main_loop_running

app = Flask(__name__)

alert_image_fname = os.path.join(APP_BASE_DIR, "assets", "alert_image.png")




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
    last_n_alerts = []
    return render_template('html/index.tpl.html', alerts=last_n_alerts, video_preview_url="/video_preview")


@app.route('/video_preview')
def video_preview():
    try:
        frm = frame_queue.get(timeout=1)
        yield Response(, mimetype='multipart/x-mixed-replace; boundary=frame')
    except:
        logging.error("Failed to get MJPEG frame from queue")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Rock detection and tracking')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--cfg', type=str, help='Config file path', default="settings.yml")
    parser.add_argument('-i', '--video_src', type=str, help='Video source')
    parser.add_argument('-o', '--output_dir', type=str, help='Output video file basepath')
    parser.add_argument('--pc_test', action='store_true', help='Test on PC')
    args = parser.parse_args()

    # load config
    loadConfig(args.cfg)
    if args.debug is True:
        config.debug = args.debug
    
    if args.video_src is not None and args.video_src != "":
        config.video_src = args.video_src

    if args.output_dir is not None and args.output_dir != "":
        config.output_dir = args.output_dir

    if args.pc_test is True:
        config.pc_test = True

    # clear tmp dir
    if os.path.exists("./tmp"):
        shutil.rmtree("./tmp")
    os.mkdir("./tmp")

    main_loop_running = True
    
    # start video fetch loop
    video_fetch_thread = threading.Thread(target=fetch_frame_loop, args=(config, frame_queue, main_loop_running_cb))
    video_fetch_thread.start()

    # start video processing loop
    video_process_thread = threading.Thread(target=process_frame_loop, args=(config, main_loop_running_cb))
    video_process_thread.start()

    print ("Application is Deinitializing...", end='', flush=True)
    main_loop_running = False
    time.sleep(0.5)

    print ("OK")
    if config.debug:
        print ("App closed.")

    app.run(host='0.0.0.0', port=8080, debug=True)


