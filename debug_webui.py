# -*- encoding: utf-8 -*-

__author__ = 'Mr.Bemani'

import sys
import os
import time
import shutil
import threading
from addict import Dict
from datetime import datetime

from flask import Flask, render_template, Response, jsonify, send_from_directory

if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
elif __file__:
    APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


main_loop_running = False
def main_loop_runing_cb():
    return main_loop_running

app = Flask(__name__)

@app.route('/')
def index():
    return "javascript:window.location.href='/webui/';"


@app.route('/webui/')
def webui():
    dt = int(datetime.now().timestamp())
    last_n_alerts = [
        [datetime.fromtimestamp(dt), 5, 10, 20],
        [datetime.fromtimestamp(dt), 5, 10, 20],
        [datetime.fromtimestamp(dt), 5, 10, 20],
        [datetime.fromtimestamp(dt), 5, 10, 20],
        [datetime.fromtimestamp(dt), 5, 10, 20],
    ]
    video_preview_url = "/assets/video_preview.png"
    return render_template('index.tpl.html', alerts=last_n_alerts, video_preview_url=video_preview_url)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('assets', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9007, debug=True)


