# -*- encoding: utf-8 -*-

__author__ = 'Mr.Bemani'

import sys
import os

from flask import 
from configure import config, loadConfig, saveConfig
from magic_happening import process_frame_loop, frame_queue


if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
elif __file__:
    APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = flask.Flask(__name__)

alert_image_fname = os.path.join(APP_BASE_DIR, "assets", "alert_image.png")

@app.route('/')
def index():
    return 'Hello, World!'


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

    print ("Application is Deinitializing...", end='', flush=True)
    main_loop_running = False
    time.sleep(0.5)

    print ("OK")
    if config.debug:
        print ("App closed.")

    app.run(host='0.0.0.0', port=5000, debug=True)


