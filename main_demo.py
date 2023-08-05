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
os.environ["PC_TEST"] = "1"
############################################################

from configure import config, loadConfig, saveConfig

from video import fetch_frame_loop
from magic_happening import process_frame_loop, PF_W, PF_H

from event_utils import load_event_log

current_frame = None

main_loop_running = False
def main_loop_running_cb():
    return main_loop_running

frame_queue = queue.Queue(2)
out_queue = queue.Queue(50)


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

    # start video process loop
    video_process_thread = threading.Thread(target=process_frame_loop, args=(config, main_loop_running_cb, frame_queue, out_queue, current_frame))
    video_process_thread.setDaemon(True)
    video_process_thread.start()
    
    ############################################################
    # start gui loop
    try:
        while True:
            cv.imshow("Rock Detection GUI Demo", current_frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt, stopping...')
    cv.destroyAllWindows()

    print ("Application is Deinitializing...", end='', flush=True)
    main_loop_running = False
    time.sleep(0.5)
    video_fetch_thread.join(timeout=3)
    video_process_thread.join(timeout=3)
    cv.destroyAllWindows()
    
    print ("OK")
    if config.debug:
        print ("Server closed.")

