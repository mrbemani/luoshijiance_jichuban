# -*- encoding: utf-8 -*-

__author__ = 'Mr.Bemani'


import os
import time
from typing import Union
from typing import Callable
import cv2
import queue
import multiprocessing as mp
from addict import Dict
import numpy as np
from datetime import datetime
from PIL import Image
import threading
import logging


# write out video file as mp4
def make_video_cv(in_image_files, video_dir, fps=30, resolution=(1280, 720), remove_images=True):
    if not os.path.isdir(video_dir):
        logging.error("video directory not found")
        return False
    try:
        mp4_file = os.path.join(video_dir, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print (mp4_file)
        video_writer = cv2.VideoWriter(mp4_file, fourcc, fps, resolution)
        for image_file in in_image_files:
            image = cv2.imread(image_file)
            video_writer.write(image)
            if remove_images:
                os.unlink(image_file)
        video_writer.release()
        return True
    except:
        logging.error("error writing video file to: " + video_dir)
        return False

def make_video_ffmpeg(record_id):
    try:
        os.system(f"cd {os.getcwd()} && python3 ./ffmpeg_make_video.py {record_id}")
        return True
    except:
        return False


# fetch frame from video source and put it into frame_queue
def fetch_frame_loop(video_src: str, keep_running: Callable, frame_put_queue: Union[queue.Queue, mp.Queue]):
    # Capture livestream
    logging.info("Initiating fetch_frame_loop with source: " + repr(video_src))
    cap = cv2.VideoCapture(video_src)
    ret = True
    while keep_running():
        #time.sleep(0.0333333) # 30 fps
        ret, frame = cap.read()
        while not ret or frame is None: # try to reconnect forever
            cap.release()
            cap = cv2.VideoCapture (video_src)
            ret, frame = cap.read()
            if ret is None or frame is None:
                time.sleep(0.5)
                continue
        try:
            frame_put_queue.put(frame, timeout=0.03)
        except queue.Full:
            try:
                frame_put_queue.get(timeout=0.1)
            except:
                pass
            frame_put_queue.put_nowait(frame)
    cap.release()


