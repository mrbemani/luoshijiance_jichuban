# -*- encoding: utf-8 -*-

__author__ = 'Mr.Bemani'


import time
from typing import Callable
import cv2
import queue
import logging



# fetch frame from video source and put it into frame_queue
def fetch_frame_loop(video_src: str, keep_running: Callable, frame_put_queue: queue.Queue, fps: int=25):
    # Capture livestream
    logging.info("Initiating fetch_frame_loop with source: " + repr(video_src))
    is_rtsp = video_src[:7] in ["rtsp://", "rtmp://"]
    while keep_running():
        cap = cv2.VideoCapture(video_src)
        while keep_running():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                frame_put_queue.put(frame, timeout=0.01)
            except:
                try:
                    frame_put_queue.get(frame, timeout=0.01)
                except:
                    pass
            if not is_rtsp:
                time.sleep(1.0/fps)
    cap.release()


