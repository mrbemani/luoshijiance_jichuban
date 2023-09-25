# -*- coding: utf-8 -*-

__author__ = "Mr.Bemani"


import threading
import os
import time
from typing import Callable
import multiprocessing as mp
import subprocess as subp

# side run ssh tunnel
# /usr/bin/autossh -M 5678 -NR '*:8887:localhost:8080' ssh_tunnel@120.25.160.160

# side run http tunnel
# /usr/bin/autossh -M 5678 -NR '*:8887:localhost:8080' http_tunnel@120.25.160.160

# side run ffmpeg record loop
def vcr_record_loop(loop_running: Callable, vcr_path: str, video_src: str):
    while loop_running():
        command_str = f"/usr/bin/ffmpeg -i \"{video_src}\" -c copy -map 0 -f segment -segment_time 5 -segment_format mp4 -reset_timestamps 0 -strftime 1 -segment_atclocktime 1 -segment_clocktime_offset 0 {vcr_path}/%s.mp4"
        ffmpeg_proc = subp.Popen(command_str, shell=True)
        ffmpeg_proc.wait()
        time.sleep(0.01)


# side run clean loop
def vcr_clean_loop(loop_running: Callable, vcr_path, expire_minute: int = 10):
    while loop_running():
        command_str = f"find {vcr_path} -type f -mmin +{expire_minute} -delete"
        os.system(command_str, shell=True)
        time.sleep(expire_minute * 60)


def start_siderun_jobs(loop_running: Callable, video_src: str, vcr_path: str, expire_minute: int = 10):
    # start subprocess
    vcr_record_proc = mp.Process(target=vcr_record_loop, 
                         args=(loop_running, 
                               vcr_path, 
                               video_src), daemon=True)
    vcr_record_proc.start()
    
    time.sleep(5)

    vcr_clean_proc = mp.Process(target=vcr_clean_loop,
                         args=(loop_running, 
                               vcr_path,
                               expire_minute), daemon=True)
    vcr_clean_proc.start()

    time.sleep(10)


    while loop_running():
        time.sleep(1)

    print ("terminating...")
    vcr_record_proc.terminate()
    vcr_clean_proc.terminate()
    
    print ("joining...")
    vcr_record_proc.join()
    vcr_clean_proc.join()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="side run jobs cli")
    parser.add_argument("--video_src", type=str, default="/dev/video0", help="video source")
    parser.add_argument("--vcr_path", type=str, default="/home/autossh/vcr", help="vcr path")
    parser.add_argument("--expire_minute", type=int, default=10, help="vcr expire minute")
    args = parser.parse_args()

    print ("Start side run jobs...")

    video_src = args.video_src
    vcr_path = args.vcr_path
    expire_minute = args.expire_minute

    running = True


    start_siderun_jobs(lambda: running, 
                        video_src,
                        vcr_path,
                        expire_minute)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
        time.sleep(1)
        print("exit")
        exit(0)

