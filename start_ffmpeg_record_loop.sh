#!/bin/bash

video_src="rtsp://admin:nd12345678@10.168.1.108/"
dest_dir="/opt/vcr"


while true; do
    ffmpeg -i $video_src -c copy -map 0 -f segment -segment_time 5 -segment_format mp4 -reset_timestamps 0 -strftime 1 -segment_atclocktime 1 -segment_clocktime_offset 0 $dest_dir/%s.mp4
    sleep 1
done

