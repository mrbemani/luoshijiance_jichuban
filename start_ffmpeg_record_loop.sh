#!/bin/bash

while true; do
    ffmpeg -i $VIDEO_SRC -c copy -map 0 -f segment -segment_time 5 -segment_format mp4 -reset_timestamps 0 -strftime 1 -segment_atclocktime 1 -segment_clocktime_offset 0 $VCR_PATH/%s.mp4
    sleep 1
done

