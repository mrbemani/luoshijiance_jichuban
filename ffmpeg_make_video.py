# -*- coding: utf-8 -*-

import sys
import os
import glob
import subprocess as subp


if __name__ == '__main__':
    record = sys.argv[1]
    print (record)
    base_path = os.path.abspath(os.path.dirname(__file__))
    outputs_path = os.path.join(base_path, 'outputs')
    record_base = os.path.join(outputs_path, record)
    output_mp4 = os.path.join(record_base, "output.mp4")
    record_frames = os.path.join(record_base, r"*.jpg")
    record_frame_count = len(glob.glob(record_frames))
    if os.path.isfile(output_mp4):
        sys.exit(1)
    if record_frame_count < 1:
        sys.exit(2)
    if not os.path.exists(output_mp4) and record_frame_count >= 5:
        try:
            print ("processing: " + record_base)
            ffmpeg_subp = subp.getoutput(f"cd /home/firefly/main_server && /usr/local/bin/ffmpeg -r 12 -f image2 -s 1920x1080 -pattern_type glob -i \"{record_frames}\" -vcodec mpeg4 -pix_fmt yuv420p \"{output_mp4}\"")
            print (f"ffmpeg_subp: {ffmpeg_subp}")
        except:
            os.system(f"cd {base_path} && rm -rf {output_mp4}")
        finally:
            os.system(f"cd {base_path} && rm -rf {record_base}/*.jpg")
