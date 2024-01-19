# -*- encoding: utf-8 -*-

import sys
import os
import time
import datetime
import json
import threading
import cv2
import api


def get_mp4_list(path):
    mp4_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mp4'):
                mp4_list.append(os.path.join(root, file))
    # sort mp4_list by timestamp
    mp4_list.sort(key=lambda x: int(x.split(os.path.sep)[-1].split('.')[0]))
    return mp4_list


def parse_trace_json(trace_json):
    with open(trace_json, 'r', encoding="utf-8") as f:
        trace_obj = json.load(f)
    tracks = []
    for track_id, track_info in trace_obj.items():
        track = {}
        track['track_id'] = track_id
        track['color'] = track_info[0]
        track['trace'] = track_info[1]
        tracks.append(track)
    return tracks


def draw_frame_overlay(frame, frame_ts, record_ts, tracks):
    for track in tracks:
        draw_color = track['color']
        for idx, trace in enumerate(track['trace']):
            if idx == 0:
                continue
            trace_ts = record_ts + trace[0]
            if trace_ts > frame_ts:
                break
            else:
                fh, fw = frame.shape[:2]
                cv2.line(frame, 
                         (int(track['trace'][idx-1][1][0] * fw), int(track['trace'][idx-1][1][1] * fh)), 
                         (int(track['trace'][idx][1][0] * fw), int(track['trace'][idx][1][1] * fh)), 
                         tuple(draw_color), 3, cv2.LINE_AA, 0)
                cv2.circle(frame,
                            (int(track['trace'][idx][1][0] * fw), int(track['trace'][idx][1][1] * fh)),
                            5,
                            tuple(draw_color),
                            -1)
    return frame


def augment_record_video(evt, fps=25, dim=(960, 540)):
    time.sleep(2.0)
    mp4_list = get_mp4_list(os.path.join('outputs', evt.record))
    trace_json = os.path.join('outputs', evt.record, 'trace.json')
    # if no mp4 files, return
    if len(mp4_list) < 1:
        return
    # else if mp4 files exists, fuse them
    mp4_start_ts = int(mp4_list[0].split(os.path.sep)[-1].split('.')[0])
    record_start_ts = evt.ts_start
    record_ts_offset = record_start_ts - mp4_start_ts
    tracks = parse_trace_json(trace_json)
    augmented_writer = cv2.VideoWriter(os.path.join('outputs', evt.record, 'augmented.mkv'), cv2.VideoWriter_fourcc(*'mp4v'), fps, dim)
    frame_gap = 1.0 / fps
    frame_idx = 0
    for mp4 in mp4_list:
        cap = cv2.VideoCapture(mp4)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            frame_ts = mp4_start_ts + frame_idx * frame_gap
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_NEAREST)
            aug_frame = draw_frame_overlay(frame, frame_ts, record_start_ts, tracks)
            augmented_writer.write(aug_frame)
            frame_idx += 1
        cap.release()
    augmented_writer.release()
    with open(os.path.join('outputs', evt.record, 'ok'), 'w', encoding="utf-8") as okfp:
        okfp.write('1')
        okfp.flush()
        os.fsync(okfp.fileno())
    # send record to server
    api.send_falling_rock_event(evt.record, tracks, 
                                evt.ts_start, evt.ts_end, 
                                evt.max_count, evt.max_volumn, 
                                evt.max_speed)


def make_fuse_output(evt):
    # call augment_record_video in a thread
    t = threading.Thread(target=augment_record_video, args=(evt,))
    t.start()


if __name__ == '__main__':
    record_ids = [d for d in os.listdir('outputs') if d.isdigit() and not os.path.exists(os.path.join('outputs', d, 'augmented.mkv'))]
    for record_id in record_ids:
        print ("processing record_id: {}".format(record_id))
        augment_record_video(record_id)
    print ("Done")

