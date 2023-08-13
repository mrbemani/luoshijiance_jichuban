# -*- encoding: utf-8 -*-

import sys
import os
import datetime
import json

def get_mp4_list(path):
    mp4_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mp4'):
                mp4_list.append(os.path.join(root, file))
    return mp4_list

def parse_trace_json(trace_json):
    with open(trace_json, 'r') as f:
        tracks = json.load(f)
    return tracks


def record_trace(record_id):

