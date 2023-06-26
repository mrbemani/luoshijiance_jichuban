# -*- coding: utf-8 -*-

__author__ = 'Mr.Bemani'


import os
from addict import Dict
import yaml


############################################################
# define default config settings
config = Dict()

config.debug = False

config.frame_dist_cm = 1.0
config.max_detection = 240

config.video_src = 0
config.rock_boundaries = []

config.tracking.min_rock_pix = 8
config.tracking.max_rock_pix = 300
config.tracking.max_rock_ratio = 2.0
config.tracking.dist_thresh = 80
config.tracking.max_skip_frame = 3
config.tracking.max_trace_length = 2
config.tracking.max_object_count = 20
############################################################

default_cfgfile = os.path.join(os.path.curdir, "settings.yml")


def loadConfig(cfgfile=None):
    global config
    if cfgfile is None:
        cfgfile = default_cfgfile
    if not os.path.isfile(cfgfile):
        return False
    try:
        config.update(yaml.load(open(cfgfile, 'r'), Loader=yaml.FullLoader))
        return True
    except Exception as e:
        print (e)
    return False


def saveConfig(cfgfile=None):
    global config
    if cfgfile is None:
        cfgfile = default_cfgfile
    if not os.path.isfile(cfgfile):
        return False
    try:
        yaml.dump(config.to_dict(), open(cfgfile, 'w+'))
        return True
    except Exception as e:
        print (e)
    return False

