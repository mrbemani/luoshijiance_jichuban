# -*- coding: utf-8 -*-

__author__ = 'Mr.Bemani'


import os
import logging
from typing import Optional
from addict import Dict
import yaml

logging.basicConfig(level=logging.WARN)

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
        config.update(yaml.load(open(cfgfile, 'r', encoding="utf-8"), Loader=yaml.FullLoader))
        return True
    except Exception as e:
        logging.error(e)
    return False


def saveConfig(config_obj: Dict, cfgfile: Optional[str]=None):
    if cfgfile is None:
        cfgfile = default_cfgfile
    if not os.path.isfile(cfgfile):
        return False
    try:
        yaml.dump(config_obj.to_dict(), open(cfgfile, 'w+'))
        return True
    except Exception as e:
        logging.error(e)
    return False

