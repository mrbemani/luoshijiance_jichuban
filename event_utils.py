# -*- encoding: utf-8 -*-

__author__ = 'Mr.Bemani'

import io
import csv
from typing import Optional, Union
from addict import Dict
from datetime import datetime
import logging


eventTypes = [
    'falling_rock',
]


def create_falling_rock_event():
    event = Dict()
    event.type = 'falling_rock'
    event.max_vol = 0
    event.total_vol = 0
    event.max_speed = 0
    event.max_count = 0
    event.ts_start = datetime.now().timestamp()
    event.ts_end = datetime.now().timestamp()
    event.record = str(int(event.ts_start*1000))
    return event

def load_event_object(evt_data: Union[list, tuple]):
    event = Dict()
    event.type = evt_data[0]
    event.max_vol = float(evt_data[1])
    event.total_vol = float(evt_data[2])
    event.max_speed = float(evt_data[3])
    event.max_count = int(evt_data[4])
    event.ts_start = float(evt_data[5])
    event.ts_end = float(evt_data[6])
    event.record = evt_data[7]
    return event


def fmt_event(event):
    event_fmt = []
    if event.type == 'falling_rock':
        event_fmt.append(event.type)
        event_fmt.append(event.max_vol)
        event_fmt.append(event.total_vol)
        event_fmt.append(event.max_speed)
        event_fmt.append(event.max_count)
        event_fmt.append(event.ts_start)
        event_fmt.append(event.ts_end)
        event_fmt.append(event.record)
    else:
        event_fmt.append("unknown")
        event_fmt.append(0)
        event_fmt.append(0)
        event_fmt.append(0)
        event_fmt.append(0)
        event_fmt.append(datetime.now().timestamp())
        event_fmt.append(datetime.now().timestamp())
        event_fmt.append("")
    return event_fmt



def load_event_log(eventlog: str, num_events: Optional[int] = 5):
    events = []
    with open(eventlog, newline='', encoding="utf-8") as f:
        csv_reader = csv.reader(f, quotechar='"')
        if num_events is None:
            for row in csv_reader:
                events.append(row)
        else:
            events = csv_reader[-num_events:]
    events.reverse()
    return events



def store_event(eventlog_store: Union[io.TextIOWrapper, str], event: Dict):
    event_row = fmt_event(event)
    if type(eventlog_store) == str:
        with open(eventlog_store, 'a', newline='', encoding="utf-8") as f:
            csv.writer(f, quotechar='"').writerow(event_row)
    elif type(eventlog_store) == io.TextIOWrapper:
        csv.writer(eventlog_store, quotechar='"').writerow(event_row)
    else:
        raise TypeError("eventlog_store must be a str or a file handle")

