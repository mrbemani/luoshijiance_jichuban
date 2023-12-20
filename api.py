# HTTP API version 1.0

__author__ = 'Mr.Bemani'

import os
import requests
import json
import threading
import event_utils as evtu
import uuid
from datetime import datetime
from addict import Dict
from configure import config

API_SERVER = os.environ.get("API_SERVER", "http://www.smallworld-network.com.cn:10001")
HTTP_FORWARD = f"http://www.smallworld-network.com.cn:" + os.environ.get("HTTP_FORWARD_PORT", "10001")
DEVICE_ID = os.environ.get("DEVICE_ID", "0000000000000000")

# 设备任务获取
def get_device_task():
    url = f"{API_SERVER}/api/v1/device/task"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "device_id": DEVICE_ID
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return json.dumps(dict(status=0, error=response.status_code, message="Bad HTTP status code"))
    except Exception as e:
        return json.dumps(dict(status=0, error=1, message=e.__str__()))

# 心跳包POST格式
def send_heartbeat(device_longitude = 0.0, device_latitude = 0.0, device_height = 0.0, 
                   device_angle = 0.0, device_battery = 0.0, device_solar_status = 0.0, 
                   device_temperature = 0.0, device_humidity = 0.0, device_pressure = 0.0, 
                   device_acceleration = 0.0, device_speed = 0.0, device_direction = 0.0):
    url = f"{API_SERVER}/api/v1/device/heartbeat"
    device_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    frame_url = f"{HTTP_FORWARD}/frames/frame_{device_time}.jpg"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "device_id": DEVICE_ID,
        "device_type": "ND-ICUBE",
        "device_version": "1.0",
        "device_status": "online",
        "device_location": config.location,
        "device_description": config.target_desc,
        "device_longitude": device_longitude,
        "device_latitude": device_latitude,
        "device_height": device_height,
        "device_angle": device_angle,
        "device_battery": device_battery,
        "device_solar_status": device_solar_status,
        "device_temperature": device_temperature,
        "device_humidity": device_humidity,
        "device_pressure": device_pressure,
        "device_acceleration": device_acceleration,
        "device_speed": device_speed,
        "device_direction": device_direction,
        "device_time": device_time,
        "frame_url": frame_url
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return json.dumps(dict(status=0, error=response.status_code, message="Bad HTTP status code"))
    except Exception as e:
        return json.dumps(dict(status=0, error=1, message=e.__str__()))

# 落石事件POST格式
def send_falling_rock_event(event_id, traces, start_time, end_time, max_count, max_volumn, max_speed):
    url = f"{API_SERVER}/api/v1/event/add"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    video_url = f"{HTTP_FORWARD}/get_record_video/{event_id}"
    video_expire = end_time + 3600.0
    data = {
        "type": "falling_rock",
        "data": {
            "device_id": DEVICE_ID,
            "event_id": event_id,
            "traces": traces,
            "start_time": datetime.fromtimestamp(start_time).strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": datetime.fromtimestamp(end_time).strftime("%Y-%m-%dT%H:%M:%S"),
            "max_count": max_count,
            "max_volumn": max_volumn,
            "max_speed": max_speed,
            "video_url": video_url,
            "video_expire": datetime.fromtimestamp(video_expire).strftime("%Y-%m-%dT%H:%M:%S")
        }
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return json.dumps(dict(status=0, error=response.status_code, message="Bad HTTP status code"))
    except Exception as e:
        return json.dumps(dict(status=0, error=1, message=e.__str__()))

# 表面变化事件POST格式
def send_surface_change_event(event_id, flow, start_time, end_time, start_image_data, end_image_data):
    url = f"{API_SERVER}/api/v1/event/add"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "type": "surface_change",
        "data": {
            "device_id": DEVICE_ID,
            "event_id": event_id,
            "flow": flow,
            "start_time": datetime.fromtimestamp(start_time).strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": datetime.fromtimestamp(end_time).strftime("%Y-%m-%dT%H:%M:%S"),
            "start_image_data": start_image_data,
            "end_image_data": end_image_data
        }
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return json.dumps(dict(status=0, error=response.status_code, message="Bad HTTP status code"))
    except Exception as e:
        return json.dumps(dict(status=0, error=1, message=e.__str__()))
