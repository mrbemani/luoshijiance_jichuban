# -*- coding: utf-8 -*-


__author__ = "Shi Qi"

import os
import sys
import logging
from datetime import datetime
import json
from addict import Dict

from typing import List

from alibabacloud_dysmsapi20170525.client import Client as Dysmsapi20170525Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dysmsapi20170525 import models as dysmsapi_20170525_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_console.client import Client as ConsoleClient
from alibabacloud_tea_util.client import Client as UtilClient

os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'] = "LTAI5tCdUXSscV33Cr9pJJh6"
os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET'] = "2GBTWVedR66txvm9FlURrRtjF1ntob"


class TSAlibabaSMS(object):
    def __init__(self):
        pass

    @staticmethod
    def create_client(
        access_key_id: str,
        access_key_secret: str,
    ) -> Dysmsapi20170525Client:
        """
        使用AK&SK初始化账号Client
        @param access_key_id:
        @param access_key_secret:
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config(
            # 必填，您的 AccessKey ID,
            access_key_id=access_key_id,
            # 必填，您的 AccessKey Secret,
            access_key_secret=access_key_secret
        )
        # 访问的域名
        config.endpoint = f'dysmsapi.aliyuncs.com'
        return Dysmsapi20170525Client(config)

    @staticmethod
    def create_client_with_sts(
        access_key_id: str,
        access_key_secret: str,
        security_token: str,
    ) -> Dysmsapi20170525Client:
        """
        使用STS鉴权方式初始化账号Client，推荐此方式。
        @param access_key_id:
        @param access_key_secret:
        @param security_token:
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config(
            # 必填，您的 AccessKey ID,
            access_key_id=access_key_id,
            # 必填，您的 AccessKey Secret,
            access_key_secret=access_key_secret,
            # 必填，您的 Security Token,
            security_token=security_token,
            # 必填，表明使用 STS 方式,
            type='sts'
        )
        # 访问的域名
        config.endpoint = f'dysmsapi.aliyuncs.com'
        return Dysmsapi20170525Client(config)

    @staticmethod
    def sendSMS(
        location: str,
        phone_number: str,
        sender_name: str,
        ts_start: int,
        cam_no: str, 
        max_count: int, 
        max_vol: int,
        max_speed: int,
        template_code: str = 'SMS_461865249'
    ) -> None:
        client = TSAlibabaSMS.create_client(os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'], os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET'])
        dt_string = datetime.fromtimestamp(ts_start).strftime("%Y-%m-%d %H:%M:%S")
        content_dict = dict(
            time=dt_string,
            location=location,
            cam_no=str(cam_no),
            num=str(max_count),
            size=str(max_vol),
            speed=str(max_speed)
        )
        send_sms_request = dysmsapi_20170525_models.SendSmsRequest(
            phone_numbers=phone_number,
            sign_name=sender_name,
            template_code=template_code,
            template_param=json.dumps(content_dict)
        )
        runtime = util_models.RuntimeOptions()
        resp = client.send_sms_with_options(send_sms_request, runtime)
        return resp


def send_sms(
    config: Dict,
    event: Dict,
) -> None:
    try:
        cfg = Dict(config)
        if cfg.sms.enable is False:
            return
        location = cfg.location
        phone_number = cfg.sms.phone
        sender_name = cfg.sms.sender
        ts_start = event.ts_start
        cam_no = cfg.camera_id
        max_count = int(event.max_count)
        max_vol = round(event.max_vol, 2)
        max_speed = round(event.max_speed, 2)
        TSAlibabaSMS.sendSMS(
            location=location,
            phone_number=phone_number,
            sender_name=sender_name,
            ts_start=ts_start,
            cam_no=cam_no,
            max_count=max_count,
            max_vol=max_vol,
            max_speed=max_speed)
    except:
        logging.error("Failed to send sms")
        logging.error(sys.exc_info()[0])
        logging.error(sys.exc_info()[1])
        logging.error(sys.exc_info()[2])


