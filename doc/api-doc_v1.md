# NanDi iCube - API Document

version: 1.0

author: David Shi

---

## HTTP API 说明

- 所有请求均为 `POST` 请求
- 所有请求均为 `JSON` 格式
- 所有请求均需要 `Authorization` 头部, 值为 `Bearer <token>`
- 所有请求均需要 `Content-Type` 头部, 值为 `application/json`
- 所有请求均需要 `Accept` 头部, 值为 `application/json`

####  Request 格式

```json
// 请求为 JSON 对象
// 根据不同请求，内容格式不同
```

#### Response 格式

```json
{
    "status": 1,        // 返回状态 1 为成功, 0 为失败
    "data": {},         // 回复数据, 根据不同请求，格式不同. 当没有回复数据时，data 为 null.
}
```


## 设备任务获取

设备定时获取任务, 任务中包含设备需要执行的任务。

支持的任务类型:

- `settings` 设置设备算法参数
- `reboot` 重启设备

```http
POST /api/v1/device/task
```

#### Request

```json
{
    "device_id": "device_unique_id_string",         // 设备唯一ID
}
```

#### Response

- reboot 任务

```json
// reboot 任务
{
    "status": 1,        // 返回状态 1 为成功, 0 为失败
    "data": {
        "type": "reboot",   // 任务类型
        "device_id": "device_unique_id_string",         // 返回发送的设备ID，用于确认 [必选]
    }
}
```

- settings 任务

```json
// settings 任务
{
    "status": 1,        // 返回状态 1 为成功, 0 为失败
    "data": {
        "type": "update",   // 任务类型
        "device_id": "device_unique_id_string",     // 返回发送的设备ID，用于确认 [必选]
        "data": {
            "device_location": "",                  // 设备位置 [可选]
            "device_description": "",               // 设备描述 [可选]
            "device_longitude": 0.0,                // 设备经度 [可选]
            "device_latitude": 0.0,                 // 设备纬度 [可选]
            "frame_dist_cm": 10.0,                  // 画面单像素对应的实际尺寸CM [可选]
            "roi_mask_image_data": "base64 png image data",    // ROI Mask 数据 BASE64 编码，图片必须是单通道二值化黑白PNG图片格式。白色区域为观察区。黑色区域为非观察区。[可选]
            "video_src": "rtsp://www.example.com/video.mp4",   // 视频源地址 [可选]
            "vcr_path": "/path/to/vcr",             // VCR 路径 [可选]
            "dist_thresh": 100,                     // 最大跟踪距离搜索阈值 [可选]
            "max_trace_length": 64,                 // 最大轨迹节点数 [可选]
            "max_skip_frames": 10,                  // 最大跟踪跳帧数 [可选]
            "min_rock_pix": 100,                    // 最小落石像素面积 [可选]
            "max_rock_pix": 1000,                   // 最大落石像素面积 [可选]
            "min_rock_speed": 1.0,                  // 最小落石速度 M/s [可选]
            "min_y_frame_motion": 0.1,              // 最小垂直位移像素距离 / 帧图像纵向分辨率。举例: 假设画面为1920x1080像素，Y方向向下位移100像素为 100/1080. [可选]
            "min_y_x_ratio": 0.1,                   // 最小垂直位移像素距离 / 水平位移像素距离。举例: 假设画面为1920x1080像素，Y方向向下位移100像素，X方向向右位移10像素，那么dY/dX = 100/10 = 10. [可选]
            "min_trace_length": 4,                  // 最小轨迹节点数 [可选]
            "max_object_count": 100,                // 最大同时掉落个数 [可选]
            "surface_change_min_magnitude": 0.5,    // 最小表面变化位移阈值 (像素，可设置为亚像素，比如: 0.7, 3.2 等值) [可选]
            "surface_change_preprocess_gaussian_kernel_size": 3, // 表面变化预处理高斯模糊核大小 [可选]
            "surface_change_preprocess_gaussian_sigma": 0.0, // 表面变化预处理高斯模糊核标准差 [可选]
            "surface_change_preprocess_median_kernel_size": 3, // 表面变化预处理中值滤波核大小 [可选]
            "surface_change_pyramid_scale": 0.5, // 表面变化金字塔缩放比例 [可选]
            "surface_change_pyramid_levels": 3, // 表面变化金字塔层数 [可选]
            "surface_change_win_size": 15, // 表面变化光流窗口大小 [可选]
            "surface_change_max_iter": 3, // 表面变化光流最大迭代次数 [可选]
            "surface_change_poly_n": 5, // 表面变化光流多项式展开阶数 [可选]
            "surface_change_poly_sigma": 1.1, // 表面变化光流多项式展开标准差 [可选]
            "surface_change_roi_mask": "base64 png image data", // 表面变化光流 ROI Mask 数据 BASE64 编码，图片必须是单通道二值化黑白PNG图片格式。白色区域为观察区。黑色区域为非观察区。[可选]
            "sms_enable": true,                     // 短信报警开关 [可选]
            "sms_phone": "sms_phone_string",        // 短信报警电话 [可选]
            "sms_sender": "sms_sender_string",      // 短信报警发送者名称 [可选]
        }
    }
}
```

## 心跳包POST格式

心跳包每隔10秒发送一次。为定时发送的包, 用于告知服务器设备在线状态

```http
POST /api/v1/device/heartbeat
```

#### Request

```json
{
    "device_id": "device_unique_id_string",         // 设备唯一ID
    "device_type": "device_type_string",            // 设备类型
    "device_version": "device_version_string",      // 设备版本
    "device_status": "device_status_string",        // 设备状态
    "device_location": "location_string",           // 设备位置
    "device_description": "description_string",     // 设备描述
    "device_longitude": 0.0,                        // 设备经度 [如果设备有GPS信息]
    "device_latitude": 0.0,                         // 设备纬度 [如果设备有GPS信息]
    "device_height": 0.0,                           // 设备高度 [如果设备有GPS信息]
    "device_angle": 0.0,                            // 设备角度 [如果设备有GPS信息]
    "device_battery": 0.0,                          // 设备电量 [如果设备有电池]
    "device_solar_status": 0.0,                     // 设备太阳能状态 [如果设备有太阳能]
    "device_temperature": 0.0,                      // 设备温度 [如果设备有温度传感器]
    "device_humidity": 0.0,                         // 设备湿度 [如果设备有湿度传感器]
    "device_pressure": 0.0,                         // 设备气压 [如果设备有气压传感器]
    "device_acceleration": 0.0,                     // 设备加速度 [如果设备有加速度传感器]
    "device_speed": 0.0,                            // 设备速度 [如果设备有速度传感器]
    "device_direction": 0.0,                        // 设备方向 [如果设备有GPS信息]
    "device_time": "2019-01-01T00:00:00",           // 设备当前系统时间
    "frame_url": "http://www.example.com/frame.jpg" // 设备当前画面URL [无画面时为 null]
}
```

#### Response

```json
{
    "status": 1,        // 返回状态 1 为成功, 0 为失败
    "data": null
}
```


## 落石事件POST格式

当设备检测到落石事件时, 会发送此事件，事件中包含落石的轨迹信息，以及落石的基本信息，如最大个数，最大体积，最大速度等

```http
POST /api/v1/event/add
```

#### Request

```json
{
    "type": "falling_rock", // 事件类型 falling_rock 为落石事件
    "data": {
        "device_id": "device_unique_id_string",
        "event_id": "event_unique_id_string",
        "traces": [
            [
                [time: float, x: int, y: int],
                [time: float, x: int, y: int],
                [time: float, x: int, y: int],
                ...
            ],
            [
                [time: float, x: int, y: int],
                [time: float, x: int, y: int],
                [time: float, x: int, y: int],
                ...
            ],
            [
                [time: float, x: int, y: int],
                [time: float, x: int, y: int],
                [time: float, x: int, y: int],
                ...
            ],
            ...
        ],
        "start_time": "2019-01-01 00:00:00", // 事件开始时间
        "end_time": "2019-01-01 00:01:00", // 事件结束时间
        "max_count": 5, // 最大同时掉落个数
        "max_volumn": 3.2, // 最大落石体积 m^3
        "max_speed": 4.1, // 最大掉落运动速度 m/s
        "video_url": "http://www.example.com/video.mp4", // 事件视频地址
        "video_expire": "2019-01-01T08:01:00", // 事件视频过期时间
    }
}
```

#### Response

```json
{
    "status": 1,        // 返回状态 1 为成功, 0 为失败
    "data": null
}
```


## 表面变化事件POST格式

当设备检测到表面变化超过阈值时, 会发送此事件，事件中包含表面变化的各点位移信息，以及表面变化的初始和当前图像以及时间。

```http
POST /api/v1/event/add
```

#### Request

```json
{
    "type": "surface_change", // 事件类型 surface_change 为表面变化事件
    "data": {
        "device_id": "device_unique_id_string",
        "event_id": "event_unique_id_string",
        "flow": [
            [x: float, y: float, dx: float, dy: float], // 位移超过阈值的点, x, y 为坐标, dx, dy 为位移
            [x: float, y: float, dx: float, dy: float],
            [x: float, y: float, dx: float, dy: float],
            [x: float, y: float, dx: float, dy: float],
            ...
        ],
        "start_time": "2019-01-01 00:00:00", // 初始参考时间
        "end_time": "2021-03-01 07:03:00", // 位移超过阈值被检测到的时间
        "start_image_data": "base_image_data_string", // 初始参考图像数据 BASE64 编码
        "end_image_data": "change_image_data_string", // 变化图像数据 BASE64 编码
    }
}
```

#### Response

```json
{
    "status": 1,        // 返回状态 1 为成功, 0 为失败
    "data": null
}
```

