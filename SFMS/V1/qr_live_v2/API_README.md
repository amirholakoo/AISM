# QR Code Scanner Flask API

این API امکان کنترل از راه دور اسکنر QR Code را فراهم می‌کند.

## نصب و راه‌اندازی

1. نصب dependencies:
```bash
pip install -r requirements.txt
```

2. اجرای API:
```bash
python flask_api.py
```

API روی پورت 5002 اجرا می‌شود: `http://localhost:5002`

## Endpoints

### 1. شروع اسکن QR
**POST** `/start`

**Request Body:**
```json
{
    "video_source": "rtsp://127.0.0.1:8554/cam1"
}
```

**Response:**
```json
{
    "success": true,
    "message": "QR scanning started successfully",
    "status": {
        "is_running": true,
        "start_time": "2024-01-15 10:30:00",
        "current_session": {
            "video_source": "rtsp://127.0.0.1:8554/cam1"
        },
        "detected_codes": [],
        "fps": 0.0
    }
}
```

### 2. توقف اسکن QR
**POST** `/stop`

**Response:**
```json
{
    "success": true,
    "message": "QR scanning stopped successfully",
    "summary": {
        "total_codes_detected": 5,
        "start_time": "2024-01-15 10:30:00",
        "end_time": "2024-01-15 11:45:00",
        "video_source": "rtsp://127.0.0.1:8554/cam1",
        "detected_codes": [
            {
                "content": "QR_CODE_1",
                "timestamp": "2024-01-15T10:35:12"
            }
        ],
        "log_file": "output/qrcodes_20240115_103000.json"
    },
    "status": {
        "is_running": false,
        "start_time": null,
        "current_session": null,
        "detected_codes": [],
        "fps": 0.0
    }
}
```

### 3. وضعیت فعلی
**GET** `/status`

**Response:**
```json
{
    "success": true,
    "status": {
        "is_running": true,
        "start_time": "2024-01-15 10:30:00",
        "current_session": {
            "video_source": "rtsp://127.0.0.1:8554/cam1"
        },
        "fps": 25.5,
        "total_codes_detected": 3,
        "latest_codes": [
            {
                "content": "QR_CODE_3",
                "timestamp": "2024-01-15T10:42:33"
            }
        ]
    }
}
```

### 4. بررسی سلامت
**GET** `/health`

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15 10:30:00",
    "service": "QR Code Scanner API"
}
```

### 5. لیست فایل‌های لاگ
**GET** `/logs`

**Response:**
```json
{
    "success": true,
    "log_files": [
        {
            "filename": "qrcodes_20240115_103000.json",
            "size": 1024,
            "modified": "2024-01-15 11:45:00"
        }
    ]
}
```

### 6. دانلود فایل لاگ
**GET** `/logs/<filename>`

فایل JSON لاگ را برمی‌گرداند.

## مثال استفاده با curl

### شروع اسکن:
```bash
curl -X POST http://localhost:5002/start \
  -H "Content-Type: application/json" \
  -d '{"video_source": "0"}'
```

### بررسی وضعیت:
```bash
curl http://localhost:5002/status
```

### توقف اسکن:
```bash
curl -X POST http://localhost:5002/stop
```

## نکات مهم

1. **Threading**: اسکن QR در یک thread جداگانه اجرا می‌شود تا API responsive باشد.

2. **Video Source**: می‌توانید از منابع مختلف استفاده کنید:
   - `0` - دوربین پیش‌فرض
   - `rtsp://...` - RTSP stream
   - `path/to/video.mp4` - فایل ویدیو

3. **Log Files**: تمام QR Code های تشخیص داده شده در فایل JSON ذخیره می‌شوند.

4. **FPS Monitoring**: نرخ فریم‌ها در زمان واقعی محاسبه و نمایش داده می‌شود.

5. **Error Handling**: در صورت قطع اتصال ویدیو، سیستم به طور خودکار تلاش می‌کند دوباره متصل شود.

## Troubleshooting

- **خطای اتصال به ویدیو**: مطمئن شوید که منبع ویدیو در دسترس است
- **پورت در حال استفاده**: اگر پورت 5002 در حال استفاده است، می‌توانید در کد تغییر دهید
- **Memory Issues**: برای اجرای طولانی مدت، سیستم منابع کافی نیاز دارد 