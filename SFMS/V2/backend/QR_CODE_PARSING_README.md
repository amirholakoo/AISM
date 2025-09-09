# QR Code Parsing در سیستم بارگیری و تخلیه

## نحوه کارکرد

سیستم QR code parsing در endpoint موجود `/api/vision/stop` پیاده‌سازی شده است و برای هر نوع عملیات (loading/unloading) به صورت جداگانه پردازش می‌شود.

## جریان کار

### 1. شروع عملیات
- کاربر عملیات بارگیری یا تخلیه جدید شروع می‌کند
- وضعیت: `started`

### 2. توقف با بینایی
- کاربر دکمه `/stop` را می‌زند
- سیستم بینایی متوقف می‌شود
- داده‌های vision output دریافت می‌شود

### 3. پردازش جداگانه

#### برای عملیات بارگیری (Loading):
سیستم به طور خودکار:
- آیتم‌ها را از `summary.detailed_product_counts` استخراج می‌کند
- QR codes شامل "Reel Number" را شناسایی و پردازش می‌کند
- فیلدهای زیر را استخراج می‌کند:
  - **Reel Number**: شماره رول
  - **Width**: عرض (mm)
  - **GSM**: گرماژ
  - **Length**: طول (m)
  - **Breaks**: تعداد شکستگی‌ها
  - **Grade**: درجه/کیفیت

#### برای عملیات تخلیه (Unloading):
سیستم به طور خودکار:
- آیتم‌ها را از `summary.detailed_product_counts` استخراج می‌کند
- محصولات تخلیه شده را ذخیره می‌کند

### 4. ذخیره در دیتابیس
- آیتم‌های پردازش شده در جداول مربوطه ذخیره می‌شوند
- منبع: `vision`
- وضعیت عملیات: `vision`

## API Endpoints

### 1. Vision Stop (اصلی)
**URL**: `POST /api/vision/stop`

**Request Body**:
```json
{
  "warehouse_id": "warehouse1",
  "operation_type": "loading"  // یا "unloading"
}
```

**Response**:
```json
{
  "success": true,
  "message": "عملیات بارگیری با موفقیت متوقف شد و 2 آیتم بارگیری ذخیره شد.",
  "loading_id": 123,
  "loading_token": "123.warehouse1.1234567890.hash",
  "summary": {...},
  "qrcodes": [...]
}
```

### 2. Vision Stop Test (تست)
**URL**: `POST /api/vision/test-stop`

**Description**: تست عملکرد بدون فراخوانی سرور بینایی واقعی

**Request Body**: همانند endpoint اصلی

## Frontend Integration

### نحوه ارسال پارامترها

Frontend باید حتماً دو پارامتر زیر را ارسال کند:

```javascript
const requestBody = {
  warehouse_id: warehouseId,        // شناسه انبار
  operation_type: 'loading'         // یا 'unloading'
};

const res = await fetch('/api/vision/stop', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(requestBody)
});
```

### صفحات Frontend

#### LoadingPage.jsx
```javascript
const requestBody = { 
  warehouse_id: warehouseId,
  operation_type: 'loading'  // ✅ درست
};
```

#### UnloadingPage.jsx
```javascript
const requestBody = { 
  warehouse_id: warehouseId, 
  operation_type: 'unloading'  // ✅ درست
};
```

#### CameraSelectionPage.jsx
```javascript
const requestBody = { 
  warehouse_id: currentWarehouseId,
  operation_type: operationType  // ✅ درست (از props گرفته می‌شود)
};
```

## ساختار کد

### توابع اصلی:
1. **`process_loading_vision_data(operation, result)`**: پردازش داده‌های بارگیری
2. **`process_loading_qr_codes(operation, result)`**: پردازش QR codes بارگیری
3. **`process_unloading_vision_data(operation, result)`**: پردازش داده‌های تخلیه

### مزایای ساختار جدید:
- **جداسازی منطق**: هر نوع عملیات جداگانه پردازش می‌شود
- **قابلیت نگهداری**: کد تمیزتر و قابل فهم‌تر
- **خطایابی بهتر**: هر بخش مستقل قابل تست است
- **انعطاف‌پذیری**: تغییرات در یک بخش روی بخش دیگر تأثیر نمی‌گذارد

## مثال داده

### برای بارگیری:
```json
{
  "qrcodes": [
    {
      "content": "Reel Number: 2156, Width: 280, GSM: 150, Length:~ 7200, Breaks: 1, Grade: Kraftliner PREMIUM",
      "timestamp": "2025-08-16T13:16:57.516264+0330"
    }
  ],
  "summary": {
    "detailed_product_counts": {
      "loaded": {
        "Product A": 5,
        "Product B": 3
      }
    }
  }
}
```

### برای تخلیه:
```json
{
  "summary": {
    "detailed_product_counts": {
      "unloaded": {
        "Product X": 2,
        "Product Y": 4
      }
    }
  }
}
```

## نکات مهم

- **تکرار**: اگر Reel Number تکراری باشد، نادیده گرفته می‌شود
- **فیلتر**: فقط QR codes با Reel Number معتبر پردازش می‌شوند
- **ذخیره**: آیتم‌های قبلی vision حذف و جدید جایگزین می‌شوند
- **وضعیت**: عملیات به وضعیت `vision` تغییر می‌کند
- **جداسازی**: loading و unloading کاملاً مستقل پردازش می‌شوند
- **پارامترهای اجباری**: `warehouse_id` و `operation_type` باید حتماً ارسال شوند

## تست

### تست بدون سرور بینایی
```bash
curl -X POST http://localhost:5000/api/vision/test-stop \
  -H 'Content-Type: application/json' \
  -d '{
    "warehouse_id": "warehouse1",
    "operation_type": "loading"
  }'
```

### تست واقعی
```bash
curl -X POST http://localhost:5000/api/vision/stop \
  -H 'Content-Type: application/json' \
  -d '{
    "warehouse_id": "warehouse1",
    "operation_type": "loading"
  }'
```

## استفاده

برای استفاده، کافی است:
1. عملیات بارگیری یا تخلیه را شروع کنید
2. دکمه `/stop` را بزنید (با پارامترهای صحیح)
3. سیستم به طور خودکار داده‌ها را پردازش و ذخیره می‌کند

## لاگ‌ها

سیستم لاگ‌های مفصل تولید می‌کند:
- پردازش آیتم‌ها
- تعداد QR codes پردازش شده
- خطاهای احتمالی
- وضعیت ذخیره‌سازی
- نوع عملیات (loading/unloading)
