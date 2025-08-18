# Routes Structure

این پوشه شامل تمام API routes است که به صورت منطقی به فایل‌های مختلف تقسیم شده‌اند.

## فایل‌های موجود:

### 1. `vision_routes.py`
- API های مربوط به سیستم بینایی
- شامل `/api/vision/start`, `/api/vision/stop`, `/api/vision/status` و غیره

### 2. `warehouse_routes.py`
- API های مربوط به مدیریت انبارها
- شامل `/api/warehouses` برای CRUD عملیات انبارها

### 3. `product_routes.py`
- API های مربوط به مدیریت محصولات
- شامل `/api/products` برای CRUD عملیات محصولات

### 4. `loading_routes.py`
- API های مربوط به عملیات بارگیری
- شامل تمام endpoint های مربوط به loading operations

### 5. `unloading_routes.py`
- API های مربوط به عملیات تخلیه
- شامل تمام endpoint های مربوط به unloading operations

### 6. `shipment_routes.py`
- API های مربوط به مدیریت محموله‌ها
- شامل `/api/shipments/*` برای دریافت اطلاعات محموله‌ها

### 7. `operation_type_routes.py`
- API های مربوط به انواع عملیات
- شامل `/api/operation-types` برای مدیریت انواع عملیات

### 8. `vision_server_routes.py`
- API های مربوط به سرورهای بینایی
- شامل `/api/vision-servers` برای مدیریت سرورهای بینایی

### 9. `ssh_routes.py`
- API های مربوط به عملیات SSH
- شامل `/api/ssh/*` برای تست اتصال و کپی دیتابیس

## نحوه استفاده:

هر فایل یک Blueprint Flask ایجاد می‌کند که در `main.py` ثبت می‌شود:

```python
from routes.vision_routes import vision_bp
app.register_blueprint(vision_bp)
```

## مزایای این ساختار:

1. **قابلیت نگهداری بهتر**: هر بخش در فایل جداگانه‌ای قرار دارد
2. **خوانایی بیشتر**: کد به راحتی قابل فهم و ویرایش است
3. **قابلیت توسعه**: اضافه کردن API های جدید راحت‌تر است
4. **تست بهتر**: هر بخش را می‌توان جداگانه تست کرد
5. **کار تیمی**: چندین توسعه‌دهنده می‌توانند همزمان روی بخش‌های مختلف کار کنند

## فایل اصلی:

فایل `main.py` جدید جایگزین `ui.py` شده و تمام blueprint ها را import و register می‌کند.
