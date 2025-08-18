# دیکشنری ترجمه پیام‌های سرور بینایی
VISION_MESSAGES = {
    # پیام‌های موفقیت
    "Video processing started successfully": "پردازش ویدیو با موفقیت شروع شد",
    "Video processing stopped successfully": "پردازش ویدیو با موفقیت متوقف شد",
    "Processing stopped.": "پردازش متوقف شد",
    "Test summary data": "داده‌های نمونه summary",
    
    # پیام‌های خطا
    "Processing is already running": "پردازش در حال اجرا است",
    "No processing session is currently running": "هیچ جلسه پردازشی در حال اجرا نیست",
    "Failed to start processing": "خطا در شروع پردازش",
    "Failed to start processing: Simulated error for warehouse 2": "خطا در شروع پردازش: خطای شبیه‌سازی برای انبار 2",
    "Failed to start processing: Random error occurred": "خطا در شروع پردازش: خطای تصادفی رخ داد",
    "Failed to start video processing. Please check the video source, model weights, and system configuration.": "خطا در شروع پردازش ویدیو. لطفاً منبع ویدیو، وزن‌های مدل و پیکربندی سیستم را بررسی کنید.",
    "Failed to stop processing": "خطا در توقف پردازش", 
    "Failed to stop processing: Simulated error for warehouse 2": "خطا در توقف پردازش: خطای شبیه‌سازی برای انبار 2",
    "Failed to get status": "خطا در دریافت وضعیت",
    "Model load failed": "خطا در بارگذاری مدل",
    "Cannot open source": "نمی‌توان منبع را باز کرد",
    "Stream ended or connection lost. Attempting to reconnect...": "جریان پایان یافت یا اتصال قطع شد. در حال تلاش برای اتصال مجدد...",
    "Processing loop error": "خطا در حلقه پردازش",
    "Failed to save snapshot": "خطا در ذخیره عکس",
    
    # پیام‌های هشدار
    "Stream ended or connection lost. Attempting to reconnect...": "جریان پایان یافت یا اتصال قطع شد. در حال تلاش برای اتصال مجدد...",
    "هیچ رویدادی برای ویرایش وجود ندارد.": "هیچ رویدادی برای ویرایش وجود ندارد.",
    "آیا اطلاعات فوق مورد تایید است؟": "آیا اطلاعات فوق مورد تایید است؟",
    
    # پیام‌های اضافی
    "Attempting to open video source with cv2.VideoCapture": "در حال تلاش برای باز کردن منبع ویدیو",
    "Successfully opened video source": "منبع ویدیو با موفقیت باز شد",
    "Failed to open video source": "خطا در باز کردن منبع ویدیو",
    "Stream ended or connection lost. Breaking inner loop to reconnect.": "جریان پایان یافت یا اتصال قطع شد. در حال تلاش برای اتصال مجدد...",
    "An exception occurred in the processing loop": "خطایی در حلقه پردازش رخ داد",
    "Processing stopped.": "پردازش متوقف شد",
    "Failed to save file": "خطا در ذخیره فایل",
    "Saved snapshot": "عکس ذخیره شد",
    "Event: loaded": "رویداد: بارگیری",
    "Event: unloaded": "رویداد: تخلیه",
    "No counting for empty forklifts": "شمارش برای لیفتراک خالی انجام نمی‌شود",
    "Global cooldown for 'loaded' events is active": "زمان انتظار برای رویدادهای بارگیری فعال است",
    "Global cooldown for 'unloaded' events is active": "زمان انتظار برای رویدادهای تخلیه فعال است"
}

def translate_vision_message(message):
    """
    ترجمه پیام‌های سرور بینایی به فارسی
    """
    return VISION_MESSAGES.get(message, message)

def translate_vision_response(response_data):
    """
    ترجمه تمام پیام‌های موجود در پاسخ سرور بینایی
    """
    if isinstance(response_data, dict):
        # ترجمه پیام اصلی
        if 'message' in response_data:
            response_data['message'] = translate_vision_message(response_data['message'])
        
        # ترجمه پیام‌های درون summary
        if 'summary' in response_data and isinstance(response_data['summary'], dict):
            summary = response_data['summary']
            if 'message' in summary:
                summary['message'] = translate_vision_message(summary['message'])
        
        # ترجمه پیام‌های درون status
        if 'status' in response_data and isinstance(response_data['status'], dict):
            status = response_data['status']
            if 'message' in status:
                status['message'] = translate_vision_message(status['message'])
    
    return response_data 