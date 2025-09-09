from flask import Blueprint, request, jsonify
import requests
from datetime import datetime
import hmac
import hashlib
import time
import json
import re
from models import db, Unloading, UnloadingItem, Loading, LoadingItem, Warehouse, VisionServer
from config import SECRET_KEY, EDIT_WINDOW_MINUTES
from translations import translate_vision_response

vision_bp = Blueprint('vision', __name__)

@vision_bp.route('/api/vision/start', methods=['POST'])
def api_vision_start():
    data = request.json or {}
    print(data)
    warehouse_id = data.get('warehouse_id')
    shipment_id = data.get('shipment_id')
    operation_type = data.get('operation_type', 'unloading')
    
    if not warehouse_id:
        return jsonify({'success': False, 'message': 'شناسه انبار مشخص نشده است.'}), 400
    
    warehouse = Warehouse.query.filter_by(id=warehouse_id).first()
    if not warehouse:
        return jsonify({'success': False, 'message': 'انبار یافت نشد.'}), 400
    
    # پیدا کردن سرور بینایی مناسب برای نوع عملیات
    vision_server = VisionServer.query.filter_by(
        id=data.get('camera_id'),
        is_active=True
    ).first()
    print(vision_server.id,"_____________")
    if not vision_server:
        return jsonify({'success': False, 'message': f'سرور بینایی برای عملیات {operation_type} یافت نشد.'}), 400
    
    try:
        print(f'{vision_server.url}/start')

        # ارسال video_source به سرور بینایی - همیشه یک JSON ارسال کن
        # استفاده از video_source از VisionServer یا پیش‌فرض
        video_source = vision_server.video_source
        print(video_source)
        # تشخیص نوع سرور بینایی بر اساس پورت URL
        if ':5001' in vision_server.url:
            # Smart Warehouse Vision - expects 'source' parameter
            vision_request = {
                'source': video_source
            }
        else:
            # QR Live Vision - expects 'video_source' parameter  
            vision_request = {
                'video_source': video_source
            }
        resp = requests.post(f'{vision_server.url}/start', json=vision_request)
        print(vision_request)

        result = resp.json()

        print(result)
        
        # ترجمه پیام‌های سرور بینایی
        result = translate_vision_response(result)
        
        # اگر سرور بینایی موفق بود، رکورد عملیات ایجاد کن
        if result.get('success'):
            if operation_type == 'loading':
                operation = Loading()
                operation.warehouse_id = warehouse_id
                operation.shipment_id = shipment_id
                operation.start_time = datetime.utcnow()
                operation.status = 'started'
                db.session.add(operation)
                db.session.commit()
                
                result['loading_id'] = operation.id
                result['message'] = result.get('message', '') + ' و رکورد بارگیری ایجاد شد.'
            else:
                operation = Unloading()
                operation.warehouse_id = warehouse_id
                operation.shipment_id = shipment_id
                operation.start_time = datetime.utcnow()
                operation.status = 'started'
                db.session.add(operation)
                db.session.commit()
                
                result['unloading_id'] = operation.id
                result['message'] = result.get('message', '') + ' و رکورد تخلیه ایجاد شد.'
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': 'خطا در ارتباط با سرویس بینایی.'}), 500

@vision_bp.route('/api/vision/status', methods=['GET'])
def api_vision_status():
    """Check if vision server is running for a specific operation type"""
    operation_type = request.args.get('operation_type', 'unloading')
    
    # پیدا کردن سرور بینایی مناسب برای نوع عملیات
    vision_server = VisionServer.query.filter_by(
        type=operation_type,
        is_active=True
    ).first()
    
    if not vision_server:
        return jsonify({'success': False, 'message': f'سرور بینایی برای عملیات {operation_type} یافت نشد.'}), 400
    
    try:
        resp = requests.get(f'{vision_server.url}/status', timeout=5)
        if resp.status_code == 200:
            result = resp.json()
            # ترجمه پیام‌های سرور بینایی
            result = translate_vision_response(result)
            return jsonify(result)
        else:
            return jsonify({'success': False, 'message': 'خطا در دریافت وضعیت سرور بینایی.'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': 'خطا در ارتباط با سرویس بینایی.'}), 500

def process_loading_vision_data(operation, result):
    """پردازش داده‌های بینایی برای عملیات بارگیری"""
    try:
        print(f"🚀 Starting process_loading_vision_data for operation {operation.id}")
        print(f"📊 Result keys: {list(result.keys())}")
        
        # استخراج آیتم‌ها از summary (API جدید)
        summary = result.get('summary')
        print(f"📋 Summary: {summary}")
        
        items = []
        if summary:
            print(f"✅ Loading Summary found")
            # از detailed_product_counts آیتم‌ها را مستقیماً بگیر
            detailed_counts = summary.get('detailed_product_counts', {})
            print(f"📊 Loading Detailed counts: {detailed_counts}")
            
            # هر محصول را جداگانه ذخیره کن
            for status_key, products in detailed_counts.items():
                print(f"🔍 Processing {status_key}: {products}")
                for product_name, count in products.items():
                    if int(count) > 0:
                        item = {
                            'name': product_name,
                            'type': status_key,  # 'loaded' یا 'unloaded'
                            'count': int(count)
                        }
                        items.append(item)
                        print(f"✅ Added loading item: {item}")
        else:
            print(f"⚠️ No summary found, trying items")
            # پشتیبانی از حالت قدیمی (items)
            items = result.get('items', [])
            print(f"📦 Items from result: {items}")
        
        print(f"📊 Final loading items: {items}")
        print(f"📊 Total items count: {len(items)}")
        
        # حذف آیتم‌های قبلی vision برای این loading
        # print(f"🗑️ Deleting previous vision items for loading {operation.id}")
        # deleted_count = LoadingItem.query.filter_by(
        #     loading_id=operation.id,
        #     source='vision'
        # ).delete()
        # print(f"🗑️ Deleted {deleted_count} previous vision items")
        
        # ایجاد آیتم‌های بارگیری از summary
        print(f"🆕 Creating loading items from summary")
        for i, item in enumerate(items):
            if int(item.get('count', 0)) > 0:
                print(f"🆕 Creating item {i+1}: {item}")
                loading_item = LoadingItem()
                loading_item.loading_id = operation.id
                loading_item.name = item.get('name', '')
                loading_item.type = item.get('type', '')
                loading_item.count = int(item.get('count', 0))
                loading_item.source = 'vision'
                loading_item.version = 1
                db.session.add(loading_item)
                print(f"✅ Added LoadingItem to session: {loading_item.name}")
        
        # پردازش QR codes برای اطلاعات اضافی
        print(f"🔍 Calling process_loading_qr_codes")
        process_loading_qr_codes(operation, result)
        
        print(f"✅ All loading items added to session")
        
        # ایجاد HMAC token برای loading
        print(f"🔑 Creating HMAC token")
        loading_id = operation.id
        timestamp = int(time.time())
        data = f"{loading_id}.{operation.warehouse_id}.{timestamp}"
        token = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        loading_token = f"{data}.{token}"
        
        print(f"✅ HMAC token created: {loading_token[:20]}...")
        
        return {
            'loading_id': operation.id,
            'loading_token': loading_token,
            'items_count': len(items)
        }
        
    except Exception as e:
        print(f"❌ Error processing loading vision data: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_loading_qr_codes(operation, result):
    """پردازش QR codes برای عملیات بارگیری"""
    try:
        print(f"🔍 Starting process_loading_qr_codes for operation {operation.id}")
        
        # بررسی qrcodes در vision output - مسیر جدید
        qrcodes = []
        if 'summary' in result and 'log_content' in result['summary']:
            qrcodes = result['summary']['log_content'].get('qrcodes', [])
            print(f"📱 Found {len(qrcodes)} QR codes in summary.log_content.qrcodes")
        else:
            # پشتیبانی از مسیر قدیمی
            qrcodes = result.get('qrcodes', [])
            print(f"📱 Found {len(qrcodes)} QR codes in result.qrcodes (legacy path)")
        
        print(f"📱 QR codes: {qrcodes}")
        
        if not qrcodes:
            print("❌ No QR codes found in vision output")
            return
        
        print(f"✅ Processing {len(qrcodes)} QR codes for loading")
        
        # ایجاد آیتم‌های جدید از QR codes
        parsed_items = []
        seen_reel_numbers = set()  # برای جلوگیری از تکرار
        
        for i, qr_code in enumerate(qrcodes):
            print(f"🔍 Processing QR code {i+1}: {qr_code}")
            content = qr_code.get('content', '')
            timestamp = qr_code.get('timestamp', '')
            
            print(f"📝 Content: {content}")
            print(f"⏰ Timestamp: {timestamp}")
            
            # فقط QR codes که شامل Reel Number هستند را پردازش کن
            if 'Reel Number:' in content:
                print(f"✅ Found Reel Number in content")
                
                # استخراج Reel Number
                reel_match = re.search(r'Reel Number:\s*(\d+)', content)
                if not reel_match:
                    print(f"❌ Invalid Reel Number format in: {content}")
                    continue
                    
                reel_number = reel_match.group(1)
                print(f"🔢 Extracted Reel Number: {reel_number}")
                
                # جلوگیری از تکرار Reel Number
                if reel_number in seen_reel_numbers:
                    print(f"⚠️ Skipping duplicate Reel Number: {reel_number}")
                    continue
                    
                seen_reel_numbers.add(reel_number)
                print(f"✅ Added {reel_number} to seen_reel_numbers")
                
                # استخراج سایر فیلدها
                width_match = re.search(r'Width:\s*(\d+)', content)
                width = int(width_match.group(1)) if width_match else None
                print(f"📏 Width: {width}")
                
                gsm_match = re.search(r'GSM:\s*(\d+)', content)
                gsm = int(gsm_match.group(1)) if gsm_match else None
                print(f"⚖️ GSM: {gsm}")
                
                length_match = re.search(r'Length:\s*~?\s*(\d+)', content)
                length = int(length_match.group(1)) if length_match else None
                print(f"📐 Length: {length}")
                
                breaks_match = re.search(r'Breaks:\s*(\d+)', content)
                breaks = int(breaks_match.group(1)) if breaks_match else None
                print(f"💔 Breaks: {breaks}")
                
                grade_match = re.search(r'Grade:\s*([^,\n\r]+)', content)
                grade = grade_match.group(1).strip() if grade_match else None
                print(f"⭐ Grade: {grade}")
                
                # ایجاد آیتم بارگیری
                print(f"🆕 Creating LoadingItem for Reel {reel_number}")
                qr_loading_item = LoadingItem()
                qr_loading_item.loading_id = operation.id
                qr_loading_item.name = f"Reel {reel_number}"  # استفاده از شماره رول به عنوان نام
                qr_loading_item.type = 'loaded'
                qr_loading_item.count = 1
                qr_loading_item.source = 'vision'
                qr_loading_item.version = 1
                
                # تنظیم فیلدهای استخراج شده
                qr_loading_item.reel_number = reel_number
                qr_loading_item.width = width
                qr_loading_item.gsm = gsm
                qr_loading_item.length = length
                qr_loading_item.breaks = breaks
                qr_loading_item.grade = grade
                
                print(f"💾 Adding LoadingItem to database session")
                db.session.add(qr_loading_item)
                
                parsed_items.append({
                    'reel_number': reel_number,
                    'width': width,
                    'gsm': gsm,
                    'length': length,
                    'breaks': breaks,
                    'grade': grade,
                    'timestamp': timestamp
                })
                
                print(f"✅ Created loading item for Reel {reel_number}")
            else:
                print(f"❌ No Reel Number found in content: {content}")
        
        print(f"📊 Total parsed items: {len(parsed_items)}")
        
        # اضافه کردن اطلاعات پردازش شده به vision_output
        if parsed_items:
            try:
                print(f"📝 Updating vision_output with parsed items")
                vision_data = json.loads(operation.vision_output) if operation.vision_output else {}
                vision_data['parsed_items'] = parsed_items
                vision_data['parsed_items_count'] = len(parsed_items)
                vision_data['processing_timestamp'] = datetime.utcnow().isoformat()
                operation.vision_output = json.dumps(vision_data, ensure_ascii=False, indent=2)
                print(f"✅ vision_output updated successfully")
            except Exception as e:
                print(f"❌ Error updating vision_output: {e}")
        
        print(f"🎉 Successfully parsed {len(parsed_items)} QR codes for loading {operation.id}")
        
    except Exception as e:
        print(f"❌ Error processing loading QR codes: {e}")
        import traceback
        traceback.print_exc()

def process_unloading_vision_data(operation, result):
    """پردازش داده‌های بینایی برای عملیات تخلیه"""
    try:
        # استخراج آیتم‌ها از summary (API جدید)
        summary = result.get('summary')
        items = []
        if summary:
            print(f"Unloading Summary: {summary}")
            # از detailed_product_counts آیتم‌ها را مستقیماً بگیر
            detailed_counts = summary.get('detailed_product_counts', {})
            print(f"Unloading Detailed counts: {detailed_counts}")
            
            # هر محصول را جداگانه ذخیره کن
            for status_key, products in detailed_counts.items():
                print(f"Processing {status_key}: {products}")
                for product_name, count in products.items():
                    if int(count) > 0:
                        item = {
                            'name': product_name,
                            'type': status_key,  # 'loaded' یا 'unloaded'
                            'count': int(count)
                        }
                        items.append(item)
                        print(f"Added unloading item: {item}")
        else:
            # پشتیبانی از حالت قدیمی (items)
            items = result.get('items', [])
        
        print(f"Final unloading items: {items}")
        
        # حذف آیتم‌های قبلی vision برای این unloading
        # UnloadingItem.query.filter_by(
        #     unloading_id=operation.id,
        #     source='vision'
        # ).delete()
        
        # ایجاد آیتم‌های تخلیه
        for item in items:
            if int(item.get('count', 0)) > 0:
                unloading_item = UnloadingItem()
                unloading_item.unloading_id = operation.id
                unloading_item.name = item.get('name', '')
                unloading_item.type = item.get('type', '')
                unloading_item.count = int(item.get('count', 0))
                unloading_item.source = 'vision'
                unloading_item.version = 1
                db.session.add(unloading_item)
        
        # ایجاد HMAC token برای unloading
        unloading_id = operation.id
        timestamp = int(time.time())
        data = f"{unloading_id}.{operation.warehouse_id}.{timestamp}"
        token = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        unloading_token = f"{data}.{token}"
        
        return {
            'unloading_id': operation.id,
            'unloading_token': unloading_token,
            'items_count': len(items)
        }
        
    except Exception as e:
        print(f"Error processing unloading vision data: {e}")
        import traceback
        traceback.print_exc()
        return None

@vision_bp.route('/api/vision/stop', methods=['POST'])
def api_vision_stop():
    data = request.json or {}
    warehouse_id = data.get('warehouse_id')
    operation_type = data.get('operation_type', 'unloading')
    camera_id = data.get('camera_id')
    
    if not warehouse_id:
        return jsonify({'success': False, 'message': 'شناسه انبار مشخص نشده است.'}), 400
    
    warehouse = Warehouse.query.filter_by(id=warehouse_id).first()
    if not warehouse:
        return jsonify({'success': False, 'message': 'انبار یافت نشد.'}), 400
    
    # پیدا کردن سرور بینایی - اگر camera_id داریم از آن استفاده کن، وگرنه از نوع عملیات
    if camera_id:
        vision_server = VisionServer.query.filter_by(
            id=camera_id,
            is_active=True
        ).first()
    else:
        # fallback to old method for backward compatibility
        vision_server = VisionServer.query.filter_by(
            type=operation_type,
            is_active=True
        ).first()
    
    if not vision_server:
        return jsonify({'success': False, 'message': f'سرور بینایی برای عملیات {operation_type} یافت نشد.'}), 400
    
    # پیدا کردن آخرین رکورد عملیات که در مرحله started هست
    if operation_type == 'loading':
        operation = Loading.query.filter_by(warehouse_id=warehouse_id, status='started').order_by(Loading.id.desc()).first()
        if not operation:
            return jsonify({'success': False, 'message': 'بارگیری فعالی برای این انبار یافت نشد.'}), 400
    else:
        operation = Unloading.query.filter_by(warehouse_id=warehouse_id, status='started').order_by(Unloading.id.desc()).first()
        if not operation:
            return jsonify({'success': False, 'message': 'تخلیه فعالی برای این انبار یافت نشد.'}), 400
    
    try:
        resp = requests.post(f'{vision_server.url}/stop', json={})
        result = resp.json()

        print('result', result)
        
        # ترجمه پیام‌های سرور بینایی
        result = translate_vision_response(result)
        
        # اگر سرور بینایی موفق بود، رکورد عملیات را آپدیت کن
        if result.get('success'):
            # آپدیت رکورد عملیات
            operation.status = 'vision'
            operation.end_time = datetime.utcnow()
            operation.user_confirm_time = datetime.utcnow()
            
            # ذخیره خروجی سیستم بینایی
            operation.vision_output = json.dumps(result, ensure_ascii=False, indent=2)
            
            # پردازش داده‌های بینایی برای عملیات
            if operation_type == 'loading':
                processed_data = process_loading_vision_data(operation, result)
                if processed_data:
                    result['loading_id'] = processed_data['loading_id']
                    result['loading_token'] = processed_data['loading_token']
                    result['message'] = result.get('message', '') + f' و {processed_data["items_count"]} آیتم بارگیری ذخیره شد.'
                else:
                    result['message'] = result.get('message', '') + ' ولی آیتم‌های بارگیری ذخیره نشدند.'
            else:
                processed_data = process_unloading_vision_data(operation, result)
                if processed_data:
                    result['unloading_id'] = processed_data['unloading_id']
                    result['unloading_token'] = processed_data['unloading_token']
                    result['message'] = result.get('message', '') + f' و {processed_data["items_count"]} آیتم تخلیه ذخیره شد.'
                else:
                    result['message'] = result.get('message', '') + ' ولی آیتم‌های تخلیه ذخیره نشدند.'
            
            # ذخیره همه تغییرات در دیتابیس
            db.session.commit()
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در ارتباط با سرویس بینایی: {str(e)}'}), 500

@vision_bp.route('/api/vision/test', methods=['GET'])
def api_vision_test():
    """Test endpoint to check vision server integration"""
    try:
        # Get first active vision server
        vision_server = VisionServer.query.filter_by(is_active=True).first()
        if not vision_server:
            return jsonify({'success': False, 'message': 'هیچ سرور بینایی فعالی تعریف نشده است.'}), 400
        
        # Test health check
        resp = requests.get(f'{vision_server.url}/health', timeout=5)
        if resp.status_code == 200:
            health_data = resp.json()
            
            # Test test_summary endpoint to verify structure
            summary_resp = requests.get(f'{vision_server.url}/test_summary', timeout=5)
            summary_test = "نامشخص"
            if summary_resp.status_code == 200:
                summary_data = summary_resp.json()
                if 'summary' in summary_data and 'detailed_product_counts' in summary_data['summary']:
                    summary_test = "✅ ساختار صحیح"
                else:
                    summary_test = "❌ ساختار نادرست"
            else:
                summary_test = "❌ در دسترس نیست"
            
            return jsonify({
                'success': True,
                'message': 'اتصال به سرور بینایی موفق است',
                'vision_server': {
                    'id': vision_server.id,
                    'name': vision_server.name,
                    'persian_name': vision_server.persian_name,
                    'type': vision_server.type,
                    'url': vision_server.url
                },
                'health_status': health_data,
                'summary_structure_test': summary_test
            })
        else:
            return jsonify({
                'success': False,
                'message': f'خطا در اتصال به سرور بینایی: {resp.status_code}',
                'vision_server': {
                    'id': vision_server.id,
                    'name': vision_server.name,
                    'persian_name': vision_server.persian_name,
                    'type': vision_server.type,
                    'url': vision_server.url
                }
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در تست سرور بینایی: {str(e)}'}), 500

@vision_bp.route('/api/vision/summary', methods=['GET'])
def api_vision_summary():
    """Get test summary from vision server to verify structure"""
    try:
        data = request.args or {}
        operation_type = data.get('operation_type', 'unloading')
        
        # Get vision server for the specified operation type
        vision_server = VisionServer.query.filter_by(
            type=operation_type,
            is_active=True
        ).first()
            
        if not vision_server:
            return jsonify({'success': False, 'message': f'هیچ سرور بینایی برای عملیات {operation_type} تعریف نشده است.'}), 400
        
        # Get test summary
        resp = requests.get(f'{vision_server.url}/test_summary', timeout=5)
        if resp.status_code == 200:
            summary_data = resp.json()
            # ترجمه پیام‌های سرور بینایی
            summary_data = translate_vision_response(summary_data)
            return jsonify({
                'success': True,
                'message': 'نمونه summary از سرور بینایی',
                'vision_server': {
                    'id': vision_server.id,
                    'name': vision_server.name,
                    'persian_name': vision_server.persian_name,
                    'type': vision_server.type,
                    'url': vision_server.url
                },
                'summary': summary_data.get('summary', {}),
                'full_response': summary_data
            })
        else:
            return jsonify({
                'success': False,
                'message': f'خطا در دریافت summary: {resp.status_code}',
                'vision_server': {
                    'id': vision_server.id,
                    'name': vision_server.name,
                    'persian_name': vision_server.persian_name,
                    'type': vision_server.type,
                    'url': vision_server.url
                }
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت summary: {str(e)}'}), 500

@vision_bp.route('/api/vision/test-stop', methods=['POST'])
def api_vision_test_stop():
    """Test endpoint to verify vision stop functionality without actually calling vision server"""
    try:
        data = request.json or {}
        warehouse_id = data.get('warehouse_id')
        operation_type = data.get('operation_type', 'unloading')
        
        if not warehouse_id:
            return jsonify({'success': False, 'message': 'شناسه انبار مشخص نشده است.'}), 400
        
        warehouse = Warehouse.query.filter_by(id=warehouse_id).first()
        if not warehouse:
            return jsonify({'success': False, 'message': 'انبار یافت نشد.'}), 400
        
        # پیدا کردن سرور بینایی مناسب برای نوع عملیات
        vision_server = VisionServer.query.filter_by(
            type=operation_type,
            is_active=True
        ).first()
        
        if not vision_server:
            return jsonify({'success': False, 'message': f'سرور بینایی برای عملیات {operation_type} یافت نشد.'}), 400
        
        # پیدا کردن آخرین رکورد عملیات که در مرحله started هست
        if operation_type == 'loading':
            operation = Loading.query.filter_by(warehouse_id=warehouse_id, status='started').order_by(Loading.id.desc()).first()
            if not operation:
                return jsonify({'success': False, 'message': 'بارگیری فعالی برای این انبار یافت نشد.'}), 400
        else:
            operation = Unloading.query.filter_by(warehouse_id=warehouse_id, status='started').order_by(Unloading.id.desc()).first()
            if not operation:
                return jsonify({'success': False, 'message': 'تخلیه فعالی برای این انبار یافت نشد.'}), 400
        
        # شبیه‌سازی پاسخ سرور بینایی
        mock_result = {
            'success': True,
            'message': f'عملیات {operation_type} با موفقیت متوقف شد',
            'summary': {
                'detailed_product_counts': {
                    'loaded' if operation_type == 'loading' else 'unloaded': {
                        'Product A': 5,
                        'Product B': 3
                    }
                }
            }
        }
        
        # اضافه کردن QR codes برای loading
        if operation_type == 'loading':
            mock_result['qrcodes'] = [
                {
                    'content': 'Reel Number: 2156, Width: 280, GSM: 150, Length:~ 7200, Breaks: 1, Grade: Kraftliner PREMIUM',
                    'timestamp': '2025-08-16T13:16:57.516264+0330'
                },
                {
                    'content': 'Reel Number: 1193, Width: 240, GSM: 130, Length:~ 6300, Breaks: 0, Grade: Testliner HOMAYOUN',
                    'timestamp': '2025-08-16T13:16:58.425159+0330'
                }
            ]
        
        # ترجمه پیام‌های سرور بینایی
        result = translate_vision_response(mock_result)
        
        # آپدیت رکورد عملیات
        operation.status = 'vision'
        operation.end_time = datetime.utcnow()
        operation.user_confirm_time = datetime.utcnow()
        
        # ذخیره خروجی سیستم بینایی
        operation.vision_output = json.dumps(result, ensure_ascii=False, indent=2)
        
        # پردازش داده‌های بینایی برای عملیات
        if operation_type == 'loading':
            processed_data = process_loading_vision_data(operation, result)
            if processed_data:
                result['loading_id'] = processed_data['loading_id']
                result['loading_token'] = processed_data['loading_token']
                result['message'] = result.get('message', '') + f' و {processed_data["items_count"]} آیتم بارگیری ذخیره شد.'
            else:
                result['message'] = result.get('message', '') + ' ولی آیتم‌های بارگیری ذخیره نشدند.'
        else:
            processed_data = process_unloading_vision_data(operation, result)
            if processed_data:
                result['unloading_id'] = processed_data['unloading_id']
                result['unloading_token'] = processed_data['unloading_token']
                result['message'] = result.get('message', '') + f' و {processed_data["items_count"]} آیتم تخلیه ذخیره شد.'
            else:
                result['message'] = result.get('message', '') + ' ولی آیتم‌های تخلیه ذخیره نشدند.'
        
        # ذخیره همه تغییرات در دیتابیس
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'تست {operation_type} با موفقیت انجام شد',
            'test_data': {
                'warehouse_id': warehouse_id,
                'operation_type': operation_type,
                'vision_server': vision_server.name,
                'operation_id': operation.id,
                'operation_status': operation.status
            },
            **result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در تست: {str(e)}'}), 500
