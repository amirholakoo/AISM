from flask import Flask, request, jsonify, send_from_directory  # type: ignore
import requests  # type: ignore
from flask_cors import CORS  # type: ignore
from flask_sqlalchemy import SQLAlchemy  # type: ignore
from datetime import datetime, timedelta
import hmac
import hashlib
import time
import os
import re
from models import db, Unloading, UnloadingItem, Loading, LoadingItem, Product, Warehouse, OperationType, VisionServer
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS, SECRET_KEY, EDIT_WINDOW_MINUTES
from translations import translate_vision_response

app = Flask(__name__, static_folder='static')
CORS(app, origins=[
    'http://localhost:5173', 
    'http://127.0.0.1:5173', 
    'http://localhost:3000', 
    'http://127.0.0.1:3000',
    'http://localhost:18888',
    'http://127.0.0.1:18888',
    'http://0.0.0.0:18888'
])
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
db.init_app(app)





@app.route('/api/vision/start', methods=['POST'])
def api_vision_start():
    data = request.json or {}
    warehouse_id = data.get('warehouse_id')
    shipment_id = data.get('shipment_id')  # دریافت shipment_id
    operation_type = data.get('operation_type', 'unloading')  # نوع عملیات
    
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
    
    try:
        print(f'{vision_server.url}/start')
        resp = requests.post(f'{vision_server.url}/start', json={})
        result = resp.json()

        print(result)
        
        # ترجمه پیام‌های سرور بینایی
        result = translate_vision_response(result)
        
        # اگر سرور بینایی موفق بود، رکورد عملیات ایجاد کن
        if result.get('success'):
            if operation_type == 'loading':
                operation = Loading()
                operation.warehouse_id = warehouse_id
                operation.shipment_id = shipment_id  # ذخیره shipment_id
                operation.start_time = datetime.utcnow()
                operation.status = 'started'
                db.session.add(operation)
                db.session.commit()
                
                result['loading_id'] = operation.id
                result['message'] = result.get('message', '') + ' و رکورد بارگیری ایجاد شد.'
            else:
                operation = Unloading()
                operation.warehouse_id = warehouse_id
                operation.shipment_id = shipment_id  # ذخیره shipment_id
                operation.start_time = datetime.utcnow()
                operation.status = 'started'
                db.session.add(operation)
                db.session.commit()
                
                result['unloading_id'] = operation.id
                result['message'] = result.get('message', '') + ' و رکورد تخلیه ایجاد شد.'
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': 'خطا در ارتباط با سرویس بینایی.'}), 500

@app.route('/api/vision/status', methods=['GET'])
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

@app.route('/api/vision/stop', methods=['POST'])
def api_vision_stop():
    data = request.json or {}
    warehouse_id = data.get('warehouse_id')
    operation_type = data.get('operation_type', 'unloading')  # نوع عملیات
    
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
    
    # پیدا کردن رکورد عملیات که در مرحله started هست
    if operation_type == 'loading':
        operation = Loading.query.filter_by(warehouse_id=warehouse_id, status='started').first()
        if not operation:
            return jsonify({'success': False, 'message': 'بارگیری فعالی برای این انبار یافت نشد.'}), 400
    else:
        operation = Unloading.query.filter_by(warehouse_id=warehouse_id, status='started').first()
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
            operation.user_confirm_time = datetime.utcnow()  # تنظیم زمان تایید برای محاسبه محدودیت زمانی
            
            # ذخیره خروجی سیستم بینایی
            import json
            operation.vision_output = json.dumps(result, ensure_ascii=False, indent=2)
            
            db.session.commit()
            
            # استخراج آیتم‌ها از summary (API جدید)
            summary = result.get('summary')
            items = []
            if summary:
                print(f"Summary: {summary}")
                # از detailed_product_counts آیتم‌ها را مستقیماً بگیر
                detailed_counts = summary.get('detailed_product_counts', {})
                print(f"Detailed counts: {detailed_counts}")
                
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
                            print(f"Added item: {item}")
                
                # اضافه کردن اطلاعات اضافی از summary
                print(f"Operation type: {summary.get('operation_type')}")
                print(f"Total products: {summary.get('total_products')}")
                print(f"Location: {summary.get('location')}")
            else:
                # پشتیبانی از حالت قدیمی (items)
                items = result.get('items', [])
            
            print(f"Final items: {items}")
            
            # ذخیره آیتم‌ها بر اساس نوع عملیات
            if operation_type == 'loading':
                for item in items:
                    if int(item.get('count', 0)) > 0:
                        loading_item = LoadingItem()
                        loading_item.loading_id = operation.id
                        loading_item.name = item.get('name', '')
                        loading_item.type = item.get('type', '')
                        loading_item.count = int(item.get('count', 0))
                        loading_item.source = 'vision'  # مشخص کردن منبع به عنوان vision
                        
                        # استخراج فیلدهای اضافی از vision output برای loading
                        # بررسی qrcodes در vision output
                        if operation.vision_output:
                            try:
                                vision_data = json.loads(operation.vision_output)
                                qrcodes = vision_data.get('qrcodes', [])
                                if qrcodes:
                                    # استفاده از اولین QR code برای استخراج اطلاعات
                                    first_qr = qrcodes[0]
                                    content = first_qr.get('content', '')
                                    
                                    # استخراج فیلدها از content
                                    if 'Reel Number:' in content:
                                        reel_match = re.search(r'Reel Number:\s*(\d+)', content)
                                        if reel_match:
                                            loading_item.reel_number = reel_match.group(1)
                                    
                                    if 'Width:' in content:
                                        width_match = re.search(r'Width:\s*(\d+)', content)
                                        if width_match:
                                            loading_item.width = int(width_match.group(1))
                                    
                                    if 'GSM:' in content:
                                        gsm_match = re.search(r'GSM:\s*(\d+)', content)
                                        if gsm_match:
                                            loading_item.gsm = int(gsm_match.group(1))
                                    
                                    if 'Length:' in content:
                                        length_match = re.search(r'Length:\s*~?\s*(\d+)', content)
                                        if length_match:
                                            loading_item.length = int(length_match.group(1))
                                    
                                    if 'Breaks:' in content:
                                        breaks_match = re.search(r'Breaks:\s*(\d+)', content)
                                        if breaks_match:
                                            loading_item.breaks = int(breaks_match.group(1))
                                    
                                    if 'Grade:' in content:
                                        grade_match = re.search(r'Grade:\s*([^,\n]+)', content)
                                        if grade_match:
                                            loading_item.grade = grade_match.group(1).strip()
                            except Exception as e:
                                print(f"Error parsing vision output for loading item: {e}")
                        
                        db.session.add(loading_item)
                db.session.commit()
                
                # ایجاد HMAC token برای loading
                loading_id = operation.id
                timestamp = int(time.time())
                data = f"{loading_id}.{warehouse_id}.{timestamp}"
                token = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
                loading_token = f"{data}.{token}"
            else:
                for item in items:
                    if int(item.get('count', 0)) > 0:
                        unloading_item = UnloadingItem()
                        unloading_item.unloading_id = operation.id
                        unloading_item.name = item.get('name', '')
                        unloading_item.type = item.get('type', '')
                        unloading_item.count = int(item.get('count', 0))
                        unloading_item.source = 'vision'  # مشخص کردن منبع به عنوان vision
                        db.session.add(unloading_item)
                db.session.commit()
                
                # ایجاد HMAC token برای unloading
                unloading_id = operation.id
                timestamp = int(time.time())
                data = f"{unloading_id}.{warehouse_id}.{timestamp}"
                token = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
                unloading_token = f"{data}.{token}"
            
            # اضافه کردن اطلاعات ذخیره شده به پاسخ
            if operation_type == 'loading':
                result['loading_id'] = operation.id
                result['loading_token'] = loading_token
                result['message'] = result.get('message', '') + ' و در دیتابیس ذخیره شد.'
            else:
                result['unloading_id'] = operation.id
                result['unloading_token'] = unloading_token
                result['message'] = result.get('message', '') + ' و در دیتابیس ذخیره شد.'
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': 'خطا در ارتباط با سرویس بینایی.'}), 500

def verify_unloading_token(token):
    """اعتبارسنجی HMAC token و برگرداندن unloading_id و warehouse_id"""
    try:
        data, hash_value = token.rsplit('.', 1)
        expected_hash = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        if hash_value == expected_hash:
            unloading_id, warehouse_id, timestamp = data.split('.')
            # چک کردن timestamp (1 ساعت اعتبار)
            if int(time.time()) - int(timestamp) < 3600:
                return int(unloading_id), warehouse_id
    except:
        pass
    return None, None

def verify_loading_token(token):
    """اعتبارسنجی HMAC token و برگرداندن loading_id و warehouse_id"""
    try:
        data, hash_value = token.rsplit('.', 1)
        expected_hash = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        if hash_value == expected_hash:
            loading_id, warehouse_id, timestamp = data.split('.')
            # چک کردن timestamp (1 ساعت اعتبار)
            if int(time.time()) - int(timestamp) < 3600:
                return int(loading_id), warehouse_id
    except:
        pass
    return None, None

@app.route('/api/unloadings', methods=['POST'])
@app.route('/api/unloadings/save', methods=['POST'])
def api_unloadings_save():
    data = request.json or {}
    warehouse_id = data.get('warehouse_id')
    shipment_id = data.get('shipment_id')  # دریافت shipment_id
    items = data.get('items')
    unloading_token = data.get('unloading_token')
    
    if not warehouse_id or not items:
        return jsonify({'success': False, 'message': 'اطلاعات ناقص است.'}), 400
    
    # اگر unloading_token ارسال شده، اعتبارسنجی کن
    if unloading_token:
        unloading_id, token_warehouse_id = verify_unloading_token(unloading_token)
        if unloading_id and token_warehouse_id == warehouse_id:
            unloading = Unloading.query.get(unloading_id)
            if unloading and unloading.status in ['vision', 'edited']:
                # حذف آیتم‌های قبلی user
                UnloadingItem.query.filter_by(unloading_id=unloading.id, source='user').delete()
                # آپدیت status به completed و افزایش version
                unloading.status = 'completed'
                unloading.version += 1
                unloading.user_confirm_time = datetime.utcnow()
                db.session.commit()
            else:
                return jsonify({'success': False, 'message': 'تخلیه یافت نشد یا در وضعیت نامعتبر است.'}), 404
        else:
            return jsonify({'success': False, 'message': 'توکن نامعتبر یا منقضی شده است.'}), 400
    else:
        # اگر token نبود، unloading جدید بساز (برای تخلیه دستی)
        unloading = Unloading()
        unloading.warehouse_id = warehouse_id
        unloading.shipment_id = shipment_id  # ذخیره shipment_id
        unloading.start_time = datetime.utcnow()
        unloading.status = 'completed'
        unloading.user_confirm_time = datetime.utcnow() + timedelta(minutes=EDIT_WINDOW_MINUTES)
        db.session.add(unloading)
        db.session.commit()
    
    # Add items
    for item in items:
        if int(item.get('count', 0)) == 0:
            continue
        unloading_item = UnloadingItem()
        unloading_item.unloading_id = unloading.id
        unloading_item.name = item.get('name', '')
        unloading_item.type = item.get('type', '')
        unloading_item.count = int(item.get('count', 0))
        unloading_item.source = 'user'  # مشخص کردن منبع به عنوان کاربر
        unloading_item.version = unloading.version
        db.session.add(unloading_item)
    db.session.commit()
    return jsonify({'success': True, 'message': 'تخلیه با موفقیت ذخیره شد!'})

@app.route('/api/warehouses', methods=['GET'])
def api_warehouses():
    """Get all warehouses"""
    try:
        warehouses = [{
            'id': w.id, 
            'name': w.name,
            'persian_name': w.persian_name,
            'is_active': w.is_active
        } for w in Warehouse.query.all()]
        
        return jsonify({
            'success': True,
            'warehouses': warehouses,
            'count': len(warehouses)
        })
    except Exception as e:
        print(f"Error in api_warehouses: {e}")
        return jsonify({
            'success': False,
            'message': f'خطا در دریافت لیست انبارها: {str(e)}',
            'warehouses': []
        }), 500

@app.route('/api/warehouses', methods=['POST'])
def api_warehouses_add():
    """Add a new warehouse"""
    try:
        data = request.json or {}
        warehouse_id = data.get('id', '').strip()
        name = data.get('name', '').strip()
        persian_name = data.get('persian_name', '').strip()
        is_active = data.get('is_active', True)
        
        if not warehouse_id:
            return jsonify({'success': False, 'message': 'شناسه انبار الزامی است.'}), 400
        
        if not name:
            return jsonify({'success': False, 'message': 'نام انگلیسی انبار الزامی است.'}), 400
        
        # بررسی تکراری نبودن شناسه
        existing_warehouse = Warehouse.query.filter_by(id=warehouse_id).first()
        if existing_warehouse:
            return jsonify({'success': False, 'message': 'این شناسه انبار قبلاً ثبت شده است.'}), 400
        
        # بررسی تکراری نبودن نام انگلیسی
        existing_name = Warehouse.query.filter_by(name=name).first()
        if existing_name:
            return jsonify({'success': False, 'message': 'این نام انگلیسی قبلاً ثبت شده است.'}), 400
        
        # ایجاد انبار جدید
        new_warehouse = Warehouse(
            id=warehouse_id,
            name=name,
            persian_name=persian_name,
            is_active=is_active
        )
        
        db.session.add(new_warehouse)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'انبار با موفقیت اضافه شد.',
            'warehouse': {
                'id': new_warehouse.id,
                'name': new_warehouse.name,
                'persian_name': new_warehouse.persian_name,
                'is_active': new_warehouse.is_active
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'خطا در اضافه کردن انبار: {str(e)}'}), 500

@app.route('/api/warehouses/sync', methods=['POST'])
def api_warehouses_sync():
    """Sync warehouses from external database"""
    try:
        from models.database import get_warehouse_tables_from_database
        
        # Create tables if they don't exist
        db.create_all()
        
        # دریافت انبارها از external database
        external_warehouses = get_warehouse_tables_from_database()
        
        # انبارهای موجود در دیتابیس اصلی
        existing_warehouses = {w.id: w for w in Warehouse.query.all()}
        existing_names = {w.name: w for w in Warehouse.query.all()}
        
        added_count = 0
        updated_count = 0
        
        for external_name in external_warehouses:
            warehouse_name = external_name.replace('Anbar_', '')
            
            # چک کردن بر اساس id و name
            if external_name in existing_warehouses or warehouse_name in existing_names:
                # انبار موجود - آپدیت نکن
                continue
            else:
                # انبار جدید - اضافه کن
                warehouse = Warehouse(
                    id=external_name,
                    name=warehouse_name,
                    persian_name=f"انبار {warehouse_name}",
                    is_active=True,
                    vision_server_url="http://127.0.0.1:5001"
                )
                db.session.add(warehouse)
                added_count += 1
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'{added_count} انبار جدید اضافه شد.',
            'added_count': added_count,
            'updated_count': updated_count
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'خطا در همگام‌سازی انبارها: {str(e)}'}), 500

@app.route('/api/warehouses/<warehouse_id>', methods=['PUT'])
def api_warehouse_update(warehouse_id):
    """Update warehouse details"""
    try:
        warehouse = Warehouse.query.get_or_404(warehouse_id)
        data = request.json or {}
        
        # آپدیت فیلدها
        if 'persian_name' in data:
            warehouse.persian_name = data['persian_name']
        if 'is_active' in data:
            warehouse.is_active = data['is_active']
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'انبار با موفقیت بروزرسانی شد.',
            'warehouse': {
                'id': warehouse.id,
                'name': warehouse.name,
                'persian_name': warehouse.persian_name,
                'is_active': warehouse.is_active
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'خطا در بروزرسانی انبار: {str(e)}'}), 500

@app.route('/api/warehouses/<warehouse_id>', methods=['DELETE'])
def api_warehouse_delete(warehouse_id):
    """Delete warehouse"""
    try:
        warehouse = Warehouse.query.get_or_404(warehouse_id)
        
        # بررسی اینکه آیا انبار در حال استفاده هست
        active_unloadings = Unloading.query.filter_by(warehouse_id=warehouse_id, status='started').count()
        if active_unloadings > 0:
            return jsonify({'success': False, 'message': 'این انبار در حال استفاده است و قابل حذف نیست.'}), 500
        
        db.session.delete(warehouse)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'انبار با موفقیت حذف شد.'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'خطا در حذف انبار: {str(e)}'}), 500




@app.route('/api/products', methods=['GET'])
def api_products():
    """Get products from internal database"""
    try:
        products = Product.query.all()
        
        result = []
        for product in products:
            result.append({
                'id': product.id,
                'name': product.name,
                'persian_name': product.persian_name,
                'vision_name': product.vision_name,
                'width': product.width,
                'gsm': product.gsm,
                'length': product.length
            })
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/products', methods=['POST'])
def api_products_create():
    """Create a new product"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'persian_name']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create new product
        new_product = Product(
            name=data['name'],
            persian_name=data['persian_name'],
            vision_name=data.get('vision_name') or data['name'].lower(),
            width=int(data['width']) if data.get('width') else None,
            gsm=int(data['gsm']) if data.get('gsm') else None,
            length=int(data['length']) if data.get('length') else None
        )
        
        db.session.add(new_product)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': {
                'id': new_product.id,
                'name': new_product.name,
                'persian_name': new_product.persian_name
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/products/<int:product_id>', methods=['PUT'])
def api_products_update(product_id):
    """Update an existing product"""
    try:
        product = Product.query.get(product_id)
        if not product:
            return jsonify({
                'success': False,
                'error': 'Product not found'
            }), 404
        
        data = request.get_json()
        
        # Update fields
        if 'name' in data:
            product.name = data['name']
            # اگر vision_name خالی باشد، از name استفاده کن
            if not data.get('vision_name'):
                product.vision_name = data['name'].lower()
        if 'persian_name' in data:
            product.persian_name = data['persian_name']
        if 'vision_name' in data:
            product.vision_name = data['vision_name']
        if 'width' in data:
            product.width = int(data['width']) if data['width'] else None
        if 'gsm' in data:
            product.gsm = int(data['gsm']) if data['gsm'] else None
        if 'length' in data:
            product.length = int(data['length']) if data['length'] else None
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': {
                'id': product.id,
                'name': product.name,
                'persian_name': product.persian_name
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/products/<int:product_id>', methods=['DELETE'])
def api_products_delete(product_id):
    """Delete a product"""
    try:
        product = Product.query.get(product_id)
        if not product:
            return jsonify({
                'success': False,
                'error': 'Product not found'
            }), 404
        
        db.session.delete(product)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Product deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/vision/test', methods=['GET'])
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

@app.route('/api/vision/summary', methods=['GET'])
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

@app.route('/api/loadings/last-completed', methods=['GET'])
def api_loadings_last_completed():
    """Get the last completed or vision loading for editing"""
    try:
        # Get the last completed, vision, or edited loading
        loading = Loading.query.filter(
            Loading.status.in_(['completed', 'vision', 'edited'])
        ).order_by(Loading.user_confirm_time.desc()).first()
        
        if not loading:
            return jsonify({'success': False, 'message': 'هیچ بارگیری تکمیل شده‌ای یافت نشد.'}), 404
    
        # Check if still within edit window based on last edit time
        can_edit = True
        remaining_minutes = 0
        
        # Use edit_time if available, otherwise use user_confirm_time
        last_edit_time = loading.edit_time or loading.user_confirm_time
        edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
        can_edit = datetime.utcnow() < edit_deadline
        remaining_minutes = max(0, int((edit_deadline - datetime.utcnow()).total_seconds() / 60))
    
        # Get latest version items (user items if version > 1, vision items if version = 1)
        items = []
        latest_version = loading.version
        
        if latest_version > 1:
            # Get user items from latest version
            for item in loading.items:
                if item.source == 'user' and item.version == latest_version:
                    items.append({
                        'name': item.name,
                        'type': item.type,
                        'count': item.count,
                        'reel_number': item.reel_number,
                        'width': item.width,
                        'gsm': item.gsm,
                        'length': item.length,
                        'breaks': item.breaks,
                        'grade': item.grade
                    })
        else:
            # Get vision items
            for item in loading.items:
                if item.source == 'vision':
                    items.append({
                        'name': item.name,
                        'type': item.type,
                        'count': item.count,
                        'reel_number': item.reel_number,
                        'width': item.width,
                        'gsm': item.gsm,
                        'length': item.length,
                        'breaks': item.breaks,
                        'grade': item.grade
                    })
        
        # ایجاد HMAC token برای loading
        loading_id = loading.id
        timestamp = int(time.time())
        data = f"{loading_id}.{loading.warehouse_id}.{timestamp}"
        token = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        loading_token = f"{data}.{token}"
        
        # تلاش برای دریافت اطلاعات shipment
        shipment_info = None
        if loading.shipment_id:
            try:
                from models.database import get_db
                from models.external_db import Shipments
                
                db = next(get_db())
                shipment = db.query(Shipments).filter(Shipments.id == loading.shipment_id).first()
                
                if shipment:
                    shipment_info = {
                        'id': shipment.id,
                        'license_number': shipment.license_number,
                        'supplier_name': shipment.supplier_name,
                        'customer_name': shipment.customer_name,
                        'status': shipment.status,
                        'shipment_type': shipment.shipment_type,
                        'weight1': shipment.weight1,
                        'weight2': shipment.weight2,
                        'net_weight': shipment.net_weight,
                        'date': shipment.date.isoformat() if shipment.date else None,
                        'receive_date': shipment.receive_date.isoformat() if shipment.receive_date else None,
                        'entry_time': shipment.entry_time.isoformat() if shipment.entry_time else None,
                        'weight1_time': shipment.weight1_time.isoformat() if shipment.weight1_time else None,
                        'weight2_time': shipment.weight2_time.isoformat() if shipment.weight2_time else None,
                        'exit_time': shipment.exit_time.isoformat() if shipment.exit_time else None,
                        'material_type': shipment.material_type,
                        'material_name': shipment.material_name,
                        'quantity': shipment.quantity,
                        'unit': shipment.unit,
                        'quality': shipment.quality,
                        'penalty': shipment.penalty,
                        'unload_location': shipment.unload_location,
                        'list_of_reels': shipment.list_of_reels,
                        'profile_name': shipment.profile_name,
                        'width': shipment.width,
                        'sales_id': shipment.sales_id,
                        'price_per_kg': shipment.price_per_kg,
                        'total_price': shipment.total_price,
                    }
            except Exception as e:
                print(f"Error getting shipment info: {e}")
                shipment_info = None
        
        return jsonify({
            'success': True,
            'loading_token': loading_token,
            'warehouse_id': loading.warehouse_id,
            'shipment_id': loading.shipment_id,
            'status': loading.status,
            'start_time': loading.start_time.isoformat() if loading.start_time else None,
            'end_time': loading.end_time.isoformat() if loading.end_time else None,
            'user_confirm_time': loading.user_confirm_time.isoformat() if loading.user_confirm_time else None,
            'edit_time': loading.edit_time.isoformat() if loading.edit_time else None,
            'version': loading.version,
            'vision_output': loading.vision_output,
            'items': items,
            'can_edit': can_edit,
            'remaining_minutes': remaining_minutes,
            'shipment_info': shipment_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت آخرین بارگیری: {str(e)}'}), 500

@app.route('/api/loadings/active', methods=['GET'])
def api_loadings_active():
    """Get active loading for a specific warehouse"""
    try:
        warehouse_id = request.args.get('warehouse_id')
        if not warehouse_id:
            return jsonify({'success': False, 'message': 'شناسه انبار مشخص نشده است.'}), 400
        
        # Get active loading for the warehouse
        loading = Loading.query.filter_by(
            warehouse_id=warehouse_id,
            status='started'
        ).first()
        
        if not loading:
            return jsonify({'success': False, 'message': 'بارگیری فعالی برای این انبار یافت نشد.'}), 404
        
        return jsonify({
            'success': True,
            'loading': {
                'id': loading.id,
                'warehouse_id': loading.warehouse_id,
                'shipment_id': loading.shipment_id,
                'status': loading.status,
                'start_time': loading.start_time.isoformat() if loading.start_time else None
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت بارگیری فعال: {str(e)}'}), 500

@app.route('/api/loadings/active-any', methods=['GET'])
def api_loadings_active_any():
    """Get any active loading across all warehouses"""
    try:
        # Get any active loading
        loading = Loading.query.filter_by(status='started').first()
        
        if not loading:
            return jsonify({'success': False, 'message': 'هیچ بارگیری فعالی یافت نشد.'}), 404
        
        return jsonify({
            'success': True,
            'loading': {
                'id': loading.id,
                'warehouse_id': loading.warehouse_id,
                'shipment_id': loading.shipment_id,
                'status': loading.status,
                'start_time': loading.start_time.isoformat() if loading.start_time else None
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت بارگیری فعال: {str(e)}'}), 500

@app.route('/api/unloadings/last-completed', methods=['GET'])
def api_unloadings_last_completed():
    """Get the last completed or vision unloading for editing"""
    try:
        # Get the last completed, vision, or edited unloading
        unloading = Unloading.query.filter(
            Unloading.status.in_(['completed', 'vision', 'edited'])
        ).order_by(Unloading.user_confirm_time.desc()).first()
        
        if not unloading:
            return jsonify({'success': False, 'message': 'هیچ تخلیه تکمیل شده‌ای یافت نشد.'}), 404
    
        # Check if still within edit window based on last edit time
        can_edit = True
        remaining_minutes = 0
        
        # Use edit_time if available, otherwise use user_confirm_time
        last_edit_time = unloading.edit_time or unloading.user_confirm_time
        edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
        can_edit = datetime.utcnow() < edit_deadline
        remaining_minutes = max(0, int((edit_deadline - datetime.utcnow()).total_seconds() / 60))
    
        # Get latest version items (user items if version > 1, vision items if version = 1)
        items = []
        latest_version = unloading.version
        
        if latest_version > 1:
            # Get user items from latest version
            for item in unloading.items:
                if item.source == 'user' and item.version == latest_version:
                    items.append({
                        'name': item.name,
                        'type': item.type,
                        'count': item.count
                    })
        else:
            # Get vision items
            for item in unloading.items:
                if item.source == 'vision':
                    items.append({
                        'name': item.name,
                        'type': item.type,
                        'count': item.count
                    })
        
        # ایجاد HMAC token برای unloading
        unloading_id = unloading.id
        timestamp = int(time.time())
        data = f"{unloading_id}.{unloading.warehouse_id}.{timestamp}"
        token = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        unloading_token = f"{data}.{token}"
        
        # تلاش برای دریافت اطلاعات shipment
        shipment_info = None
        if unloading.shipment_id:
            # اگر shipment_id در جدول unloadings وجود دارد، از جدول shipments بخوان
            try:
                from models.database import get_db
                from models.external_db import Shipments
                
                db = next(get_db())
                shipment = db.query(Shipments).filter(Shipments.id == unloading.shipment_id).first()
                
                if shipment:
                    shipment_info = {
                        'id': shipment.id,
                        'license_number': shipment.license_number,
                        'supplier_name': shipment.supplier_name,
                        'customer_name': shipment.customer_name,
                        'status': shipment.status,
                        'shipment_type': shipment.shipment_type,
                        'weight1': shipment.weight1,
                        'weight2': shipment.weight2,
                        'net_weight': shipment.net_weight,
                        'date': shipment.date.isoformat() if shipment.date else None,
                        'receive_date': shipment.receive_date.isoformat() if shipment.receive_date else None,
                        'entry_time': shipment.entry_time.isoformat() if shipment.entry_time else None,
                        'weight1_time': shipment.weight1_time.isoformat() if shipment.weight1_time else None,
                        'weight2_time': shipment.weight2_time.isoformat() if shipment.weight2_time else None,
                        'exit_time': shipment.exit_time.isoformat() if shipment.exit_time else None,
                        'material_type': shipment.material_type,
                        'material_name': shipment.material_name,
                        'quantity': shipment.quantity,
                        'unit': shipment.unit,
                        'quality': shipment.quality,
                        'penalty': shipment.penalty,
                        'unload_location': shipment.unload_location,
                        'list_of_reels': shipment.list_of_reels,
                        'profile_name': shipment.profile_name,
                        'width': shipment.width,
                        'sales_id': shipment.sales_id,
                        'price_per_kg': shipment.price_per_kg,
                        'total_price': shipment.total_price,
                        'extra_cost': shipment.extra_cost,
                        'vat': shipment.vat,
                        'invoice_status': shipment.invoice_status,
                        'payment_status': shipment.payment_status,
                        'document_info': shipment.document_info,
                        'comments': shipment.comments,
                        'cancellation_reason': shipment.cancellation_reason,
                        'username': shipment.username,
                        'logs': shipment.logs,
                        'purchase_id_id': shipment.purchase_id_id
                    }
            except Exception as e:
                print(f"خطا در خواندن اطلاعات shipment از دیتابیس خارجی: {str(e)}")
            finally:
                if 'db' in locals():
                    db.close()
    
        return jsonify({
            'success': True,
            'id': unloading.id,
            'warehouse_id': unloading.warehouse_id,
            'warehouse_name': unloading.warehouse.persian_name or unloading.warehouse.name,
            'warehouse_english_name': unloading.warehouse.name,
            'shipment_id': unloading.shipment_id,
            'shipment_info': shipment_info,  # اطلاعات کامل shipment
            'status': unloading.status,
            'version': latest_version,
            'items': items,
            'can_edit': can_edit,
            'remaining_minutes': remaining_minutes,
            'start_time': unloading.start_time.isoformat() if unloading.start_time else None,
            'end_time': unloading.end_time.isoformat() if unloading.end_time else None,
            'user_confirm_time': unloading.user_confirm_time.isoformat() if unloading.user_confirm_time else None,
            'edit_time': unloading.edit_time.isoformat() if unloading.edit_time else None,
            'vision_output': unloading.vision_output,
            'token': unloading_token
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت آخرین تخلیه: {str(e)}'}), 500

@app.route('/api/unloadings/<token>', methods=['GET'])
def api_unloading_by_token(token):
    """Get unloading by token"""
    try:
        # Parse token to get unloading_id
        parts = token.split('.')
        if len(parts) != 4:
            return jsonify({'success': False, 'message': 'توکن نامعتبر است.'}), 400
        
        unloading_id, warehouse_id, timestamp, signature = parts
        unloading_id = int(unloading_id)
        
        # Verify token
        data = f"{unloading_id}.{warehouse_id}.{timestamp}"
        expected_signature = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        
        if signature != expected_signature:
            return jsonify({'success': False, 'message': 'توکن نامعتبر است.'}), 400
        
        # Get unloading
        unloading = Unloading.query.get(unloading_id)
        if not unloading:
            return jsonify({'success': False, 'message': 'تخلیه یافت نشد.'}), 404
        
        # Check if still editable based on last edit time
        can_edit = True
        remaining_minutes = 0
        
        if unloading.status in ['completed', 'vision', 'edited']:
            # Use edit_time if available, otherwise use user_confirm_time
            last_edit_time = unloading.edit_time or unloading.user_confirm_time
            edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
            now = datetime.utcnow()
            
            # اگر زمان ویرایش تمام شده، edit_time رو ریست کن
            if now > edit_deadline:
                unloading.edit_time = now
                last_edit_time = now
                db.session.commit()
            
            # محاسبه مجدد بعد از ریست
            edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
            can_edit = datetime.utcnow() < edit_deadline
            remaining_minutes = max(0, int((edit_deadline - datetime.utcnow()).total_seconds() / 60))
        
        # Get all items for all versions
        items = []
        for item in unloading.items:
            items.append({
                'name': item.name,
                'type': item.type,
                'count': item.count,
                'source': item.source,
                'version': item.version
            })
        
        # تلاش برای دریافت اطلاعات shipment
        shipment_info = None
        if unloading.shipment_id:
            # اگر shipment_id در جدول unloadings وجود دارد، از جدول shipments بخوان
            try:
                from models.database import get_db
                from models.external_db import Shipments
                
                db = next(get_db())
                shipment = db.query(Shipments).filter(Shipments.id == unloading.shipment_id).first()
                
                if shipment:
                    shipment_info = {
                        'id': shipment.id,
                        'license_number': shipment.license_number,
                        'supplier_name': shipment.supplier_name,
                        'customer_name': shipment.customer_name,
                        'status': shipment.status,
                        'shipment_type': shipment.shipment_type,
                        'weight1': shipment.weight1,
                        'weight2': shipment.weight2,
                        'net_weight': shipment.net_weight,
                        'date': shipment.date.isoformat() if shipment.date else None,
                        'receive_date': shipment.receive_date.isoformat() if shipment.receive_date else None,
                        'entry_time': shipment.entry_time.isoformat() if shipment.entry_time else None,
                        'weight1_time': shipment.weight1_time.isoformat() if shipment.weight1_time else None,
                        'weight2_time': shipment.weight2_time.isoformat() if shipment.weight2_time else None,
                        'exit_time': shipment.exit_time.isoformat() if shipment.exit_time else None,
                        'material_type': shipment.material_type,
                        'material_name': shipment.material_name,
                        'quantity': shipment.quantity,
                        'unit': shipment.unit,
                        'quality': shipment.quality,
                        'penalty': shipment.penalty,
                        'unload_location': shipment.unload_location,
                        'list_of_reels': shipment.list_of_reels,
                        'profile_name': shipment.profile_name,
                        'width': shipment.width,
                        'sales_id': shipment.sales_id,
                        'price_per_kg': shipment.price_per_kg,
                        'total_price': shipment.total_price,
                        'extra_cost': shipment.extra_cost,
                        'vat': shipment.vat,
                        'invoice_status': shipment.invoice_status,
                        'payment_status': shipment.payment_status,
                        'document_info': shipment.document_info,
                        'comments': shipment.comments,
                        'cancellation_reason': shipment.cancellation_reason,
                        'username': shipment.username,
                        'logs': shipment.logs,
                        'purchase_id_id': shipment.purchase_id_id
                    }
            except Exception as e:
                print(f"خطا در خواندن اطلاعات shipment از دیتابیس خارجی: {str(e)}")
            finally:
                if 'db' in locals():
                    db.close()
        
        return jsonify({
            'success': True,
            'id': unloading.id,
            'warehouse_id': unloading.warehouse_id,
            'warehouse_name': unloading.warehouse.persian_name or unloading.warehouse.name,
            'warehouse_english_name': unloading.warehouse.name,
            'shipment_id': unloading.shipment_id,
            'shipment_info': shipment_info,  # اطلاعات کامل shipment
            'status': unloading.status,
            'version': unloading.version,
            'token': token,
            'can_edit': can_edit,
            'remaining_minutes': remaining_minutes,
            'items': items,
            'start_time': unloading.start_time.isoformat() if unloading.start_time else None,
            'end_time': unloading.end_time.isoformat() if unloading.end_time else None,
            'user_confirm_time': unloading.user_confirm_time.isoformat() if unloading.user_confirm_time else None,
            'edit_time': unloading.edit_time.isoformat() if unloading.edit_time else None,
            'vision_output': unloading.vision_output
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت تخلیه: {str(e)}'}), 500

@app.route('/api/unloadings/<token>/shipment', methods=['GET'])
def api_unloading_shipment_by_token(token):
    """Get shipment details for an unloading by token"""
    try:
        # Parse token to get unloading_id
        parts = token.split('.')
        if len(parts) != 4:
            return jsonify({'success': False, 'message': 'توکن نامعتبر است.'}), 400
        
        unloading_id, warehouse_id, timestamp, signature = parts
        unloading_id = int(unloading_id)
        
        # Verify token
        data = f"{unloading_id}.{warehouse_id}.{timestamp}"
        expected_signature = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        
        if signature != expected_signature:
            return jsonify({'success': False, 'message': 'توکن نامعتبر است.'}), 400
        
        # Get unloading
        unloading = Unloading.query.get(unloading_id)
        if not unloading:
            return jsonify({'success': False, 'message': 'تخلیه یافت نشد.'}), 404
        
        # اگر shipment_id در جدول unloadings وجود دارد، از جدول shipments بخوان
        if unloading.shipment_id:
            try:
                from models.database import get_db
                from models.external_db import Shipments
                
                db = next(get_db())
                shipment = db.query(Shipments).filter(Shipments.id == unloading.shipment_id).first()
                
                if shipment:
                    result = {
                        'id': shipment.id,
                        'license_number': shipment.license_number,
                        'supplier_name': shipment.supplier_name,
                        'customer_name': shipment.customer_name,
                        'status': shipment.status,
                        'shipment_type': shipment.shipment_type,
                        'weight1': shipment.weight1,
                        'weight2': shipment.weight2,
                        'net_weight': shipment.net_weight,
                        'date': shipment.date.isoformat() if shipment.date else None,
                        'receive_date': shipment.receive_date.isoformat() if shipment.receive_date else None,
                        'entry_time': shipment.entry_time.isoformat() if shipment.entry_time else None,
                        'weight1_time': shipment.weight1_time.isoformat() if shipment.weight1_time else None,
                        'weight2_time': shipment.weight2_time.isoformat() if shipment.weight2_time else None,
                        'exit_time': shipment.exit_time.isoformat() if shipment.exit_time else None,
                        'material_type': shipment.material_type,
                        'material_name': shipment.material_name,
                        'quantity': shipment.quantity,
                        'unit': shipment.unit,
                        'quality': shipment.quality,
                        'penalty': shipment.penalty,
                        'unload_location': shipment.unload_location,
                        'list_of_reels': shipment.list_of_reels,
                        'profile_name': shipment.profile_name,
                        'width': shipment.width,
                        'sales_id': shipment.sales_id,
                        'price_per_kg': shipment.price_per_kg,
                        'total_price': shipment.total_price,
                        'extra_cost': shipment.extra_cost,
                        'vat': shipment.vat,
                        'invoice_status': shipment.invoice_status,
                        'payment_status': shipment.payment_status,
                        'document_info': shipment.document_info,
                        'comments': shipment.comments,
                        'cancellation_reason': shipment.cancellation_reason,
                        'username': shipment.username,
                        'logs': shipment.logs,
                        'purchase_id_id': shipment.purchase_id_id
                    }
                    
                    return jsonify({
                        'success': True,
                        'data': result,
                        'source': 'database'
                    })
                else:
                    return jsonify({
                        'success': False, 
                        'message': 'محموله در دیتابیس یافت نشد'
                    }), 404
                    
            except Exception as e:
                return jsonify({
                    'success': False, 
                    'message': f'خطا در خواندن از دیتابیس خارجی: {str(e)}'
                }), 500
            finally:
                if 'db' in locals():
                    db.close()
        else:
            # اگر shipment_id وجود ندارد، پیام مناسب برگردان
            return jsonify({
                'success': False,
                'message': 'برای این بارگیری محموله‌ای انتخاب نشده است.',
                'source': 'none'
            }), 404
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت اطلاعات محموله: {str(e)}'}), 500

@app.route('/api/unloadings/active', methods=['GET'])
def api_unloadings_active():
    """Check for active unloading records for a specific warehouse"""
    warehouse_id = request.args.get('warehouse_id')
    if not warehouse_id:
        return jsonify({'success': False, 'message': 'شناسه انبار مشخص نشده است.'}), 400
    
    try:
        # Get the latest unloading record for this warehouse and check its status
        # unloading = Unloading.query.filter_by(warehouse_id=warehouse_id, status='started').first()
        # if unloading:
        unloading = Unloading.query.filter_by(warehouse_id=warehouse_id).order_by(Unloading.id.desc()).first()
        if unloading and unloading.status == 'started':
            return jsonify({
                'success': True,
                'unloading': {
                    'id': unloading.id,
                    'warehouse_id': unloading.warehouse_id,
                    'warehouse_name': unloading.warehouse.persian_name or unloading.warehouse.name,
                    'warehouse_english_name': unloading.warehouse.name,
                    'shipment_id': unloading.shipment_id,
                    'start_time': unloading.start_time.isoformat() if unloading.start_time else None,
                    'status': unloading.status
                }
            })
        else:
            return jsonify({'success': True, 'unloading': None})
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در بررسی تخلیه فعال: {str(e)}'}), 500

@app.route('/api/unloadings/active-any', methods=['GET'])
def api_unloadings_active_any():
    """Check for active unloading records across all warehouses"""
    try:
        # Look for any unloading record with status 'started'
        unloading = Unloading.query.filter_by(status='started').first()
        if unloading:
            return jsonify({
                'success': True,
                'unloading': {
                    'id': unloading.id,
                    'warehouse_id': unloading.warehouse_id,
                    'warehouse_name': unloading.warehouse.persian_name or unloading.warehouse.name,
                    'warehouse_english_name': unloading.warehouse.name,
                    'shipment_id': unloading.shipment_id,
                    'start_time': unloading.start_time.isoformat() if unloading.start_time else None,
                    'status': unloading.status
                }
            })
        else:
            return jsonify({'success': True, 'unloading': None})
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در بررسی تخلیه فعال: {str(e)}'}), 500


@app.route('/api/unloadings/edit', methods=['PUT', 'POST'])
def api_unloadings_edit():
    data = request.json or {}
    unloading_token = data.get('unloading_token')
    items = data.get('items')
    
    if not unloading_token:
        return jsonify({'success': False, 'message': 'اطلاعات ناقص است.'}), 400
    
    # اگر items خالی باشد، آرایه خالی در نظر بگیریم
    if not items:
        items = []
    
    # اعتبارسنجی token
    unloading_id, warehouse_id = verify_unloading_token(unloading_token)
    if not unloading_id:
        return jsonify({'success': False, 'message': 'توکن نامعتبر یا منقضی شده است.'}), 400
    
    unloading = Unloading.query.get(unloading_id)
    if not unloading:
        return jsonify({'success': False, 'message': 'تخلیه یافت نشد.'}), 404
    
    if unloading.status not in ['completed', 'vision', 'edited']:
        return jsonify({'success': False, 'message': 'فقط تخلیه‌های تکمیل شده یا در انتظار تایید قابل ویرایش هستند.'}), 400
    
    # چک کردن زمان ویرایش بر اساس آخرین ویرایش
    now = datetime.utcnow()
    last_edit_time = unloading.edit_time or unloading.user_confirm_time
    edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
    
    # اگر زمان ویرایش تمام شده، edit_time رو ریست کن تا کاربر بتونه دوباره ذخیره کنه
    if now > edit_deadline:
        unloading.edit_time = now
        last_edit_time = now
    
    # افزایش version و تغییر status بر اساس وضعیت فعلی
    unloading.version += 1
    
    # تغییر status بر اساس وضعیت فعلی
    if unloading.status == 'vision':
        unloading.status = 'completed'
    elif unloading.status == 'completed':
        unloading.status = 'edited'
    # اگر status قبلاً 'edited' بوده، همان 'edited' باقی می‌ماند
    
    # ذخیره تغییرات edit_time و version
    db.session.commit()
    
    # حذف آیتم‌های user از نسخه فعلی (نه همه نسخه‌ها)
    UnloadingItem.query.filter_by(unloading_id=unloading.id, source='user', version=unloading.version).delete()
    
    # اضافه کردن آیتم‌های جدید
    for item in items:
        count_value = item.get('count', 0)
        # اگر مقدار خالی باشد، آن را نادیده بگیر
        if count_value == '':
            continue
        unloading_item = UnloadingItem()
        unloading_item.unloading_id = unloading.id
        unloading_item.name = item.get('name', '')
        unloading_item.type = item.get('type', '')
        unloading_item.count = int(count_value)
        unloading_item.source = 'user'
        unloading_item.version = unloading.version
        db.session.add(unloading_item)
    
    db.session.commit()
    
    # محاسبه زمان ویرایش بر اساس آخرین edit_time (که ممکنه ریست شده باشه)
    edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
    can_edit = datetime.utcnow() < edit_deadline
    remaining_minutes = max(0, int((edit_deadline - datetime.utcnow()).total_seconds() / 60))
    
    return jsonify({
        'success': True, 
        'message': 'تخلیه با موفقیت ویرایش شد!',
        'version': unloading.version,
        'can_edit': can_edit,
        'remaining_minutes': remaining_minutes
    })

@app.route('/api/unloadings/<int:unloading_id>/history', methods=['GET'])
def api_unloading_history(unloading_id):
    """Get version history of a loading"""
    loading = Loading.query.get(loading_id)
    if not loading:
        return jsonify({'success': False, 'message': 'بارگیری یافت نشد.'}), 404
    
    # Group items by version
    version_history = {}
    for item in loading.items:
        version = item.version
        if version not in version_history:
            version_history[version] = {
                'version': version,
                'source': item.source,
                'items': []
            }
        
        version_history[version]['items'].append({
            'name': item.name,
            'type': item.type,
            'count': item.count,
            'source': item.source
        })
    
    # Sort by version
    sorted_history = sorted(version_history.values(), key=lambda x: x['version'])
    
    return jsonify({
        'success': True,
        'loading_id': loading.id,
        'warehouse_name': loading.warehouse.name,
        'shipment_id': loading.shipment_id,
        'current_version': loading.version,
        'history': sorted_history
    })

@app.route('/api/unloadings/<int:unloading_id>/details', methods=['GET'])
def api_unloading_details(unloading_id):
    """Get detailed information about a specific loading including vision data"""
    loading = Loading.query.get(loading_id)
    if not loading:
        return jsonify({'success': False, 'message': 'بارگیری یافت نشد.'}), 404
    
    # Get all items for this loading
    items = []
    vision_items = []
    user_items = []
    
    for item in loading.items:
        item_data = {
            'name': item.name,
            'type': item.type,
            'count': item.count,
            'source': item.source,
            'version': item.version
        }
        items.append(item_data)
        
        if item.source == 'vision':
            vision_items.append(item_data)
        elif item.source == 'user':
            user_items.append(item_data)
    
    # Calculate totals
    total_loaded = sum(item['count'] for item in items if item['type'] == 'loaded')
    total_unloaded = sum(item['count'] for item in items if item['type'] == 'unloaded')
    
    return jsonify({
        'success': True,
        'loading': {
            'id': loading.id,
            'warehouse': loading.warehouse,
            'shipment_id': loading.shipment_id,
            'status': loading.status,
            'start_time': loading.start_time.isoformat() if loading.start_time else None,
            'end_time': loading.end_time.isoformat() if loading.end_time else None,
            'user_confirm_time': loading.user_confirm_time.isoformat() if loading.user_confirm_time else None,
            'edit_time': loading.edit_time.isoformat() if loading.edit_time else None,
            'version': loading.version,
            'vision_output': loading.vision_output
        },
        'items': {
            'all': items,
            'vision': vision_items,
            'user': user_items
        },
        'totals': {
            'loaded': total_loaded,
            'unloaded': total_unloaded,
            'total_items': len(items)
        }
    })

@app.route('/api/loadings/list', methods=['GET'])
def api_loadings_list():
    """Get list of all loadings with basic information"""
    try:
        # Get query parameters
        status = request.args.get('status', '')
        warehouse_id = request.args.get('warehouse_id', '')
        limit = request.args.get('limit', 50, type=int)
        
        # Build query
        query = Loading.query
        
        if status:
            query = query.filter(Loading.status == status)
        if warehouse_id:
            query = query.filter(Loading.warehouse_id == warehouse_id)
        
        # Order by most recent first
        loadings = query.order_by(Loading.start_time.desc()).limit(limit).all()
        
        result = []
        for loading in loadings:
            # Calculate basic totals
            total_loaded = sum(item.count for item in loading.items if item.type == 'loaded')
            total_unloaded = sum(item.count for item in loading.items if item.type == 'unloaded')
            
            loading_info = {
                'id': loading.id,
                'warehouse': loading.warehouse,
                'shipment_id': loading.shipment_id,
                'status': loading.status,
                'start_time': loading.start_time.isoformat() if loading.start_time else None,
                'end_time': loading.end_time.isoformat() if loading.end_time else None,
                'user_confirm_time': loading.user_confirm_time.isoformat() if loading.user_confirm_time else None,
                'version': loading.version,
                'totals': {
                    'loaded': total_loaded,
                    'unloaded': total_unloaded,
                    'total_items': len(loading.items)
                }
            }
            result.append(loading_info)
        
        return jsonify({
            'success': True,
            'loadings': result,
            'count': len(result)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت لیست بارگیری‌ها: {str(e)}'}), 500

@app.route('/api/loadings/all', methods=['GET'])
def api_loadings_all():
    """Get all loadings with detailed information including items and versions"""
    try:
        # Get query parameters for pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Validate parameters
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 100:
            per_page = 10
        
        # Get total count
        total_count = Loading.query.count()
        
        # Get paginated loadings ordered by most recent first
        loadings = Loading.query.order_by(Loading.start_time.desc()).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        result = []
        for loading in loadings:
            # Get warehouse information
            warehouse = Warehouse.query.get(loading.warehouse_id)
            warehouse_name = warehouse.persian_name if warehouse else f"انبار {loading.warehouse_id}"
            
            # Calculate counts from the latest user version (or vision if no user version)
            latest_version = loading.version
            latest_items = []
            
            if latest_version > 1:
                # Get items from latest user version
                latest_items = [item for item in loading.items if item.source == 'user' and item.version == latest_version]
            else:
                # Get vision items if no user version exists
                latest_items = [item for item in loading.items if item.source == 'vision']
            
            # Calculate counts from latest version only
            items_count = len(latest_items)
            loaded_count = sum(item.count for item in latest_items if item.type == 'loaded')
            unloaded_count = sum(item.count for item in latest_items if item.type == 'unloaded')
            
            # Determine loading type based on status and version
            loading_type = 'vision'
            if loading.status == 'edited':
                loading_type = 'user'
            elif loading.version and loading.version > 1:
                loading_type = 'history'
            
            loading_info = {
                'token': str(loading.id),  # Use ID as token for now
                'warehouse_id': loading.warehouse_id,
                'warehouse_name': warehouse_name,
                'shipment_id': loading.shipment_id,
                'status': loading.status,
                'type': loading_type,
                'version': loading.version or 1,
                'created_at': loading.start_time.isoformat() if loading.start_time else None,
                'updated_at': loading.end_time.isoformat() if loading.end_time else None,
                'items_count': items_count,
                'loaded_count': loaded_count,
                'unloaded_count': unloaded_count
            }
            result.append(loading_info)
        
        return jsonify({
            'success': True,
            'loadings': result,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_count,
                'pages': loadings.pages,
                'has_next': loadings.has_next,
                'has_prev': loadings.has_prev,
                'next_num': loadings.next_num,
                'prev_num': loadings.prev_num
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت لیست بارگیری‌ها: {str(e)}'}), 500

@app.route('/api/loadings/<int:loading_id>/items', methods=['GET'])
def api_loading_items(loading_id):
    """Get detailed items for a specific loading"""
    try:
        loading = Loading.query.get(loading_id)
        if not loading:
            return jsonify({'success': False, 'message': 'بارگیری یافت نشد'}), 404
        
        # Check if all_versions parameter is provided
        all_versions = request.args.get('all_versions', 'false').lower() == 'true'
        
        items = []
        
        if all_versions:
            # Get all items for all versions
            for item in loading.items:
                items.append({
                    'id': item.id,
                    'name': item.name,
                    'type': item.type,
                    'count': item.count,
                    'source': item.source,
                    'version': item.version
                })
        else:
            # Get items from latest version only (default behavior)
            latest_version = loading.version
            
            if latest_version > 1:
                # Get user items from latest version
                for item in loading.items:
                    if item.source == 'user' and item.version == latest_version:
                        items.append({
                            'id': item.id,
                            'name': item.name,
                            'type': item.type,
                            'count': item.count,
                            'source': item.source,
                            'version': item.version
                        })
            else:
                # Get vision items if no user version exists
                for item in loading.items:
                    if item.source == 'vision':
                        items.append({
                            'id': item.id,
                            'name': item.name,
                            'type': item.type,
                            'count': item.count,
                            'source': item.source,
                            'version': item.version
                        })
        
        return jsonify({
            'success': True,
            'loading_id': loading_id,
            'items': items,
            'count': len(items),
            'start_time': loading.start_time.isoformat() if loading.start_time else None,
            'end_time': loading.end_time.isoformat() if loading.end_time else None,
            'user_confirm_time': loading.user_confirm_time.isoformat() if loading.user_confirm_time else None,
            'edit_time': loading.edit_time.isoformat() if loading.edit_time else None,
            'vision_output': loading.vision_output
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت آیتم‌های بارگیری: {str(e)}'}), 500

@app.route('/api/loadings/<int:loading_id>/all-items', methods=['GET'])
def api_loading_all_items(loading_id):
    """Get all items for all versions of a specific loading"""
    try:
        loading = Loading.query.get(loading_id)
        if not loading:
            return jsonify({'success': False, 'message': 'بارگیری یافت نشد'}), 404
        
        # Get all items for all versions
        items = []
        for item in loading.items:
            items.append({
                'id': item.id,
                'name': item.name,
                'type': item.type,
                'count': item.count,
                'source': item.source,
                'version': item.version
            })
        
        return jsonify({
            'success': True,
            'loading_id': loading_id,
            'items': items,
            'count': len(items),
            'vision_output': loading.vision_output
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت آیتم‌های بارگیری: {str(e)}'}), 500

@app.route('/api/loadings/<token>', methods=['GET'])
def api_loading_by_token(token):
    """Get loading by token for editing"""
    try:
        loading_id, warehouse_id = verify_loading_token(token)
        if not loading_id:
            return jsonify({'success': False, 'message': 'توکن نامعتبر یا منقضی شده است.'}), 400
        
        loading = Loading.query.get(loading_id)
        if not loading:
            return jsonify({'success': False, 'message': 'بارگیری یافت نشد.'}), 404
        
        # Check if still within edit window
        can_edit = True
        remaining_minutes = 0
        
        # Use edit_time if available, otherwise use user_confirm_time
        last_edit_time = loading.edit_time or loading.user_confirm_time
        if last_edit_time:
            edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
            can_edit = datetime.utcnow() < edit_deadline
            remaining_minutes = max(0, int((edit_deadline - datetime.utcnow()).total_seconds() / 60))
        
        # Get latest version items
        items = []
        latest_version = loading.version
        
        if latest_version > 1:
            # Get user items from latest version
            for item in loading.items:
                if item.source == 'user' and item.version == latest_version:
                    items.append({
                        'name': item.name,
                        'type': item.type,
                        'count': item.count,
                        'source': item.source,
                        'version': item.version,
                        'reel_number': item.reel_number,
                        'width': item.width,
                        'gsm': item.gsm,
                        'length': item.length,
                        'breaks': item.breaks,
                        'grade': item.grade
                    })
        else:
            # Get vision items
            for item in loading.items:
                if item.source == 'vision':
                    items.append({
                        'name': item.name,
                        'type': item.type,
                        'count': item.count,
                        'source': item.source,
                        'version': item.version,
                        'reel_number': item.reel_number,
                        'width': item.width,
                        'gsm': item.gsm,
                        'length': item.length,
                        'breaks': item.breaks,
                        'grade': item.grade
                    })
        
        return jsonify({
            'success': True,
            'id': loading.id,
            'warehouse_id': loading.warehouse_id,
            'shipment_id': loading.shipment_id,
            'status': loading.status,
            'start_time': loading.start_time.isoformat() if loading.start_time else None,
            'end_time': loading.end_time.isoformat() if loading.end_time else None,
            'user_confirm_time': loading.user_confirm_time.isoformat() if loading.user_confirm_time else None,
            'edit_time': loading.edit_time.isoformat() if loading.edit_time else None,
            'version': loading.version,
            'vision_output': loading.vision_output,
            'items': items,
            'can_edit': can_edit,
            'remaining_minutes': remaining_minutes
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت بارگیری: {str(e)}'}), 500

@app.route('/api/loadings/<token>/shipment', methods=['GET'])
def api_loading_shipment_by_token(token):
    """Get shipment information for a loading by token"""
    try:
        loading_id, warehouse_id = verify_loading_token(token)
        if not loading_id:
            return jsonify({'success': False, 'message': 'توکن نامعتبر یا منقضی شده است.'}), 400
        
        loading = Loading.query.get(loading_id)
        if not loading:
            return jsonify({'success': False, 'message': 'بارگیری یافت نشد.'}), 404
        
        if not loading.shipment_id:
            return jsonify({'success': False, 'message': 'اطلاعات محموله در دسترس نیست.'}), 404
        
        # Get shipment details from external database
        try:
            from models.external_db import Shipments
            shipment = Shipments.query.get(loading.shipment_id)
            if shipment:
                return jsonify({
                    'success': True,
                    'data': {
                        'id': shipment.id,
                        'name': shipment.name,
                        'description': getattr(shipment, 'description', ''),
                        'date': shipment.date.isoformat() if hasattr(shipment, 'date') and shipment.date else None
                    }
                })
            else:
                return jsonify({'success': False, 'message': 'محموله یافت نشد.'}), 404
        except Exception as e:
            print(f"Error getting shipment details: {e}")
            return jsonify({'success': False, 'message': 'خطا در دریافت جزئیات محموله.'}), 500
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت اطلاعات محموله: {str(e)}'}), 500

@app.route('/api/loadings/edit', methods=['PUT', 'POST'])
def api_loadings_edit():
    """Edit loading items and return new token"""
    try:
        data = request.json or {}
        loading_token = data.get('loading_token')
        items = data.get('items')
        
        if not loading_token or not items:
            return jsonify({'success': False, 'message': 'اطلاعات ناقص است.'}), 400
        
        loading_id, warehouse_id = verify_loading_token(loading_token)
        if not loading_id:
            return jsonify({'success': False, 'message': 'توکن نامعتبر یا منقضی شده است.'}), 400
        
        loading = Loading.query.get(loading_id)
        if not loading:
            return jsonify({'success': False, 'message': 'بارگیری یافت نشد.'}), 404
        
        if loading.status not in ['vision', 'edited']:
            return jsonify({'success': False, 'message': 'بارگیری در وضعیت نامعتبر برای ویرایش است.'}), 400
        
        # Check if still within edit window
        last_edit_time = loading.edit_time or loading.user_confirm_time
        if last_edit_time:
            edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
            if datetime.utcnow() >= edit_deadline:
                return jsonify({'success': False, 'message': 'زمان ویرایش به پایان رسیده است.'}), 400
        
        # Delete previous user items
        LoadingItem.query.filter_by(loading_id=loading.id, source='user').delete()
        
        # Add new user items
        for item in items:
            if int(item.get('count', 0)) > 0:
                loading_item = LoadingItem()
                loading_item.loading_id = loading.id
                loading_item.name = item.get('name', '')
                loading_item.type = item.get('type', '')
                loading_item.count = int(item.get('count', 0))
                loading_item.source = 'user'
                loading_item.version = loading.version + 1
                loading_item.reel_number = item.get('reel_number')
                loading_item.width = item.get('width')
                loading_item.gsm = item.get('gsm')
                loading_item.length = item.get('length')
                loading_item.breaks = item.get('breaks')
                loading_item.grade = item.get('grade')
                db.session.add(loading_item)
        
        # Update loading status and version
        loading.status = 'edited'
        loading.version += 1
        loading.edit_time = datetime.utcnow()
        
        db.session.commit()
        
        # Generate new token
        timestamp = int(time.time())
        data_string = f"{loading.id}.{warehouse_id}.{timestamp}"
        new_token = f"{data_string}.{hmac.new(SECRET_KEY.encode(), data_string.encode(), hashlib.sha256).hexdigest()}"
        
        return jsonify({
            'success': True,
            'message': 'بارگیری با موفقیت ویرایش شد.',
            'loading_token': new_token,
            'can_edit': True,
            'remaining_minutes': EDIT_WINDOW_MINUTES
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در ویرایش بارگیری: {str(e)}'}), 500

@app.route('/api/loadings/save', methods=['POST'])
def api_loadings_save():
    """Save loading items and mark as completed"""
    try:
        data = request.json or {}
        loading_token = data.get('loading_token')
        items = data.get('items')
        
        if not loading_token or not items:
            return jsonify({'success': False, 'message': 'اطلاعات ناقص است.'}), 400
        
        loading_id, warehouse_id = verify_loading_token(loading_token)
        if not loading_id:
            return jsonify({'success': False, 'message': 'توکن نامعتبر یا منقضی شده است.'}), 400
        
        loading = Loading.query.get(loading_id)
        if not loading:
            return jsonify({'success': False, 'message': 'بارگیری یافت نشد.'}), 404
        
        if loading.status not in ['vision', 'edited']:
            return jsonify({'success': False, 'message': 'بارگیری در وضعیت نامعتبر برای ذخیره است.'}), 400
        
        # Check if still within edit window
        last_edit_time = loading.edit_time or loading.user_confirm_time
        if last_edit_time:
            edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
            if datetime.utcnow() >= edit_deadline:
                return jsonify({'success': False, 'message': 'زمان ویرایش به پایان رسیده است.'}), 400
        
        # Delete previous user items
        LoadingItem.query.filter_by(loading_id=loading.id, source='user').delete()
        
        # Add new user items
        for item in items:
            if int(item.get('count', 0)) > 0:
                loading_item = LoadingItem()
                loading_item.loading_id = loading.id
                loading_item.name = item.get('name', '')
                loading_item.type = item.get('type', '')
                loading_item.count = int(item.get('count', 0))
                loading_item.source = 'user'
                loading_item.version = loading.version + 1
                loading_item.reel_number = item.get('reel_number')
                loading_item.width = item.get('width')
                loading_item.gsm = item.get('gsm')
                loading_item.length = item.get('length')
                loading_item.breaks = item.get('breaks')
                loading_item.grade = item.get('grade')
                db.session.add(loading_item)
        
        # Update loading status and version
        loading.status = 'completed'
        loading.version += 1
        loading.user_confirm_time = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'بارگیری با موفقیت ذخیره شد!'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در ذخیره بارگیری: {str(e)}'}), 500

@app.route('/api/operations/all', methods=['GET'])
def api_operations_all():
    """Get all operations (both loadings and unloadings) with detailed information"""
    try:
        # Get query parameters for pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        operation_type = request.args.get('operation_type', '')  # 'loading', 'unloading', or empty for all
        
        # Validate parameters
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 100:
            per_page = 10
        
        result = []
        
        # Get loadings if requested (when operation_type is empty or 'loading')
        if operation_type in ['', 'loading']:
            # Get total count for loadings
            loadings_query = Loading.query
            total_loadings = loadings_query.count()
            
            # Get paginated loadings
            loadings = loadings_query.order_by(Loading.start_time.desc()).paginate(
                page=page, 
                per_page=per_page, 
                error_out=False
            )
            
            for loading in loadings:
                # Get warehouse information
                warehouse = Warehouse.query.get(loading.warehouse_id)
                warehouse_name = warehouse.persian_name if warehouse else f"انبار {loading.warehouse_id}"
                
                # Calculate counts from the latest user version (or vision if no user version)
                latest_version = loading.version
                latest_items = []
                
                if latest_version > 1:
                    # Get items from latest user version
                    latest_items = [item for item in loading.items if item.source == 'user' and item.version == latest_version]
                else:
                    # Get vision items if no user version exists
                    latest_items = [item for item in loading.items if item.source == 'vision']
                
                # Calculate counts from latest version only
                items_count = len(latest_items)
                loaded_count = sum(item.count for item in latest_items if item.type == 'loaded')
                unloaded_count = sum(item.count for item in latest_items if item.type == 'unloaded')
                
                result.append({
                    'id': loading.id,
                    'type': 'loading',
                    'warehouse_id': loading.warehouse_id,
                    'warehouse_name': warehouse_name,
                    'warehouse_english_name': warehouse.name if warehouse else loading.warehouse_id,
                    'shipment_id': loading.shipment_id,
                    'status': loading.status,
                    'version': loading.version,
                    'start_time': loading.start_time.isoformat() if loading.start_time else None,
                    'end_time': loading.end_time.isoformat() if loading.end_time else None,
                    'user_confirm_time': loading.user_confirm_time.isoformat() if loading.user_confirm_time else None,
                    'edit_time': loading.edit_time.isoformat() if loading.edit_time else None,
                    'vision_output': loading.vision_output,
                    'items_count': items_count,
                    'loaded_count': loaded_count,
                    'unloaded_count': unloaded_count,
                    'token': f"loading_{loading.id}_{int(loading.start_time.timestamp()) if loading.start_time else 0}"
                })
        
        # Get unloadings if requested
        if operation_type in ['', 'unloading']:
            # Get total count for unloadings
            unloadings_query = Unloading.query
            total_unloadings = unloadings_query.count()
            
            # Get paginated unloadings
            unloadings = unloadings_query.order_by(Unloading.start_time.desc()).paginate(
                page=page, 
                per_page=per_page, 
                error_out=False
            )
            
            for unloading in unloadings:
                # Get warehouse information
                warehouse = Warehouse.query.get(unloading.warehouse_id)
                warehouse_name = warehouse.persian_name if warehouse else f"انبار {unloading.warehouse_id}"
                
                # Calculate counts from the latest user version (or vision if no user version)
                latest_version = unloading.version
                latest_items = []
                
                if latest_version > 1:
                    # Get items from latest user version
                    latest_items = [item for item in unloading.items if item.source == 'user' and item.version == latest_version]
                else:
                    # Get vision items if no user version exists
                    latest_items = [item for item in unloading.items if item.source == 'vision']
                
                # Calculate counts from latest version only
                items_count = len(latest_items)
                loaded_count = sum(item.count for item in latest_items if item.type == 'loaded')
                unloaded_count = sum(item.count for item in latest_items if item.type == 'unloaded')
                
                result.append({
                    'id': unloading.id,
                    'type': 'unloading',
                    'warehouse_id': unloading.warehouse_id,
                    'warehouse_name': warehouse_name,
                    'warehouse_english_name': warehouse.name if warehouse else unloading.warehouse_id,
                    'shipment_id': unloading.shipment_id,
                    'status': unloading.status,
                    'version': unloading.version,
                    'start_time': unloading.start_time.isoformat() if unloading.start_time else None,
                    'end_time': unloading.end_time.isoformat() if unloading.end_time else None,
                    'user_confirm_time': unloading.user_confirm_time.isoformat() if unloading.user_confirm_time else None,
                    'edit_time': unloading.edit_time.isoformat() if unloading.edit_time else None,
                    'vision_output': unloading.vision_output,
                    'items_count': items_count,
                    'loaded_count': loaded_count,
                    'unloaded_count': unloaded_count,
                    'token': f"unloading_{unloading.id}_{int(unloading.start_time.timestamp()) if unloading.start_time else 0}"
                })
        
        # Sort combined results by start_time (most recent first)
        result.sort(key=lambda x: x['start_time'] or '', reverse=True)
        
        # Apply pagination to combined results
        total_count = len(result)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_result = result[start_idx:end_idx]
        
        return jsonify({
            'success': True,
            'operations': paginated_result,
            'count': len(paginated_result),
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_count,
                'pages': (total_count + per_page - 1) // per_page,
                'has_next': end_idx < total_count,
                'has_prev': page > 1,
                'next_num': page + 1 if end_idx < total_count else None,
                'prev_num': page - 1 if page > 1 else None
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت لیست عملیات: {str(e)}'}), 500

@app.route('/api/shipments/latest', methods=['GET'])
def api_shipments_latest():
    """Get latest shipments from external database"""
    try:
        from models.database import get_db, get_latest_loaded_unloaded_shipments
        
        # Get query parameters
        limit = int(request.args.get('limit', 10))
        status = request.args.get('status')  # Optional filter by status
        
        db = next(get_db())
        
        # Get latest loaded/unloaded shipments
        shipments = get_latest_loaded_unloaded_shipments(db, limit=limit)
        
        # Filter by status if provided
        if status:
            shipments = [s for s in shipments if s.status == status]
        
        result = []
        for shipment in shipments:
            result.append({
                'id': shipment.id,
                'license_number': shipment.license_number,
                'supplier_name': shipment.supplier_name,
                'customer_name': shipment.customer_name,
                'status': shipment.status,
                'shipment_type': shipment.shipment_type,
                'weight1': shipment.weight1,
                'weight2': shipment.weight2,
                'net_weight': shipment.net_weight,
                'date': shipment.date.isoformat() if shipment.date else None,
                'receive_date': shipment.receive_date.isoformat() if shipment.receive_date else None,
                'material_type': shipment.material_type,
                'material_name': shipment.material_name,
                'quantity': shipment.quantity,
                'unit': shipment.unit
            })
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'خطا در دریافت محموله‌ها: {str(e)}'
        }), 500
    finally:
        if 'db' in locals():
            db.close()

@app.route('/api/shipments/for-unloading', methods=['GET'])
def api_shipments_for_unloading():
    """Get shipments for unloading operation (Incoming shipments with LoadingUnloading status)"""
    try:
        from models.database import get_db, get_shipments_for_unloading
        
        # Get query parameters
        limit = int(request.args.get('limit', 50))
        
        db = next(get_db())
        
        # Get shipments for unloading
        shipments = get_shipments_for_unloading(db, limit=limit)
        
        result = []
        for shipment in shipments:
            result.append({
                'id': shipment.id,
                'license_number': shipment.license_number,
                'supplier_name': shipment.supplier_name,
                'customer_name': shipment.customer_name,
                'status': shipment.status,
                'shipment_type': shipment.shipment_type,
                'weight1': shipment.weight1,
                'weight2': shipment.weight2,
                'net_weight': shipment.net_weight,
                'date': shipment.date.isoformat() if shipment.date else None,
                'receive_date': shipment.receive_date.isoformat() if shipment.receive_date else None,
                'material_type': shipment.material_type,
                'material_name': shipment.material_name,
                'quantity': shipment.quantity,
                'unit': shipment.unit
            })
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'خطا در دریافت محموله‌ها: {str(e)}'
        }), 500
    finally:
        if 'db' in locals():
            db.close()

@app.route('/api/shipments/for-loading', methods=['GET'])
def api_shipments_for_loading():
    """Get shipments for loading operation (Outgoing shipments with LoadingUnloading status)"""
    try:
        from models.database import get_db, get_shipments_for_loading
        
        # Get query parameters
        limit = int(request.args.get('limit', 50))
        
        db = next(get_db())
        
        # Get shipments for loading
        shipments = get_shipments_for_loading(db, limit=limit)
        
        result = []
        for shipment in shipments:
            result.append({
                'id': shipment.id,
                'license_number': shipment.license_number,
                'supplier_name': shipment.supplier_name,
                'customer_name': shipment.customer_name,
                'status': shipment.status,
                'shipment_type': shipment.shipment_type,
                'weight1': shipment.weight1,
                'weight2': shipment.weight2,
                'net_weight': shipment.net_weight,
                'date': shipment.date.isoformat() if shipment.date else None,
                'receive_date': shipment.receive_date.isoformat() if shipment.receive_date else None,
                'material_type': shipment.material_type,
                'material_name': shipment.material_name,
                'quantity': shipment.quantity,
                'unit': shipment.unit
            })
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'خطا در دریافت محموله‌ها: {str(e)}'
        }), 500
    finally:
        if 'db' in locals():
            db.close()

@app.route('/api/shipments/<int:shipment_id>', methods=['GET'])
def api_shipment_detail(shipment_id):
    """Get specific shipment details"""
    try:
        from models.database import get_db
        from models.external_db import Shipments
        
        db = next(get_db())
        shipment = db.query(Shipments).filter(Shipments.id == shipment_id).first()
        
        if not shipment:
            return jsonify({
                'success': False,
                'message': 'محموله یافت نشد'
            }), 404
        
        result = {
            'id': shipment.id,
            'license_number': shipment.license_number,
            'supplier_name': shipment.supplier_name,
            'customer_name': shipment.customer_name,
            'status': shipment.status,
            'shipment_type': shipment.shipment_type,
            'weight1': shipment.weight1,
            'weight2': shipment.weight2,
            'net_weight': shipment.net_weight,
            'date': shipment.date.isoformat() if shipment.date else None,
            'receive_date': shipment.receive_date.isoformat() if shipment.receive_date else None,
            'entry_time': shipment.entry_time.isoformat() if shipment.entry_time else None,
            'weight1_time': shipment.weight1_time.isoformat() if shipment.weight1_time else None,
            'weight2_time': shipment.weight2_time.isoformat() if shipment.weight2_time else None,
            'exit_time': shipment.exit_time.isoformat() if shipment.exit_time else None,
            'material_type': shipment.material_type,
            'material_name': shipment.material_name,
            'quantity': shipment.quantity,
            'unit': shipment.unit,
            'quality': shipment.quality,
            'penalty': shipment.penalty,
            'unload_location': shipment.unload_location,
            'list_of_reels': shipment.list_of_reels,
            'profile_name': shipment.profile_name,
            'width': shipment.width,
            'price_per_kg': shipment.price_per_kg,
            'total_price': shipment.total_price,
            'extra_cost': shipment.extra_cost,
            'vat': shipment.vat,
            'invoice_status': shipment.invoice_status,
            'payment_status': shipment.payment_status,
            'document_info': shipment.document_info,
            'comments': shipment.comments
        }
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'خطا در دریافت جزئیات محموله: {str(e)}'
        }), 500
    finally:
        if 'db' in locals():
            db.close()

@app.route('/api/database/close', methods=['POST'])
def api_database_close():
    """Close external database connections to allow file replacement"""
    try:
        from models.database import close_database_connection
        
        success = close_database_connection()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'اتصالات دیتابیس با موفقیت بسته شد. حالا می‌توانید فایل دیتابیس را جایگزین کنید.'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'خطا در بستن اتصالات دیتابیس'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'خطا در بستن اتصالات دیتابیس: {str(e)}'
        }), 500

@app.route('/api/database/status', methods=['GET'])
def api_database_status():
    """Check if external database is accessible"""
    try:
        from models.database import get_db, get_shipments_count
        
        db = next(get_db())
        count = get_shipments_count(db)
        
        return jsonify({
            'success': True,
            'message': 'دیتابیس قابل دسترسی است',
            'data': {
                'shipments_count': count,
                'status': 'connected'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'دیتابیس قابل دسترسی نیست: {str(e)}',
            'data': {
                'status': 'disconnected'
            }
        }), 500
    finally:
        if 'db' in locals():
            db.close()

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    try:
        return jsonify({
            'success': True,
            'message': 'Server is running',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/ssh/test', methods=['GET'])
def api_ssh_test():
    """Test SSH connection to remote server"""
    try:
        from ssh_operations import test_ssh_connection
        
        success = test_ssh_connection()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'SSH connection test successful'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'SSH connection test failed'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'SSH test error: {str(e)}'
        }), 500

@app.route('/api/ssh/copy-database', methods=['POST'])
def api_ssh_copy_database():
    """Copy database file from remote server via SSH"""
    try:
        from ssh_operations import copy_database_via_ssh
        
        # Get filename from request (optional)
        data = request.json or {}
        filename = data.get('filename', 'localnew.sqlite3')
        
        success = copy_database_via_ssh(filename)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Database file {filename} copied successfully from remote server'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to copy database file {filename}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'SSH copy error: {str(e)}'
        }), 500

@app.route('/api/ssh/list-files', methods=['GET'])
def api_ssh_list_files():
    """List files in remote server home directory"""
    try:
        from ssh_operations import list_remote_database_files
        
        success = list_remote_database_files()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Remote files listed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to list remote files'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'SSH list files error: {str(e)}'
        }), 500

@app.route('/api/ssh/health', methods=['GET'])
def api_ssh_health():
    """Check remote server health via SSH"""
    try:
        from ssh_operations import test_ssh_connection
        
        success = test_ssh_connection()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Remote server is accessible via SSH'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Remote server is not accessible via SSH'
            }), 503
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'SSH health check error: {str(e)}'
        }), 503

# Operation Types endpoints
@app.route('/api/operation-types', methods=['GET'])
def api_operation_types():
    """Get all operation types"""
    try:
        operation_types = OperationType.query.order_by(OperationType.order).all()
        
        result = []
        for op_type in operation_types:
            result.append({
                'id': op_type.id,
                'name': op_type.name,
                'persian_name': op_type.persian_name,
                'icon': op_type.icon,
                'color': op_type.color,
                'is_enabled': op_type.is_enabled,
                'is_available': op_type.is_available,
                'description': op_type.description,
                'order': op_type.order
            })
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/operation-types', methods=['POST'])
def api_operation_types_create():
    """Create a new operation type"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'persian_name']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Check if operation type already exists
        existing = OperationType.query.filter_by(name=data['name']).first()
        if existing:
            return jsonify({
                'success': False,
                'error': 'Operation type with this name already exists'
            }), 400
        
        # Create new operation type
        new_operation_type = OperationType(
            name=data['name'],
            persian_name=data['persian_name'],
            icon=data.get('icon'),
            color=data.get('color'),
            is_enabled=data.get('is_enabled', True),
            is_available=data.get('is_available', True),
            description=data.get('description'),
            order=data.get('order', 0)
        )
        
        db.session.add(new_operation_type)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Operation type created successfully',
            'data': {
                'id': new_operation_type.id,
                'name': new_operation_type.name,
                'persian_name': new_operation_type.persian_name,
                'icon': new_operation_type.icon,
                'color': new_operation_type.color,
                'is_enabled': new_operation_type.is_enabled,
                'is_available': new_operation_type.is_available,
                'description': new_operation_type.description,
                'order': new_operation_type.order
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/operation-types/<int:operation_type_id>', methods=['PUT'])
def api_operation_types_update(operation_type_id):
    """Update an operation type"""
    try:
        operation_type = OperationType.query.get_or_404(operation_type_id)
        data = request.get_json()
        
        # Update fields
        if 'persian_name' in data:
            operation_type.persian_name = data['persian_name']
        if 'icon' in data:
            operation_type.icon = data['icon']
        if 'color' in data:
            operation_type.color = data['color']
        if 'is_enabled' in data:
            operation_type.is_enabled = data['is_enabled']
        if 'is_available' in data:
            operation_type.is_available = data['is_available']
        if 'description' in data:
            operation_type.description = data['description']
        if 'order' in data:
            operation_type.order = data['order']
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Operation type updated successfully',
            'data': {
                'id': operation_type.id,
                'name': operation_type.name,
                'persian_name': operation_type.persian_name,
                'icon': operation_type.icon,
                'color': operation_type.color,
                'is_enabled': operation_type.is_enabled,
                'is_available': operation_type.is_available,
                'description': operation_type.description,
                'order': operation_type.order
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/operation-types/<int:operation_type_id>', methods=['DELETE'])
def api_operation_types_delete(operation_type_id):
    """Delete an operation type"""
    try:
        operation_type = OperationType.query.get_or_404(operation_type_id)
        
        db.session.delete(operation_type)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Operation type deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Vision Servers endpoints
@app.route('/api/vision-servers', methods=['GET'])
def api_vision_servers():
    """Get all vision servers"""
    try:
        vision_servers = VisionServer.query.all()
        
        result = []
        for server in vision_servers:
            # Get warehouse IDs for this server
            warehouse_ids = [warehouse.id for warehouse in server.warehouses]
            
            result.append({
                'id': server.id,
                'name': server.name,
                'persian_name': server.persian_name,
                'url': server.url,
                'type': server.type,
                'is_active': server.is_active,
                'warehouse_ids': warehouse_ids
            })
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/vision-servers', methods=['POST'])
def api_vision_servers_create():
    """Create a new vision server"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'url', 'type']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Check if vision server already exists
        existing = VisionServer.query.filter_by(name=data['name']).first()
        if existing:
            return jsonify({
                'success': False,
                'error': 'Vision server with this name already exists'
            }), 400
        
        # Create new vision server
        new_vision_server = VisionServer(
            name=data['name'],
            persian_name=data.get('persian_name'),
            url=data['url'],
            type=data['type'],
            is_active=data.get('is_active', True)
        )
        
        db.session.add(new_vision_server)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Vision server created successfully',
            'data': {
                'id': new_vision_server.id,
                'name': new_vision_server.name,
                'persian_name': new_vision_server.persian_name,
                'url': new_vision_server.url,
                'type': new_vision_server.type,
                'is_active': new_vision_server.is_active
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/vision-servers/<int:vision_server_id>', methods=['PUT'])
def api_vision_servers_update(vision_server_id):
    """Update a vision server"""
    try:
        vision_server = VisionServer.query.get_or_404(vision_server_id)
        data = request.get_json()
        
        # Update fields
        if 'persian_name' in data:
            vision_server.persian_name = data['persian_name']
        if 'url' in data:
            vision_server.url = data['url']
        if 'type' in data:
            vision_server.type = data['type']
        if 'is_active' in data:
            vision_server.is_active = data['is_active']
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Vision server updated successfully',
            'data': {
                'id': vision_server.id,
                'name': vision_server.name,
                'persian_name': vision_server.persian_name,
                'url': vision_server.url,
                'type': vision_server.type,
                'is_active': vision_server.is_active
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/vision-servers/<int:vision_server_id>', methods=['DELETE'])
def api_vision_servers_delete(vision_server_id):
    """Delete a vision server"""
    try:
        vision_server = VisionServer.query.get_or_404(vision_server_id)
        
        db.session.delete(vision_server)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Vision server deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/vision-servers/assignments', methods=['POST'])
def api_vision_servers_assignments():
    """Update warehouse assignments for vision servers"""
    try:
        data = request.get_json()
        assignments = data.get('assignments', {})
        
        # Clear all existing assignments
        db.session.execute(db.text('DELETE FROM warehouse_vision_server'))
        
        # Add new assignments
        for server_id, warehouse_ids in assignments.items():
            if warehouse_ids:  # Only process if there are warehouse IDs
                vision_server = VisionServer.query.get(int(server_id))
                if vision_server:
                    for warehouse_id in warehouse_ids:
                        warehouse = Warehouse.query.get(str(warehouse_id))
                        if warehouse:
                            vision_server.warehouses.append(warehouse)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Warehouse assignments updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/vision-servers/warehouse/<warehouse_id>', methods=['GET'])
def api_vision_servers_by_warehouse(warehouse_id):
    """Get vision servers assigned to a specific warehouse"""
    try:
        # Find the warehouse
        warehouse = Warehouse.query.filter_by(id=warehouse_id).first()
        if not warehouse:
            return jsonify({
                'success': False,
                'error': 'Warehouse not found'
            }), 404
        
        # Get vision servers assigned to this warehouse
        vision_servers = warehouse.vision_servers
        
        result = []
        for server in vision_servers:
            result.append({
                'id': server.id,
                'name': server.name,
                'persian_name': server.persian_name,
                'url': server.url,
                'type': server.type,
                'operation_type': server.type,  # برای سازگاری با frontend
                'is_active': server.is_active,
                'is_enabled': server.is_active,  # برای سازگاری با frontend
                'is_available': server.is_active,  # برای سازگاری با frontend
                'location': server.persian_name or server.name,  # استفاده از نام به عنوان محل
                'warehouse_id': warehouse_id
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



# Static file serving routes for React Router (must be at the end to avoid conflicts with API routes)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve React app for all routes, allowing client-side routing to work"""
    # Check if the path is for a static file that exists
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # For all other paths, serve index.html to let React Router handle routing
        return send_from_directory('static', 'index.html')

@app.route('/api/debug/operations', methods=['GET'])
def api_debug_operations():
    """Debug endpoint to check the status of operations in database"""
    try:
        warehouse_id = request.args.get('warehouse_id')
        
        # Get loadings
        loadings_query = Loading.query
        if warehouse_id:
            loadings_query = loadings_query.filter(Loading.warehouse_id == warehouse_id)
        loadings = loadings_query.all()
        
        # Get unloadings
        unloadings_query = Unloading.query
        if warehouse_id:
            unloadings_query = unloadings_query.filter(Unloading.warehouse_id == warehouse_id)
        unloadings = unloadings_query.all()
        
        loadings_data = []
        for loading in loadings:
            loadings_data.append({
                'id': loading.id,
                'warehouse_id': loading.warehouse_id,
                'status': loading.status,
                'start_time': loading.start_time.isoformat() if loading.start_time else None,
                'end_time': loading.end_time.isoformat() if loading.end_time else None,
                'shipment_id': loading.shipment_id
            })
        
        unloadings_data = []
        for unloading in unloadings:
            unloadings_data.append({
                'id': unloading.id,
                'warehouse_id': unloading.warehouse_id,
                'status': unloading.status,
                'start_time': unloading.start_time.isoformat() if unloading.start_time else None,
                'end_time': unloading.end_time.isoformat() if unloading.end_time else None,
                'shipment_id': unloading.shipment_id
            })
        
        return jsonify({
            'success': True,
            'loadings': loadings_data,
            'unloadings': unloadings_data,
            'loadings_count': len(loadings_data),
            'unloadings_count': len(unloadings_data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در بررسی عملیات: {str(e)}'}), 500

@app.route('/api/debug/vision-servers', methods=['GET'])
def api_debug_vision_servers():
    """Debug endpoint to check vision server status"""
    try:
        operation_type = request.args.get('operation_type', 'loading')
        
        # Get vision server for the operation type
        vision_server = VisionServer.query.filter_by(
            type=operation_type,
            is_active=True
        ).first()
        
        if not vision_server:
            return jsonify({
                'success': False,
                'message': f'سرور بینایی برای عملیات {operation_type} یافت نشد.',
                'vision_servers': []
            })
        
        # Test connection to vision server
        try:
            resp = requests.get(f'{vision_server.url}/status', timeout=5)
            server_status = {
                'url': vision_server.url,
                'type': vision_server.type,
                'is_active': vision_server.is_active,
                'connection_status': 'connected' if resp.status_code == 200 else 'error',
                'response_status': resp.status_code,
                'response_data': resp.json() if resp.status_code == 200 else None
            }
        except Exception as e:
            server_status = {
                'url': vision_server.url,
                'type': vision_server.type,
                'is_active': vision_server.is_active,
                'connection_status': 'error',
                'error': str(e)
            }
        
        return jsonify({
            'success': True,
            'vision_server': server_status,
            'message': f'وضعیت سرور بینایی {operation_type} بررسی شد.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در بررسی سرور بینایی: {str(e)}'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print('db.sqlite3 created or already exists.')
    app.run(debug=True, host='0.0.0.0', port=18888) 