from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import hmac
import hashlib
import time
from models import db, Unloading, UnloadingItem, Warehouse
from config import SECRET_KEY, EDIT_WINDOW_MINUTES

unloading_bp = Blueprint('unloading', __name__)

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

@unloading_bp.route('/api/unloadings', methods=['POST'])
@unloading_bp.route('/api/unloadings/save', methods=['POST'])
def api_unloadings_save():
    data = request.json or {}
    warehouse_id = data.get('warehouse_id')
    shipment_id = data.get('shipment_id')
    items = data.get('items')
    unloading_token = data.get('unloading_token')
    
    print(f"DEBUG: Save request received - warehouse_id: {warehouse_id}, items count: {len(items) if items else 0}, token: {unloading_token}")
    
    if not warehouse_id or not items:
        return jsonify({'success': False, 'message': 'اطلاعات ناقص است.'}), 400
    
    try:
        # اگر unloading_token ارسال شده، اعتبارسنجی کن
        if unloading_token:
            unloading_id, token_warehouse_id = verify_unloading_token(unloading_token)
            print(f"DEBUG: Token verification - unloading_id: {unloading_id}, token_warehouse_id: {token_warehouse_id}")
            
            if unloading_id and token_warehouse_id == warehouse_id:
                unloading = Unloading.query.get(unloading_id)
                if unloading and unloading.status in ['vision', 'edited', 'completed']:
                    print(f"DEBUG: Found unloading - id: {unloading.id}, status: {unloading.status}, version: {unloading.version}")
                    # Don't delete previous user items - keep version history
                    # UnloadingItem.query.filter_by(unloading_id=unloading.id, source='user').delete()
                    # آپدیت status به completed و افزایش version
                    unloading.status = 'completed'
                    unloading.version += 1
                    unloading.user_confirm_time = datetime.utcnow()
                    print(f"DEBUG: Updated unloading - new status: {unloading.status}, new version: {unloading.version}")
                else:
                    return jsonify({'success': False, 'message': 'تخلیه یافت نشد یا در وضعیت نامعتبر است.'}), 404
            else:
                return jsonify({'success': False, 'message': 'توکن نامعتبر یا منقضی شده است.'}), 400
        else:
            # اگر token نبود، unloading جدید بساز (برای تخلیه دستی)
            unloading = Unloading()
            unloading.warehouse_id = warehouse_id
            unloading.shipment_id = shipment_id
            unloading.start_time = datetime.utcnow()
            unloading.status = 'completed'
            unloading.user_confirm_time = datetime.utcnow() + timedelta(minutes=EDIT_WINDOW_MINUTES)
            db.session.add(unloading)
            print(f"DEBUG: Created new unloading - warehouse_id: {warehouse_id}")
        
        # Add items
        added_count = 0
        for item in items:
            if int(item.get('count', 0)) == 0:
                continue
            unloading_item = UnloadingItem()
            unloading_item.unloading_id = unloading.id
            unloading_item.name = item.get('name', '')
            unloading_item.type = item.get('type', '')
            unloading_item.count = int(item.get('count', 0))
            unloading_item.source = 'user'
            unloading_item.version = unloading.version
            db.session.add(unloading_item)
            added_count += 1
        
        print(f"DEBUG: Added {added_count} items")
        
        # ذخیره همه تغییرات در یک commit
        db.session.commit()
        print(f"DEBUG: Database commit successful")
        return jsonify({'success': True, 'message': 'تخلیه با موفقیت ذخیره شد!'})
        
    except Exception as e:
        # Rollback در صورت بروز خطا
        db.session.rollback()
        print(f"ERROR in unloading save: {str(e)}")
        print(f"DEBUG: Rollback completed")
        return jsonify({'success': False, 'message': f'خطا در ذخیره تخلیه: {str(e)}'}), 500

@unloading_bp.route('/api/unloadings/last-completed', methods=['GET'])
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
        now = datetime.utcnow()
        
        # اگر زمان ویرایش تمام شده، edit_time رو ریست کن تا کاربر بتونه دوباره ویرایش کنه
        if now > edit_deadline:
            unloading.edit_time = now
            last_edit_time = now
            db.session.commit()
        
        # محاسبه مجدد بعد از ریست
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
            'shipment_info': shipment_info,
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

@unloading_bp.route('/api/unloadings/active', methods=['GET'])
def api_unloadings_active():
    """Check for active unloading records for a specific warehouse"""
    warehouse_id = request.args.get('warehouse_id')
    if not warehouse_id:
        return jsonify({'success': False, 'message': 'شناسه انبار مشخص نشده است.'}), 400
    
    try:
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

@unloading_bp.route('/api/unloadings/active-any', methods=['GET'])
def api_unloadings_active_any():
    """Check for active unloading records across all warehouses"""
    try:
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

@unloading_bp.route('/api/unloadings/edit', methods=['PUT', 'POST'])
def api_unloadings_edit():
    data = request.json or {}
    unloading_token = data.get('unloading_token')
    items = data.get('items')
    
    print(f"DEBUG: Edit request received - token: {unloading_token}, items count: {len(items) if items else 0}")
    
    if not unloading_token:
        return jsonify({'success': False, 'message': 'اطلاعات ناقص است.'}), 400
    
    # اگر items خالی باشد، آرایه خالی در نظر بگیریم
    if not items:
        items = []
    
    # اعتبارسنجی token
    unloading_id, warehouse_id = verify_unloading_token(unloading_token)
    print(f"DEBUG: Token verification - unloading_id: {unloading_id}, warehouse_id: {warehouse_id}")
    
    if not unloading_id:
        return jsonify({'success': False, 'message': 'توکن نامعتبر یا منقضی شده است.'}), 400
    
    unloading = Unloading.query.get(unloading_id)
    if not unloading:
        return jsonify({'success': False, 'message': 'تخلیه یافت نشد.'}), 404
    
    print(f"DEBUG: Found unloading - id: {unloading.id}, status: {unloading.status}, version: {unloading.version}")
    
    if unloading.status not in ['completed', 'vision', 'edited']:
        return jsonify({'success': False, 'message': 'فقط تخلیه‌های تکمیل شده یا در انتظار تایید قابل ویرایش هستند.'}), 400
    
    try:
        # چک کردن زمان ویرایش بر اساس آخرین ویرایش
        now = datetime.utcnow()
        last_edit_time = unloading.edit_time or unloading.user_confirm_time
        edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
        
        print(f"DEBUG: Edit time check - last_edit_time: {last_edit_time}, edit_deadline: {edit_deadline}, now: {now}")
        
        # اگر زمان ویرایش تمام شده، edit_time رو ریست کن تا کاربر بتونه دوباره ذخیره کنه
        if now > edit_deadline:
            unloading.edit_time = now
            last_edit_time = now
            print(f"DEBUG: Reset edit_time to: {now}")
        
        # افزایش version و تغییر status بر اساس وضعیت فعلی
        unloading.version += 1
        print(f"DEBUG: Increased version to: {unloading.version}")
        
        # تغییر status بر اساس وضعیت فعلی
        if unloading.status == 'vision':
            unloading.status = 'completed'
        elif unloading.status == 'completed':
            unloading.status = 'edited'
        
        print(f"DEBUG: Status changed to: {unloading.status}")
        
        # حذف آیتم‌های user از نسخه فعلی (نه همه نسخه‌ها)
        deleted_count = UnloadingItem.query.filter_by(unloading_id=unloading.id, source='user', version=unloading.version).delete()
        print(f"DEBUG: Deleted {deleted_count} user items for version {unloading.version}")
        
        # اضافه کردن آیتم‌های جدید
        added_count = 0
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
            added_count += 1
        
        print(f"DEBUG: Added {added_count} new items")
        
        # ذخیره همه تغییرات در یک commit
        db.session.commit()
        print(f"DEBUG: Database commit successful")
        
        # محاسبه زمان ویرایش بر اساس آخرین edit_time (که ممکنه ریست شده باشه)
        edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
        can_edit = datetime.utcnow() < edit_deadline
        remaining_minutes = max(0, int((edit_deadline - datetime.utcnow()).total_seconds() / 60))
        
        print(f"DEBUG: Final calculation - can_edit: {can_edit}, remaining_minutes: {remaining_minutes}")
        
        return jsonify({
            'success': True, 
            'message': 'تخلیه با موفقیت ویرایش شد!',
            'version': unloading.version,
            'can_edit': can_edit,
            'remaining_minutes': remaining_minutes
        })
        
    except Exception as e:
        # Rollback در صورت بروز خطا
        db.session.rollback()
        print(f"ERROR in unloading edit: {str(e)}")
        print(f"DEBUG: Rollback completed")
        return jsonify({'success': False, 'message': f'خطا در ویرایش تخلیه: {str(e)}'}), 500

@unloading_bp.route('/api/unloadings/<token>', methods=['GET'])
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
            
            # اگر زمان ویرایش تمام شده، edit_time رو ریست کن تا کاربر بتونه دوباره ویرایش کنه
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
        print(f"DEBUG: unloading.items count: {len(unloading.items) if unloading.items else 0}")
        print(f"DEBUG: unloading.items: {unloading.items}")
        
        for item in unloading.items:
            print(f"DEBUG: Processing item: {item.name}, {item.type}, {item.count}, {item.source}, {item.version}")
            items.append({
                'name': item.name,
                'type': item.type,
                'count': item.count,
                'source': item.source,
                'version': item.version
            })
        
        print(f"DEBUG: Final items array: {items}")
        
        # Get shipment info if available
        shipment_info = None
        if unloading.shipment_id:
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
                        'total_with_vat': shipment.total_with_vat,
                        'notes': shipment.notes
                    }
            except Exception as e:
                print(f"Error loading shipment info: {e}")
        
        return jsonify({
            'success': True,
            'data': {
                'id': unloading.id,
                'warehouse_id': unloading.warehouse_id,
                'shipment_id': unloading.shipment_id,
                'start_time': unloading.start_time.isoformat() if unloading.start_time else None,
                'end_time': unloading.end_time.isoformat() if unloading.end_time else None,
                'user_confirm_time': unloading.user_confirm_time.isoformat() if unloading.user_confirm_time else None,
                'edit_time': unloading.edit_time.isoformat() if unloading.edit_time else None,
                'status': unloading.status,
                'version': unloading.version,
                'vision_output': unloading.vision_output,
                'items': items,
                'can_edit': can_edit,
                'remaining_minutes': remaining_minutes,
                'shipment_info': shipment_info
            }
        })
        
    except Exception as e:
        print(f"Error in api_unloading_by_token: {e}")
        return jsonify({'success': False, 'message': 'خطا در دریافت اطلاعات تخلیه'}), 500

@unloading_bp.route('/api/unloadings/<token>/shipment', methods=['GET'])
def api_unloading_shipment_by_token(token):
    """Get shipment info for unloading by token"""
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
        
        # Get shipment info if available
        shipment_info = None
        if unloading.shipment_id:
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
                        'total_with_vat': shipment.total_with_vat,
                        'notes': shipment.notes
                    }
            except Exception as e:
                print(f"Error loading shipment info: {e}")
        
        if not shipment_info:
            return jsonify({'success': False, 'message': 'اطلاعات محموله یافت نشد.'}), 404
        
        return jsonify({
            'success': True,
            'data': shipment_info
        })
        
    except Exception as e:
        print(f"Error in api_unloading_shipment_by_token: {e}")
        return jsonify({'success': False, 'message': 'خطا در دریافت اطلاعات محموله'}), 500

@unloading_bp.route('/api/unloadings/<int:unloading_id>/items', methods=['GET'])
def api_unloading_items(unloading_id):
    """Get detailed items for a specific unloading"""
    try:
        unloading = Unloading.query.get(unloading_id)
        if not unloading:
            return jsonify({'success': False, 'message': 'تخلیه یافت نشد'}), 404
        
        # Check if all_versions parameter is provided
        all_versions = request.args.get('all_versions', 'false').lower() == 'true'
        
        items = []
        
        if all_versions:
            # Get all items for all versions
            for item in unloading.items:
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
            latest_version = unloading.version
            
            if latest_version > 1:
                # Get user items from latest version
                for item in unloading.items:
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
                for item in unloading.items:
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
            'unloading_id': unloading_id,
            'items': items,
            'count': len(items),
            'start_time': unloading.start_time.isoformat() if unloading.start_time else None,
            'end_time': unloading.end_time.isoformat() if unloading.end_time else None,
            'user_confirm_time': unloading.user_confirm_time.isoformat() if unloading.user_confirm_time else None,
            'edit_time': unloading.edit_time.isoformat() if unloading.edit_time else None,
            'vision_output': unloading.vision_output
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت آیتم‌های تخلیه: {str(e)}'}), 500
