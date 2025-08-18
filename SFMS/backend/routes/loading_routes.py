from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import hmac
import hashlib
import time
from models import db, Loading, LoadingItem, Warehouse, VisionServer
from config import SECRET_KEY, EDIT_WINDOW_MINUTES

loading_bp = Blueprint('loading', __name__)

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

@loading_bp.route('/api/loadings/last-completed', methods=['GET'])
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
        now = datetime.utcnow()
        
        # اگر زمان ویرایش تمام شده، edit_time رو ریست کن تا کاربر بتونه دوباره ویرایش کنه
        if now > edit_deadline:
            loading.edit_time = now
            last_edit_time = now
            db.session.commit()
        
        # محاسبه مجدد بعد از ریست
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

@loading_bp.route('/api/loadings/active', methods=['GET'])
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

@loading_bp.route('/api/loadings/<int:loading_id>/items', methods=['GET', 'POST'])
def api_loading_items(loading_id):
    """Get all items for a specific loading or add a new item"""
    if request.method == 'GET':
        try:
            loading = Loading.query.get(loading_id)
            if not loading:
                return jsonify({'success': False, 'message': 'بارگیری یافت نشد.'}), 404
            
            items = []
            for item in loading.items:
                items.append({
                    'id': item.id,
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
                'items': items
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'خطا در دریافت آیتم‌های بارگیری: {str(e)}'}), 500
    
    elif request.method == 'POST':
        try:
            loading = Loading.query.get(loading_id)
            if not loading:
                return jsonify({'success': False, 'message': 'بارگیری یافت نشد.'}), 404
            
            data = request.json
            if not data:
                return jsonify({'success': False, 'message': 'داده‌ای ارسال نشده است.'}), 400
            
            # Create new loading item
            new_item = LoadingItem()
            new_item.loading_id = loading_id
            new_item.name = data.get('name', '')
            new_item.type = data.get('type', 'loaded')
            new_item.count = int(data.get('count', 1))
            new_item.source = 'user'
            new_item.version = loading.version + 1
            
            # Additional fields
            new_item.reel_number = data.get('reel_number')
            new_item.width = data.get('width')
            new_item.gsm = data.get('gsm')
            new_item.length = data.get('length')
            new_item.breaks = data.get('breaks', 0)
            new_item.grade = data.get('grade')
            
            db.session.add(new_item)
            
            # Update loading version
            loading.version = new_item.version
            loading.edit_time = datetime.utcnow()
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'آیتم با موفقیت اضافه شد.',
                'item': {
                    'id': new_item.id,
                    'name': new_item.name,
                    'type': new_item.type,
                    'count': new_item.count,
                    'source': new_item.source,
                    'version': new_item.version,
                    'reel_number': new_item.reel_number,
                    'width': new_item.width,
                    'gsm': new_item.gsm,
                    'length': new_item.length,
                    'breaks': new_item.breaks,
                    'grade': new_item.grade
                }
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'خطا در اضافه کردن آیتم: {str(e)}'}), 500

@loading_bp.route('/api/loadings/items/<int:item_id>', methods=['PUT'])
def api_update_loading_item(item_id):
    """Update a loading item"""
    try:
        item = LoadingItem.query.get(item_id)
        if not item:
            return jsonify({'success': False, 'message': 'آیتم یافت نشد.'}), 404
        
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'داده‌ای ارسال نشده است.'}), 400
        
        # Update item fields
        item.name = data.get('name', item.name)
        item.type = data.get('type', item.type)
        item.count = int(data.get('count', item.count))
        item.reel_number = data.get('reel_number', item.reel_number)
        item.width = data.get('width', item.width)
        item.gsm = data.get('gsm', item.gsm)
        item.length = data.get('length', item.length)
        item.breaks = data.get('breaks', item.breaks)
        item.grade = data.get('grade', item.grade)
        
        # Update loading edit time
        if item.loading:
            item.loading.edit_time = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'آیتم با موفقیت بروزرسانی شد.',
            'item': {
                'id': item.id,
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
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در بروزرسانی آیتم: {str(e)}'}), 500

@loading_bp.route('/api/loadings/items/<int:item_id>', methods=['DELETE'])
def api_delete_loading_item(item_id):
    """Delete a loading item"""
    try:
        item = LoadingItem.query.get(item_id)
        if not item:
            return jsonify({'success': False, 'message': 'آیتم یافت نشد.'}), 404
        
        loading_id = item.loading_id
        
        # Delete the item
        db.session.delete(item)
        
        # Update loading edit time
        if item.loading:
            item.loading.edit_time = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'آیتم با موفقیت حذف شد.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در حذف آیتم: {str(e)}'}), 500

@loading_bp.route('/api/loadings/active-any', methods=['GET'])
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

@loading_bp.route('/api/loadings/list', methods=['GET'])
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

@loading_bp.route('/api/loadings/all', methods=['GET'])
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



@loading_bp.route('/api/loadings/<int:loading_id>/all-items', methods=['GET'])
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

@loading_bp.route('/api/loadings/<token>', methods=['GET'])
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
            now = datetime.utcnow()
            
            # اگر زمان ویرایش تمام شده، edit_time رو ریست کن تا کاربر بتونه دوباره ویرایش کنه
            if now > edit_deadline:
                loading.edit_time = now
                last_edit_time = now
                db.session.commit()
            
            # محاسبه مجدد بعد از ریست
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

@loading_bp.route('/api/loadings/<token>/shipment', methods=['GET'])
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

@loading_bp.route('/api/loadings/edit', methods=['PUT', 'POST'])
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
        
        if loading.status not in ['vision', 'edited', 'completed']:
            return jsonify({'success': False, 'message': 'بارگیری در وضعیت نامعتبر برای ویرایش است.'}), 400
        
        # Check if still within edit window
        last_edit_time = loading.edit_time or loading.user_confirm_time
        if last_edit_time:
            edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
            now = datetime.utcnow()
            
            # اگر زمان ویرایش تمام شده، edit_time رو ریست کن تا کاربر بتونه دوباره ویرایش کنه
            if now >= edit_deadline:
                loading.edit_time = now
                last_edit_time = now
                db.session.commit()
                # ادامه عملیات ویرایش
        
        # Don't delete previous user items - keep version history
        # LoadingItem.query.filter_by(loading_id=loading.id, source='user').delete()
        
        # Add new user items with new version
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

@loading_bp.route('/api/loadings/save', methods=['POST'])
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
        
        if loading.status not in ['vision', 'edited', 'completed']:
            return jsonify({'success': False, 'message': 'بارگیری در وضعیت نامعتبر برای ذخیره است.'}), 400
        
        # Check if still within edit window
        last_edit_time = loading.edit_time or loading.user_confirm_time
        if last_edit_time:
            edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
            now = datetime.utcnow()
            
            # اگر زمان ویرایش تمام شده، edit_time رو ریست کن تا کاربر بتونه دوباره ویرایش کنه
            if now >= edit_deadline:
                loading.edit_time = now
                last_edit_time = now
                db.session.commit()
                # ادامه عملیات ویرایش
        
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

@loading_bp.route('/api/operations/last-completed', methods=['GET'])
def api_operations_last_completed():
    """Get the last completed operation (loading or unloading) for editing"""
    try:
        from routes.unloading_routes import Unloading
        
        # Get the last completed loading
        last_loading = Loading.query.filter(
            Loading.status.in_(['completed', 'vision', 'edited'])
        ).order_by(Loading.user_confirm_time.desc()).first()
        
        # Get the last completed unloading
        last_unloading = Unloading.query.filter(
            Unloading.status.in_(['completed', 'vision', 'edited'])
        ).order_by(Unloading.user_confirm_time.desc()).first()
        
        # Determine which operation is more recent
        if not last_loading and not last_unloading:
            return jsonify({'success': False, 'message': 'هیچ عملیات تکمیل شده‌ای یافت نشد.'}), 404
        
        if not last_loading:
            # Only unloading exists
            operation_type = 'unloading'
            operation = last_unloading
            operation_time = last_unloading.user_confirm_time
        elif not last_unloading:
            # Only loading exists
            operation_type = 'loading'
            operation = last_loading
            operation_time = last_loading.user_confirm_time
        else:
            # Both exist, compare times
            if last_loading.user_confirm_time > last_unloading.user_confirm_time:
                operation_type = 'loading'
                operation = last_loading
                operation_time = last_loading.user_confirm_time
            else:
                operation_type = 'unloading'
                operation = last_unloading
                operation_time = last_unloading.user_confirm_time
        
        # Check if still within edit window based on last edit time
        can_edit = True
        remaining_minutes = 0
        
        # Use edit_time if available, otherwise use user_confirm_time
        last_edit_time = operation.edit_time or operation.user_confirm_time
        edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
        now = datetime.utcnow()
        
        # اگر زمان ویرایش تمام شده، edit_time رو ریست کن تا کاربر بتونه دوباره ویرایش کنه
        if now > edit_deadline:
            operation.edit_time = now
            last_edit_time = now
            db.session.commit()
        
        # محاسبه مجدد بعد از ریست
        edit_deadline = last_edit_time + timedelta(minutes=EDIT_WINDOW_MINUTES)
        can_edit = datetime.utcnow() < edit_deadline
        remaining_minutes = max(0, int((edit_deadline - datetime.utcnow()).total_seconds() / 60))
        
        # Get latest version items
        items = []
        latest_version = operation.version
        
        if latest_version > 1:
            # Get user items from latest version
            for item in operation.items:
                if item.source == 'user' and item.version == latest_version:
                    if operation_type == 'loading':
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
                    else:  # unloading
                        items.append({
                            'name': item.name,
                            'type': item.type,
                            'count': item.count
                        })
        else:
            # Get vision items
            for item in operation.items:
                if item.source == 'vision':
                    if operation_type == 'loading':
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
                    else:  # unloading
                        items.append({
                            'name': item.name,
                            'type': item.type,
                            'count': item.count
                        })
        
        # Create HMAC token
        operation_id = operation.id
        timestamp = int(time.time())
        data = f"{operation_id}.{operation.warehouse_id}.{timestamp}"
        token = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        operation_token = f"{data}.{token}"
        
        # Get shipment info
        shipment_info = None
        if operation.shipment_id:
            try:
                from models.database import get_db
                from models.external_db import Shipments
                
                db = next(get_db())
                shipment = db.query(Shipments).filter(Shipments.id == operation.shipment_id).first()
                
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
                print(f"Error getting shipment info: {e}")
                shipment_info = None
            finally:
                if 'db' in locals():
                    db.close()
        
        # Prepare response based on operation type
        if operation_type == 'loading':
            return jsonify({
                'success': True,
                'type': 'loading',
                'token': operation_token,
                'warehouse_id': operation.warehouse_id,
                'warehouse_name': operation.warehouse.persian_name or operation.warehouse.name,
                'warehouse_english_name': operation.warehouse.name,
                'shipment_id': operation.shipment_id,
                'shipment_info': shipment_info,
                'status': operation.status,
                'start_time': operation.start_time.isoformat() if operation.start_time else None,
                'end_time': operation.end_time.isoformat() if operation.end_time else None,
                'user_confirm_time': operation.user_confirm_time.isoformat() if operation.user_confirm_time else None,
                'edit_time': operation.edit_time.isoformat() if operation.edit_time else None,
                'version': operation.version,
                'vision_output': operation.vision_output,
                'items': items,
                'can_edit': can_edit,
                'remaining_minutes': remaining_minutes
            })
        else:  # unloading
            return jsonify({
                'success': True,
                'type': 'unloading',
                'token': operation_token,
                'warehouse_id': operation.warehouse_id,
                'warehouse_name': operation.warehouse.persian_name or operation.warehouse.name,
                'warehouse_english_name': operation.warehouse.name,
                'shipment_id': operation.shipment_id,
                'shipment_info': shipment_info,
                'status': operation.status,
                'version': operation.version,
                'items': items,
                'can_edit': can_edit,
                'remaining_minutes': remaining_minutes,
                'start_time': operation.start_time.isoformat() if operation.start_time else None,
                'end_time': operation.end_time.isoformat() if operation.end_time else None,
                'user_confirm_time': operation.user_confirm_time.isoformat() if operation.user_confirm_time else None,
                'edit_time': operation.edit_time.isoformat() if operation.edit_time else None,
                'vision_output': operation.vision_output
            })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت آخرین عملیات: {str(e)}'}), 500




