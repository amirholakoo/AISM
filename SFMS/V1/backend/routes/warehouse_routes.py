from flask import Blueprint, request, jsonify
from models import db, Warehouse
from models.database import get_warehouse_tables_from_database

warehouse_bp = Blueprint('warehouse', __name__)

@warehouse_bp.route('/api/warehouses', methods=['GET'])
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

@warehouse_bp.route('/api/warehouses', methods=['POST'])
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

@warehouse_bp.route('/api/warehouses/sync', methods=['POST'])
def api_warehouses_sync():
    """Sync warehouses from external database"""
    try:
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

@warehouse_bp.route('/api/warehouses/<warehouse_id>', methods=['PUT'])
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

@warehouse_bp.route('/api/warehouses/<warehouse_id>', methods=['DELETE'])
def api_warehouse_delete(warehouse_id):
    """Delete warehouse"""
    try:
        warehouse = Warehouse.query.get_or_404(warehouse_id)
        
        # بررسی اینکه آیا انبار در حال استفاده هست
        from models import Unloading
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
