from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import requests
from models import db, Loading, Unloading, Warehouse, VisionServer
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS

# Import all blueprints
from routes.vision_routes import vision_bp
from routes.warehouse_routes import warehouse_bp
from routes.product_routes import product_bp
from routes.loading_routes import loading_bp
from routes.unloading_routes import unloading_bp
from routes.shipment_routes import shipment_bp
from routes.operation_type_routes import operation_type_bp
from routes.vision_server_routes import vision_server_bp
from routes.ssh_routes import ssh_bp

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

# Register all blueprints
app.register_blueprint(vision_bp)
app.register_blueprint(warehouse_bp)
app.register_blueprint(product_bp)
app.register_blueprint(loading_bp)
app.register_blueprint(unloading_bp)
app.register_blueprint(shipment_bp)
app.register_blueprint(operation_type_bp)
app.register_blueprint(vision_server_bp)
app.register_blueprint(ssh_bp)

# Additional routes that don't fit into specific categories
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
        
        # Calculate total count for the requested operation type
        total_count = 0
        if operation_type == 'loading':
            total_count = Loading.query.count()
        elif operation_type == 'unloading':
            total_count = Unloading.query.count()
        else:
            # For empty operation_type, get total of both
            total_count = Loading.query.count() + Unloading.query.count()
        
        # Calculate pagination info
        total_pages = (total_count + per_page - 1) // per_page
        has_next = page < total_pages
        has_prev = page > 1
        
        # If no operations exist, return empty result
        if total_count == 0:
            return jsonify({
                'success': True,
                'operations': [],
                'count': 0,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': 0,
                    'pages': 0,
                    'has_next': False,
                    'has_prev': False,
                    'next_num': None,
                    'prev_num': None
                }
            })
        
        result = []
        
        if operation_type == 'loading':
            # Only loadings - simple pagination
            loadings = Loading.query.order_by(Loading.start_time.desc()).offset((page - 1) * per_page).limit(per_page).all()
            
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
                
        elif operation_type == 'unloading':
            # Only unloadings - simple pagination
            unloadings = Unloading.query.order_by(Unloading.start_time.desc()).offset((page - 1) * per_page).limit(per_page).all()
            
            for unloading in unloadings:
                # Get warehouse information
                warehouse = Warehouse.query.get(unloading.warehouse_id)
                warehouse_name = warehouse.persian_name if warehouse else f"انبار {unloading.warehouse_id}"
                
                # Calculate counts from the latest user version (or vision if no user version)
                latest_version = unloading.version
                latest_items = []
                
                if latest_version > 1:
                    # Get user items from latest version
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
        else:
            # Mixed operations - need to fetch all data first, then sort and paginate
            # For mixed operations, we need to fetch all data first, then sort and paginate
            # This is not ideal for performance but necessary for proper sorting across types
            
            # Get all loadings and unloadings (we'll paginate after sorting)
            loadings = Loading.query.order_by(Loading.start_time.desc()).all()
            unloadings = Unloading.query.order_by(Unloading.start_time.desc()).all()
            
            # Process loadings
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
            
            # Process unloadings
            for unloading in unloadings:
                # Get warehouse information
                warehouse = Warehouse.query.get(unloading.warehouse_id)
                warehouse_name = warehouse.persian_name if warehouse else f"انبار {unloading.warehouse_id}"
                
                # Calculate counts from the latest user version (or vision if no user version)
                latest_version = unloading.version
                latest_items = []
                
                if latest_version > 1:
                    # Get user items from latest version
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
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            result = result[start_idx:end_idx]
        
        final_pagination = {
            'page': page,
            'per_page': per_page,
            'total': total_count,
            'pages': total_pages,
            'has_next': has_next,
            'has_prev': has_prev,
            'next_num': page + 1 if has_next else None,
            'prev_num': page - 1 if has_prev else None
        }
        
        return jsonify({
            'success': True,
            'operations': result,
            'count': len(result),
            'pagination': final_pagination
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'خطا در دریافت لیست عملیات: {str(e)}'}), 500

# Debug endpoints
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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print('db.sqlite3 created or already exists.')
    app.run(debug=True, host='0.0.0.0', port=18888)
