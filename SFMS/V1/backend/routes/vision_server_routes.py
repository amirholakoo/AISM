from flask import Blueprint, request, jsonify
from models import db, VisionServer, Warehouse

vision_server_bp = Blueprint('vision_server', __name__)

@vision_server_bp.route('/api/vision-servers', methods=['GET'])
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
                'video_source': server.video_source,
                'warehouse_ids': warehouse_ids,
                # Additional fields for frontend compatibility
                'operation_type': server.type,
                'is_enabled': server.is_active,
                'is_available': server.is_active,
                'location': server.persian_name or server.name
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

@vision_server_bp.route('/api/vision-servers', methods=['POST'])
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
            is_active=data.get('is_active', True),
            video_source=data.get('video_source', 'picamera')
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
                'is_active': new_vision_server.is_active,
                'video_source': new_vision_server.video_source
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@vision_server_bp.route('/api/vision-servers/<int:vision_server_id>', methods=['PUT'])
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
        if 'video_source' in data:
            vision_server.video_source = data['video_source']
        
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
                'is_active': vision_server.is_active,
                'video_source': vision_server.video_source
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@vision_server_bp.route('/api/vision-servers/<int:vision_server_id>', methods=['DELETE'])
def api_vision_servers_delete(vision_server_id):
    """Delete a vision server and all its warehouse assignments"""
    try:
        vision_server = VisionServer.query.get_or_404(vision_server_id)
        
        # Get warehouse assignments before deletion for logging
        warehouse_names = [w.name for w in vision_server.warehouses]
        
        # Clear all warehouse assignments for this vision server
        # This will automatically remove records from warehouse_vision_server table
        vision_server.warehouses.clear()
        
        # Delete the vision server
        db.session.delete(vision_server)
        db.session.commit()
        
        # Log the deletion
        print(f"🗑️  Deleted vision server '{vision_server.name}' (ID: {vision_server_id})")
        if warehouse_names:
            print(f"   - Removed assignments from warehouses: {', '.join(warehouse_names)}")
        
        return jsonify({
            'success': True,
            'message': 'Vision server deleted successfully',
            'deleted_assignments': len(warehouse_names)
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error deleting vision server {vision_server_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@vision_server_bp.route('/api/vision-servers/assignments', methods=['GET'])
def api_vision_servers_assignments_get():
    """Get all warehouse assignments for vision servers"""
    try:
        # Get all vision servers with their warehouse assignments
        vision_servers = VisionServer.query.all()
        
        assignments = {}
        for server in vision_servers:
            warehouse_ids = [w.id for w in server.warehouses]
            assignments[str(server.id)] = warehouse_ids
        
        print(f"🔍 GET assignments - Found {len(assignments)} server assignments:")
        for server_id, warehouse_ids in assignments.items():
            print(f"   Server {server_id}: {warehouse_ids}")
        
        return jsonify(assignments)
        
    except Exception as e:
        print(f"❌ Error getting assignments: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@vision_server_bp.route('/api/vision-servers/assignments', methods=['POST'])
def api_vision_servers_assignments():
    """Update warehouse assignments for vision servers"""
    try:
        data = request.get_json()
        assignments = data.get('assignments', {})
        
        print(f"💾 POST assignments - Received: {assignments}")
        
        # Clear all existing assignments
        db.session.execute(db.text('DELETE FROM warehouse_vision_server'))
        print("🗑️  Cleared existing assignments")
        
        # Add new assignments
        for server_id, warehouse_ids in assignments.items():
            if warehouse_ids:  # Only process if there are warehouse IDs
                vision_server = VisionServer.query.get(int(server_id))
                if vision_server:
                    print(f"   Adding server {server_id} to warehouses: {warehouse_ids}")
                    for warehouse_id in warehouse_ids:
                        warehouse = Warehouse.query.get(str(warehouse_id))
                        if warehouse:
                            vision_server.warehouses.append(warehouse)
                            print(f"     ✓ Server {server_id} -> Warehouse {warehouse_id}")
                        else:
                            print(f"     ❌ Warehouse {warehouse_id} not found")
                else:
                    print(f"   ❌ Vision server {server_id} not found")
        
        db.session.commit()
        print("✅ Assignments saved to database")
        
        return jsonify({
            'success': True,
            'message': 'Warehouse assignments updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error saving assignments: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@vision_server_bp.route('/api/vision-servers/warehouse/<warehouse_id>', methods=['GET'])
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

@vision_server_bp.route('/api/warehouses', methods=['GET'])
def api_warehouses():
    """Get all warehouses"""
    try:
        warehouses = Warehouse.query.all()
        
        result = []
        for warehouse in warehouses:
            result.append({
                'id': warehouse.id,
                'name': warehouse.name,
                'persian_name': warehouse.persian_name,
                'is_active': warehouse.is_active
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

@vision_server_bp.route('/api/test', methods=['GET'])
def api_test():
    """Test endpoint to verify the API is working"""
    try:
        from datetime import datetime
        return jsonify({
            'success': True,
            'message': 'Vision server API is working correctly',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
