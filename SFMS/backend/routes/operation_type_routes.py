from flask import Blueprint, request, jsonify
from models import db, OperationType

operation_type_bp = Blueprint('operation_type', __name__)

@operation_type_bp.route('/api/operation-types', methods=['GET'])
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

@operation_type_bp.route('/api/operation-types', methods=['POST'])
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

@operation_type_bp.route('/api/operation-types/<int:operation_type_id>', methods=['PUT'])
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

@operation_type_bp.route('/api/operation-types/<int:operation_type_id>', methods=['DELETE'])
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
