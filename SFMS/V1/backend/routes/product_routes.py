from flask import Blueprint, request, jsonify
from models import db, Product

product_bp = Blueprint('product', __name__)

@product_bp.route('/api/products', methods=['GET'])
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

@product_bp.route('/api/products', methods=['POST'])
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

@product_bp.route('/api/products/<int:product_id>', methods=['PUT'])
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

@product_bp.route('/api/products/<int:product_id>', methods=['DELETE'])
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
