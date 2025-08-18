from flask import Blueprint, request, jsonify
from models.database import get_db, get_latest_loaded_unloaded_shipments, get_shipments_for_unloading, get_shipments_for_loading

shipment_bp = Blueprint('shipment', __name__)

@shipment_bp.route('/api/shipments/latest', methods=['GET'])
def api_shipments_latest():
    """Get latest shipments from external database"""
    try:
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

@shipment_bp.route('/api/shipments/for-unloading', methods=['GET'])
def api_shipments_for_unloading():
    """Get shipments for unloading operation (Incoming shipments with LoadingUnloading status)"""
    try:
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

@shipment_bp.route('/api/shipments/for-loading', methods=['GET'])
def api_shipments_for_loading():
    """Get shipments for loading operation (Outgoing shipments with LoadingUnloading status)"""
    try:
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

@shipment_bp.route('/api/shipments/<int:shipment_id>', methods=['GET'])
def api_shipment_detail(shipment_id):
    """Get specific shipment details"""
    try:
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
