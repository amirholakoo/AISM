from flask import Blueprint, request, jsonify
from ssh_operations import test_ssh_connection, copy_database_via_ssh, list_remote_database_files

ssh_bp = Blueprint('ssh', __name__)

@ssh_bp.route('/api/ssh/test', methods=['GET'])
def api_ssh_test():
    """Test SSH connection to remote server"""
    try:
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

@ssh_bp.route('/api/ssh/copy-database', methods=['POST'])
def api_ssh_copy_database():
    """Copy database file from remote server via SSH"""
    try:
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

@ssh_bp.route('/api/ssh/list-files', methods=['GET'])
def api_ssh_list_files():
    """List files in remote server home directory"""
    try:
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

@ssh_bp.route('/api/ssh/health', methods=['GET'])
def api_ssh_health():
    """Check remote server health via SSH"""
    try:
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
