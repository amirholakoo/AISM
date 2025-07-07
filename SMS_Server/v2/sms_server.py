"""
SMS Server with SIM800C Integration
Flask application providing SMS verification API endpoints with improved error handling
"""

from flask import Flask, request, jsonify
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler
from models import db, SMSVerification, SMSLog
from sim800 import controller as sim800c
from functools import wraps
from config import config
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sms_server')
handler = RotatingFileHandler('sms_server.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
logger.addHandler(handler)

def create_app(config_name='default'):
    """Application factory"""
app = Flask(__name__)
    app.config.from_object(config[config_name])
    app.logger.addHandler(handler)
    
    # Initialize extensions
    db.init_app(app)
    
    return app

app = create_app(os.getenv('FLASK_ENV', 'development'))

def require_api_key(f):
    """Decorator to require API key"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != app.config['API_KEY']:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def initialize_database():
    """Initialize database tables"""
    try:
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

def initialize_sim800c():
    """Initialize SIM800C connection with retry logic"""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logger.info(f"Initializing SIM800C (attempt {attempt + 1}/{max_attempts})")
            if sim800c.connect():
                logger.info("SIM800C initialized successfully")
                return
            else:
                logger.warning(f"SIM800C initialization attempt {attempt + 1} failed")
                if attempt < max_attempts - 1:
                    time.sleep(5)
        except Exception as e:
            logger.error(f"SIM800C initialization error on attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                time.sleep(5)
    
    logger.error("Failed to initialize SIM800C after all attempts")

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle unhandled exceptions"""
    logger.error(f"Unhandled error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def index():
    """Index page with server status"""
    try:
        # Get SIM status
        sim_status = "Ready" if sim800c.check_sim_status() else "Not Ready"
        signal_strength, signal_status = sim800c.get_signal_strength()
        
        return jsonify({
            'service': 'SMS Server',
            'status': 'running',
            'timestamp': datetime.utcnow().isoformat(),
            'sim_status': sim_status,
            'signal_strength': f"{signal_strength}% ({signal_status})",
            'endpoints': {
                'health_check': '/health',
                'send_verification': '/api/v1/verify/send',
                'check_verification': '/api/v1/verify/check',
                'server_status': '/api/v1/status'
            }
        })
    except Exception as e:
        logger.error(f"Index page error: {e}")
        return jsonify({
            'service': 'SMS Server',
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status"""
    try:
        sim_status = "Ready" if sim800c.check_sim_status() else "Not Ready"
        signal_strength, signal_status = sim800c.get_signal_strength()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'sim_status': sim_status,
            'signal_strength': f"{signal_strength}% ({signal_status})",
            'connection_status': 'connected' if sim800c.is_connected else 'disconnected'
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/v1/verify/send', methods=['POST'])
@require_api_key
def send_verification():
    """Send SMS verification with improved error handling and retry logic"""
    try:
        data = request.get_json()
        if not data or 'phone_number' not in data or 'message' not in data:
            return jsonify({'error': 'Missing phone_number or message'}), 400
        
        phone_number = data['phone_number']
        message = data['message']
        
        # Validate phone number format
        if not phone_number or len(phone_number) < 10:
            return jsonify({'error': 'Invalid phone number format'}), 400
        
        # Check if SIM800C is connected
        if not sim800c.is_connected:
            logger.warning("SIM800C not connected, attempting to reconnect...")
            if not sim800c.connect():
                logger.error("Failed to reconnect to SIM800C")
                SMSLog.log_sms(phone_number, message, 'failed', 'SIM800C connection failed')
                return jsonify({'error': 'SMS service unavailable'}), 503
        
        # Check signal strength before sending
        signal_strength, signal_status = sim800c.get_signal_strength()
        logger.info(f"Current signal strength: {signal_strength}% ({signal_status})")
        
        if signal_strength < 5:
            logger.warning(f"Signal too weak for SMS: {signal_strength}% ({signal_status})")
            SMSLog.log_sms(phone_number, message, 'failed', f'Signal too weak: {signal_strength}%')
            return jsonify({'error': 'Signal too weak for SMS sending'}), 503
        
        # Send SMS with retry logic
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Sending SMS attempt {attempt + 1}/{max_attempts}")
                
                if sim800c.send_sms(phone_number, message):
                    SMSLog.log_sms(phone_number, message, 'success')
                    return jsonify({
                        'success': True,
                        'message': 'SMS sent successfully',
                        'attempt': attempt + 1,
                        'signal_strength': f"{signal_strength}% ({signal_status})"
                    })
                else:
                    last_error = f"SMS sending failed on attempt {attempt + 1}"
                    logger.warning(last_error)
                    
                    if attempt < max_attempts - 1:
                        logger.info("Waiting 3 seconds before retry...")
                        time.sleep(3)
                        
                        # Check connection before retry
                        if not sim800c.is_connected:
                            logger.warning("Connection lost, attempting to reconnect...")
                            sim800c.connect()
                            
            except Exception as e:
                last_error = f"SMS sending error on attempt {attempt + 1}: {str(e)}"
                logger.error(last_error)
                
                if attempt < max_attempts - 1:
                    logger.info("Waiting 3 seconds before retry...")
                    time.sleep(3)
        
        # All attempts failed
        error_msg = last_error or 'Failed to send SMS after all attempts'
        SMSLog.log_sms(phone_number, message, 'failed', error_msg)
        return jsonify({'error': error_msg}), 500
        
    except Exception as e:
        logger.error(f"Error in send_verification: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/v1/verify/check', methods=['POST'])
@require_api_key
def verify_code():
    """Check verification code"""
    try:
        data = request.get_json()
        if not data or 'phone_number' not in data or 'code' not in data:
            return jsonify({'error': 'Missing phone_number or code'}), 400
        
        phone_number = data['phone_number']
        code = data['code']
        
        # Find verification record
        verification = SMSVerification.query.filter_by(
            phone_number=phone_number,
            code=code,
            is_verified=False
        ).first()
        
        if not verification:
            return jsonify({'error': 'Invalid verification code'}), 400
        
        # Check if code is expired
        if verification.is_expired():
            return jsonify({'error': 'Verification code expired'}), 400
        
        # Mark as verified
        verification.is_verified = True
        verification.verified_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Phone number verified successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in verify_code: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/v1/status', methods=['GET'])
@require_api_key
def get_status():
    """Get comprehensive SMS server status"""
    try:
        sim_status = "Ready" if sim800c.check_sim_status() else "Not Ready"
        signal_strength, signal_status = sim800c.get_signal_strength()
        
        # Get recent logs
        recent_logs = SMSLog.query.order_by(
            SMSLog.created_at.desc()
        ).limit(10).all()
        
        # Get statistics
        total_sms = SMSLog.query.count()
        successful_sms = SMSLog.query.filter_by(status='success').count()
        failed_sms = SMSLog.query.filter_by(status='failed').count()
        
        return jsonify({
            'sim_status': sim_status,
            'signal_strength': f"{signal_strength}% ({signal_status})",
            'connection_status': 'connected' if sim800c.is_connected else 'disconnected',
            'statistics': {
                'total_sms': total_sms,
                'successful_sms': successful_sms,
                'failed_sms': failed_sms,
                'success_rate': f"{(successful_sms/total_sms*100):.1f}%" if total_sms > 0 else "0%"
            },
            'recent_logs': [
                {
                    'phone_number': log.phone_number,
                    'status': log.status,
                    'error_message': log.error_message,
                    'created_at': log.created_at.isoformat()
                }
                for log in recent_logs
            ]
        })
    except Exception as e:
        logger.error(f"Error in get_status: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    import sys
    port = 5003
    
    # Check for port argument
    if len(sys.argv) > 1 and sys.argv[1] == '--port' and len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            port = 5003
    
    # Initialize database and SIM800C
    initialize_database()
    initialize_sim800c()
    
    app.run(host='0.0.0.0', port=port, debug=False)
