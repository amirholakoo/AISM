"""
QR Code Scanner Camera Simulator
Simulates QR code scanning camera for testing and development purposes.
Outputs the exact same structure as the real QR scanner API.

Usage:
python qr_vision_simulator.py --port 5006 --id 1
"""

import json
import threading
import time
import random
import os
from datetime import datetime
from typing import Dict, Optional, List
from flask import Flask, jsonify, request
from flask_cors import CORS
import pytz

# Global variables for configuration
PORT = None
CAMERA_ID = None
TIMEZONE = pytz.timezone('Asia/Tehran')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global scanning state
scanning_status = {
    "is_running": False,
    "start_time": None,
    "current_session": None,
    "detected_codes": [],
    "fps": 0.0
}

# Simulated QR scanner state
simulated_scanner = {
    "is_running": False,
    "detected_codes": [],
    "total_frames_processed": 0,
    "start_time": None,
    "fps_counter": 0,
    "last_fps_time": None
}

# Sample QR code contents for simulation
SAMPLE_QR_CONTENTS = [
    "Reel Number: 1482,\nWidth: 240,\nGSM: 130,\nLength:~ 6300,\nBreaks: 0,\nGrade: Testliner HOMAYOUN",
    "Reel Number: 1193,\nWidth: 240,\nGSM: 130,\nLength:~ 6300,\nBreaks: 0,\nGrade: Testliner HOMAYOUN",
    "Reel Number: 2156,\nWidth: 280,\nGSM: 150,\nLength:~ 7200,\nBreaks: 1,\nGrade: Kraftliner PREMIUM",
    "Reel Number: 892,\nWidth: 200,\nGSM: 120,\nLength:~ 5800,\nBreaks: 0,\nGrade: Testliner STANDARD",
    "Reel Number: 3341,\nWidth: 320,\nGSM: 180,\nLength:~ 8500,\nBreaks: 2,\nGrade: Kraftliner HEAVY",
    "TEST",
    "PRODUCTION_LINE_A",
    "QUALITY_CHECK_PASS",
    "WAREHOUSE_LOCATION_3",
    "BATCH_NUMBER_2025_001"
]

def create_log_content():
    """Create simulated log content similar to the JSON output format."""
    if not simulated_scanner["detected_codes"]:
        return {
            "run_start_time": "N/A",
            "run_finish_time": "N/A",
            "qrcodes": []
        }
    
    start_time = simulated_scanner["start_time"] or datetime.now(TIMEZONE)
    end_time = datetime.now(TIMEZONE)
    
    # Format detected codes for log
    qrcodes = []
    for code_data in simulated_scanner["detected_codes"]:
        qrcodes.append({
            "content": code_data["content"],
            "timestamp": code_data["timestamp"]
        })
    
    return {
        "run_start_time": start_time.strftime('%Y-%m-%dT%H:%M:%S.%f%z'),
        "run_finish_time": end_time.strftime('%Y-%m-%dT%H:%M:%S.%f%z'),
        "qrcodes": qrcodes
    }

def simulate_scanning_loop():
    """Simulate the QR code scanning loop in a separate thread."""
    frame_count = 0
    last_fps_update = time.time()
    
    while simulated_scanner["is_running"]:
        # Simulate frame processing
        frame_count += 1
        simulated_scanner["total_frames_processed"] += 1
        
        # Simulate random QR code detection (10% chance per frame)
        if random.random() < 0.1:  # 10% chance of detection
            timestamp = datetime.now(TIMEZONE).strftime('%Y-%m-%dT%H:%M:%S.%f%z')
            content = random.choice(SAMPLE_QR_CONTENTS)
            
            # Check if this content was already detected (avoid duplicates)
            if not any(code["content"] == content for code in simulated_scanner["detected_codes"]):
                detected_code = {
                    "content": content,
                    "timestamp": timestamp,
                    "confidence": random.uniform(0.85, 0.99),
                    "bbox": [random.randint(100, 500), random.randint(100, 400), 
                            random.randint(50, 150), random.randint(50, 150)]
                }
                
                simulated_scanner["detected_codes"].append(detected_code)
                scanning_status["detected_codes"].append(detected_code)
                
                print(f"🔍 Detected QR Code: {content[:50]}...")
        
        # Update FPS every second
        current_time = time.time()
        if current_time - last_fps_update >= 1.0:
            fps = frame_count / (current_time - last_fps_update)
            simulated_scanner["fps_counter"] = fps
            scanning_status["fps"] = round(fps, 2)
            frame_count = 0
            last_fps_update = current_time
        
        time.sleep(0.1)  # Simulate 10 FPS processing
    
    simulated_scanner["is_running"] = False

@app.route('/start', methods=['POST'])
def start_scanning():
    """
    Start QR code scanning with provided parameters.
    
    Expected JSON payload:
    {
        "video_source": "picamera"  # or any other supported source
    }
    """
    global scanning_status, simulated_scanner
    
    try:
        # Check if already running
        if scanning_status["is_running"]:
            return jsonify({
                "success": False,
                "message": "QR scanning is already running",
                "status": scanning_status
            }), 400
        
        # Get parameters from request
        data = request.get_json() or {}
        video_source = data.get("video_source", "picamera")
        warehouse_id = data.get("warehouse_id", "unknown")
        operation_type = data.get("operation_type", "unknown")
        
        print(f"🎥 Received start request:")
        print(f"  - Video source: {video_source}")
        print(f"  - Warehouse ID: {warehouse_id}")
        print(f"  - Operation type: {operation_type}")
        
        # Reset scanner state
        simulated_scanner["detected_codes"] = []
        simulated_scanner["total_frames_processed"] = 0
        simulated_scanner["start_time"] = datetime.now(TIMEZONE)
        simulated_scanner["is_running"] = True
        
        # Start scanning in daemon thread
        threading.Thread(target=simulate_scanning_loop, daemon=True).start()
        
        # Update status
        scanning_status["is_running"] = True
        scanning_status["start_time"] = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
        scanning_status["current_session"] = {
            "video_source": video_source,
            "camera_id": CAMERA_ID,
            "start_time": scanning_status["start_time"]
        }
        scanning_status["detected_codes"] = []
        scanning_status["fps"] = 0.0
        
        print(f"✅ QR scanning started successfully with source: {video_source}")
        
        return jsonify({
            "success": True,
            "message": "QR scanning started successfully",
            "status": scanning_status
        }), 200
        
    except Exception as e:
        print(f"❌ Error in start_scanning: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Failed to start scanning: {str(e)}",
            "status": scanning_status
        }), 500

@app.route('/stop', methods=['POST'])
def stop_scanning():
    """
    Stop QR code scanning and return session summary.
    Returns the exact same structure as the real QR scanner API.
    """
    global scanning_status, simulated_scanner
    
    try:
        # Check if not running
        if not scanning_status["is_running"]:
            return jsonify({
                "success": False,
                "message": "No scanning session is currently running",
                "status": scanning_status
            }), 400
        
        # Stop scanning
        simulated_scanner["is_running"] = False
        time.sleep(0.5)  # Allow scanning loop to finish
        
        # Calculate processing statistics
        if simulated_scanner["start_time"]:
            duration = (datetime.now(TIMEZONE) - simulated_scanner["start_time"]).total_seconds()
            total_frames = simulated_scanner["total_frames_processed"]
            average_fps = total_frames / duration if duration > 0 else 0
            average_detection_time_ms = 50.0  # Simulated average detection time
        else:
            average_fps = 0
            average_detection_time_ms = 0
            total_frames = 0
        
        # Create session summary
        session_summary = {
            "total_qr_codes_detected": len(simulated_scanner["detected_codes"]),
            "total_frames_processed": total_frames,
            "average_fps": round(average_fps, 2),
            "average_detection_time_ms": average_detection_time_ms,
            "log_file": f"qrcodes_{datetime.now(TIMEZONE).strftime('%Y%m%d_%H%M%S')}.json"
        }
        
        # Create summary using session data (exact format from flask_api.py)
        summary = {
            "total_codes_detected": session_summary["total_qr_codes_detected"],
            "start_time": scanning_status["start_time"],
            "end_time": datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'),
            "video_source": scanning_status["current_session"]["video_source"],
            "detected_codes": scanning_status["detected_codes"],
            "log_file": session_summary["log_file"],
            "log_content": create_log_content(),
            "processing_stats": {
                "total_frames_processed": session_summary["total_frames_processed"],
                "average_fps": session_summary["average_fps"],
                "average_detection_time_ms": session_summary["average_detection_time_ms"]
            }
        }
        
        # Update status
        scanning_status["is_running"] = False
        scanning_status["start_time"] = None
        scanning_status["current_session"] = None
        scanning_status["detected_codes"] = []
        scanning_status["fps"] = 0.0
        
        print("✅ QR scanning stopped successfully")
        
        return jsonify({
            "success": True,
            "message": "QR scanning stopped successfully",
            "summary": summary,
            "status": scanning_status
        }), 200
        
    except Exception as e:
        print(f"❌ Error in stop_scanning: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Failed to stop scanning: {str(e)}",
            "status": scanning_status
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """
    Get current scanning status and statistics.
    """
    global scanning_status, simulated_scanner
    
    try:
        status_info = {
            "is_running": scanning_status["is_running"],
            "start_time": scanning_status["start_time"],
            "current_session": scanning_status["current_session"],
            "fps": scanning_status["fps"],
            "total_codes_detected": len(scanning_status["detected_codes"]),
            "latest_codes": scanning_status["detected_codes"][-10:] if scanning_status["detected_codes"] else []  # Last 10 codes
        }
        
        # Add processing information if available
        if simulated_scanner["is_running"]:
            status_info["processor_status"] = {
                "total_frames_processed": simulated_scanner["total_frames_processed"],
                "current_fps": simulated_scanner["fps_counter"]
            }
        
        return jsonify({
            "success": True,
            "status": status_info
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to get status: {str(e)}",
            "status": scanning_status
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'),
        "service": f"QR Code Scanner Camera Simulator - Camera {CAMERA_ID}"
    }), 200

@app.route('/camera_info', methods=['GET'])
def get_camera_info():
    """
    Get camera information.
    """
    return jsonify({
        "camera_id": CAMERA_ID,
        "port": PORT,
        "timezone": str(TIMEZONE),
        "available_qr_samples": len(SAMPLE_QR_CONTENTS)
    }), 200

@app.route('/test_summary', methods=['GET'])
def test_summary():
    """
    Test endpoint to generate a sample summary for testing.
    Returns the exact same structure as the main server.
    """
    # Create some test detected codes
    test_codes = [
        {
            "content": "Reel Number: 1482,\nWidth: 240,\nGSM: 130,\nLength:~ 6300,\nBreaks: 0,\nGrade: Testliner HOMAYOUN",
            "timestamp": "2025-08-04T15:04:44.512821+03:30",
            "confidence": 0.95,
            "bbox": [200, 150, 120, 120]
        },
        {
            "content": "Reel Number: 1193,\nWidth: 240,\nGSM: 130,\nLength:~ 6300,\nBreaks: 0,\nGrade: Testliner HOMAYOUN",
            "timestamp": "2025-08-04T15:04:47.603510+03:30",
            "confidence": 0.92,
            "bbox": [350, 200, 110, 110]
        },
        {
            "content": "TEST",
            "timestamp": "2025-08-04T15:04:57.107927+03:30",
            "confidence": 0.88,
            "bbox": [450, 300, 80, 80]
        }
    ]
    
    # Set test data
    simulated_scanner["detected_codes"] = test_codes
    simulated_scanner["total_frames_processed"] = 150
    simulated_scanner["start_time"] = datetime.now(TIMEZONE)
    
    # Create session summary
    session_summary = {
        "total_qr_codes_detected": len(test_codes),
        "total_frames_processed": 150,
        "average_fps": 15.0,
        "average_detection_time_ms": 45.0,
        "log_file": "test_qrcodes.json"
    }
    
    # Create summary (exact format from flask_api.py)
    summary = {
        "total_codes_detected": session_summary["total_qr_codes_detected"],
        "start_time": "2025-08-04 15:04:25",
        "end_time": datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'),
        "video_source": "picamera",
        "detected_codes": test_codes,
        "log_file": session_summary["log_file"],
        "log_content": create_log_content(),
        "processing_stats": {
            "total_frames_processed": session_summary["total_frames_processed"],
            "average_fps": session_summary["average_fps"],
            "average_detection_time_ms": session_summary["average_detection_time_ms"]
        }
    }
    
    return jsonify({
        "success": True,
        "message": "QR scanning stopped successfully",
        "summary": summary,
        "status": {
            "current_session": None,
            "is_running": False,
            "start_time": None
        }
    }), 200

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='QR Code Scanner Camera Simulator')
    parser.add_argument('--port', type=int, default=5002, help='Port number to run the server on')
    parser.add_argument('--id', type=int, required=True, help='Camera ID (required)')
    args = parser.parse_args()
    
    PORT = args.port
    CAMERA_ID = args.id
    
    print("🚀 Starting QR Code Scanner Camera Simulator...")
    print(f"📷 Camera ID: {CAMERA_ID}")
    print(f"🌐 Port: {PORT}")
    print(f"⏰ Timezone: {TIMEZONE}")
    print("\n📡 Available endpoints:")
    print("   POST /start - Start QR scanning")
    print("   POST /stop  - Stop QR scanning")
    print("   GET  /status - Get current status")
    print("   GET  /health - Health check")
    print("   GET  /camera_info - Get camera information")
    print("   GET  /test_summary - Generate test summary")
    print(f"\n🌐 Server will start on http://localhost:{PORT}")
    print("\n🧪 Test the API:")
    print(f"   curl http://localhost:{PORT}/test_summary")
    print(f"   curl -X POST http://localhost:{PORT}/start")
    print(f"   curl -X POST http://localhost:{PORT}/stop")
    print(f"\n📋 Sample QR codes available: {len(SAMPLE_QR_CONTENTS)}")
    
    app.run(host='0.0.0.0', port=PORT, debug=True)
