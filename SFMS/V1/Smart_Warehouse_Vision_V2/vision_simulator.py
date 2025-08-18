"""
Smart Warehouse CV API - Backend Version
Compatible with the new Flask API structure from vision/flask_api.py
"""

import json
import threading
import time
import random
from datetime import datetime
from typing import Dict, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

# Global variables for warehouse configuration
PORT = None
WAREHOUSE_ID = None
WAREHOUSE_LOCATIONS = {
    1: "انبار سنگین",
    2: "انبار سبک", 
    3: "انبار بسته‌بندی",
    4: "انبار ذخیره‌سازی"
}

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global processing state
processing_status = {
    "is_running": False,
    "start_time": None,
    "current_session": None
}

def get_products_from_database():
    """
    Get product types from database or return default products for simulation.
    Returns a list of product names that can be used in simulation.
    """
    # برای شبیه‌سازی، محصولات ثابت برمی‌گردانیم
    # در حالت واقعی، این تابع از دیتابیس محصولات را می‌خواند
    return ['neshaste', 'pack']

# Simulated video processor state
simulated_processor = {
    "is_running": False,
    "tracker": {
        "counts": {"in": 0, "out": 0},
        "events": []
    }
}

def create_simulated_summary():
    """Create a simulated session summary similar to the real VideoProcessor."""
    if not simulated_processor["tracker"]["events"]:
        return {
            "total_products": 0,
            "operation_type": "none",
            "start_time": "N/A",
            "end_time": "N/A",
            "location": WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "Unknown"),
            "detailed_product_counts": {"loaded": {}, "unloaded": {}},
            "events": {}
        }
    
    events = simulated_processor["tracker"]["events"]
    counts = simulated_processor["tracker"]["counts"]
    
    # Calculate detailed product counts - ساختار واقعی
    detailed_product_counts = {"loaded": {}, "unloaded": {}}
    for event in events:
        status = event[1]  # 'loaded' or 'unloaded'
        product_type = event[4]  # product type
        
        if status in detailed_product_counts:
            detailed_product_counts[status][product_type] = detailed_product_counts[status].get(product_type, 0) + 1
    
    # Determine operation type
    loaded_count = counts.get('in', 0)
    unloaded_count = counts.get('out', 0)
    total_products = max(loaded_count, unloaded_count)
    
    if loaded_count > unloaded_count:
        operation_type = "loaded"
    elif unloaded_count > loaded_count:
        operation_type = "unloaded"
    else:
        operation_type = "balanced" if loaded_count > 0 else "none"
    
    # Format events for output - ساختار دقیق سرور اصلی
    formatted_events = {}
    for i, event in enumerate(events):
        formatted_events[str(i)] = {
            "timestamp": event[0],
            "status": event[1],
            "track_id": event[2],
            "location": event[3],
            "product_type": event[4]
        }
    
    return {
        "total_products": total_products,
        "operation_type": operation_type,
        "start_time": events[0][0] if events else "N/A",
        "end_time": events[-1][0] if events else "N/A",
        "location": WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "Unknown"),
        "detailed_product_counts": detailed_product_counts,
        "events": formatted_events
    }

def simulate_processing_loop():
    """Simulate the video processing loop in a separate thread."""
    # Get products from database
    product_types = get_products_from_database()
    track_id_counter = 0  # شروع از 0 مثل سرور اصلی
    
    # برای تخلیه، neshaste تعداد 2 تا و pack مقدار 5 تا تولید کن
    neshaste_count = 0
    pack_count = 0
    
    while simulated_processor["is_running"]:
        # Simulate events - neshaste: 2, pack: 5
        if random.random() < 0.6:  # 60% chance of event per iteration
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # برای تخلیه، neshaste و pack را به ترتیب تولید کن
            if neshaste_count < 2:
                status = 'unloaded'
                product_type = 'neshaste'
                neshaste_count += 1
            elif pack_count < 5:
                status = 'loaded'
                product_type = 'pack'
                pack_count += 1
            else:
                # اگر همه آیتم‌ها تولید شدند، متوقف شو
                break
            
            location = WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "Unknown")
            
            # Create event
            event = (timestamp, status, track_id_counter, location, product_type)
            simulated_processor["tracker"]["events"].append(event)
            
            # Update counts
            if status == 'loaded':
                simulated_processor["tracker"]["counts"]["in"] += 1
            else:
                simulated_processor["tracker"]["counts"]["out"] += 1
            
            track_id_counter += 1
        
        time.sleep(1.0)  # Simulate processing delay
    
    simulated_processor["is_running"] = False

@app.route('/start', methods=['POST'])
def start_processing():
    """
    Start video processing with provided parameters.
    
    Expected JSON payload:
    {
        "source": "output_filtered_3.mp4",
        "weights": "pallete_yolov8n.pt",
        "line_x": 900,
        "frame_skip": 3,
        "iou_thresh": 0.3,
        "conf_thresh": 0.3,
        "location": "انبار سنگین",
        "model_input_size": 1280
    }
    """
    global processing_status, simulated_processor
    
    try:
        # Check if already running
        if processing_status["is_running"]:
            return jsonify({
                "success": False,
                "message": "Processing is already running",
                "status": processing_status
            }), 400
        
        # Get parameters from request
        data = request.get_json() or {}
        
        # Use default values if not provided
        source = data.get("source", "output_filtered_3.mp4")
        weights = data.get("weights", "pallete_yolov8n.pt")
        line_x = data.get("line_x", 900)
        frame_skip = data.get("frame_skip", 3)
        iou_thresh = data.get("iou_thresh", 0.3)
        conf_thresh = data.get("conf_thresh", 0.3)
        location = data.get("location", WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"))
        model_input_size = data.get("model_input_size", 1280)
        
        # Reset simulated processor
        simulated_processor["tracker"]["counts"] = {"in": 0, "out": 0}
        simulated_processor["tracker"]["events"] = []
        simulated_processor["is_running"] = True
        
        # Start processing in daemon thread
        threading.Thread(target=simulate_processing_loop, daemon=True).start()
        
        # Update status
        processing_status["is_running"] = True
        processing_status["start_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        processing_status["current_session"] = {
            "source": source,
            "location": location,
            "line_x": line_x,
            "frame_skip": frame_skip,
            "iou_thresh": iou_thresh,
            "conf_thresh": conf_thresh,
            "model_input_size": model_input_size,
            "warehouse_id": WAREHOUSE_ID
        }
        
        return jsonify({
            "success": True,
            "message": "Video processing started successfully",
            "status": processing_status
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to start processing: {str(e)}",
            "status": processing_status
        }), 500

@app.route('/stop', methods=['POST'])
def stop_processing():
    """
    Stop video processing and return session summary.
    
    Returns the same summary format as the real VideoProcessor.
    """
    global processing_status, simulated_processor
    
    try:
        # Check if not running
        if not processing_status["is_running"]:
            return jsonify({
                "success": False,
                "message": "No processing session is currently running",
                "status": processing_status
            }), 400
        
        # Stop processing
        simulated_processor["is_running"] = False
        time.sleep(0.5)  # Allow processing loop to finish
        
        # اگر هیچ رویدادی وجود ندارد، رویدادهای تست اضافه کن
        if not simulated_processor["tracker"]["events"]:
            # neshaste تعداد 2 تا و pack مقدار 5 تا
            test_events = [
                ('2025-07-17 15:52:53', 'unloaded', 0, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'pack'),
                ('2025-07-17 15:52:54', 'loaded', 1, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'neshaste'),

            ]
            simulated_processor["tracker"]["events"] = test_events
            simulated_processor["tracker"]["counts"] = {"in": 5, "out": 2}
        
        # Get summary
        summary = create_simulated_summary()
        
        # Update status
        processing_status["is_running"] = False
        processing_status["start_time"] = None
        processing_status["current_session"] = None
        
        return jsonify({
            "success": True,
            "message": "Video processing stopped successfully",
            "summary": summary,
            "status": processing_status
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to stop processing: {str(e)}",
            "status": processing_status
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """
    Get current processing status and statistics.
    """
    global processing_status, simulated_processor
    
    try:
        status_info = {
            "is_running": processing_status["is_running"],
            "start_time": processing_status["start_time"],
            "current_session": processing_status["current_session"]
        }
        
        # Add live statistics if running
        if processing_status["is_running"] and simulated_processor["is_running"]:
            counts = dict(simulated_processor["tracker"]["counts"])
            events = list(simulated_processor["tracker"]["events"])
            
            status_info["live_stats"] = {
                "counts": counts,
                "total_events": len(events),
                "latest_events": events[-5:] if events else []  # Last 5 events
            }
        
        return jsonify({
            "success": True,
            "status": status_info
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to get status: {str(e)}",
            "status": processing_status
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "service": f"Smart Warehouse CV API - Warehouse {WAREHOUSE_ID}",
        "warehouse_location": WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "Unknown")
    }), 200

@app.route('/warehouse_info', methods=['GET'])
def get_warehouse_info():
    """
    Get warehouse information.
    """
    return jsonify({
        "warehouse_id": WAREHOUSE_ID,
        "warehouse_location": WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "Unknown"),
        "port": PORT,
        "available_locations": WAREHOUSE_LOCATIONS
    }), 200

@app.route('/test_summary', methods=['GET'])
def test_summary():
    """
    Test endpoint to generate a sample summary for testing.
    Returns the exact same structure as the main server.
    """
    # Get products from database for test events
    product_types = get_products_from_database()
    
    # Create some test events - ساختار دقیق سرور اصلی
    # neshaste تعداد 2 تا و pack مقدار 5 تا
    test_events = [
        ('2025-07-17 15:52:54', 'unloaded', 1, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'neshaste'),
        ('2025-07-17 15:53:25', 'loaded', 3, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'pack'),
    ]
    
    # Set test data - neshaste: 2, pack: 5
    simulated_processor["tracker"]["events"] = test_events
    simulated_processor["tracker"]["counts"] = {"in": 5, "out": 2}
    
    # Generate summary
    summary = create_simulated_summary()
    
    return jsonify({
        "success": True,
        "message": "Video processing stopped successfully",
        "status": {
            "current_session": None,
            "is_running": False,
            "start_time": None
        },
        "summary": summary
    }), 200

@app.route('/unloading_test', methods=['GET'])
def unloading_test():
    """
    Test endpoint specifically for unloading operations.
    Returns neshaste: 2, pack: 5 for testing.
    """
    # Create test events for unloading
    test_events = [
        ('2025-07-17 15:52:53', 'unloaded', 0, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'neshaste'),
        ('2025-07-17 15:52:54', 'unloaded', 1, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'neshaste'),
        ('2025-07-17 15:53:24', 'loaded', 2, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'pack'),
        ('2025-07-17 15:53:25', 'loaded', 3, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'pack'),
        ('2025-07-17 15:53:26', 'loaded', 4, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'pack'),
        ('2025-07-17 15:53:27', 'loaded', 5, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'pack'),
        ('2025-07-17 15:53:28', 'loaded', 6, WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID if WAREHOUSE_ID is not None else 1, "انبار سنگین"), 'pack'),
    ]
    
    # Set test data
    simulated_processor["tracker"]["events"] = test_events
    simulated_processor["tracker"]["counts"] = {"in": 5, "out": 2}
    
    # Generate summary
    summary = create_simulated_summary()
    
    return jsonify({
        "success": True,
        "message": "Unloading test data generated successfully",
        "summary": summary,
        "test_data": {
            "neshaste_count": 2,
            "pack_count": 5,
            "total_unloaded": 2,
            "total_loaded": 5
        }
    }), 200

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Smart Warehouse CV API - Backend Version')
    parser.add_argument('--port', type=int, default=5005, help='Port number to run the server on')
    parser.add_argument('--id', type=int, required=True, help='Warehouse ID (required)')
    args = parser.parse_args()
    
    PORT = args.port
    WAREHOUSE_ID = args.id
    
    print("🚀 Starting Smart Warehouse CV Backend API...")
    print(f"📡 Warehouse ID: {WAREHOUSE_ID}")
    print(f"📍 Location: {WAREHOUSE_LOCATIONS.get(WAREHOUSE_ID, 'Unknown')}")
    print(f"🌐 Port: {PORT}")
    print("\n📡 Available endpoints:")
    print("   POST /start - Start video processing")
    print("   POST /stop  - Stop video processing")
    print("   GET  /status - Get current status")
    print("   GET  /health - Health check")
    print("   GET  /warehouse_info - Get warehouse information")
    print("   GET  /test_summary - Generate test summary")
    print("   GET  /unloading_test - Generate unloading test data (neshaste: 2, pack: 5)")
    print(f"\n🌐 Server will start on http://localhost:{PORT}")
    print("\n🧪 Test the API:")
    print(f"   curl http://localhost:{PORT}/test_summary")
    print(f"   curl http://localhost:{PORT}/unloading_test")
    print(f"   curl -X POST http://localhost:{PORT}/start")
    print(f"   curl -X POST http://localhost:{PORT}/stop")
    
    app.run(host='0.0.0.0', port=PORT, debug=True) 