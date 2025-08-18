"""
Flask API for Smart Warehouse CV System
Provides remote control endpoints for starting and stopping video processing.
"""

import json
import threading
import time
import os
from datetime import datetime
from typing import Dict, Optional
import time

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config.warehouse_config import WarehouseConfig
from core.video_processor import VideoProcessor
from core.logging_config import setup_logging

# Set up logging
setup_logging()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global video processor instance
video_processor: Optional[VideoProcessor] = None
processing_status = {
    "is_running": False,
    "start_time": None,
    "current_session": None
}

def create_video_processor():
    """Create and return a new VideoProcessor instance."""
    return VideoProcessor()

@app.route('/start', methods=['POST'])
def start_processing():
    """
    Start video processing with provided parameters.
    
    Expected JSON payload:
    {
        "source": "output_filtered_3.mp4",  # or camera URL
        "weights": "final_yolov11n_640_100epoch.pt",
        "frame_skip": 1,
        "conf_thresh": 0.65,
        "location": "انبار تست",
        "model_input_size": 256
    }
    """
    global video_processor, processing_status
    
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
        source = data.get("source", "rtsp://127.0.0.1:8554/cam1")
        weights = data.get("weights", WarehouseConfig.WEIGHTS_DEFAULT)
        frame_skip = data.get("frame_skip", WarehouseConfig.FRAME_SKIP_DEFAULT)
        conf_thresh = data.get("conf_thresh", WarehouseConfig.CONF_THRESH_DEFAULT)
        location = data.get("location", "انبار تست")
        model_input_size = data.get("model_input_size", WarehouseConfig.MODEL_INPUT_SIZE_DEFAULT)

        # Create new video processor
        video_processor = create_video_processor()
        
        # Start processing and capture the result
        processing_result = video_processor.start_processing(
            source=source,
            weights_path=weights,
            frame_skip=frame_skip,
            conf_thresh=conf_thresh,
            location=location,
            model_input_size=model_input_size
        )

        # Check if processing started successfully
        if not processing_result:
            # Clean up
            processing_status["is_running"] = False
            processing_status["start_time"] = None
            processing_status["current_session"] = None
            video_processor = None
            
            return jsonify({
                "success": False,
                "message": "Failed to start video processing. Please check the video source, model weights, and system configuration.",
                "status": processing_status
            }), 500

        # Update status
        processing_status["is_running"] = True
        processing_status["start_time"] = datetime.now(WarehouseConfig.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
        processing_status["current_session"] = {
            "source": source,
            "location": location,
            "frame_skip": frame_skip,
            "conf_thresh": conf_thresh,
            "model_input_size": model_input_size
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
    
    Returns the same summary format as the Streamlit UI.
    """
    global video_processor, processing_status
    
    try:
        # Check if not running
        if not processing_status["is_running"] or video_processor is None:
            return jsonify({
                "success": False,
                "message": "No processing session is currently running",
                "status": processing_status
            }), 400
        
        # Stop processing and get summary
        summary = video_processor.stop_processing()
        
        # Update status
        processing_status["is_running"] = False
        processing_status["start_time"] = None
        processing_status["current_session"] = None
        
        # Clean up
        video_processor = None
        
        # Save summary to output file
        try:
            # Create outputs directory if it doesn't exist
            outputs_dir = "outputs"
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.now(WarehouseConfig.TIMEZONE).strftime('%Y%m%d_%H%M%S')
            filename = f"session_summary_{timestamp}.json"
            filepath = os.path.join(outputs_dir, filename)
            
            # Save summary to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            # Add file path to response
            summary["output_file"] = filepath
            
        except Exception as e:
            # Log the error but don't fail the request
            print(f"Warning: Failed to save summary to file: {e}")
        
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
    global video_processor, processing_status
    
    try:
        status_info = {
            "is_running": processing_status["is_running"],
            "start_time": processing_status["start_time"],
            "current_session": processing_status["current_session"]
        }
        
        # Add live statistics if running
        if processing_status["is_running"] and video_processor and video_processor.tracker:
            counts = dict(video_processor.tracker.counts)
            events = list(video_processor.tracker.events)
            
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
        "timestamp": datetime.now(WarehouseConfig.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'),
        "service": "Smart Warehouse CV API"
    }), 200

@app.route('/snapshots/<filename>', methods=['GET'])
def get_snapshot(filename):
    """
    Serve snapshot images from the snapshots directory.
    
    Args:
        filename: The name of the snapshot file to serve
        
    Returns:
        The image file if found, 404 if not found
    """
    try:
        # Ensure the snapshots directory exists
        snapshots_dir = "snapshots"
        if not os.path.exists(snapshots_dir):
            return jsonify({
                "success": False,
                "message": "Snapshots directory not found"
            }), 404
        
        # Check if the file exists
        filepath = os.path.join(snapshots_dir, filename)
        if not os.path.exists(filepath):
            return jsonify({
                "success": False,
                "message": f"Snapshot file '{filename}' not found"
            }), 404
        
        # Serve the file
        return send_from_directory(snapshots_dir, filename)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error serving snapshot: {str(e)}"
        }), 500

@app.route('/test_summary', methods=['GET'])
def test_summary():
    """
    Test endpoint that returns a sample summary for testing purposes.
    """
    sample_summary = {
        "total_products": 15,
        "operation_type": "loaded",
        "start_time": "2024-01-15 10:30:00",
        "end_time": "2024-01-15 11:45:00",
        "location": "انبار سنگین",
        "detailed_product_counts": {
            "loaded": {
                "sulfat": 8,
                "neshaste": 5,
                "pack_material": 2
            },
            "unloaded": {
                "sulfat": 3,
                "neshaste": 1
            }
        },
        "events": {
            "0": {
                "timestamp": "2024-01-15 10:35:12",
                "status": "loaded",
                "track_id": 1,
                "location": "انبار سنگین",
                "product_type": "sulfat"
            },
            "1": {
                "timestamp": "2024-01-15 10:42:33",
                "status": "loaded",
                "track_id": 2,
                "location": "انبار سنگین",
                "product_type": "neshaste"
            }
        }
    }
    
    return jsonify({
        "success": True,
        "message": "Test summary data",
        "summary": sample_summary
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Smart Warehouse CV Flask API...")
    print("📡 Available endpoints:")
    print("   POST /start - Start video processing")
    print("   POST /stop  - Stop video processing")
    print("   GET  /status - Get current status")
    print("   GET  /health - Health check")
    print("   GET  /snapshots/<filename> - Serve snapshot images")
    print("\n🌐 Server will start on http://localhost:5001")
    
    app.run(host='0.0.0.0', port=5001, debug=False) 

