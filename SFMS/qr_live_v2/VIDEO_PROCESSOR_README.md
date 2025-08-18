# Video Processor for QR Code Scanner

## Overview
`video_processor.py` is a comprehensive video processing module designed specifically for QR code scanning. It provides a robust, multi-threaded solution for video capture, QR detection, and logging.

## Features

### 🎯 **Multi-Threaded Architecture**
- Separate frame grabber and processing threads
- Queue-based frame buffering
- Non-blocking frame processing

### 🔄 **Robust Video Capture**
- Automatic RTSP/HTTP stream detection
- Retry logic for connection failures
- Optimized for RTSP streams with FFMPEG backend

### 📊 **Real-Time Processing**
- Frame-by-frame QR code detection
- Performance metrics (FPS, detection time)
- Live status monitoring

### 🛡️ **Error Handling**
- Consecutive failure tracking
- Graceful degradation
- Automatic resource cleanup

### 📝 **Integrated Logging**
- Automatic QR code logging
- Session summaries
- Performance statistics

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Source  │───▶│  Frame Grabber  │───▶│  Frame Queue    │
│  (RTSP/Camera)  │    │     Thread      │    │   (Buffer)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   QR Scanner    │◀───│  Processing     │◀───│  Frame Queue    │
│   & Logger      │    │     Thread      │    │   (Buffer)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Usage

### Basic Usage
```python
from video_processor import VideoProcessor
from config.settings import QRScannerConfig

# Create configuration
config = QRScannerConfig()

# Create video processor
processor = VideoProcessor(config)

# Start processing
if processor.start_processing("rtsp://localhost:8554/cam1"):
    # Get status
    status = processor.get_status()
    print(f"Processing FPS: {status['processing_fps']}")
    
    # Stop processing
    summary = processor.stop_processing()
    print(f"Detected {summary['total_qr_codes_detected']} QR codes")
```

### Integration with Flask API
The video processor is automatically used by the Flask API:

```python
# In QRScanningThread
self.video_processor = VideoProcessor(config)
self.video_processor.start_processing(video_source)
```

## API Reference

### Methods

#### `start_processing(source: str) -> bool`
Starts video processing with the specified source.
- **Args**: `source` - Video source (RTSP URL, camera index, file path)
- **Returns**: `bool` - True if successful, False otherwise

#### `stop_processing() -> Dict`
Stops processing and returns session summary.
- **Returns**: `Dict` - Session summary with statistics

#### `get_status() -> Dict`
Gets current processing status.
- **Returns**: `Dict` - Current status information

#### `get_latest_frame() -> Optional[cv2.Mat]`
Gets the latest processed frame.
- **Returns**: `cv2.Mat` or `None` - Latest frame if available

### Status Information
```python
{
    "is_running": bool,
    "processing_fps": float,
    "detection_time_ms": float,
    "frames_processed": int,
    "qr_codes_detected": int,
    "queue_size": int,
    "has_latest_frame": bool
}
```

### Session Summary
```python
{
    "total_frames_processed": int,
    "total_qr_codes_detected": int,
    "average_fps": float,
    "average_detection_time_ms": float,
    "log_file": str,
    "session_duration": str
}
```

## Configuration

The video processor uses the following configuration from `QRScannerConfig`:

- **TIMEZONE**: For timestamp generation
- **RECONNECT_DELAY_SECONDS**: For connection retries
- **QR detection settings**: From QRScanner configuration

## Performance Optimizations

### RTSP Streams
- Uses FFMPEG backend for best performance
- TCP transport for reliability
- Optimized buffer settings

### Frame Processing
- Queue-based buffering prevents frame drops
- Separate threads for capture and processing
- Configurable buffer sizes

### Memory Management
- Automatic cleanup on stop
- Frame queue management
- Resource release on errors

## Error Handling

### Connection Failures
- Automatic retry with exponential backoff
- Multiple backend attempts
- Graceful fallback to alternative sources

### Processing Errors
- Consecutive failure tracking
- Automatic thread cleanup
- Error logging with context

### Resource Management
- Automatic cleanup on destruction
- Thread timeout handling
- Memory leak prevention

## Integration Points

### QR Scanner
- Integrates with `QRScanner` for detection
- Passes frames for processing
- Receives detection results

### JSON Logger
- Integrates with `JsonLogManager` for logging
- Automatic record creation
- Session finalization

### Flask API
- Used by `QRScanningThread`
- Provides status information
- Session management

## Threading Model

### Frame Grabber Thread
- Dedicated thread for video capture
- Non-blocking frame queuing
- Error recovery and retry logic

### Processing Thread
- Main QR detection loop
- Performance monitoring
- Status updates

### Thread Safety
- Queue-based communication
- Thread-safe status updates
- Proper cleanup and joining

## Monitoring and Debugging

### Logging
- Comprehensive logging with emojis
- Performance metrics
- Error tracking

### Status Monitoring
- Real-time FPS monitoring
- Queue size tracking
- Detection performance

### Debug Information
- Frame shape and source info
- Processing statistics
- Error context

## Best Practices

1. **Always call `stop_processing()`** to ensure proper cleanup
2. **Monitor queue size** to detect processing bottlenecks
3. **Use appropriate buffer sizes** for your use case
4. **Handle connection failures** gracefully
5. **Monitor performance metrics** for optimization

## Troubleshooting

### Common Issues

#### Connection Failures
- Check RTSP URL validity
- Verify network connectivity
- Try different backends

#### Low FPS
- Reduce buffer size
- Check processing performance
- Monitor queue size

#### Memory Issues
- Ensure proper cleanup
- Monitor frame queue size
- Check for memory leaks

### Debug Steps
1. Check connection status
2. Monitor processing FPS
3. Verify frame queue size
4. Review error logs
5. Test with different sources
