# Vision Servers - Compatible with Main Server

## Overview

Both vision servers (`vision.py` and `vision_new.py`) have been updated to return the exact same output structure as the main server.

## Main Server Output Structure

The main server returns this structure:

```json
{
  "message": "Video processing stopped successfully",
  "status": {
    "current_session": null,
    "is_running": false,
    "start_time": null
  },
  "success": true,
  "summary": {
    "detailed_product_counts": {
      "loaded": {"neshaste": 1},
      "unloaded": {"neshaste": 1}
    },
    "end_time": "2025-07-17 15:53:24",
    "events": {
      "0": {
        "location": "انبار سنگین",
        "product_type": "neshaste",
        "status": "unloaded",
        "timestamp": "2025-07-17 15:52:53",
        "track_id": 0
      },
      "1": {
        "location": "انبار سنگین",
        "product_type": "neshaste",
        "status": "loaded",
        "timestamp": "2025-07-17 15:53:24",
        "track_id": 3
      }
    },
    "location": "انبار سنگین",
    "operation_type": "balanced",
    "start_time": "2025-07-17 15:52:53",
    "total_products": 1
  }
}
```

## Updated Servers

### 1. Old Vision Server (`vision.py`)
- **Port**: 5001 (as configured in seed.py)
- **Features**: Simple simulation with random events
- **Endpoints**: `/start`, `/stop`, `/test_summary`

### 2. New Vision Server (`vision_new.py`)
- **Port**: 5005 (default)
- **Features**: More advanced simulation with threading
- **Endpoints**: `/start`, `/stop`, `/status`, `/health`, `/warehouse_info`, `/test_summary`

## Running the Servers

### Old Vision Server
```bash
cd backend
python vision.py --port 5001 --id 1
```

### New Vision Server
```bash
cd backend
python vision_new.py --port 5005 --id 1
```

## Testing

Use the test script to verify both servers work correctly:

```bash
cd backend
python test_vision_output.py
```

## Key Changes Made

1. **Unified Output Structure**: Both servers now return the exact same structure as the main server
2. **Detailed Product Counts**: Instead of simple `items` array, now returns `detailed_product_counts` with `loaded` and `unloaded` keys
3. **Events Structure**: Proper events dictionary with track IDs, timestamps, and product types
4. **Status Information**: Consistent status object with session information
5. **Test Endpoints**: Both servers have `/test_summary` endpoints for easy testing

## Backend Integration

The backend (`ui.py`) is already configured to work with this structure and will:
- Extract product counts from `detailed_product_counts.loaded` and `detailed_product_counts.unloaded`
- Store individual product counts without combining them
- Handle the new event structure properly

## Product Names

Both servers use the same product names as defined in `seed.py`:
- `sulfat`
- `neshaste` 
- `alum`
- `resin`
- `titanium_dioxide`
- `sodium_hydroxide`

## Warehouse Locations

Both servers use the same warehouse location mapping:
- 1: "انبار سنگین"
- 2: "انبار سبک"
- 3: "انبار بسته‌بندی"
- 4: "انبار ذخیره‌سازی" 