# Vision Output Storage Implementation

## Overview
This document describes the implementation of the `vision_output` field in the loadings table to store the complete output from the vision system.

## Changes Made

### 1. Database Model Update
**File:** `backend/models/db.py`

Added the `vision_output` field to the `Loading` model:
```python
class Loading(db.Model):
    __tablename__ = 'loadings'
    # ... existing fields ...
    vision_output = db.Column(db.Text, nullable=True)  # خروجی سیستم بینایی
    # ... rest of fields ...
```

### 2. Database Migration
**File:** `backend/migrate_add_vision_output.py`

The migration script adds the `vision_output` column to the existing `loadings` table:
- Column type: `TEXT`
- Nullable: `True`
- Purpose: Store complete JSON output from vision system

### 3. API Endpoint Updates

#### Vision Stop Endpoint
**File:** `backend/ui.py` - `/api/vision/stop`

Updated to save the complete vision output when processing stops:
```python
# ذخیره خروجی سیستم بینایی
import json
loading.vision_output = json.dumps(result, ensure_ascii=False, indent=2)
```

#### Loading Data Endpoints
Updated the following endpoints to include `vision_output` in their responses:

1. **`/api/loadings/<token>`** - Get loading by token
2. **`/api/loadings/last-completed`** - Get last completed loading
3. **`/api/loadings/<int:loading_id>/details`** - Get loading details
4. **`/api/loadings/<int:loading_id>/items`** - Get loading items
5. **`/api/loadings/<int:loading_id>/all-items`** - Get all loading items for all versions

All these endpoints now return the `vision_output` field in their JSON responses.

## Data Structure

The `vision_output` field stores a JSON string containing the complete response from the vision system, typically including:

```json
{
  "success": true,
  "message": "Vision processing completed successfully",
  "summary": {
    "operation_type": "loading",
    "total_products": 5,
    "location": "Warehouse A",
    "detailed_product_counts": {
      "loaded": {
        "Product A": 10,
        "Product B": 5
      },
      "unloaded": {
        "Product C": 3
      }
    }
  },
  "items": [
    {"name": "Product A", "type": "loaded", "count": 10},
    {"name": "Product B", "type": "loaded", "count": 5},
    {"name": "Product C", "type": "unloaded", "count": 3}
  ]
}
```

## Usage

### When Vision Output is Stored
The vision output is automatically stored when:
1. A vision processing session is stopped via `/api/vision/stop`
2. The vision system returns a successful response
3. The loading status is updated to 'vision'

### Accessing Vision Output
The vision output can be accessed through:
1. **API Endpoints**: All loading detail endpoints now include `vision_output`
2. **Database**: Direct query of the `loadings` table
3. **Frontend**: The vision output is available in loading data responses

### Example API Response
```json
{
  "success": true,
  "id": 1,
  "warehouse_id": "warehouse_1",
  "status": "vision",
  "vision_output": "{\"success\": true, \"message\": \"...\", ...}",
  "items": [...],
  ...
}
```

### Example Items Endpoint Response
```json
{
  "success": true,
  "loading_id": 1,
  "items": [
    {
      "id": 1,
      "name": "Product A",
      "type": "loaded",
      "count": 10,
      "source": "vision",
      "version": 1
    }
  ],
  "count": 1,
  "start_time": "2024-01-01T10:00:00",
  "end_time": "2024-01-01T10:30:00",
  "user_confirm_time": "2024-01-01T10:30:00",
  "edit_time": null,
  "vision_output": "{\"success\": true, \"message\": \"...\", ...}"
}
```

## Testing

A test script has been created to verify the implementation:
**File:** `backend/test_vision_output_integration.py`

The test verifies:
1. Database column exists
2. Field can be updated
3. JSON data can be stored and retrieved
4. Data integrity is maintained

Run the test with:
```bash
python test_vision_output_integration.py
```

## Migration

To apply the database changes, run:
```bash
python migrate_add_vision_output.py
```

The migration is idempotent and safe to run multiple times.

## Benefits

1. **Complete Data Preservation**: All vision system output is preserved
2. **Debugging Support**: Full vision response available for troubleshooting
3. **Audit Trail**: Complete record of what the vision system detected
4. **API Consistency**: All loading endpoints now include vision data
5. **Future Extensibility**: Easy to add new vision output fields

## Notes

- The `vision_output` field is nullable, so existing loadings without vision data will have `null` values
- The JSON is stored with proper Unicode support (`ensure_ascii=False`)
- The data is formatted with indentation for readability
- The field is included in all relevant API responses for consistency 