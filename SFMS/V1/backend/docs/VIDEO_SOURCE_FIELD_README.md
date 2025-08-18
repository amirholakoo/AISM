# Video Source Field Addition to VisionServer Model

## Overview

A new field `video_source` has been added to the `VisionServer` model to specify the video source type for each vision server.

## Changes Made

### 1. Model Update (`backend/models/db.py`)

Added the `video_source` field to the `VisionServer` class:

```python
class VisionServer(db.Model):
    __tablename__ = 'vision_servers'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    persian_name = db.Column(db.String(100), nullable=True)
    url = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    video_source = db.Column(db.String(50), nullable=False, default='picamera')  # NEW FIELD
```

### 2. API Routes Update (`backend/routes/vision_server_routes.py`)

Updated all CRUD operations to handle the new field:

- **GET**: Returns `video_source` in server data
- **POST**: Accepts `video_source` parameter (defaults to 'picamera')
- **PUT**: Allows updating `video_source` field

### 3. Seed Data Update (`backend/seed.py`)

Updated vision server creation to include `video_source` field:

```python
vision_servers_data = [
    {
        "name": "unloading_vision_server",
        "persian_name": "سرور بینایی تخلیه",
        "url": "http://localhost:5001",
        "type": "unloading",
        "is_active": True,
        "video_source": "picamera"  # NEW FIELD
    },
    # ... other servers
]
```

### 4. Migration Script (`backend/migrate_add_video_source_field.py`)

Created a migration script to add the field to existing databases:

```bash
cd backend
python migrate_add_video_source_field.py
```

### 5. Test Script (`backend/test_video_source_field.py`)

Created a test script to verify the field works correctly:

```bash
cd backend
python test_video_source_field.py
```

## Field Properties

- **Name**: `video_source`
- **Type**: `String(50)`
- **Default Value**: `'picamera'`
- **Nullable**: `False`
- **Description**: Specifies the video source type for the vision server

## Common Video Source Values

- `picamera` - Raspberry Pi Camera Module
- `webcam` - USB Webcam
- `file` - Video file input
- `rtsp` - RTSP stream
- `http` - HTTP stream
- `custom` - Custom video source

## Usage Examples

### Creating a Vision Server with Custom Video Source

```python
new_server = VisionServer(
    name="custom_vision_server",
    url="http://localhost:5004",
    type="unloading",
    video_source="webcam"  # Custom video source
)
```

### Updating Video Source

```python
# Via API
PUT /api/vision-servers/{id}
{
    "video_source": "rtsp"
}

# Via Python
server = VisionServer.query.get(server_id)
server.video_source = "rtsp"
db.session.commit()
```

## Database Migration

If you have an existing database, run the migration script:

```bash
cd backend
python migrate_add_video_source_field.py
```

This will:
1. Add the `video_source` column to the `vision_servers` table
2. Set the default value `'picamera'` for all existing records
3. Verify the migration was successful

## Testing

After making changes, test the functionality:

```bash
cd backend
python test_video_source_field.py
```

This will verify:
- Field exists in the model
- Default value works correctly
- Custom values can be set
- Existing servers have the field

## Frontend Integration

The frontend will automatically receive the `video_source` field in API responses. Update your frontend components to display or edit this field as needed.

## Backward Compatibility

- Existing code will continue to work
- The field defaults to `'picamera'` for backward compatibility
- No breaking changes to existing APIs
