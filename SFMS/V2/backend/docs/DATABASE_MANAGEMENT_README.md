# External Database Management

This document explains how to manage the external SQLite database connections for file replacement via SSH.

## Overview

The external database (`external_db/localnew.sqlite3`) is used by the SFMS application to read shipment and warehouse data. When you need to update this database with fresh data from a remote server, you need to:

1. Close all database connections
2. Replace the database file via SSH
3. Re-establish connections

## API Endpoints

### Close Database Connections
```http
POST /api/database/close
```

**Response:**
```json
{
  "success": true,
  "message": "اتصالات دیتابیس با موفقیت بسته شد. حالا می‌توانید فایل دیتابیس را جایگزین کنید."
}
```

### Check Database Status
```http
GET /api/database/status
```

**Response:**
```json
{
  "success": true,
  "message": "دیتابیس قابل دسترسی است",
  "data": {
    "shipments_count": 1234,
    "status": "connected"
  }
}
```

## Usage Methods

### Method 1: Using the Database Manager Script

Run the interactive database manager:

```bash
cd backend
python database_manager.py
```

This provides a menu with options:
1. Close database connections
2. Check database status
3. Replace database file (manual)
4. Full workflow (close → replace → check)
5. Exit

### Method 2: Using the Simple Close Script

Quickly close database connections:

```bash
cd backend
python close_db.py
```

### Method 3: Using curl

Close database connections via HTTP:

```bash
curl -X POST http://127.0.0.1:18888/api/database/close
```

Check database status:

```bash
curl http://127.0.0.1:18888/api/database/status
```

### Method 4: Using Python Requests

```python
import requests

# Close database connections
response = requests.post("http://127.0.0.1:18888/api/database/close")
if response.status_code == 200:
    print("Database connections closed")

# Check status
response = requests.get("http://127.0.0.1:18888/api/database/status")
if response.status_code == 200:
    data = response.json()
    print(f"Shipments count: {data['data']['shipments_count']}")
```

## Complete Workflow

### Step 1: Close Database Connections
```bash
python close_db.py
```

### Step 2: Replace Database File via SSH
```bash
# Example SSH command (replace with your actual server details)
scp user@remote-server:/path/to/localnew.sqlite3 external_db/
```

### Step 3: Verify Database is Accessible
```bash
curl http://127.0.0.1:18888/api/database/status
```

## Automated Workflow

You can create a shell script to automate the entire process:

```bash
#!/bin/bash
# update_database.sh

echo "🔒 Closing database connections..."
python close_db.py

if [ $? -eq 0 ]; then
    echo "⏳ Waiting 2 seconds..."
    sleep 2
    
    echo "🔄 Downloading new database file..."
    scp user@remote-server:/path/to/localnew.sqlite3 external_db/
    
    if [ $? -eq 0 ]; then
        echo "⏳ Waiting 1 second..."
        sleep 1
        
        echo "🔍 Checking database status..."
        curl -s http://127.0.0.1:18888/api/database/status | python -m json.tool
    else
        echo "❌ Failed to download database file"
        exit 1
    fi
else
    echo "❌ Failed to close database connections"
    exit 1
fi
```

## Troubleshooting

### Error: "Cannot connect to API server"
- Make sure the Flask application is running on port 18888
- Check if the server is accessible at `http://127.0.0.1:18888`

### Error: "Database not accessible"
- The database file might be corrupted
- Check file permissions on `external_db/localnew.sqlite3`
- Verify the file exists and is readable

### Error: "File is locked"
- Some other process might be using the database
- Restart the Flask application
- Check for any other Python processes using the database

### Error: "Permission denied" during SSH
- Check SSH key configuration
- Verify server credentials
- Ensure the remote file path is correct

## Technical Details

### Database Connection Management

The application uses SQLAlchemy with the following configuration:

```python
# From models/database.py
engine = create_engine(
    EXTERNAL_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False
)
```

### Connection Closing Process

When `close_database_connection()` is called:

1. All active sessions are closed using `SessionLocal.close_all()`
2. The engine is disposed using `engine.dispose()`
3. All file handles are released
4. The database file can now be safely replaced

### Automatic Reconnection

After the database file is replaced, the next API call will automatically:
1. Create a new engine instance
2. Establish new connections
3. Resume normal operation

## Security Notes

- The database close endpoint should be protected in production
- Consider adding authentication for database management operations
- Monitor database file access and modifications
- Keep backup copies of the database file

## Performance Considerations

- Closing and reopening connections has minimal overhead
- The database file is read-only in this application
- Connection pooling is disabled for SQLite (StaticPool)
- File I/O performance depends on the database file size 