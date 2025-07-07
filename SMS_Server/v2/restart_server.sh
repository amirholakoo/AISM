#!/bin/bash

# SMS Server Restart Script
# This script stops any running SMS server processes and starts a new one

echo "🔄 Restarting SMS Server with improved error handling..."

# Kill any existing SMS server processes
echo "🛑 Stopping existing SMS server processes..."
pkill -f "python.*sms_server.py" || true
pkill -f "python.*SMS_Server" || true

# Wait for processes to stop
sleep 2

# Check if port 5003 is still in use
if lsof -i :5003 > /dev/null 2>&1; then
    echo "⚠️  Port 5003 still in use, killing processes..."
    lsof -ti :5003 | xargs kill -9 || true
    sleep 2
fi

# Navigate to SMS_Server directory
cd /home/admin/Downloads/IOMS/SMS_Server

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the SMS server
echo "🚀 Starting SMS Server on port 5003..."
export FLASK_ENV=production
export FLASK_APP=sms_server.py
python sms_server.py --port 5003 &

# Get the process ID
SMS_PID=$!
echo "📱 SMS Server started with PID: $SMS_PID"

# Wait a moment for the server to start
sleep 3

# Check if server is running
if ps -p $SMS_PID > /dev/null; then
    echo "✅ SMS Server is running successfully!"
    echo "📊 Server accessible at: http://192.168.1.60:5003"
    echo "🔍 Health check: http://192.168.1.60:5003/health"
    echo "📋 Logs: tail -f sms_server.log"
else
    echo "❌ Failed to start SMS Server"
    exit 1
fi

echo "🎉 SMS Server restart completed!" 