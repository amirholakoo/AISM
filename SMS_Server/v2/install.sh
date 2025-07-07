#!/bin/bash

# IOMS SMS Server Installation Script
# This script automates the installation and setup process

set -e  # Exit on any error

echo "🚀 IOMS SMS Server Installation Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root!"
    exit 1
fi

print_info "Starting SMS Server installation..."

# Step 1: Update system
print_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_status "System updated"

# Step 2: Install dependencies
print_info "Installing Python dependencies..."
sudo apt install python3-pip python3-venv python3-dev build-essential -y
print_status "Dependencies installed"

# Step 3: Check if we're in the right directory
if [ ! -f "sms_server.py" ]; then
    print_error "Please run this script from the SMS_Server directory"
    exit 1
fi

# Step 4: Create virtual environment
print_info "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Step 5: Activate virtual environment and install packages
print_info "Installing Python packages..."
source venv/bin/activate
pip install --upgrade pip
pip install flask pyserial sqlalchemy requests
print_status "Python packages installed"

# Step 6: Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env configuration file..."
    cat > .env << EOF
# SMS Server Configuration
FLASK_ENV=production
SECRET_KEY=ioms_secret_key_2025
API_KEY=ioms_sms_server_2025

# SIM800C Configuration
SIM800C_PORT=/dev/ttyAMA0
SIM800C_BAUDRATE=115200

# Database
DATABASE_URL=sqlite:///sms_server.db

# Logging
LOG_LEVEL=INFO
EOF
    print_status ".env file created"
else
    print_warning ".env file already exists"
fi

# Step 7: Make scripts executable
print_info "Making scripts executable..."
chmod +x restart_server.sh
chmod +x test_sim800c.py
print_status "Scripts made executable"

# Step 8: Create backup directory
print_info "Creating backup directory..."
mkdir -p backup
print_status "Backup directory created"

# Step 9: Check serial port
print_info "Checking serial port configuration..."
if [ -e "/dev/ttyAMA0" ]; then
    print_status "Serial port /dev/ttyAMA0 exists"
    
    # Check if user is in dialout group
    if groups | grep -q dialout; then
        print_status "User is in dialout group"
    else
        print_warning "Adding user to dialout group..."
        sudo usermod -a -G dialout $USER
        print_status "User added to dialout group (reboot required)"
    fi
else
    print_error "Serial port /dev/ttyAMA0 not found!"
    print_info "Please enable serial port using: sudo raspi-config"
    print_info "Navigate to: Interface Options > Serial Port"
    print_info "Enable serial port hardware: Yes"
    print_info "Enable serial console: No"
    exit 1
fi

# Step 10: Test hardware connection
print_info "Testing SIM800C hardware connection..."
if python test_sim800c.py; then
    print_status "SIM800C hardware test successful!"
else
    print_warning "SIM800C hardware test failed - please check connections"
    print_info "This is normal if the SIM800C module is not connected yet"
fi

# Step 11: Create systemd service (optional)
read -p "Do you want to create a systemd service for automatic startup? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Creating systemd service..."
    sudo tee /etc/systemd/system/sms-server.service > /dev/null << EOF
[Unit]
Description=IOMS SMS Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python sms_server.py --port 5003
Restart=always
RestartSec=10
Environment=PATH=$(pwd)/venv/bin

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable sms-server.service
    print_status "Systemd service created and enabled"
    
    print_info "You can now control the service with:"
    print_info "  sudo systemctl start sms-server"
    print_info "  sudo systemctl stop sms-server"
    print_info "  sudo systemctl status sms-server"
fi

echo
echo "🎉 Installation Complete!"
echo "======================="
print_status "SMS Server is ready to use!"
echo
print_info "Next steps:"
echo "1. Connect your SIM800C module to the Raspberry Pi"
echo "2. Insert an active SIM card"
echo "3. Test the connection: python test_sim800c.py"
echo "4. Start the server: ./restart_server.sh"
echo "5. Check health: curl http://192.168.1.60:5003/health"
echo
print_info "Documentation:"
echo "- Full documentation: README.md"
echo "- API documentation: See README.md API section"
echo "- Troubleshooting: See README.md troubleshooting section"
echo
print_info "Useful commands:"
echo "- Start server: ./restart_server.sh"
echo "- Test hardware: python test_sim800c.py"
echo "- View logs: tail -f sms_server.log"
echo "- Check health: curl http://192.168.1.60:5003/health"
echo
print_warning "If you added user to dialout group, please reboot your system!"
echo
print_status "Happy SMS sending! 📱" 