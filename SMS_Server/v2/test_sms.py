#!/usr/bin/env python3
"""
Test script for SMS Server
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:5003"
API_KEY = os.getenv('API_KEY')

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"   SIM Status: {data['sim_status']}")
            print(f"   Signal: {data['signal_strength']}")
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

def test_send_sms():
    """Test SMS sending"""
    phone = input("Enter phone number (with country code, e.g., +1234567890): ")
    message = input("Enter test message: ")
    
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY
    }
    
    data = {
        'phone_number': phone,
        'message': message
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/verify/send", 
                               json=data, headers=headers, timeout=30)
        if response.status_code == 200:
            print("✅ SMS sent successfully!")
            return True
        else:
            error = response.json().get('error', 'Unknown error')
            print(f"❌ SMS sending failed: {error}")
            return False
    except Exception as e:
        print(f"❌ SMS sending error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 SMS Server Test Script")
    print("========================")
    
    if not API_KEY:
        print("❌ API_KEY not found in environment")
        exit(1)
    
    print(f"Testing server at: {BASE_URL}")
    print(f"API Key: {API_KEY[:8]}...")
    print()
    
    # Test health
    if test_health():
        print()
        choice = input("Do you want to test SMS sending? (y/n): ")
        if choice.lower() == 'y':
            test_send_sms()
    
    print("\n🏁 Test completed")
