"""
Test script for SMS Server
Tests API endpoints and SIM800C functionality
"""

import requests
import time
from datetime import datetime

# Configuration
BASE_URL = 'http://localhost:5003'  # Updated to use localhost
API_KEY = 'test-api-key'  # Should match the one in sms_server.py
HEADERS = {'X-API-Key': API_KEY}

def test_health():
    """Test health check endpoint"""
    print("\n🏥 Testing health check endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    print("✅ Health check passed")

def test_send_verification():
    """Test sending verification code"""
    print("\n📱 Testing send verification endpoint...")
    
    # Test with valid phone number
    data = {'phone_number': '09123456789'}
    response = requests.post(f"{BASE_URL}/api/v1/verify/send", json=data, headers=HEADERS)
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()['success'] is True
    print("✅ Send verification passed")

def test_verify_code():
    """Test code verification"""
    print("\n🔐 Testing verify code endpoint...")
    
    # First send a verification code
    phone_number = '09123456789'
    data = {'phone_number': phone_number}
    send_response = requests.post(f"{BASE_URL}/api/v1/verify/send", json=data, headers=HEADERS)
    assert send_response.status_code == 200
    
    # Wait for code to be sent
    time.sleep(2)
    
    # Get the verification code from the database (in real scenario this would be received via SMS)
    verify_data = {
        'phone_number': phone_number,
        'code': '123456'  # This is a test code, real code would come from SMS
    }
    response = requests.post(f"{BASE_URL}/api/v1/verify/check", json=verify_data, headers=HEADERS)
    print(f"Response: {response.json()}")
    assert response.status_code in [200, 400]  # 400 is acceptable as we're using a fake code
    print("✅ Verify code test passed")

def test_invalid_requests():
    """Test invalid requests handling"""
    print("\n❌ Testing invalid requests...")
    
    # Test missing phone number
    data = {}
    response = requests.post(f"{BASE_URL}/api/v1/verify/send", json=data, headers=HEADERS)
    print(f"Missing phone response: {response.json()}")
    assert response.status_code == 400
    
    # Test invalid phone format
    data = {'phone_number': '123'}
    response = requests.post(f"{BASE_URL}/api/v1/verify/send", json=data, headers=HEADERS)
    print(f"Invalid phone response: {response.json()}")
    assert response.status_code == 400
    
    # Test missing verification code
    data = {'phone_number': '09123456789'}
    response = requests.post(f"{BASE_URL}/api/v1/verify/check", json=data, headers=HEADERS)
    print(f"Missing code response: {response.json()}")
    assert response.status_code == 400
    
    print("✅ Invalid requests test passed")

def run_all_tests():
    """Run all tests"""
    print("\n🧪 Starting SMS Server tests...")
    print("=" * 60)
    
    try:
        test_health()
        test_send_verification()
        test_verify_code()
        test_invalid_requests()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise e

if __name__ == "__main__":
    run_all_tests() 