#!/usr/bin/env python3
"""
Simple SIM800C test script to check hardware communication
"""

import serial
import time
import sys

def test_sim800c():
    """Test SIM800C communication"""
    port = "/dev/ttyAMA0"
    baudrate = 115200
    
    print(f"Testing SIM800C on {port} at {baudrate} baud...")
    
    try:
        # Open serial connection
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=3,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        
        print("✅ Serial connection opened successfully")
        
        # Wait for module to stabilize
        time.sleep(2)
        
        # Test AT command
        print("📡 Testing AT command...")
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        ser.write(b'AT\r\n')
        time.sleep(1)
        
        response = ""
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                response += chunk.decode('utf-8', errors='ignore')
                if 'OK' in response:
                    break
            time.sleep(0.1)
        
        if 'OK' in response:
            print("✅ AT command successful")
            print(f"Response: {response.strip()}")
        else:
            print("❌ AT command failed")
            print(f"Response: {response}")
            return False
        
        # Test signal strength
        print("📶 Testing signal strength...")
        ser.reset_input_buffer()
        ser.write(b'AT+CSQ\r\n')
        time.sleep(1)
        
        response = ""
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                response += chunk.decode('utf-8', errors='ignore')
                if 'OK' in response:
                    break
            time.sleep(0.1)
        
        print(f"Signal response: {response.strip()}")
        
        # Test SIM status
        print("📱 Testing SIM status...")
        ser.reset_input_buffer()
        ser.write(b'AT+CPIN?\r\n')
        time.sleep(1)
        
        response = ""
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                response += chunk.decode('utf-8', errors='ignore')
                if 'OK' in response or 'ERROR' in response:
                    break
            time.sleep(0.1)
        
        print(f"SIM status: {response.strip()}")
        
        # Test network registration
        print("🌐 Testing network registration...")
        ser.reset_input_buffer()
        ser.write(b'AT+CREG?\r\n')
        time.sleep(1)
        
        response = ""
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                response += chunk.decode('utf-8', errors='ignore')
                if 'OK' in response or 'ERROR' in response:
                    break
            time.sleep(0.1)
        
        print(f"Network registration: {response.strip()}")
        
        ser.close()
        print("✅ SIM800C test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_sim800c() 