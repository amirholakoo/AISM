"""
SIM800C Controller for SMS functionality with improved error handling and reliability
"""

import serial
import time
import threading
import logging
from datetime import datetime
import os
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sms_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SIM800CController:
    def __init__(self, port="/dev/ttyAMA0", baudrate=115200, test_mode=False):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.lock = threading.Lock()
        self.is_connected = False
        self.test_mode = test_mode or os.getenv('SMS_TEST_MODE', 'false').lower() == 'true'
        self.max_retries = 3
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
    def connect(self):
        """Establish connection to SIM800C with retry logic"""
        if self.test_mode:
            logger.info("Running in test mode - simulating SIM800C connection")
            self.is_connected = True
            return True
        
        self.connection_attempts = 0
        while self.connection_attempts < self.max_connection_attempts:
            self.connection_attempts += 1
            logger.info(f"Connecting to SIM800C on {self.port} (attempt {self.connection_attempts})")
            
            try:
                # Close any existing connection
                if self.serial_conn:
                    try:
                        self.serial_conn.close()
                    except:
                        pass
                    self.serial_conn = None
                    self.is_connected = False
                
                # Create new connection
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=3,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    xonxoff=False,
                    rtscts=False,
                    dsrdtr=False
                )
                
                # Wait for module to stabilize
                time.sleep(3)
                
                # Test basic communication with retries
                if not self._test_at_command():
                    logger.error("No response to AT command after retries")
                    if self.connection_attempts < self.max_connection_attempts:
                        time.sleep(2)
                        continue
                    else:
                        logger.error(f"Max connection attempts ({self.max_connection_attempts}) reached")
                        return False
                
                # Check if SIM is ready
                if not self._check_sim_ready():
                    logger.warning("SIM card not ready, but continuing...")
                
                # Initialize SMS settings
                if not self._initialize_sms_settings():
                    logger.error("Failed to initialize SMS settings")
                    if self.connection_attempts < self.max_connection_attempts:
                        time.sleep(2)
                        continue
                    else:
                        return False
                
                # Check signal strength
                signal, status = self.get_signal_strength()
                logger.info(f"Signal strength: {signal}% ({status})")
                
                self.is_connected = True
                logger.info("SIM800C connection established and configured")
                return True
                
            except Exception as e:
                logger.error(f"Connection attempt {self.connection_attempts} failed: {e}")
                if self.serial_conn:
                    try:
                        self.serial_conn.close()
                    except:
                        pass
                    self.serial_conn = None
                
                if self.connection_attempts < self.max_connection_attempts:
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"Max connection attempts ({self.max_connection_attempts}) reached")
                    return False
        
        self.is_connected = False
        return False
    
    def _test_at_command(self):
        """Test AT command with retries"""
        for attempt in range(3):
            try:
                if not self.serial_conn or not self.serial_conn.is_open:
                    logger.error("No serial connection available")
                    return False
                
                # Clear buffers
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()
                
                # Send AT command
                self.serial_conn.write(b'AT\r\n')
                time.sleep(1)
                
                # Read response
                response = ""
                start_time = time.time()
                while time.time() - start_time < 3:
                    if self.serial_conn.in_waiting > 0:
                        chunk = self.serial_conn.read(self.serial_conn.in_waiting)
                        response += chunk.decode('utf-8', errors='ignore')
                        if 'OK' in response:
                            logger.debug(f"AT command successful: {response.strip()}")
                            return True
                    time.sleep(0.1)
                
                logger.warning(f"AT command attempt {attempt + 1} failed - response: {response}")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"AT command test error: {e}")
                time.sleep(1)
        
        return False
    
    def _check_sim_ready(self):
        """Check if SIM card is ready"""
        response = self.send_command("AT+CPIN?", wait_time=2)
        if response and ("READY" in response or "+CPIN: READY" in response):
            return True
        logger.warning("SIM card status check failed")
        return False
    
    def _initialize_sms_settings(self):
        """Initialize SMS settings"""
        # Set to text mode
        if not self.send_command("AT+CMGF=1", wait_time=1):
            logger.error("Failed to set text mode")
            return False
        
        # Set character set
        if not self.send_command('AT+CSCS="GSM"', wait_time=1):
            logger.error("Failed to set character set")
            return False
        
        # Set SMS storage to SIM card
        if not self.send_command('AT+CPMS="SM","SM","SM"', wait_time=1):
            logger.warning("Failed to set SMS storage (continuing anyway)")
        
        return True
    
    def send_command(self, command, wait_time=1, expect_response=True):
        """Send AT command and get response with improved error handling"""
        if self.test_mode:
            logger.debug(f"Test mode - simulating command: {command}")
            if command == "AT":
                return "OK"
            elif command == "AT+CPIN?":
                return "+CPIN: READY\r\nOK"
            elif command == "AT+CSQ":
                return "+CSQ: 25,0\r\nOK"
            return "OK"
        
        with self.lock:
            if not self.serial_conn:
                logger.error("No serial connection available")
                return None
            
            if not self.serial_conn.is_open:
                logger.error("Serial connection is not open")
                return None
            
            try:
                # Clear input buffer
                self.serial_conn.reset_input_buffer()
                
                # Send command
                cmd = f"{command}\r\n"
                logger.debug(f"Sending: {command}")
                self.serial_conn.write(cmd.encode())
                
                if expect_response:
                    time.sleep(wait_time)
                    
                    # Read response with timeout
                    response = ""
                    start_time = time.time()
                    timeout = wait_time + 2  # Add extra time for response
                    
                    while time.time() - start_time < timeout:
                        if self.serial_conn.in_waiting > 0:
                            chunk = self.serial_conn.read(self.serial_conn.in_waiting)
                            response += chunk.decode('utf-8', errors='ignore')
                            
                            # Check for complete response
                            if 'OK' in response or 'ERROR' in response:
                                break
                        time.sleep(0.1)
                    
                    if response:
                        logger.debug(f"Response: {response.strip()}")
                        return response.strip()
                    else:
                        logger.warning("No response received")
                        return None
                else:
                    return "OK"
                    
            except Exception as e:
                logger.error(f"Command '{command}' failed: {e}")
                return None
    
    def get_signal_strength(self):
        """Get signal strength with improved parsing"""
        if self.test_mode:
            return 80, "Good"
        
        response = self.send_command("AT+CSQ", wait_time=2)
        if response and "+CSQ:" in response:
            try:
                # Use regex to extract signal value
                match = re.search(r'\+CSQ:\s*(\d+),\d+', response)
                if match:
                    signal = int(match.group(1))
                    if signal == 99:
                        return 0, "Unknown"
                    # Convert to percentage (0-31 -> 0-100%)
                    percentage = int((signal / 31) * 100)
                    status = "Excellent" if percentage > 80 else \
                             "Good" if percentage > 60 else \
                             "Fair" if percentage > 40 else \
                             "Poor"
                    return percentage, status
                else:
                    logger.warning("Could not parse signal strength response")
                    return 0, "Parse Error"
            except Exception as e:
                logger.error(f"Error parsing signal strength: {e}")
                return 0, "Error"
        
        logger.warning("No signal strength response")
        return 0, "No response"
    
    def send_sms(self, phone_number, message):
        """Send SMS message with improved error handling and retry logic"""
        if self.test_mode:
            logger.info(f"Test mode - simulating SMS to {phone_number}: {message}")
            return True
        
        max_attempts = 3
        for attempt in range(max_attempts):
            logger.info(f"Sending SMS to {phone_number} (Attempt {attempt + 1}/{max_attempts})")
            
            try:
                # Check connection first
                if not self.is_connected or not self.serial_conn:
                    logger.warning("Connection lost, attempting to reconnect...")
                    if not self.connect():
                        logger.error("Failed to reconnect")
                        continue
                
                # Test if module is responsive
                if not self.send_command("AT", wait_time=1):
                    logger.error("Module not responding")
                    continue
                
                # Check signal quality
                signal, status = self.get_signal_strength()
                logger.info(f"Signal strength: {signal}% ({status})")
                
                if signal < 5:  # Very weak signal
                    logger.error(f"SMS sending failed on attempt {attempt + 1}: Signal too weak: {signal}% ({status})")
                    if attempt < max_attempts - 1:
                        logger.info("Waiting 5 seconds before retry...")
                        time.sleep(5)
                    continue
                
                # Reinitialize SMS settings
                if not self._initialize_sms_settings():
                    logger.error("Failed to reinitialize SMS settings")
                    continue
                
                # Send SMS command
                sms_cmd = f'AT+CMGS="{phone_number}"'
                response = self.send_command(sms_cmd, wait_time=3)
                
                if not response or ">" not in response:
                    logger.error("Failed to get SMS prompt")
                    continue
                
                # Send message content
                success = self._send_message_content(message)
                if success:
                    logger.info("SMS sent successfully")
                    return True
                else:
                    logger.error(f"Failed to send message content on attempt {attempt + 1}")
                    if attempt < max_attempts - 1:
                        logger.info("Waiting 5 seconds before retry...")
                        time.sleep(5)
                
            except Exception as e:
                logger.error(f"SMS sending error on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    logger.info("Waiting 5 seconds before retry...")
                    time.sleep(5)
        
        logger.error(f"SMS sending failed after {max_attempts} attempts")
        return False
    
    def _send_message_content(self, message):
        """Send SMS message content with improved response handling"""
        try:
            if not self.serial_conn:
                logger.error("No serial connection")
                return False
            
            # Clear buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # Send message with termination character
            full_message = f"{message}\x1A"
            self.serial_conn.write(full_message.encode('utf-8', errors='ignore'))
            
            # Wait for response with extended timeout
            response = ""
            start_time = time.time()
            timeout = 30  # 30 second timeout for network operations
            
            while time.time() - start_time < timeout:
                if self.serial_conn.in_waiting > 0:
                    chunk = self.serial_conn.read(self.serial_conn.in_waiting)
                    response += chunk.decode('utf-8', errors='ignore')
                    
                    # Check for successful response
                    if "+CMGS:" in response and "OK" in response:
                        logger.info("SMS sent successfully")
                        return True
                    elif "ERROR" in response:
                        logger.error(f"SMS sending failed with error: {response}")
                        return False
                
                time.sleep(0.5)
            
            logger.error("SMS sending timed out")
            return False
            
        except Exception as e:
            logger.error(f"Failed to send message content: {e}")
            return False
    
    def check_sim_status(self):
        """Check if SIM card is ready"""
        if self.test_mode:
            return True
        
        response = self.send_command("AT+CPIN?", wait_time=2)
        return response and ("READY" in response or "+CPIN: READY" in response)
    
    def close(self):
        """Close serial connection"""
        if not self.test_mode and self.serial_conn:
            try:
                self.serial_conn.close()
            except:
                pass
            self.is_connected = False
            logger.info("Serial connection closed")

# Create global controller instance
controller = SIM800CController(test_mode=False) 