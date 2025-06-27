# sms_server.py

from flask import Flask, request, render_template_string
import serial
import random
import time

# Setup serial connection with SIM800
sim800 = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

# Initialize Flask
app = Flask(__name__)

# HTML Template for input
html_form = '''
<!DOCTYPE html>
<html>
<head><title>SMS Login</title></head>
<body>
  <h2>Enter Your Phone Number</h2>
  <form method="POST">
    <input type="text" name="phone" placeholder="e.g. +989123456789" required>
    <button type="submit">Send Password</button>
  </form>
  {% if sent %}
    <p style="color:green;">SMS sent to {{ number }}</p>
  {% endif %}
</body>
</html>
'''

def send_sms(number, message):
    # Send AT commands to SIM800
    sim800.write(b'AT\r')
    time.sleep(1)
    sim800.write(b'AT+CMGF=1\r')  # Set to text mode
    time.sleep(1)
    sim800.write(('AT+CMGS="{}"\r'.format(number)).encode())
    time.sleep(1)
    sim800.write((message + '\x1A').encode())  # End with Ctrl+Z
    time.sleep(3)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        number = request.form['phone']
        password = str(random.randint(100000, 999999))
        send_sms(number, f"Your login code is: {password}")
        return render_template_string(html_form, sent=True, number=number)
    return render_template_string(html_form, sent=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
