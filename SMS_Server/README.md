
# 📱 SMS Verification Server using Raspberry Pi 4 + SIM800

This project sets up a simple **Flask web server** on a **Raspberry Pi 4** that:
- Asks users to enter their **phone number**
- Sends a **randomly generated password** via **SMS** using a SIM800 module

---

## 🧰 Hardware Requirements

- ✅ Raspberry Pi 4 (any model)
- ✅ SIM800 module (SIM800L, SIM800C, etc.)
- ✅ SIM card with SMS capability (inserted in SIM800)
- ✅ 5V 2A power supply for SIM800
- ✅ Jumper wires or USB-to-serial cable

---

## 🧑‍💻 Software Requirements

- Raspberry Pi OS (Lite or Full)
- Python 3
- Flask
- pyserial
- Optional: `minicom` for manual SIM800 testing

### 📦 Installation

```bash
sudo apt update
sudo apt install python3-pip python3-flask python3-serial minicom
````

---

## 🔌 Wiring (if using GPIO UART)

| SIM800 Pin            | Raspberry Pi GPIO |
| --------------------- | ----------------- |
| VCC (use 5V external) | 5V external       |
| GND                   | GND               |
| TX                    | GPIO15 (RXD)      |
| RX                    | GPIO14 (TXD)      |

⚠️ Use a **logic level shifter** or voltage divider on SIM800 RX if it's not 5V-tolerant.

---

## 🔧 Enable Serial Interface

Run:

```bash
sudo raspi-config
```

* Go to: `Interface Options` → `Serial Port`

  * Login shell over serial: **No**
  * Enable serial port hardware: **Yes**

Then:

```bash
sudo reboot
```

---

## 🚀 Run the Flask App

Create and run the app:

```bash
python3 sms_server.py
```

Now visit the web interface from another device:

```
http://<your-raspberry-ip>:8080
```

---

## 📝 `sms_server.py` Highlights

* Uses Flask to serve a web form for phone number input
* Generates a random 6-digit password
* Sends the password via SMS using SIM800 with AT commands
* Displays confirmation on the browser

---

## 🧪 Debugging the SIM800

Use `minicom`:

```bash
sudo minicom -b 9600 -o -D /dev/serial0
```

Then test commands:

```plaintext
AT
AT+CMGF=1
AT+CMGS="+989123456789"
> Hello world ⏎ Ctrl+Z
```

Watch for `+CMGS:` and `OK` as success indicators.

---

## ✅ Working AT Commands in Python

```python
sim800.write(b'AT\r')
time.sleep(1)
sim800.write(b'AT+CMGF=1\r')
time.sleep(1)
sim800.write(b'AT+CMGS="+989123456789"\r')
time.sleep(1)
sim800.write(b'Message text\x1A')
```

---

## ⚠️ Notes and Tips

* Always use international phone format (e.g. `+98912...`)
* Check network signal: `AT+CSQ` (above 10 is OK)
* Check registration: `AT+CREG?` → `0,1` or `0,5`
* SIM PIN needed? Use `AT+CPIN="1234"` if required

---

## 📌 To Do (Next Steps)

* ✅ Web interface with phone form
* ✅ SMS sending via SIM800
* ⏳ Add verification/login after SMS
* ⏳ Save phone/code in database (SQLite)
* ⏳ Code expiration and resend timer
* ⏳ Frontend improvements (CSS)

---

## 🧑‍🎓 Author

Created by Amir with assistance from ChatGPT.
Simple, scalable, and beginner-friendly! ☺️

---

