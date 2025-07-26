---

### **1\. Check the Current Status ⚡️**

First, see if power saving is currently enabled. The default network interface name is likely wlan0.  
Open a terminal and run the following command:

Bash

iw dev wlan0 get power\_save

You will see one of two outputs:

* Power save: on  
* Power save: off

---

### **2\. Disable Wi-Fi Power Saving ⚙️**

You can choose a temporary or permanent solution.

#### **Temporary (Until Next Reboot)**

To disable power saving for your current session, run this command:

Bash

sudo iw dev wlan0 set power\_save off

This change will be lost when you restart your Raspberry Pi.

#### **Permanent (Survives Reboots)**

To permanently disable Wi-Fi power saving, you'll create a configuration file for NetworkManager.

1. Create and edit a new configuration file using the nano text editor:  
   Bash  
   sudo nano /etc/NetworkManager/conf.d/default-wifi-powersave-on.conf

2. Add the following lines to the file:  
   Ini, TOML  
   \[connection\]  
   wifi.powersave \= 2

   *Note: A value of 2 explicitly disables power saving. A value of 3 would enable it.*  
3. Save the file and exit nano by pressing Ctrl+X, then Y, then Enter.  
4. To apply the change, either restart your Raspberry Pi or restart the NetworkManager service with this command:  
   Bash  
   sudo systemctl restart NetworkManager

---

### **3\. Verify the Change Is Permanent ✅**

After you've applied the permanent fix and rebooted your Raspberry Pi, run the check command again to make sure the setting has been applied correctly.

Bash

iw dev wlan0 get power\_save

The output should now permanently be:  
Power save: off
