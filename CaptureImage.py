import time
import subprocess
import os

###### MAKE SURE COUNT LIMITED IS FALSE FOR DEMO!!! ######
CountLimited = False

# Read IP from file
IPAddress = "IP.NOT.RETRIEVED"

AbsPath = "/home/jPi0/Documents/SmartSight/SendImage/"

IPFile = AbsPath + "LaptopIP.txt"
if os.path.exists(IPFile):
    with open(IPFile, "r") as f:
        IPAddress = f.read()
else:
    print("ERROR: ip not retrieved.")
    quit()
print(IPAddress)

# Add IP to known hosts (if necessary)
# ~/.ssh/known_hosts
KnownHostsFile = "/home/jPi0/.ssh/known_hosts"

if os.path.exists(KnownHostsFile):
    # Check if it's in the known_hosts.
    IPAlreadyKnown = False
    with open(KnownHostsFile, "r") as file:
        content = file.read()
        if IPAddress in content:
            IPAlreadyKnown = True
            print("IP not added to known hosts - Already known.")
        file.close
    if not IPAlreadyKnown:
        with open(KnownHostsFile, "a") as f:
            f.write(f"{IPAddress} ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBAiYBXgYrqtYsFkQc55uNSTHUwLJMSwJ74T2eWQ0LCYVwHOE5W4GQoBgoCwGW6wzQLN1h0h/nCoiB67Bpce6EXw=\n")
            print(f"IP {IPAddress} added to known hosts.")
else:
    print(f"{KnownHostsFile} Does not exist")

# command to start the camera process
#CameraCmd = "libcamera-still -t 0 --signal --datetime"
outputName = "output.jpg"
CameraCmd = ["libcamera-still", "-t", "0", "--signal", "-o", outputName, "--width", "1296", "--height", "972", "--shutter", "10000"]
TakePictureCmd = ["kill", "-SIGUSR1"]
StopCmd = ["kill", "-SIGUSR2"]

# Use IP in transfer.
SendCmd = ["scp", "-i", "~/.ssh/rp0keyjacob", outputName, f"jacob@[{IPAddress}]:/C:/Users/Jacob/Documents/UNT/SmartSight/EdgeProgram/image.jpg"]

activationTime = 10.0
framerate = 4.0
sendDelay = (1.0/framerate)/4.0 # Time to delay between sending image capture signal and beginning image transfer.

# Start camera process
process = subprocess.Popen(CameraCmd)

# Get pid
pid = process.pid

StopCmd.append(str(pid))
TakePictureCmd.append(str(pid))
print("activating")
time.sleep(activationTime)
print("activated")
counter = 0
while (True):
    time.sleep((1.0/framerate) - sendDelay)
    subprocess.run(TakePictureCmd)
    time.sleep(sendDelay)
    
    subprocess.run(SendCmd)
    
    print("To stop, run:")
    print("kill" + " -SIGUSR2 " + str(pid))
    print(SendCmd)
    counter += 1
    if CountLimited and counter > 150:
        break
subprocess.run(StopCmd)
