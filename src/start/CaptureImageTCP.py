import time
import subprocess
import os
import socket
import json

###### MAKE SURE COUNT LIMITED IS FALSE FOR DEMO!!! ######
CountLimited = False
CountLimit = 225

# Read IP from file
IPAddress = "IP.NOT.RETRIEVED"
SSHServerPubKey = "KEY NOT AVAILABLE"
EdgeABSPath = "PathNotAvailable"
EdgeUsername = "UsernameNotAvailable"
AbsPath = "/home/jPi0/Documents/SmartSight/SendImage/"
lines = []
IPFile = AbsPath + "LaptopIP.txt"
if os.path.exists(IPFile):
    with open(IPFile, "r") as f:
        lines = f.readlines()
else:
    print("ERROR: ip not retrieved.")
    quit()
IPAddress = lines[0]
print(f"IPAddress: {IPAddress}")
SSHServerPubKey = lines[1]
print(f"SSHServerPubKey: {SSHServerPubKey}")
EdgeABSPath = lines[2]
print(f"EdgeABSPath: {EdgeABSPath}")
EdgeUsername = lines[3]
print(f"EdgeUsername: {EdgeUsername}")

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
            f.write(f"{IPAddress} {SSHServerPubKey}\n")
            print(f"IP {IPAddress} added to known hosts.")
else:
    print(f"{KnownHostsFile} Does not exist")

# command to start the camera process
# CameraCmd = "libcamera-still -t 0 --signal --datetime"
outputName = "output.jpg"
CameraCmd = ["libcamera-still", "-t", "0", "--signal", "-o", outputName, "--width", "1296", "--height", "972", "--shutter", "10000"]
TakePictureCmd = ["kill", "-SIGUSR1"]
StopCmd = ["kill", "-SIGUSR2"]
IPAddress = IPAddress.strip()
# Use IP in transfer.
# SendCmd = ["scp", "-i", "~/.ssh/PC-rPi0Key", outputName, f"{EdgeUsername}@{IPAddress}:/{EdgeABSPath}/image.jpg"]

HOST = IPAddress  # Server IP address
PORT = 8888  # Same port as server
FILE_PATH = outputName  # File to send



import socket
 
def resolve_address(host):
    try:
        addr_info = socket.getaddrinfo(host, None, socket.AF_UNSPEC)
        ip_addresses = [info[4][0] for info in addr_info]
 
        # Filter out IPv4 and IPv6 addresses separately
        ipv4 = [ip for ip in ip_addresses if ':' not in ip]  # IPv4 lacks ':'
        ipv6 = [ip for ip in ip_addresses if ':' in ip]  # IPv6 contains ':'
 
        if ipv6:
            return ipv6[0], socket.AF_INET6  # Prefer IPv6 if available
        elif ipv4:
            return ipv4[0], socket.AF_INET  # Otherwise, use IPv4
    except socket.gaierror:
        print("Unable to resolve hostname.")
        return None, None

# sendfile as client.
def sendTCP():
    try:
        ip_address, family = resolve_address(HOST)
        client_socket = socket.socket(family, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        
        with open("output.jpg", 'rb') as file:
            client_socket.sendfile(file)
            client_socket.close()
    except socket.error as e:
        print(f"Socket error: {e}")
    client_socket = None
    # except socket.error as e:
    #     print("socket failed to connect using {sockaddr}: {e}")

activationTime = 10.0
framerate = 4.0
sendDelay = (1.0/framerate) # Time to delay between sending image capture signal and beginning image transfer.

# Start camera process
process = subprocess.Popen(CameraCmd)

# Get pid
pid = process.pid

# So we can close it with the following command in case of an emergency: kill "$(< pid.txt)"
with open("pid.txt", "w") as file:
    file.write(f"{pid}")

StopCmd.append(str(pid))
TakePictureCmd.append(str(pid))
print("activating")
time.sleep(activationTime)
print("activated")
counter = 0
while (True):
    time.sleep(0.15)
    subprocess.run(TakePictureCmd)
    time.sleep(0.15) 
    
    sendTCP()
    
    print("To stop, run:")
    print("kill" + " -SIGUSR2 " + str(pid))
    # print(SendCmd)
    counter += 1
    if CountLimited and counter > CountLimit:
        break
subprocess.run(StopCmd)
