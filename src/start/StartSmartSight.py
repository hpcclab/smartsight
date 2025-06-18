import paramiko
import os
import wmi
import subprocess
import socket
import ipaddress
import Server

# Start Smart Sight (WINDOWS EDGE PROGRAM)

# TODO: SHOULD BE TRUE FOR FINAL PRESENTATION (If remote start is working by then)
StartCapture = False

# # Example usage
# add_ip_to_known_hosts('192.168.1.1')
# Create a WMI client
c = wmi.WMI()

IPFile = "LaptopIP.txt"

# Get the network adapters
adapters = c.Win32_NetworkAdapterConfiguration(IPEnabled=True)
IPAddress = "LAPTOP.IP.NOT.FOUND."
# for adapter in adapters:
#     print(f"Adapter: {adapter.Description}")
#     print(f"IP Address: {adapter.IPAddress[1]}")
#     print(f"Subnet Mask: {adapter.IPSubnet[0]}")
#     print(f"Default Gateway: {adapter.DefaultIPGateway[0]}")
#     print(f"DNS Servers: {adapter.DNSServerSearchOrder}")
#     print("----------")

def is_ipv4_address(string):
    try:
        ipaddress.IPv4Address(string)
        return True
    except ipaddress.AddressValueError:
        return False

NeedPercent = False
GadgetNotFound = True
IPv6NotFound = True
for adapter in adapters:
    if "Gadget" in str(adapter.Description):
        GadgetNotFound = False
        for ip in adapter.IPAddress:
            if "fe80" in str(ip) or is_ipv4_address(ip):
                IPAddress = str(ip)
                IPv6NotFound = False
                if "fe80" in str(ip):
                    NeedPercent = True
if GadgetNotFound:
    print("ERROR: Gaget network adapter NOT FOUND")
    quit()
if IPv6NotFound:
    print("ERROR USB ipv6 NOT FOUND")
    quit()


EdgeABSPath = os.path.abspath(__file__)
EdgeUsername = os.environ.get('USERNAME')
EdgeABSPath = EdgeABSPath.rsplit('\\', 1)[0]

# PubKeyPath = "C:\ProgramData\ssh\ssh_host_ecdsa_key.pub"
# with open(PubKeyPath, "r") as f:
#     SSHServerPubKey = f.read()
#     f.close()
# SSHServerPubKey = SSHServerPubKey.split('=', 1)[0]
# SSHServerPubKey += "="

SSHServerPubKey = "NonAdminFriendlyPlaceholder="

if NeedPercent:
    IPAddress += "%usb0"
print(IPAddress)
with open(IPFile, "w") as f:
    f.write(IPAddress + "\n")
    f.write(SSHServerPubKey + "\n")
    f.write(EdgeABSPath + "\n")
    f.write(EdgeUsername)

    f.close()

# Check for the device

# Define the host to ping
host = "raspberrypi.local"

# Define the command to run if the ping is successful
command = ["echo", "Ping successful!"]

# Function to ping a host
def ping(host):
    try:
        # Use subprocess to ping the host
        output = subprocess.check_output(["ping", "-n", "1", host], stderr=subprocess.STDOUT, universal_newlines=True)
        return True
    except subprocess.CalledProcessError:
        return False

# Try pinging until we get a hit.
while True:
  # Check if the ping is successful
  if ping(host):
      # Run the command if the ping is successful
      print("Ping success")
      break
  else:
      print("Ping failed. Retrying...")

# Start the Camera capture application
DeviceIP = "None"
# Resolve mDNS hostname to IP
while True: # Continue trying to resolve hostname until it works.
  try:
      DeviceIP = socket.gethostbyname(host)
  except socket.gaierror:
      print(f"Failed to resolve hostname: {host}")
      continue
  break
print(f"Success resolving hostname: {host} to {DeviceIP}")
    

# SendCmd = ["scp", "-i", "C:\Users\jacob\.ssh\PC-rPi0Key", "LaptopIP.txt", "jPi0@raspberrypi.local:~/Documents/SmartSight/SendImage/LaptopIP.txt"]
SendCmd = ["scp", "-i", "C:\\Users\\jacob\\.ssh\\PC-rPi0Key", "LaptopIP.txt", "jPi0@" + DeviceIP + ":~/Documents/SmartSight/SendImage/LaptopIP.txt"]
result = subprocess.run(SendCmd)
print(result.stdout)


if StartCapture:

    # SSH server credentials
    hostname = DeviceIP # Raspberry Pi IP
    port = 22  # Default SSH port
    username = "jPi0"  # Replace with your username

    # Command to execute
    command = "python ~/Documents/SmartSight/SendImage/CaptureImage.py"  # Replace with the command you want to run

    # Create an SSH client
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the Raspberry Pi
        ssh_client.connect(hostname, port, username, key_filename="C:\\Users\\jacob\\.ssh\\PC-rPi0Key")
        
        # Execute the command
        stdin, stdout, stderr = ssh_client.exec_command(command)
        
        # Print the output
        print(stdout.read().decode())
        print(stderr.read().decode())
        
        
    finally:
        # Close the connection
        ssh_client.close()
