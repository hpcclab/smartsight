#!/bin/bash

# Navigate to the project directory
cd ~/Documents/SmartSight/gst/ || exit

# Activate the virtual environment
source env/bin/activate

# Run the python script
python gstServer.py

# Deactivate the environment after the script finishes (optional)
deactivate