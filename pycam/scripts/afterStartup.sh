#!/bin/bash
# file: afterStartup.sh
#
# This script will be executed in background after Witty Pi 3 gets initialized.
# If you want to run your commands after boot, you can place them here.
#

/usr/bin/python3 /home/pi/pycam/scripts/remote_pi_on.py > /home/pi/wittypi/pi_on.log
/usr/bin/python3 /home/pi/pycam/scripts/sync_time.py
