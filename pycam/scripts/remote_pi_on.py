# -*- coding: utf-8 -*-

"""Turns the other raspberry pi on through gpio pin"""

# Update python path so that pycam module can be found
import sys
sys.path.append('/home/pi/')

import RPi.GPIO as GPIO
import os
import time
from pycam.utils import read_file
from pycam.setupclasses import FileLocator, ConfigInfo
import datetime

# Use BCM rather than board numbers
GPIO.setmode(GPIO.BCM)

# Set GPIO pin
channel = 23

# Setup as output
GPIO.setup(channel, GPIO.OUT)

# Get ip address
config = read_file(FileLocator.CONFIG)
pi_ip = config[ConfigInfo.pi_ip].split(',')
stat_dict = {}
stat_dict_on = {}
for ip in pi_ip:
    stat_dict_on[ip] = True
    stat_dict[ip] = False


while stat_dict != stat_dict_on:
    # Send pulse to turn off pi
    GPIO.output(channel, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(channel, GPIO.LOW)
    time.sleep(20)

    # For each pi attempt to connect. If we can't we flag that this pi is now turned off
    for ip in pi_ip:
        if not stat_dict[ip]:
            date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ret = os.system("ping -w 1 {}".format(ip))
            if ret == 0:
                pass
                # with open(FileLocator.MAIN_LOG_PI, 'a', newline='\n') as f:
                #     f.write("{} remote_pi_on.py: {} now turned on\n".format(date_str, ip))
                stat_dict[ip] = True
            else:
                with open(FileLocator.MAIN_LOG_PI, 'a', newline='\n') as f:
                    f.write("{} remote_pi_on.py: {} no longer reachable\n".format(date_str, ip))


# Once turned on we shut it down again (since we seem to have SSH errors from startup
# Set GPIO pin
channel_off = 16

# Send pulse to turn off pi
GPIO.setup(channel_off, GPIO.OUT)
GPIO.output(channel_off, GPIO.HIGH)
time.sleep(0.2)
GPIO.output(channel_off, GPIO.LOW)
time.sleep(0.2)
GPIO.output(channel_off, GPIO.HIGH)
time.sleep(30)

# ------------------------------------------------------------
# Then reboot again
stat_dict = {}
stat_dict_on = {}
for ip in pi_ip:
    stat_dict_on[ip] = True
    stat_dict[ip] = False

while stat_dict != stat_dict_on:
    # Send pulse to turn off pi
    GPIO.output(channel, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(channel, GPIO.LOW)
    time.sleep(20)

    # For each pi attempt to connect. If we can't we flag that this pi is now turned off
    for ip in pi_ip:
        if not stat_dict[ip]:
            date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ret = os.system("ping -w 1 {}".format(ip))
            if ret == 0:
                with open(FileLocator.MAIN_LOG_PI, 'a', newline='\n') as f:
                    f.write("{} remote_pi_on.py: {} now turned on\n".format(date_str, ip))
                stat_dict[ip] = True
            else:
                with open(FileLocator.MAIN_LOG_PI, 'a', newline='\n') as f:
                    f.write("{} remote_pi_on.py: {} no longer reachable\n".format(date_str, ip))
print('Exiting pi startup script')

# Cleanup at the end
GPIO.cleanup()