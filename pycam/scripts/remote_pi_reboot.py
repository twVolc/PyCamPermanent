#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Turns the other raspberry pi on through SSH"""

# Update python path so that pycam module can be found
import sys
sys.path.append('/home/pi/')

from pycam.utils import read_file
from pycam.networking import ssh
from pycam.setupclasses import FileLocator, ConfigInfo

import datetime
date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Get ip address
config = read_file(FileLocator.CONFIG)
pi_ip = config[ConfigInfo.pi_ip].split(',')

for ip in pi_ip:
    try:
        cli = ssh.open_ssh(ip)
        ssh.ssh_cmd(cli, 'sudo reboot now', background=False)
        ssh.close_ssh(cli)
    except BaseException as e:
        with open(FileLocator.MAIN_LOG_PI, 'a', newline='\n') as f:
            f.write('{} Error in pi SSH reboot, it must already be off or there is an issue with the network: {}\n'.format(date_str, e))