#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Turns the other raspberry pi on through SSH"""

# Update python path so that pycam module can be found
import sys
sys.path.append('/home/pi/')

from pycam.utils import read_file
from pycam.networking import ssh
from pycam.setupclasses import FileLocator, ConfigInfo

# Get ip address
config = read_file(FileLocator.CONFIG)
pi_ip = config[ConfigInfo.pi_ip].split(',')

for ip in pi_ip:
    try:
        cli = ssh.open_ssh(ip)
        ssh.ssh_cmd(cli, 'sudo shutdown -h now', background=False)
        ssh.close_ssh(cli)
    except BaseException as e:
        print('Error in pi SSH shutdown, it must already be off or there is an issue with the network:')
        print(e)