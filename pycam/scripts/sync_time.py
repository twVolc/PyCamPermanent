# -*- coding: utf-8 -*-

"""
Syncs time of local pi with remote pi.
This runs on startup and then ntp should keep the system accurate.
"""

import sys
sys.path.append('/home/pi/')

import datetime
from pycam.utils import read_file
from pycam.networking import ssh
from pycam.setupclasses import FileLocator, ConfigInfo

time_format = '%H:%M:%S %Y-%m-%d'

# Get ip address
config = read_file(FileLocator.CONFIG)
pi_ip = config[ConfigInfo.pi_ip].split(',')

for ip in pi_ip:
    cli = ssh.open_ssh(ip)
    current_time = datetime.datetime.now().strftime(time_format)
    stdin, stdout, stderr = ssh.ssh_cmd(cli, 'sudo date -s "' + current_time + '"', background=False)
    ssh.close_ssh(cli)
