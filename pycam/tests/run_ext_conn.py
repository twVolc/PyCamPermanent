# -*- coding:utf-8 -*-

"""Script to test running an external connection to pycam and passing it commands"""

import sys
sys.path.append('C:\\Users\\tw9616\\Documents\\PostDock\\Permanent Camera\\PyCamPermanent\\')

from pycam.networking.sockets import SocketClient
from pycam.utils import read_file
from pycam.setupclasses import FileLocator, ConfigInfo
import json

# Read configuration file which contains important information for various things
config = read_file('C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\pycam\\conf\\config.txt')

host_ip = config[ConfigInfo.host_ip]
port = int(config[ConfigInfo.port_ext])

# Setup socket and connect to it
print('Creating socket for {} on port {}'.format(host_ip, port))
sock_ext = SocketClient(host_ip, port)
sock_ext.connect_socket()
print(sock_ext.connect_stat)

while True:
    # Ask user for input
    cmd = input('Enter command dictionary to send to PyCam (Q:1 to Exit). String require double quotes: ')

    cmd_dict = json.loads(cmd)

    if 'Q' in cmd_dict:
        sys.exit()
    else:
        cmd_bytes = sock_ext.encode_comms(cmd_dict)
        sock_ext.send_comms(sock_ext.sock, cmd_bytes)

        # Test closing socket
        sock_ext.send_comms(sock_ext.sock, sock_ext.encode_comms({'EXT': 1}))


    ret_comm = sock_ext.recv_comms(sock_ext.sock)
    ret_dict = sock_ext.decode_comms(ret_comm)

    print(ret_dict)


