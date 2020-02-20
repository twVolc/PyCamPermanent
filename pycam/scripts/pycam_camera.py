# -*- coding: utf-8 -*-

"""Script to be run on the camera pi.
- Deals with socket communication and camera control.
"""

from pycam.controllers import Camera
from pycam.networking.sockets import PiSocketCam, PiSocketCamComms, read_network_file
from pycam.setupclasses import FileLocator
from pycam.utils import read_file

import threading

# Setup communication socket (for transferring imaging commands)
serv_ip, port = read_network_file(FileLocator.NET_COMM_FILE)
sock_comms = PiSocketCam(serv_ip, port)

config = read_file(FileLocator.CONFIG_CAM)

# Setup camera object
cam = Camera(band=config['band'])

# Setup image transfer socket
serv_ip, port = read_network_file(FileLocator.NET_TRANSFER_FILE)
sock_trf = PiSocketCam(serv_ip, port, camera=cam)

# Setup comms socket
serv_ip, port = read_file(FileLocator.NET_COMM_FILE)
sock_comms = PiSocketCamComms(serv_ip, port, camera=cam)


""" NEED SOMETHING LIKE THIS FOR DEALING WITH COMMS - this has been stolen directly from pycam_masterpi so needs rewriting"""
for key in comm_cmd:
    # Call correct method determined by 3 character code from comms message
    getattr(comms_funcs, key + '_comm')(comm_cmd[key], sock_serv_ext.get_connection(i), sock_dict, config)