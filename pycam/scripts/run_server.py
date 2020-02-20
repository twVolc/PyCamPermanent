#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
Script to be run on Raspberry Pi server.
Script opens server socket and receives images from the camera socket
"""

from pycam.networking.sockets import SocketServer
from pycam.controllers import Camera

# Create Camera object
cam = Camera()

# Instantiate server socket
sock_serv = SocketServer('169.254.10.180', 12345)

# Open socket and listen for connection and accept 1st connection
sock_serv.open_socket()
sock_serv.acc_connection()
print('Got Connection')

while True:
    # Receive image data
    image, filename = sock_serv.recv_img()
    print('Got image data')

    # Save image
    cam.image = image
    cam.save_current_image(filename)
