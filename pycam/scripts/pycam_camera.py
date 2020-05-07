#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
Script to be run on the camera pi.
- Deals with socket communication and camera control.
"""

# Update python path so that pycam module can be found
import sys
sys.path.append('/home/pi/')

from pycam.controllers import Camera
from pycam.networking.sockets import PiSocketCam, PiSocketCamComms, read_network_file, recv_comms, send_imgs, \
    CommConnection, ImgSendConnection
from pycam.setupclasses import FileLocator
from pycam.utils import read_file

import threading
import queue
import time

# Read config file
config = read_file(FileLocator.CONFIG_CAM)

# -----------------------------------------------------------------
# Setup camera object
cam = Camera(band=config['band'])

# Initialise camera (may need to set shutter speed first?)
cam.initialise_camera()

# Setup thread for controlling camera capture
cam.interactive_capture()

# Start up continuous capture straight away
cam.capture_q.put({'start_cont': True})

# ------------------------------------------------------------------

# ----------------------------------------------------------------
# Setup image transfer socket
serv_ip, port = read_network_file(FileLocator.NET_TRANSFER_FILE)
sock_trf = PiSocketCam(serv_ip, port, camera=cam)
sock_trf.connect_socket()

# Start image sending thread
trf_conn = ImgSendConnection(sock_trf, cam.img_q, acc_conn=False)
trf_conn.thread_func()


# trf_event = threading.Event()
# thread_trf = threading.Thread(target=send_imgs, args=(sock_trf, cam.img_q, trf_event,))
# thread_trf.daemon = True
# thread_trf.start()
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Setup comms socket
serv_ip, port = read_network_file(FileLocator.NET_COMM_FILE)
sock_comms = PiSocketCamComms(serv_ip, port, camera=cam)
sock_comms.connect_socket()

# Setup comm connection and start the thread_func for communications
comm_connection = CommConnection(sock_comms, acc_conn=False)
comm_connection.connection = sock_comms.sock
comm_connection.thread_func()

# Pass tsfr and conn connection objects to comms socket object
sock_comms.transfer_connection = trf_conn
sock_comms.comm_connection = comm_connection
# -----------------------------------------------------------------

"""Final loop where all processes are carried out - mainly comms"""
while True:

    # --------------------------------------------------------------------------------------------
    # Receive comms message and act on it if we have something
    try:
        # Check message queue (taken from tuple at position [1])
        comm_cmd = comm_connection.q.get(block=False)
        if comm_cmd:

            # Loop through each command code in the dictionary, carrying our the commands individually
            for key in comm_cmd:
                # Call correct method determined by 3 character code from comms message
                # getattr(sock_comms, key + '_comm')(comm_cmd[key])
                try:
                    getattr(sock_comms, key)(comm_cmd[key])
                except AttributeError:
                    continue

                # if key == 'EXT':
                #     # time.sleep(10)
                #
                #     # Ensure transfer thread closes down
                #     trf_conn.event.set()
                #     trf_conn.q.put(['close', 1])
                #
                #     # Wait for comms connection to thread to close
                #     while comm_connection.working:
                #         pass
                #
                #     # Wait for image transfer thread to close
                #     while trf_conn.working:
                #         pass
                #
                #     # Wait for camera capture thread to close
                #     cam.capture_thread.join()
                #
                #     # Exit script by breaking loop
                #     sys.exit()

    except queue.Empty:
        pass
    # ---------------------------------------------------------------------------------------------

    # If our comms thread has died we try to reset it
    if not comm_connection.working:
        sock_comms.connect_socket()
        comm_connection.connection = sock_comms.sock
        comm_connection.thread_func()
