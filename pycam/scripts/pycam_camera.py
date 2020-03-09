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
from pycam.networking.sockets import PiSocketCam, PiSocketCamComms, read_network_file, recv_comms, send_imgs
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
capt_thread = threading.Thread(target=cam.interactive_capture, args=())
capt_thread.daemon = True
capt_thread.start()

# Start up continuous capture straight away
cam.capture_q.put({'start_cont': True})

# ------------------------------------------------------------------

# ----------------------------------------------------------------
# Setup image transfer socket
serv_ip, port = read_network_file(FileLocator.NET_TRANSFER_FILE)
sock_trf = PiSocketCam(serv_ip, port, camera=cam)
sock_trf.connect_socket()

# Start image sending thread
trf_event = threading.Event()
thread_trf = threading.Thread(target=send_imgs, args=(sock_trf, cam.img_q, trf_event,))
thread_trf.daemon = True
thread_trf.start()
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Setup comms socket
serv_ip, port = read_file(FileLocator.NET_COMM_FILE)
sock_comms = PiSocketCamComms(serv_ip, port, camera=cam)
sock_comms.connect_socket()
q_comm = queue.Queue()              # Queue for putting received comms in
comm_event = threading.Event()      # Event to shut thread down

# Start comms receiving thread
thread_comm = threading.Thread(target=recv_comms, args=(sock_comms, sock_comms.sock, q_comm, comm_event,))
thread_comm.daemon = True
thread_comm.start()
# -----------------------------------------------------------------

"""Final loop where all processes are carried out - mainly comms"""
while True:

    # --------------------------------------------------------------------------------------------
    # Receive comms message and act on it if we have something
    try:
        # Check message queue (taken from tuple at position [1])
        comm_cmd = q_comm.get(block=False)
        if comm_cmd:

            # Loop through each command code in the dictionary, carrying our the commands individually
            for key in comm_cmd:
                # Call correct method determined by 3 character code from comms message
                getattr(sock_comms, key + '_comm')(comm_cmd[key])

                if key == 'EXT':
                    # Close down all threads
                    comm_event.set()
                    trf_event.set()
                    thread_comm.join()
                    thread_trf.join()
                    capt_thread.join()

                    # Close sockets
                    sock_comms.close_socket()
                    sock_trf.close_socket()

                    # Exit script by breaking loop
                    break

    except queue.Empty():
        pass
    # ---------------------------------------------------------------------------------------------

    # If our comms thread has died we try to reset it
    if not thread_comm.is_alive():
        sock_comms.connect_socket()
        comm_event = threading.Event()  # Event to shut thread down

        # Start comms receiving thread
        thread_comm = threading.Thread(target=recv_comms, args=(sock_comms, sock_comms.sock, q_comm, comm_event,))
        thread_comm.daemon = True
        thread_comm.start()