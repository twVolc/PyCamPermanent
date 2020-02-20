# -*- coding: utf-8 -*-

"""Master script to be run on server pi for interfacing with worker pis and any external connection (such as a laptop
connected via ethernet)"""

from pycam.networking.sockets import SocketServer, CommsFuncs, recv_save_imgs, recv_save_spectra, recv_comms
from pycam.controllers import CameraSpecs
from pycam.setupclasses import FileLocator
from pycam.utils import read_file
from pycam.networking.ssh import open_ssh, close_ssh, ssh_cmd, file_upload

import threading
import queue
import socket

# ======================================================================================================================
#  NETWORK SETUP
# ======================================================================================================================
# Read configuration file which contains important
config = read_file('.\\config.txt')

# Extract ip addresses of camera pis and of server (local) pi
pi_ip = config['pi_ip'].split(',')

# Loop through remote pis and start scripts on them
remote_scripts = config['remote_scripts'].split(',')    # Remote scripts are held in the config file
ssh_clients = []
for ip in pi_ip:
    ssh_clients.append(open_ssh(ip))

    # First make sure the up-to-date specs file is present on the pi
    file_upload(ssh_clients[-1], config['cam_specs'], FileLocator.PYCAM_ROOT + FileLocator.CONFIG_CAM)

    # Loop through scripts and send command to start them
    for script in remote_scripts:
        ssh_cmd(ssh_clients[-1], script)

    # Close session
    # If other changes are needed later this line can be removed and clients should still be accessible in list
    close_ssh(ssh_clients[-1])

# ----------------------
# Setup Transfer port
# ----------------------
# Get network parameters for socket setup
# local_ip = config['host_ip']
local_ip = socket.gethostbyname(socket.gethostname())
port_transfer = int(config['port_transfer'])

# Open socket for server
sock_serv_transfer = SocketServer(local_ip, port_transfer)
sock_serv_transfer.open_socket()

# Get first 3 connections which should be the 2 cameras and 1 spectrometer
for i in range(3):
    sock_serv_transfer.acc_connection()

# ------------------------
# Setup communication port
# ------------------------
# Open socket on specified port
port_comm = int(config['port_comm'])
sock_serv_comm = SocketServer(local_ip, port_comm)
sock_serv_comm.open_socket()

# Get first 3 connections for communication - 2 cameras and 1 spectrometer
for i in range(3):
    sock_serv_comm.acc_connection()

# ------------------------
# Setup external port
# ------------------------
port_ext = int(config['port_ext'])
sock_serv_ext = SocketServer(local_ip, port_ext)
sock_serv_ext.open_socket()

# Create threads for accepting 2 new connections (one may be local computer conn, other may be wireless)
ext_comm_threads = []
for i in range(2):
    ext_comm_threads[i] = threading.Thread(target=sock_serv_ext.acc_connection, args=())
    ext_comm_threads[i].daemon = True
    ext_comm_threads[i].start()
receiving_ext_comms = []     # List holding thread names for all external comms which are currently being received

# Instantiate CommsFuncs object for controlling execution of external communication requests
comms_funcs = CommsFuncs()
# ======================================================================================================================

# ======================================================================================================================
# INSTRUMENT SETUP
# ======================================================================================================================
# Load Camera specs parameters
cam_specs = CameraSpecs(config['cam_specs'])
# ======================================================================================================================

# ======================================================================================================================
# FINAL LOOP - for dealing with communication between pis and any external computers
# ======================================================================================================================
# Setup event for controlling recv/save threads
recv_event = threading.Event()

# Use recv_save_imgs() in sockets to automate receiving and saving images from 2 cameras
save_threads_img = []
for i in range(2):
    save_threads_img.append(threading.Thread(
        target=recv_save_imgs, args=(sock_serv_transfer, sock_serv_transfer.get_connection(ip=pi_ip[i]), recv_event,)))
    save_threads_img[i].daemon = True
    save_threads_img[i].start()

# Do same for spectrum, which is on the local pi
save_thread_spec = threading.Thread(
    target=recv_save_spectra, args=(sock_serv_transfer, sock_serv_transfer.get_connection(ip=local_ip), recv_event,))
save_thread_spec.daemon = True
save_thread_spec.start()

while True:
    # NEXT DEV!!! Organise communication protocol here, using port_comms port from config. pycam_camera.py will need to setup a socket for this communication too, and have a way of dealing with this communication
    # If we have external computers receive their messages and act on them

    if len(sock_serv_ext.connections) > 0:

        # If we have more ext connections than we are receiving messages from, setup new thread to receive from most
        # recent connection
        # Determine how many new connections we need to setup receivers for
        num_conns = len(sock_serv_ext.connections) - receiving_ext_comms
        for i in range(num_conns):

            q = [queue.Queue(), queue.Queue]        # Generate queues to be passed to recevier thread
            name = 'Ext-comm thread: %i' % i        # Give thread a name

            # Start thread for receiving communications from external instruments
            t = threading.Thread(target=recv_comms,
                                 args=(sock_serv_ext, sock_serv_ext.connections[-1*i][0], q[0], q[1], ),
                                 name=name)
            t.daemon = True
            t.start()

            # A tuple is appended to receiving comms, representing that thread/connection (thread name, mess_q, close_q)
            receiving_ext_comms.append((name, q[0], q[1]))

        # Generate dictionary with latest connection object (may be obsolete redoing this each loop of the objects
        # auto-update within a dictionary? i.e. do they change when changed outside of the dictionary object?
        # Dictionary is used by all comms functions
        sock_dict = {'tsfr': sock_serv_transfer, 'comm': sock_serv_comm, 'ext': sock_serv_ext}

        # Check message queue in each comm port
        for i in range(len(receiving_ext_comms)):
            comm = receiving_ext_comms[i]

            try:
                # Check message queue (taken from tuple at position [1])
                comm_cmd = comm[1].get(block=False)
                if comm_cmd:

                    # Loop through each command code in the dictionary, carrying our the commands individually
                    for key in comm_cmd:
                        # Call correct method determined by 3 character code from comms message
                        getattr(comms_funcs, key)(comm_cmd[key], sock_serv_ext.get_connection(i), sock_dict, config)

            except queue.Empty:
                pass




        # Close everything if requested - THIS NEEDS REMOVING AND PLACING ELSEWHERE I THINK.
        if 'EXT' in comm_cmd.keys():
            if comm_cmd['EXT']:
                recv_event.set()    # Close threads for receiving images and spectra
                #DO EVERYTHING ELSE TO SHUTDOWN CAMERA

        # Pass message to pis if requested

    # Receive data from Pis


