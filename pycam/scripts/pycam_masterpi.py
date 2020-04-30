#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""Master script to be run on server pi for interfacing with worker pis and any external connection (such as a laptop
connected via ethernet)"""

# Update python path so that pycam module can be found
import sys
sys.path.append('/home/pi/')

from pycam.networking.sockets import SocketServer, CommsFuncs, recv_save_imgs, recv_save_spectra, recv_comms, \
    acc_connection, SaveSocketError, ImgRecvConnection, SpecRecvConnection, CommConnection
from pycam.controllers import CameraSpecs, SpecSpecs
from pycam.setupclasses import FileLocator, ConfigInfo
from pycam.utils import read_file
from pycam.networking.ssh import open_ssh, close_ssh, ssh_cmd, file_upload

import threading
import queue
import socket
import subprocess
import os
import time

# Read configuration file which contains important information for various things
config = read_file(FileLocator.CONFIG)

# ======================================================================================================================
#  NETWORK SETUP
# ======================================================================================================================
# Extract ip addresses of camera pis and of server (local) pi
pi_ip = config[ConfigInfo.pi_ip].split(',')
host_ip = config[ConfigInfo.host_ip]

# ======================================================================================================================
# RUN EXTERNAL SCRIPTS TO START INSTRUMENTS AND OPEN THEIR SOCKETS
# ======================================================================================================================
# Loop through remote pis and start scripts on them
remote_scripts = config[ConfigInfo.remote_scripts].split(',')    # Remote scripts are held in the config file
ssh_clients = []
print('Running remote scripts...')
for ip in pi_ip:
    ssh_clients.append(open_ssh(ip))

    # First make sure the up-to-date specs file is present on the pi
    # POSSIBLY DONT DO THIS, AS WE MAY WANT THE REMOTE PROGRAM TO SAVE CAMERA PROPERTIES LOCALLY ON SHUTDOWN?
    file_upload(ssh_clients[-1], config[ConfigInfo.cam_specs], FileLocator.CONFIG_CAM)

    # Loop through scripts and send command to start them
    for script in remote_scripts:
        ssh_cmd(ssh_clients[-1], script)

    # Sleep so the kill_process.py has time to finish, as we don't want to kill the new camera script
    time.sleep(5)

    # Run core camera script
    ssh_cmd(ssh_clients[-1], config[ConfigInfo.cam_script])

    # Close session
    # If other changes are needed later this line can be removed and clients should still be accessible in list
    close_ssh(ssh_clients[-1])

# Run any other local scrips that need running
print('Running local scripts...')
local_scripts = config[ConfigInfo.local_scripts].split(',')
for script in local_scripts:
    script = script.split()
    script.append('&')
    subprocess.run(script, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Sleep so the kill_process.py has time to finish, as we don't want to kill the new spectrometer script
time.sleep(5)

# Run spectrometer script on local machine in background
subprocess.Popen(['python3', config[ConfigInfo.spec_script], '&'])
# subprocess.Popen(['python3', './pycam_spectrometer.py', '&'])
# os.system('python3 ./pycam_spectrometer &')
# ======================================================================================================================

# ======================================================================================================================
# Accept connections - now that external scripts to create clients have been run, there should be connections to accept
# ======================================================================================================================
# Open sockets for image/spectra transfer
port_transfer = int(config['port_transfer'])
sock_serv_transfer = SocketServer(host_ip, port_transfer)
sock_serv_transfer.open_socket()

# Open socket for communication with pis (cameras and spectrometer)
port_comm = int(config['port_comm'])
sock_serv_comm = SocketServer(host_ip, port_comm)
sock_serv_comm.open_socket()

# -------------------
# Transfer socket
# ------------------
# Get first 3 connections for transfer which should be the 2 cameras and 1 spectrometer
for i in range(3):
    print('Getting connection')
    connection = sock_serv_transfer.acc_connection()
    print('Got connection {}'.format(connection))

# Use recv_save_imgs() in sockets to automate receiving and saving images from 2 cameras
save_connections = dict()
for i in range(2):
    # Setup connection objects and start thread to run image transfer
    save_connections[pi_ip[i]] = ImgRecvConnection(sock_serv_transfer, acc_conn=False)

    # Set connection to be one of the camera IP connections
    save_connections[pi_ip[i]].connection = sock_serv_transfer.get_connection(ip=pi_ip[i])

    # Start save thread
    save_connections[pi_ip[i]].thread_func()

# Do same for spectrum, which is on the local pi
save_connections[host_ip] = SpecRecvConnection(sock_serv_transfer, acc_conn=False)

# Set connection to that of the host_ip spectrometer connection
save_connections[host_ip].connection = sock_serv_transfer.get_connection(ip=host_ip)

# Start save thread
save_connections[host_ip].thread_func()

# -----------------------
# Communications socket
# -----------------------
# Get first 3 connections for transfer which should be the 2 cameras and 1 spectrometer
for i in range(3):
    print('Getting connection')
    connection = sock_serv_comm.acc_connection()
    print('Got connection {}'.format(connection))

# Dictionary holding the connection for internal communications (not external comms)
comms_connections = dict()
for i in range(2):
    # Setup connection objects and start thread to run communcations
    comms_connections[pi_ip[i]] = CommConnection(sock_serv_comm, acc_conn=False)

    # Set connection to be one of the camera IP connections
    comms_connections[pi_ip[i]].connection = sock_serv_comm.get_connection(ip=pi_ip[i])

    # Start communications thread
    comms_connections[pi_ip[i]].thread_func()

# Do same for spectrometer, which is on the local pi
comms_connections[host_ip] = CommConnection(sock_serv_comm, acc_conn=False)

# Set connection to that of the host_ip spectrometer connection
comms_connections[host_ip].connection = sock_serv_comm.get_connection(ip=host_ip)

# Start comms thread
comms_connections[host_ip].thread_func()
# -----------------------------------------------------------------------------------------------------------

# ----------------------------------
# Setup external communication port
# ----------------------------------
port_ext = int(config['port_ext'])
sock_serv_ext = SocketServer(host_ip, port_ext)
sock_serv_ext.open_socket()

# Create obects for accepting and controlling 2 new connections (one may be local computer conn, other may be wireless)
ext_comms = [CommConnection(sock_serv_ext, acc_conn=True), CommConnection(sock_serv_ext, acc_conn=True)]

# Instantiate CommsFuncs object for controlling execution of external communication requests
comms_funcs = CommsFuncs()
# ======================================================================================================================

# ======================================================================================================================
# INSTRUMENT SETUP
# ======================================================================================================================
# Load Camera specs parameters - possibly not actually needed to be held here
cam_specs = CameraSpecs(FileLocator.CONFIG_CAM)
spec_specs = SpecSpecs(FileLocator.CONFIG_SPEC)
# ======================================================================================================================

# ======================================================================================================================
# FINAL LOOP - for dealing with communication between pis and any external computers
# ======================================================================================================================
while True:
    # Need to check that send_recv threads to cameras/spectrometers are still running. If not. Restart them.
    for key in save_connections:
        if not save_connections[key].working:
            # Probably need to write in a reconnection attempt into pycam_spectrometer and pycam_camera. If not,
            # I will need to assume the program has closed and restart it - then all connection would need to be
            # redone.
            pass

    # If a CommConnection object is neither waiting to accept a connection or recieving data from a connection, we
    # must have lost that connection, so we close that connection just to make sure, and then setup the object
    # to accept a new connection
    for comm_connection in ext_comms:
        if not comm_connection.working and not comm_connection.accepting:
            # Connection has probably already been close, but try closing it anyway
            try:
                sock_serv_ext.close_connection(ip=comm_connection.ip)
            except socket.error:
                pass

            # Setup object to accept new connection
            comm_connection.acc_connection()

    # Generate dictionary with latest connection object (may be obsolete redoing this each loop if the objects
    # auto-update within a dictionary? i.e. do they change when changed outside of the dictionary object?
    # Dictionary is used by all comms functions
    sock_dict = {'tsfr': sock_serv_transfer, 'comm': sock_serv_comm, 'ext': sock_serv_ext}

    # Check message queue in each comm port
    for comm_connection in ext_comms:
        try:
            # Check message queue (taken from tuple at position [1])
            comm_cmd = comm_connection.q.get(block=False)
            print(comm_cmd)
            if comm_cmd:

                # Forward command to all communication sockets (2 cameras and 1 spectrometer)
                sock_serv_comm.send_to_all(comms_connections[key])

                """An easier way of doing this may be to just forward it all to all instruments and let them
                decide how to act on it from there? It would mean not separating each message into individual
                keys at this point"""
                # Loop through each command code in the dictionary, carrying our the commands individually
                for key in comm_cmd:
                    if key is not 'ERR':
                        # Call correct method determined by 3 character code from comms message
                        getattr(comms_funcs, key)(comm_cmd[key], comm_connection.connection, sock_dict, config)

                # If spectrometer restart is requested we need to reset all socket communications associated with
                # the spectrometer and setup new ones
                # Restarting the program itself is handled by the pycam_spectrometer script, so we don't need to do this
                if 'RSS' in comm_cmd.keys():
                    if comm_cmd['RSS']:

                        # Always do transfer socket first and then comms socket (so scripts don't hang trying to
                        # connect for something when the socket isn't listening - may not be necessary as the
                        # socket listen command can listen without accepting)
                        # Close old transfer socket
                        sock_serv_transfer.close_connection(ip=host_ip)

                        # Wait until receiving function has finished, which should be immediately after the
                        # connection is closed
                        while save_connections[host_ip].working:
                            pass

                        # Accept new spectrum transfer connection and begin receiving loop automatically
                        save_connections[host_ip].acc_connection()

                        # First remove previous spectrometer connection
                        sock_serv_comm.close_connection(ip=host_ip)

                        # Wait for receiving function to close
                        while comms_connections[host_ip].working:
                            pass

                        # Accept new connection and start receiving comms
                        comms_connections[host_ip].acc_connection()

                # As with spectrometer we need to do the same with the cameras if restart is requested
                # Restarting the program itself is handled by the pycam_camera script, so we don't need to do this
                if 'RSC' in comm_cmd.keys():
                    if comm_cmd['RSC']:
                        for ip in pi_ip:
                            # Close image transfer connection at defined ip address
                            sock_serv_transfer.close_connection(ip=ip)

                            # Wait for save thread to finish
                            while save_connections[ip].working:
                                pass

                            # Setup new transfer connection and begin receiving automatically
                            save_connections[ip].acc_connection()

                            # Close/remove previous comms connection for ip address
                            sock_serv_comm.close_connection(ip=ip)

                            # Wait for comm thread to finish
                            while comms_connections[ip].working:
                                pass

                            # Accept new connection and start receiving comms
                            comms_connections[ip].acc_connection()

                # ------------------------------------------------------------------------------------
                # Close everything if requested
                if 'EXT' in comm_cmd.keys():
                    if comm_cmd['EXT']:
                        # Let EXT comms be extended to other RPis by sleeping briefly
                        time.sleep(3)

                        # Close all sockets (must close connections before socket)
                        for conn in sock_serv_ext.connections[:]:
                            sock_serv_ext.close_connection(connection=conn[0])
                        sock_serv_ext.close_socket()

                        for conn in sock_serv_comm.connections[:]:
                            sock_serv_comm.close_connection(connection=conn[0])
                        sock_serv_comm.close_socket()

                        for conn in sock_serv_transfer.connections[:]:
                            sock_serv_transfer.close_connection(connection=conn[0])
                        sock_serv_transfer.close_socket()

                        # Wait for all threads to finish (closing sockets should cause this)
                        for comm_connection in ext_comms:
                            while comm_connection.working or comm_connection.accepting:
                                pass

                        for connection in save_connections:
                            while save_connections[connection].working or save_connections[connection].accepting:
                                pass

                        for connection in comms_connections:
                            while comms_connections[connection].working or comms_connections[connection].accepting:
                                pass

                        sys.exit(0)
                # --------------------------------------------------------------------------------------

        except queue.Empty:
            pass

    # Receive data from pis and simply forward them on to remote computers
    for key in comms_connections:
        comm = comms_connections[key].q

        try:
            # Get message from queue if there is one
            comm_cmd = comm.get(block=False)
            if comm_cmd:

                # Forward message to all external comm ports
                sock_serv_ext.send_to_all(comm_cmd)

        except queue.Empty:
            pass




