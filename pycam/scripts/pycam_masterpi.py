#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""Master script to be run on server pi for interfacing with worker pis and any external connection (such as a laptop
connected via ethernet)"""

# Update python path so that pycam module can be found
import sys
sys.path.append('/home/pi/')

from pycam.networking.sockets import SocketServer, CommsFuncs, recv_save_imgs, recv_save_spectra, recv_comms, \
    acc_connection
from pycam.controllers import CameraSpecs
from pycam.setupclasses import FileLocator, ConfigInfo
from pycam.utils import read_file
from pycam.networking.ssh import open_ssh, close_ssh, ssh_cmd, file_upload

import threading
import queue
import socket
import subprocess

# Read configuration file which contains important information for various things
config = read_file(FileLocator.CONFIG)

# ======================================================================================================================
#  NETWORK SETUP
# ======================================================================================================================
# Extract ip addresses of camera pis and of server (local) pi
pi_ip = config[ConfigInfo.pi_ip].split(',')

# ----------------------------------
# Open sockets so they are listening
# ----------------------------------
# Get network parameters for socket setup
# host_ip = config['host_ip']
# host_ip = socket.gethostbyname(socket.gethostname())
host_ip = config[ConfigInfo.host_ip]
port_transfer = int(config['port_transfer'])

# Open sockets for image/spectra transfer
sock_serv_transfer = SocketServer(host_ip, port_transfer)
sock_serv_transfer.open_socket()

# Open socket for communication with pis (cameras and spectrometer)
port_comm = int(config['port_comm'])
sock_serv_comm = SocketServer(host_ip, port_comm)
sock_serv_comm.open_socket()

# ======================================================================================================================
# RUN EXTERNAL SCRIPTS TO START INSTRUMENTS AND OPEN THEIR SOCKETS
# ======================================================================================================================
# Loop through remote pis and start scripts on them
remote_scripts = config[ConfigInfo.remote_scripts].split(',')    # Remote scripts are held in the config file
ssh_clients = []
for ip in pi_ip:
    ssh_clients.append(open_ssh(ip))

    # First make sure the up-to-date specs file is present on the pi
    # POSSIBLY DONT DO THIS, AS WE MAY WANT THE REMOTE PROGRAM TO SAVE CAMERA PROPERTIES LOCALLY ON SHUTDOWN?
    file_upload(ssh_clients[-1], config[ConfigInfo.cam_specs], FileLocator.CONFIG_CAM)

    # Loop through scripts and send command to start them
    for script in remote_scripts:
        ssh_cmd(ssh_clients[-1], script)

    # Run core camera script
    ssh_cmd(ssh_clients[-1], config[ConfigInfo.cam_script])

    # Close session
    # If other changes are needed later this line can be removed and clients should still be accessible in list
    close_ssh(ssh_clients[-1])

# Run any other local scrips that need running
for script in config[ConfigInfo.local_scripts]:
    subprocess.run([script])

# Run spectrometer script on local machine in background
subprocess.run(config[ConfigInfo.spec_script] + ' &')
# ======================================================================================================================

# ======================================================================================================================
# Accept connections - now that external scripts to create clients have been run, there should be connections to accept
# ======================================================================================================================
# -------------------
# Transfer socket
# ------------------
# Get first 3 connections for transfer which should be the 2 cameras and 1 spectrometer
for i in range(3):
    sock_serv_transfer.acc_connection()

# Use recv_save_imgs() in sockets to automate receiving and saving images from 2 cameras
save_threads = dict()
for i in range(2):
    # Setup event for controlling recv/save threads
    recv_event = threading.Event()

    # Staart thread
    save_threads[pi_ip[i]] = (threading.Thread(
        target=recv_save_imgs, args=(sock_serv_transfer, sock_serv_transfer.get_connection(ip=pi_ip[i]), recv_event,)),
        recv_event)
    save_threads[pi_ip[i]][0].daemon = True
    save_threads[pi_ip[i]][0].start()

# Do same for spectrum, which is on the local pi
recv_event = threading.Event()  # Have different recv_event so we can stop this thread separately to camera thread
save_threads[host_ip] = (threading.Thread(
    target=recv_save_spectra, args=(sock_serv_transfer, sock_serv_transfer.get_connection(ip=host_ip), recv_event,)),
                         recv_event)
save_threads[host_ip][0].daemon = True
save_threads[host_ip][0].start()

# -----------------------
# Communications socket
# -----------------------
# List to hold thread info for comm ports
cam_spec_comms = dict()

# Get first 3 connections for communication - 2 cameras and 1 spectrometer, then setup thread to listen to them
for i in range(3):
    # Accept connection and set up thread
    ret_tup = acc_connection(sock_serv_comm, recv_comms)

    # Now the connection has been accepted, get most recent connection ip and use as key to set dictionary value
    cam_spec_comms[sock_serv_comm.get_ip(conn_num=-1)] = ret_tup

    # sock_serv_comm.acc_connection()
    #
    # # Thread communication functions for cameras and spectrometer
    # q = [queue.Queue(), threading.Event()]      # Generate queue and event to be passed to receiver thread
    #
    # # Start thread for receiving communications from external instruments
    # t = threading.Thread(target=recv_comms,
    #                      args=(sock_serv_comm, sock_serv_comm.connections[i][0], q[0], q[1], ))
    # t.daemon = True
    # t.start()
    #
    # # A tuple is appended to receiving comms, representing that thread/connection (thread name, mess_q, close_q, ip)
    # cam_spec_comms[sock_serv_comm.get_ip(conn_num=i)] = (t, q[0], q[1])
# -----------------------------------------------------------------------------------------------------------

# ----------------------------------
# Setup external communication port
# ----------------------------------
port_ext = int(config['port_ext'])
sock_serv_ext = SocketServer(host_ip, port_ext)
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
# Load Camera specs parameters - possibly not actually needed to be held here
cam_specs = CameraSpecs(config[ConfigInfo.cam_specs])
# ======================================================================================================================

# ======================================================================================================================
# FINAL LOOP - for dealing with communication between pis and any external computers
# ======================================================================================================================
while True:

    if len(sock_serv_ext.connections) > 0:

        # If we have more recorded threads than we have socket connections a comm connection must have been closed,
        # so we update the list to reflect this and run thread to accept a new connection
        # If we have a thread which has closed down because there is no connection, we remove it from the
        # receiving_ext_comms list. NOTE: This assume the thread closes when the connection closes - CHECK THIS
        for i in range(len(receiving_ext_comms)):
            if not receiving_ext_comms[i][0].is_alive():
                # Remove that connection (we are assuming it is dead)
                sock_serv_ext.close_connection(ip=receiving_ext_comms[i][3])

                del receiving_ext_comms[i]

        # Setup thread to receive new connection if any have closed
        # (we need to be ready to accept new connections after one closes)
        # These threads should close as soon as they have received a connection
        for i in range(len(ext_comm_threads)):
            if not ext_comm_threads[i].is_alive():
                ext_comm_threads[i] = threading.Thread(target=sock_serv_ext.acc_connection, args=())
                ext_comm_threads[i].daemon = True
                ext_comm_threads[i].start()

        # If we have more ext connections than we are receiving messages from, setup new thread to receive from most
        # recent connection
        # Determine how many new connections we need to setup receivers for
        num_conns = len(sock_serv_ext.connections) - len(receiving_ext_comms)
        for i in range(num_conns):

            # Generate queue and event to be passed to receiver thread
            q = [queue.Queue(), threading.Event()]

            # Calculate the connection index for the connections we need to setup new receivers for
            # We start at the oldest connection that hasn't been setup and as we loop through we move towards the most
            # recent connections
            conn_num = -num_conns + i

            # Retrieve connection ip
            conn_ip = sock_serv_ext.get_ip(conn_num=conn_num)

            # Start thread for receiving communications from external instruments
            t = threading.Thread(target=recv_comms,
                                 args=(sock_serv_ext, sock_serv_ext.connections[conn_num][0], q[0], q[1], ))
            t.daemon = True
            t.start()

            # A tuple is appended to receiving comms, representing that thread/connection (thread name, mess_q, close_q)
            receiving_ext_comms.append((t, q[0], q[1], conn_ip))

        # Generate dictionary with latest connection object (may be obsolete redoing this each loop if the objects
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

                    """An easier way of doing this may be to just forward it all to all instruments and let them
                    decide how to act on it from there? It would mean not separating each message into individual
                    keys at this point"""
                    # Loop through each command code in the dictionary, carrying our the commands individually
                    for key in comm_cmd:
                        # Call correct method determined by 3 character code from comms message
                        getattr(comms_funcs, key)(comm_cmd[key], sock_serv_ext.get_connection(i), sock_dict, config)

                    # If spectrometer restart is requested we need to reset all socket communications associated with
                    # the spectrometer and setup new ones
                    if 'RSS' in comm_cmd.keys():
                        if comm_cmd['RSS']:
                            # First join previous threads
                            cam_spec_comms[host_ip][2].set()
                            save_threads[host_ip][1].set()
                            cam_spec_comms[host_ip][0].join()
                            save_threads[host_ip][0].join()

                            # Always do transfer socket first and then comms socket (so scripts don't hang trying to
                            # connect for something when the socket isn't listening - may not be necessary as the socket
                            # listen command can listen without accepting)
                            # Close old transfer socket
                            sock_serv_transfer.close_connection(ip=host_ip)

                            # Setup new transfer socket
                            sock_serv_transfer.acc_connection()

                            recv_event = threading.Event()
                            save_threads[host_ip] = (threading.Thread(
                                target=recv_save_spectra,
                                args=(sock_serv_transfer, sock_serv_transfer.get_connection(ip=host_ip), recv_event,)),
                                                     recv_event)
                            save_threads[host_ip][0].daemon = True
                            save_threads[host_ip][0].start()

                            # First remove previous spectrometer connection
                            sock_serv_comm.close_connection(ip=host_ip)

                            # Accept new connection and start receiving comms
                            cam_spec_comms[host_ip] = acc_connection(sock_serv_comm, recv_comms)

                    # As with spectrometer we need to do the same with the cameras if restart is requested
                    if 'RSC' in comm_cmd.keys():
                        if comm_cmd['RSC']:
                            for ip in pi_ip:
                                # Join previous threads
                                cam_spec_comms[ip][2].set()
                                save_threads[ip][1].set()
                                cam_spec_comms[ip][0].join()
                                save_threads[ip][0].join()

                                # Close/remove old transfer socket
                                sock_serv_transfer.close_connection(ip=ip)

                                # Setup new transfer socket
                                sock_serv_transfer.acc_connection()

                                # We can't be certain the first accepted conn will be the ip we are working with
                                # So just get the ip of the most recent connection
                                last_ip_conn = sock_serv_transfer.get_ip(conn_num=-1)

                                recv_event = threading.Event()
                                save_threads[last_ip_conn] = (threading.Thread(
                                    target=recv_save_spectra,
                                    args=(sock_serv_transfer, sock_serv_transfer.get_connection(ip=last_ip_conn),
                                    recv_event,)), recv_event)
                                save_threads[last_ip_conn][0].daemon = True
                                save_threads[last_ip_conn][0].start()

                                # Close/remove previous connection for ip address
                                sock_serv_comm.close_connection(ip=ip)

                                # Accept new connection and start receiving comms
                                ret_tup = acc_connection(sock_serv_comm, recv_comms)
                                cam_spec_comms[sock_serv_comm.get_ip(conn_num=-1)] = ret_tup

                    # ------------------------------------------------------------------------------------
                    # Close everything if requested
                    if 'EXT' in comm_cmd.keys():
                        if comm_cmd['EXT']:
                            for key in save_threads:
                                save_threads[key][1].set()      # Close threads for receiving images and spectra

                            # Close threads for comms in all cases
                            for comm_tup in receiving_ext_comms:
                                comm_tup[2].set()

                            # Close all sockets
                            sock_serv_ext.close_socket()
                            sock_serv_comm.close_socket()
                            sock_serv_transfer.close_socket()

                            # Wait for all threads to finish
                            for thread in ext_comm_threads:
                                thread.join()
                            for thread_key in save_threads:
                                save_threads[thread_key][0].join()
                    # --------------------------------------------------------------------------------------

            except queue.Empty:
                pass

    # Receive data from pis and simply forward them on to remote computers
    for key in cam_spec_comms:
        comm = cam_spec_comms[key]

        try:
            # Get message from queue if there is one
            comm_cmd = comm[1].get(block=False)
            if comm_cmd:

                # Forward message to all external comm ports
                sock_serv_ext.send_to_all(comm_cmd)

        except queue.Empty:
            pass



