# -*- coding: utf-8 -*-

"""
Script runs in a similar was to pycam_masterpi, initiating all other neccessary scripts, but it then just requests
that the instruments run their dark capture procedure (acquirign darks at all shutter speeds). Following this the
script shutsdown.
NOTE: pycam_masterpi should not be running at the time of starting this script - it will be killed if it is. This is
preferred to just connecting to that pycam_masterpi instance and requesting dark images because pycam_masterpi could
later be killed by a scheduled kill request
Also the Raspberry Pi must be turned on at the time that this script is scheduled to start!
"""

import sys
sys.path.append('/home/pi/')
import os
import subprocess
import time
import queue
import socket
from pycam.utils import read_file, StorageMount, write_file
from pycam.setupclasses import FileLocator, ConfigInfo, CameraSpecs, SpecSpecs
from pycam.networking.sockets import SocketClient, SocketServer, ImgRecvConnection, SpecRecvConnection, \
    SocketNames, CommConnection, MasterComms, CommsFuncs
from pycam.networking.ssh import open_ssh, close_ssh, ssh_cmd, file_upload
import atexit


# Read configuration file which contains important information for various things
config = read_file(FileLocator.CONFIG)
host_ip = config[ConfigInfo.host_ip]
pi_ip = config[ConfigInfo.pi_ip].split(',')

# Setup mount object
storage_mount = StorageMount()
storage_mount.mount_dev()
atexit.register(storage_mount.unmount_dev)      # Unmount device when script closes

# Kill pycam if it is already running
stop_script = config[ConfigInfo.stop_script]
stop_script_name = os.path.split(stop_script)[-1]
subprocess.call(['python3', stop_script_name, '&'])         # Use call() as this waits until the script completes

# ======================================================================================================================
# SOCKET SERVER SETUP
# Open sockets for image/spectra transfer
sock_serv_transfer = SocketServer(host_ip, None)
sock_serv_transfer.get_port_list('transfer_ports')
sock_serv_transfer.get_port()       # Check which port is available from port list

# Open socket for communication with pis (cameras and spectrometer)
sock_serv_comm = SocketServer(host_ip, None)
sock_serv_comm.get_port_list('comm_ports')
sock_serv_comm.get_port()           # Check which port is available from port list

# Write port info to file
write_file(FileLocator.NET_COMM_FILE, {'ip_address': sock_serv_comm.host_ip, 'port': sock_serv_comm.port})
write_file(FileLocator.NET_TRANSFER_FILE, {'ip_address': sock_serv_transfer.host_ip, 'port': sock_serv_transfer.port})

# ======================================================================================================================
# RUN EXTERNAL SCRIPTS TO START INSTRUMENTS AND OPEN THEIR SOCKETS
# ======================================================================================================================
# Loop through remote pis and start scripts on them
remote_scripts = config[ConfigInfo.remote_scripts].split(',')    # Remote scripts are held in the config file
ssh_clients = []
print('Running remote scripts...')
for ip in pi_ip:
    ssh_clients.append(open_ssh(ip))

    # Upload network port information files
    file_upload(ssh_clients[-1], FileLocator.NET_COMM_FILE, FileLocator.NET_COMM_FILE)
    file_upload(ssh_clients[-1], FileLocator.NET_TRANSFER_FILE, FileLocator.NET_TRANSFER_FILE)

    # First make sure the up-to-date specs file is present on the pi
    # POSSIBLY DONT DO THIS, AS WE MAY WANT THE REMOTE PROGRAM TO SAVE CAMERA PROPERTIES LOCALLY ON SHUTDOWN?
    file_upload(ssh_clients[-1], config[ConfigInfo.cam_specs], FileLocator.CONFIG_CAM)

    # Loop through scripts and send command to start them
    for script in remote_scripts:
        ssh_cmd(ssh_clients[-1], 'python3 ' + script)

    # Sleep so the kill_process.py has time to finish, as we don't want to kill the new camera script
    time.sleep(5)

    # Run core camera script
    ssh_cmd(ssh_clients[-1], 'python3 ' + config[ConfigInfo.cam_script] + ' 0' + ' > pycam_camera.out 2>&1')

    # Run core spectrometer script
    ssh_cmd(ssh_clients[-1], 'python3 ' + config[ConfigInfo.spec_script] + ' 0' + ' > pycam_spectrometer.out 2>&1')

    # Close session
    # If other changes are needed later this line can be removed and clients should still be accessible in list
    close_ssh(ssh_clients[-1])

# Run any other local scrips that need running
print('Running local scripts...')
local_scripts = config[ConfigInfo.local_scripts].split(',')
for script in local_scripts:
    script = script.split()
    script.append('&')
    py_cmd = ['python3']
    py_cmd.extend(script)
    subprocess.run(py_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Sleep so the kill_process.py has time to finish, as we don't want to kill the new spectrometer script
time.sleep(5)

# Run camera script on local machine in the background
subprocess.Popen(['python3', config[ConfigInfo.cam_script], '0', '&'])
# ======================================================================================================================

# ======================================================================================================================
# Accept connections - now that external scripts to create clients have been run, there should be connections to accept
# ======================================================================================================================

sock_serv_transfer.open_socket(bind=False)


sock_serv_comm.open_socket(bind=False)

# -------------------
# Transfer socket
# ------------------
# Get first 3 connections for transfer which should be the 2 cameras and 1 spectrometer
for i in range(3):
    print('Getting data transfer connection: {}'.format(i))
    connection = sock_serv_transfer.acc_connection()
    print('Got data transfer connection: {}'.format(i))

# Use recv_save_imgs() in sockets to automate receiving and saving images from 2 cameras
save_connections = dict()
# for i in range(2):
for ip in pi_ip:
    # Camera
    # Setup connection objects and start thread to run image transfer
    save_connections[ip + '_CM2'] = ImgRecvConnection(sock_serv_transfer, acc_conn=False,
                                                      storage_mount=storage_mount, backup=True)

    # Set connection to be one of the camera IP connections
    save_connections[ip + '_CM2'].connection = sock_serv_transfer.conn_dict[(ip, 'CM2')][0]

    # Start save thread
    save_connections[ip + '_CM2'].thread_func()

    # Spectrometer
    # Setup connection objects and start thread to run image transfer
    save_connections[ip + '_SPC'] = SpecRecvConnection(sock_serv_transfer, acc_conn=False,
                                                       storage_mount=storage_mount, backup=True)

    # Set connection to be one of the camera IP connections
    save_connections[ip + '_SPC'].connection = sock_serv_transfer.conn_dict[(ip, 'SPC')][0]

    # Start save thread
    save_connections[ip + '_SPC'].thread_func()

# Camera for master pi
# Do same for spectrum, which is on the local pi
save_connections[host_ip] = ImgRecvConnection(sock_serv_transfer, acc_conn=False,
                                              storage_mount=storage_mount, backup=True)

# Set connection to that of the host_ip spectrometer connection
save_connections[host_ip].connection = sock_serv_transfer.conn_dict[(host_ip, 'CM1')][0]

# Start save thread
save_connections[host_ip].thread_func()

# -----------------------
# Communications socket
# -----------------------
# Get first 3 connections for transfer which should be the 2 cameras and 1 spectrometer
for i in range(3):
    connection = sock_serv_comm.acc_connection()

# Dictionary holding the connection for internal communications (not external comms)
comms_connections = dict()
for idn in ['CM2', 'SPC']:
    # Setup connection objects and start thread to run communcations
    comms_connections[pi_ip[0] + '_{}'.format(idn)] = CommConnection(sock_serv_comm, acc_conn=False)

    # Set connection to be one of the camera IP connections
    comms_connections[pi_ip[0] + '_{}'.format(idn)].connection = sock_serv_comm.conn_dict[(pi_ip[0], idn)][0]

    # Start communications thread
    comms_connections[pi_ip[0] + '_{}'.format(idn)].thread_func()

# Do same for spectrometer, which is on the local pi
comms_connections[host_ip] = CommConnection(sock_serv_comm, acc_conn=False)

# Set connection to that of the host_ip spectrometer connection
comms_connections[host_ip].connection = sock_serv_comm.conn_dict[(host_ip, 'CM1')][0]

# Start comms thread
comms_connections[host_ip].thread_func()

# Set up socket dictionary - the classes are mutable so the dictionary should carry any changes to the servers made
# through time
sock_dict = {SocketNames.transfer: sock_serv_transfer, SocketNames.comm: sock_serv_comm}
# ----------------------------------------------------------------------------------------------------------------------

# ==============================================================================
# MAIN CODE
# ===============================================================================
# Forward dark imaging command to all communication sockets (2 cameras and 1 spectrometer)
dark_capt_cmd = {'DKC': 1, 'DKS': 1}
sock_serv_comm.send_to_all(dark_capt_cmd)

# Loop through each q and receive the decoded message which should represent a finished dark sequence
for conn in comms_connections:
    while True:
        resp = comms_connections[conn].q.get(block=True)
        # Either camera of spectrometer finish must be in the command, then we break to go to next conn.
        if 'DFC' in resp:
            if resp['DFC']:
                break
        elif 'DFS' in resp:
            if resp['DFS']:
                break

# Shutdown everything
sock_serv_comm.send_to_all({'EXT': 1})

# Loop though each server closing connections and sockets
for server in sock_dict:
    for conn in sock_dict[server].connections[:]:
        sock_dict[server].close_connection(connection=conn[0])
    sock_dict[server].close_socket()

# Wait for connections to finish, then we can close all
timeout = 10
time_start = time.time()
for conn in save_connections:
    while save_connections[conn].working or save_connections[conn].accepting:
        # Add a timeout so if we are waiting for too long we just close things without waiting
        time_wait = time.time() - time_start
        if time_wait > timeout:
            print(' Reached timeout limit waiting for shutdown')
            break

time_start = time.time()
for conn in comms_connections:
    while comms_connections[conn].working or comms_connections[conn].accepting:
        # Add a timeout so if we are waiting for too long we just close things without waiting
        time_wait = time.time() - time_start
        if time_wait > timeout:
            print(' Reached timeout limit waiting for shutdown')
            break
