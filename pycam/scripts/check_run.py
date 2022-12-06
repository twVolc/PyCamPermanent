# -*- coding: utf-8 -*-

"""
Script to check the instrument is running correctly at the start of each day
> Read crontab file to find when pycam will start
> Check data types being acquired
> If not all 3 data types are being acquired, restart the program or system
"""
import datetime
import time
import sys
sys.path.append('/home/pi/')
import os

from pycam.setupclasses import CameraSpecs, SpecSpecs, FileLocator, ConfigInfo
from pycam.networking.sockets import SocketClient, ExternalSendConnection, ExternalRecvConnection
from pycam.networking.ssh import open_ssh, close_ssh, ssh_cmd
from pycam.io_py import read_script_crontab, write_script_crontab, reboot_remote_pi
from pycam.utils import read_file


def check_data(sleep=150):
    """Check if data exists"""
    # Get specifications of spectrometer and camera settings
    spec_specs = SpecSpecs()
    cam_specs = CameraSpecs()

    # Create dictionary where each key is the string to look for and the value is the location of the string in the filename
    data_dict = {spec_specs.file_coadd: spec_specs.file_coadd_loc,
                      cam_specs.file_filterids['on']: cam_specs.file_fltr_loc,
                      cam_specs.file_filterids['off']: cam_specs.file_fltr_loc}

    # Get current list of images in
    all_dat_old = os.listdir(FileLocator.IMG_SPEC_PATH)

    # Sleep for 1.5 minutes to allow script to start running properly
    time.sleep(sleep)

    # Check data
    all_dat = os.listdir(FileLocator.IMG_SPEC_PATH)
    all_dat_new = [x for x in all_dat if x not in all_dat_old]

    # Check all 3 data types to make sure we're acquiring everything
    data_bools = [False] * 3

    # Loop through each image to check what data type it is
    for data_file in all_dat_new:
        for i, dat_str in enumerate(data_dict):
            # Extract string
            data_string = data_file.split('_')[data_dict[dat_str]]
            if dat_str in data_string:
                data_bools[i] = True

    # If we have all data types, there are no issues so close script
    if data_bools == [True] * 3:
        return True
    else:
        return False


# Get script start/stop times from crontab file
cfg = read_file(FileLocator.CONFIG)
start_script = cfg[ConfigInfo.start_script]
stop_script = cfg[ConfigInfo.stop_script]
scripts = read_script_crontab(FileLocator.SCRIPT_SCHEDULE_PI, [start_script, stop_script])

start_script_time = datetime.datetime.now()
start_script_time = start_script_time.replace(hour=scripts[start_script][0],
                                              minute=scripts[start_script][1], second=0, microsecond=0)
stop_script_time = datetime.datetime.now()
stop_script_time = stop_script_time.replace(hour=scripts[stop_script][0],
                                            minute=scripts[stop_script][1], second=0, microsecond=0)

# Wait until pycam script has been run
if start_script_time < stop_script_time:
    while datetime.datetime.now() < start_script_time or datetime.datetime.now() > stop_script_time:
        time.sleep(60)
        # Refresh times in case we move into a new day
        start_script_time = datetime.datetime.now()
        start_script_time = start_script_time.replace(hour=scripts[start_script][0],
                                                      minute=scripts[start_script][1], microsecond=0)
        stop_script_time = datetime.datetime.now()
        stop_script_time = stop_script_time.replace(hour=scripts[stop_script][0],
                                                    minute=scripts[stop_script][1], microsecond=0)

elif start_script_time > stop_script_time:
    while datetime.datetime.now() < start_script_time and datetime.datetime.now() > stop_script_time:
        time.sleep(60)
        # Refresh times in case we move into a new day
        start_script_time = datetime.datetime.now()
        start_script_time = start_script_time.replace(hour=scripts[start_script][0],
                                                      minute=scripts[start_script][1], microsecond=0)
        stop_script_time = datetime.datetime.now()
        stop_script_time = stop_script_time.replace(hour=scripts[stop_script][0],
                                                    minute=scripts[stop_script][1], microsecond=0)
else:
    with open(FileLocator.ERROR_LOG_PI, 'a') as f:
        f.write('ERROR! check_run.py: Pycam start and stop times are the same, this is likely to lead to unexpected behaviour. \n')
    sys.exit()

# Check data, if True is returned we have all data so no issues
if check_data():
    sys.exit()


# If there are not all data types, we need to connect to the system and correct it
# Socket client
sock = SocketClient(host_ip=cfg[ConfigInfo.host_ip], port=int(cfg[ConfigInfo.port_ext]))
try:
    sock.close_socket()
    sock.connect_socket_timeout(5)

    # Setup recv comms connection object
    recv_comms = ExternalRecvConnection(sock=sock, acc_conn=False)
    recv_comms.thread_func()

    # # Setup send comms connection object
    send_comms = ExternalSendConnection(sock=sock, acc_conn=False)
    send_comms.thread_func()

    # Send exit
    send_comms.q.put({'SPC': 1, 'SPS': 1})
    resp = recv_comms.q.get(block=True)

    time.sleep(5)

    # Restart acquisition
    send_comms.q.put({'STC': 1, 'STS': 1})
    resp = recv_comms.q.get(block=True)

    time.sleep(10)

    # Check data
    if check_data(sleep=30):
        sys.exit()

    # Stop pycam
    send_comms.q.put({'SPC': 1, 'SPS': 1})
    resp = recv_comms.q.get(block=True)
    send_comms.q.put({'EXT': 1})


except ConnectionError:
    with open(FileLocator.ERROR_LOG_PI, 'a') as f:
        f.write('check_run.py: Error connecting to pycam on port {}.\n'.format(int(cfg[ConfigInfo.port_ext])))

except BaseException as e:
    with open(FileLocator.ERROR_LOG_PI, 'a') as f:
        f.write('check_run.py: Error {}.\n'.format(e))


# Reboot remote pi. Then start the remote_picam start script on that pi, then reboot master_pi
slave_ip = cfg[ConfigInfo.pi_ip]
reboot_remote_pi(pi_ip=[slave_ip])

# Start slave pi's reboot script which starts both the check_run.py script and the pycam_masterpi.py script
ssh_cli = open_ssh(slave_ip)

stdin, stdout, stderr = ssh_cmd(ssh_cli, 'python3 ' + FileLocator.REMOTE_PI_RUN_PYCAM)

close_ssh(ssh_cli)









