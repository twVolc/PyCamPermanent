# -*- coding: utf-8 -*-

"""Contains all global variables, mainly relating to sockets and the GUI"""

from pycam.utils import read_file, StorageMount
from pycam.setupclasses import FileLocator, ConfigInfo, SpecSpecs, CameraSpecs
from .misc import Indicator
from pycam.networking.sockets import SocketClient, ExternalRecvConnection, ExternalSendConnection
from pycam.networking.FTP import FTPClient, CurrentDirectories
from .settings import GUISettings
import os
import copy

# ======================================================================================================================
# SOCKET
# =====================================================================================================================
# Configuration dictionary
config = read_file(FileLocator.CONFIG_WINDOWS)

# Socket client
sock = SocketClient(host_ip=config[ConfigInfo.host_ip], port=int(config[ConfigInfo.port_ext]))

# Setup recv comms connection object
recv_comms = ExternalRecvConnection(sock=sock, acc_conn=False)

# # Setup send comms connection object
send_comms = ExternalSendConnection(sock=sock, acc_conn=False)

# Connection indicator
indicator = Indicator()

# Current directory objects
current_dir_img = CurrentDirectories(root=os.path.join(config[ConfigInfo.local_data_dir] + 'Images/'),
                                     specs=CameraSpecs())
current_dir_spec = CurrentDirectories(root=os.path.join(config[ConfigInfo.local_data_dir] + 'Spectra/'),
                                      specs=SpecSpecs())

# FTP client
ftp_client = FTPClient(img_dir=current_dir_img, spec_dir=current_dir_spec, network_info=config,
                       storage_mount=StorageMount())

# FTP client for 2nd pi
config_2 = copy.deepcopy(config)
config_2[ConfigInfo.host_ip] = config_2[ConfigInfo.pi_ip].split(',')[0]
# ftp_client_2 = FTPClient(img_dir=current_dir_img, spec_dir=current_dir_spec, network_info=config_2)
ftp_client_2 = None
# ======================================================================================================================

# ==============================
# GUI SETTINGS
# ==============================
gui_setts = GUISettings(FileLocator.GUI_SETTINGS)
fig_face_colour = 'gainsboro'
axes_colour = 'black'

