# -*- coding: utf-8 -*-

"""Contains all global variables, mainly relating to sockets and the GUI"""

from pycam.utils import read_file
from pycam.setupclasses import FileLocator, ConfigInfo
from .misc import Indicator
from pycam.networking.sockets import SocketClient, ExternalRecvConnection, ExternalSendConnection
from pycam.networking.FTP import FTPClient
from .settings import GUISettings

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

# FTP client
ftp_client = FTPClient(network_info=config)

# ======================================================================================================================

# ==============================
# GUI SETTINGS
# ==============================
gui_setts = GUISettings(FileLocator.GUI_SETTINGS)
fig_face_colour = 'gainsboro'
axes_colour = 'black'

