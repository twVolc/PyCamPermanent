# -*- coding: utf-8 -*-

"""Script to be run on the spectrometer pi.
- Deals with socket communication and spectrometer control.
"""

from pycam.controllers import Spectrometer
from pycam.networking.sockets import PiSocketSpec, PiSocketSpecComms, read_network_file, recv_comms, send_spectra
from pycam.setupclasses import FileLocator
from pycam.utils import read_file

import threading
import queue

# Read config file
config = read_file(FileLocator.CONFIG_SPEC)

# Setup camera object
spec = Spectrometer()

# ----------------------------------------------------------------
# Setup image transfer socket
serv_ip, port = read_network_file(FileLocator.NET_TRANSFER_FILE)
sock_trf = PiSocketSpec(serv_ip, port, spectrometer=spec)
sock_trf.connect_socket()

# Start spectra sending thread
trf_event = threading.Event()
thread_trf = threading.Thread(target=send_spectra, args=(sock_trf, spec.spec_q, trf_event,))
thread_trf.daemon = True
thread_trf.start()
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Setup comms socket
serv_ip, port = read_file(FileLocator.NET_COMM_FILE)
sock_comms = PiSocketSpecComms(serv_ip, port, spectrometer=spec)
sock_comms.connect_socket()
q_comm = queue.Queue()              # Queue for putting received comms in
comm_event = threading.Event()      # Event to shut thread down

# Start comms receiving thread
thread_comm = threading.Thread(target=recv_comms, args=(sock_comms, sock_comms.sock, q_comm, comm_event,))
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

                    # Close sockets
                    sock_comms.close_socket()
                    sock_trf.close_socket()

                    # Exit script by breaking loop
                    break

    except queue.Empty():
        pass
    # ---------------------------------------------------------------------------------------------
