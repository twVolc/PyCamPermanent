# -*- coding: utf-8 -*-

"""
Socket setup and control for Raspberry Pi network and connection to the remote camera from local site.
"""
import socket
from .controllers import Camera, Spectrometer

END_STR = '\r\n'


class SocketClient:
    """Object for setup of a client socket for network communication using the low-level socket interface

    Parameters
    ----------
    host_ip: str
        IP address of server
    port: int
        Communication port
    end_str: str
        End of message string
    """
    def __init__(self, host_ip, port, end_str=END_STR):
        self.host_ip = host_ip                          # IP address of server
        self.port = port                                # Communication port
        self.server_addr = (self.host_ip, self.port)    # Tuple packaging for later use
        self.connect_stat = False                       # Bool for defining if object has a connection

        self.end_str = end_str                          # String to signify end of message

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # Socket object

    def open_socket(self):
        """Opens socket by attempting to make connection with host"""
        try:
            while not self.connect_stat:
                try:
                    self.sock.connect(self.server_addr)  # Attempting to connect to the server
                    self.connect_stat = True
                except OSError:
                    pass
        except Exception as e:
            with open('client_socket_error.log', 'a') as f:
                f.write('ERROR: ' + str(e) + '\n')
        return

    def close_socket(self):
        """Closes socket by disconnecting from host"""
        self.sock.close()
        self.connect_stat = False


class PiSocketCam(SocketClient):
    """Subclass of :class: SocketClient for specific use on PiCam end"""
    def __init__(self, host_ip, port, end_str=END_STR, camera=Camera):
        super().__init__(host_ip, port, end_str)

        self.camera = camera        # Camera object for interface/control

    def send_img(self):
        """Sends current image to server"""


class PiSocketSpec(SocketClient):
    """Subclass of :class: SocketClient for specific use on spectrometer end"""
    def __init__(self, host_ip, port, end_str=END_STR, spectrometer=Spectrometer):
        super().__init__(host_ip, port, end_str)

        self.spectrometer = spectrometer        # Spectrometer object for interface/control

    def send_spectrum(self):
        """Sends current spectrum to server"""
        pass
