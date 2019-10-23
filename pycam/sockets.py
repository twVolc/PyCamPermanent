# -*- coding: utf-8 -*-

"""
Socket setup and control for Raspberry Pi network and connection to the remote camera from local site.
"""
import socket
import numpy as np
from .controllers import Camera, Spectrometer
from .setupclasses import CameraSpecs, SpecSpecs

import time

class SendRecvSpecs:
    """Simple class containing some message separators for sending and receiving messages via sockets"""
    ret_char = '\r\n'
    end_str = bytes("END" + ret_char, 'utf-8')
    len_end_str = len(end_str)

    header_char = 'H_DATASIZE='     # Header start for comms
    header_num_size = 8             # Size of number in digits for header
    header_size = len(header_char) + len(ret_char) + header_num_size

    filename_start = b'FILENAME='
    filename_end = b'FILE_END'


class SocketClient(SendRecvSpecs):
    """Object for setup of a client socket for network communication using the low-level socket interface

    Parameters
    ----------
    host_ip: str
        IP address of server
    port: int
        Communication port
    """
    def __init__(self, host_ip, port):
        self.host_ip = host_ip                          # IP address of server
        self.port = port                                # Communication port
        self.server_addr = (self.host_ip, self.port)    # Tuple packaging for later use
        self.connect_stat = False                       # Bool for defining if object has a connection

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # Socket object

    def connect_socket(self):
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

    def generate_header(self, msg_size):
        """Generates a header with the given message size and returns byte array version"""
        header = self.header_char + str(msg_size).rjust(self.header_num_size, '0') + self.ret_char
        return header.encode()


class PiSocketCam(SocketClient):
    """Subclass of :class: SocketClient for specific use on PiCam end"""
    def __init__(self, host_ip, port, camera=Camera):
        super().__init__(host_ip, port)

        self.camera = camera        # Camera object for interface/control

    def send_img(self, filename=None, image=None):
        """Sends current image to server

        Parameters
        ----------
        filename: str
            filename for the image. If None it is taken from the socket's camera attribute
        image: np.array
            image to be sent. If None it is taken from the socket's camera attribute
        """
        # Get filename and image if they aren't provided as arguments to the method.
        if filename is None:
            filename = self.camera.filename
        if image is None:
            image = self.camera.image

        # Convert image to bytes for sending
        img_bytes = image.tobytes()

        # Encode filename
        filename_bytes = self.filename_start + bytes(filename, 'utf-8') + self.filename_end

        # Calculate size of message
        msg_size = len(filename_bytes) + len(img_bytes) + self.len_end_str

        # Format header containing length of message information
        header = self.generate_header(msg_size)

        # Send header, filename, image bytes and then end string
        self.sock.sendall(header)
        self.sock.sendall(filename_bytes)
        self.sock.sendall(img_bytes)
        self.sock.sendall(self.end_str)


class PiSocketSpec(SocketClient):
    """Subclass of :class: SocketClient for specific use on spectrometer end"""
    def __init__(self, host_ip, port, spectrometer=None):
        super().__init__(host_ip, port)

        self.spectrometer = spectrometer        # Spectrometer object for interface/control

    def send_spectrum(self):
        """Sends current spectrum to server"""
        # Convert wavelength and spectrum arrays to bytes
        wave_bytes = self.spectrometer.wavelengths.tobytes()
        spec_bytes = self.spectrometer.spectrum.tobytes()

        filename_bytes = self.filename_start + bytes(self.spectrometer.filename, 'utf-8') + self.filename_end

        # Calculate message size
        msg_size = len(filename_bytes) + len(wave_bytes) + len(spec_bytes) + self.len_end_str

        # Generate header
        header = self.generate_header(msg_size)

        self.sock.sendall(header)
        self.sock.sendall(filename_bytes)
        self.sock.sendall(wave_bytes + spec_bytes)
        self.sock.sendall(self.end_str)


class SocketServer(SendRecvSpecs):
    """Object for setup of a host socket for network communication using the low-level socket interface

    Parameters
    ----------
    host_ip: str
        IP address of server
    port: int
        Communication port
    """
    def __init__(self, host_ip, port):
        self.host_ip = host_ip              # IP address of host
        self.port = port                    # Communication port
        self.server_addr = (host_ip, port)  # Server address
        self.connections = []               # List holding connections
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # Socket object

        self.camera = CameraSpecs()         # Camera specifications
        self.spectrometer = SpecSpecs()     # Spectrometer specifications

    def open_socket(self, backlog=5):
        """Opens socket and listens for connection

        Parameters
        ----------
        backlog: int
            Number of unaccepted connections allowed before refusing new connections"""
        # Bind to socket
        self.sock.bind(self.server_addr)

        # Listen for connection (backlog=5 connections - default)
        self.sock.listen(backlog)

    def close_socket(self):
        """Closes socket"""
        self.sock.close()

    def acc_connection(self):
        """Accept connection and add to listen"""
        # Establish connection with client and append to list of connections
        self.connections.append(self.sock.accept())

    def recv_data(self):
        """Receives and decodes header"""
        # Receive header and decode
        header = self.connections[0][0].recv(self.header_size)
        header_txt = header.decode()
        bytes_to_recv = int(header_txt.split('=')[1])

        # Receive the rest of the incoming message (needs a while loop to ensure all data is received
        # as packets can come incomplete)
        data_buff = bytearray()
        while len(data_buff) < bytes_to_recv:
            data_buff += self.connections[0][0].recv(bytes_to_recv - len(data_buff))

        return data_buff

    def extract_data(self, data_buff):
        """Extracts filename from data buffer

        Parameters
        ----------
        data_buff: bytes
            Data from socket stream containing filename and img/spectrum data"""
        # Extract filename
        [file_buff, data_buff] = data_buff.split(self.filename_end)
        filename = file_buff.split(self.filename_start)[1].decode()

        # Extract data by removing end string from message
        data = data_buff.split(self.end_str)[0]

        return filename, data

    def recv_img(self):
        """Receives image from PiSocketCam socket"""
        # Receive image data
        data_buff = self.recv_data()

        # Extract filename from the data
        [filename, data] = self.extract_data(data_buff)

        # Reshape data into image array
        img = np.frombuffer(data, dtype='uint16').reshape(self.camera.pix_num_y, self.camera.pix_num_x)

        return img, filename

    def recv_spectrum(self):
        """Receives spectrum from PiSocketSpec"""
        data_buff = self.recv_data()

        # Extract filename
        [filename, data] = self.extract_data(data_buff)

        # Extract wavelengths and spectrum
        data = np.frombuffer(data, dtype='uint16')
        wavelengths = data[:self.spectrometer.pix_num]
        spectrum = data[self.spectrometer.pix_num:]

        return wavelengths, spectrum, filename

