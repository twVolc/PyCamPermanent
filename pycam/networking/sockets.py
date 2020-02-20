# -*- coding: utf-8 -*-

"""
Socket setup and control for Raspberry Pi network and connection to the remote camera from local site.
"""
import socket
import struct
import numpy as np
from pycam.controllers import Camera, Spectrometer
from pycam.setupclasses import CameraSpecs, SpecSpecs, FileLocator
from pycam.utils import check_filename
from pycam.savefuncs import save_img, save_spectrum
import time
import queue


def read_network_file(filename):
    """Reads IP address and port from text file

    Parameters
    ----------
    filename: str
        file to be read from

    :returns
    ip_address: str
        IP address corresponding to the server IP
    port: int
        communication port"""
    # Check we have a text file
    try:
        check_filename(filename, 'txt')
    except ValueError:
        raise

    ip_addr = None
    port = None

    # Read file and extract ip address and port if present
    with open(filename, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            if 'ip_address=' in line:
                ip_addr = line.split('=')[1].split('#')[0].strip('\n')
            if 'port=' in line:
                port = int(line.split('=')[1].split('#')[0].strip('\n'))

    return ip_addr, port


class SendRecvSpecs:
    """Simple class containing some message separators for sending and receiving messages via sockets"""
    encoding = 'utf-8'
    ret_char = '\r\n'
    end_str = bytes("END" + ret_char, encoding)
    len_end_str = len(end_str)

    header_char = 'H_DATASIZE='     # Header start for comms
    header_num_size = 8             # Size of number in digits for header
    header_size = len(header_char) + len(ret_char) + header_num_size

    filename_start = b'FILENAME='
    filename_end = b'FILE_END'

    pack_fmt = struct.Struct('I I f I I')     # Format of message for communication
    pack_info = ('ss', 'type', 'framerate', 'ppmm', 'auto_ss', 'exit')     # Format specifications of message


class CommsFuncs(SendRecvSpecs):
    """Holds all functions relating to communication procedures"""

    def __init__(self):
        # Create dictionary for communication protocol. Dictionary contains:
        # character code as key, value as tuple (type, range of accepted values)
        # All values are converted to ASCII before being sent over the network
        self.cmd_dict = {
            'SSC': (int, [1, 6001]),            # Shutter speed (ms) camera [min, max]
            'SSS': (int, [1, 6001]),            # Shutter speed (ms) spectrometer [min, max]
            'FRC': (float, [0.0, 1.0]),         # Framerate camera [min, max]
            'FRS': (float, [0.0, 10.0]),        # Framerate spectrometer [min, max]
            'AUT': (bool, 1),                   # Auto-shutter speed [options]
            'SMN': (float, [0.0, 0.9]),         # Minimum saturation accepted before adjusting shutter speed
            'SMX': (float, [0.1, 1.0]),         # Maximum saturation accepted before adjusting shutter speed
            'WMN': (int, [300, 400]),           # Minimum wavelength of spectra to check saturation
            'WMX': (int, [300, 400]),           # Maximum wavelength of spectra to check saturation
            'SXS': (float, [0.1, 1.0]),         # Maximum saturation accepted for spectra before adjusting int. time
            'TYP': (str, [a for a in CameraSpecs().file_img_type]),  # Type of image
            'STP': (bool, 1),           # Stops continuous image and spectra acquisitions
            'STT': (bool, 1),           # Starts continuous image and spectra acquisitions
            'EXT': (bool, 1),           # Close program (should only be succeeded by 1, to confirm close request)
            'RST': (bool, 1),           # Restart entire system
            'RSS': (bool, 1),           # Restart spectrometer (restarts pycam_spectrometer.py)
            'RSC': (bool, 1),           # Restart camera (with this request the remote pis are fully restarted)
            'LOG': (int, [0, 5]),       # Various status report requests:
                                        # 0 - Full log
                                        # 1 - Battery log
                                        # 2 - Temperature log
            }
        self.cmd_list = self.cmd_dict.keys()          # List of keys
        self.cmd_dict['ERR'] = (str, self.cmd_list)   # Error flag, which provides the key in which an error was found

    """ All of the following functions accept the same input arguments. Some are used and others aren't, but each 
    accepts the same number to allow external function calls without knowing which function is being called"""
    def comms_img_cmd(self, cmd_dict, socks, config):
        """Sends the command and associated value to both remote pis

        Parameters
        ----------
        cmd_dict: dict
            Dictionary containing commands
        socks: dict
            Dictionary of all sockets currently open, for forwarding commands
        config: dict
            Dictionary of any other configuration parameters which may be required
        """
        # Loops through ips for external pis (these are the two camera pis)
        for ip in config['pi_ip'].split(','):
            # Get the retrieve the correct connection to send to
            conn = socks['comm'].get_connection(ip=ip)

            # Send command to function for sending over socket
            cmd = socks['comm'].encode_comms(cmd_dict)
            socks['comm'].send_comms(conn, cmd)

    def comms_spec_cmd(self, cmd_dict, socks):
        """Sends the command and associated value to both remote pis

        Parameters
        ----------
        cmd_dict: dict
            Dictionary containing commands
        socks: dict
            Dictionary of all sockets currently open, for forwarding commands
        config: dict
            Dictionary of any other configuration parameters which may be required
        """
        # Get the retrieve the correct connection to send to
        conn = socks['comm'].get_connection(ip=socket.gethostbyname(socket.gethostname()))

        # Send command to function for sending over socket
        cmd = socks['comm'].encode_comms(cmd_dict)
        socks['comm'].send_comms(conn, cmd)

    def SSC(self, value, connection, socks, config):
        """Acts on SSC command"""
        self.comms_img_cmd({'SSC': value}, socks, config)

    def SSS(self, value, connection, socks, config):
        """Acts on SSS command"""
        self.comms_spec_cmd({'SSS': value}, socks)

    def FRC(self, value, connection, socks, config):
        """Acts on FRC command"""
        self.comms_img_cmd({'FRC': value}, socks, config)

    def FRS(self, value, connection, socks, config):
        """Acts on FRS command"""
        self.comms_spec_cmd({'FRS': value}, socks)

    def AUT(self, value, connection, socks, config):
        """Acts on AUT command"""
        self.comms_img_cmd({'AUT': value}, socks, config)

    def SMN(self, value, connection, socks, config):
        """Acts on SMN command"""
        self.comms_img_cmd({'SMN': value}, socks, config)

    def SMX(self, value, connection, socks, config):
        """Acts on SMX command"""
        self.comms_img_cmd({'SMX': value}, socks, config)

    def WMN(self, value, connection, socks, config):
        """Acts on WMN command"""
        self.comms_spec_cmd({'WMN': value}, socks)

    def WMX(self, value, connection, socks, config):
        """Acts on WMX command"""
        self.comms_spec_cmd({'WMX': value}, socks)

    def SXS(self, value, connection, socks, config):
        """Acts on SXS command"""
        self.comms_spec_cmd({'SXS': value}, socks)

    def TYP(self, value, connection, socks, config):
        """Acts on TYP command"""
        self.comms_img_cmd({'TYP': value}, socks, config)

    def STP(self, value, connection, socks, config):
        """Acts on STP command, stopping acquisitions on cameras and spectrometer"""
        self.comms_img_cmd({'STP': value}, socks, config)
        self.comms_spec_cmd({'STP': value}, socks)

    def STT(self, value, connection, socks, config):
        """Acts on STP command, stopping acquisitions on cameras and spectrometer"""
        self.comms_img_cmd({'STT': value}, socks, config)
        self.comms_spec_cmd({'STT': value}, socks)

    def EXT(self, value, connection, socks, config):
        """Acts on EXT command, closing everything down"""
        if value:
            # Send exit command to raspberry pis
            self.comms_img_cmd({'EXT': 1}, socks, config)

            # Send exit command to spectrometer script (at local ip address)
            self.comms_spec_cmd({'EXT': 1}, socks, config)

            return
        else:
            return

    def RST(self, value, connection, socks, config):
        """Acts on RST command, restarts entire system"""
        pass

    def RSS(self, value, connection, socks, config):
        """Acts on RSS command, restarts pycam_spectrometer.py script"""
        self.comms_spec_cmd({'RSS': value}, socks)

    def RSC(self, value, connection, socks, config):
        """Acts on STP command, restarts pycam_camera.py script"""
        self.comms_img_cmd({'RSC': value}, socks, config)

    def LOG(self, value, connection, socks, config):
        """Acts on LOG command, sending the specified log back to the connection"""
        pass


class SocketMeths(CommsFuncs):
    """Class holding generic methods used by both servers and clients
    Like decoding messages?"""
    def __init__(self):
        super().__init__()

    def encode_comms(self, message):
        """Encode message into a single byte array

        Parameters
        ----------
        message: dict
            Dictionary containing messages as the key and associated value to send

        """
        # Instantiate byte array
        cmd_bytes = bytearray()

        # Loop through messages and convert the values to strings, then append it to the byte array preceded by the key
        for key in message:
            if self.cmd_dict[key][0] is bool:
                cmd = str(int(message[key]))

            elif self.cmd_dict[key][0] is float:
                cmd = '{:.2f}'.format(message[key])

            else:
                cmd = '{}'.format(message[key])

            # Append key and cmd to bytearray
            cmd_bytes += bytes(key + ' ' + cmd + ' ', 'utf-8')

        # Add end_str bytes
        cmd_bytes += self.end_str

        return cmd_bytes

    def decode_comms(self, message):
        """Decodes string from network communication, to extract information and check it is correct.
        Returns a dictionary of decoded commands

        Parameters
        ----------
        message: str
            Message which is expected to be in the form defined by SendRecvSpecs.cmd_dict"""
        mess_list = message.split()

        cmd_ret = {'ERR': []}
        for i in range(len(mess_list)):
            if mess_list[i] in self.cmd_list:
                # NEXT DEV - PUT A CHECK IN THAT THE COMMAND IS VALID FOR THE SPECIFIC KEY BEFORE ADDING IT TO cmd_dict
                # Add command to dictionary

                # If we have a bool, check that we have either 1 or 0 as command, if not, it is not valid and is ignored
                if self.cmd_dict[mess_list[i]][0] is bool:

                    # Flag error with command if it is not recognised
                    if mess_list[i+1] not in ['1', '0']:
                        cmd_ret['ERR'].append(mess_list[i])
                        continue
                    else:
                        cmd = bool(int(mess_list[i+1]))

                # If we have a str, check it within the accepted str list
                elif self.cmd_dict[mess_list[i]][0] is str:

                    # Flag error with command if it is not recognised
                    if mess_list[i+1] not in self.cmd_dict[mess_list[i]][1]:
                        cmd_ret['ERR'].append(mess_list[i])
                        continue
                    else:
                        cmd = mess_list[i+1]

                # Otherwise we convert message to its type and then test that outcome is within defined bounds
                else:
                    cmd = self.cmd_dict[mess_list[i]][0](mess_list[i+1])

                    # Flag error with command if it is not recognised
                    if cmd < self.cmd_dict[mess_list[i]][1][0] or cmd > self.cmd_dict[mess_list[i]][1][-1]:
                        cmd_ret['ERR'].append(mess_list[i])
                        continue

                cmd_ret[mess_list[i]] = cmd

        return cmd_ret

    @staticmethod
    def send_comms(connection, cmd_bytes):
        """Sends byte array over connection

        Parameters
        ----------
        connection
            Object which has sendall() function. If client this will be the socket itself, if server this will be the
            connection
        cmd_bytes: bytearray

        """
        if hasattr(connection, 'sendall'):
            if callable(connection.sendall):
                connection.sendall(cmd_bytes)
        else:
            raise AttributeError('Object {} has no sendall command'.format(connection))

    def recv_comms(self, connection):
        """Receives data without a header until end string is encountered

        Parameters
        ----------
        connection
            Object which has recv() function. If client this will be the socket itself, if server this will be the
            connection
        """
        data_buff = bytearray()  # Instantiate empty byte array to append received data to

        while True:
            # Receive data and add to buffer
            data_buff += connection.recv(1)

            # Once we have a full message, with end_str, we return it after removing the end_str and decoding to a str
            if self.end_str in data_buff:
                end_idx = data_buff.find(self.end_str)
                return data_buff[:end_idx].decode(self.encoding)


class SocketClient(SocketMeths):
    """Object for setup of a client socket for network communication using the low-level socket interface

    Parameters
    ----------
    host_ip: str
        IP address of server
    port: int
        Communication port
    """
    def __init__(self, host_ip, port):
        super().__init__()

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

    def recv_img_cmd(self):
        """Receives command from server for imaging"""
        msg = bytearray()
        while len(msg) < self.pack_fmt.size:
            msg += self.sock.recv(self.pack_fmt.size - len(msg))

        # Decode message into dictionary
        command = self._decode_msg(msg)

        return command

    def _decode_msg(self, msg):
        """Decodes message into dictionary"""
        # Unpack data from bytes
        data = self.pack_fmt.unpack(msg)

        # Unpack into dictionary
        dictionary = dict()
        for i in range(len(self.pack_info)):
            dictionary[self.pack_info[i]] = data[i]

        return dictionary


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
    """Subclass of :class: SocketClient for specific use on spectrometer end

    Parameters
    ----------
    spectrometer: Spectrometer
        Reference to spectrometer object for controlling attributes and acquisition settings"""
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


class PiSocketCamComms(SocketClient):
    """Subclass of :class: SocketClient for specific use on spectrometer end handling comms

    Parameters
    ----------
    camera: Camera
        Reference to spectrometer object for controlling attributes and acquisition settings
    """
    def __init__(self, host_ip, port, camera=Camera):
        super().__init__(host_ip, port)

        self.camera = camera        # Camera object for interface/control

    def SSC_comm(self, value):
        """Acts on SSC command

        Parameters
        ----------
        value: int
            Value to set camera shutter speed to
        """
        try:
            self.camera.set_shutter_speed(value)
            comm = self.encode_comms({'SSC': 1})
        except:
            comm = self.encode_comms({'SSC': 0})
        finally:
            self.send_comms(self.sock, comm)

    def FRC_comm(self, value):
        """Acts on FRC command

        Parameters
        ----------
        value: float
            Value to set camera shutter speed to
        """
        try:
            self.camera.set_cam_framerate(value)
            comm = self.encode_comms({'FRC': 1})
        except:
            comm = self.encode_comms({'FRC': 0})
        finally:
            self.send_comms(self.sock, comm)

    def AUT_comm(self, value):
        """Acts on AUT command"""
        if value:
            self.camera.auto_ss = True
            comm = self.encode_comms({'AUT': True})
        else:
            self.camera.auto_ss = False
            comm = self.encode_comms({'AUT': False})
        self.send_comms(self.sock, comm)

    def SMN_comm(self, value):
        """Acts on SMN command"""
        if value < self.camera.max_saturation:
            self.camera.min_saturation = value
            comm = self.encode_comms({'SMN': 1})
        else:
            comm = self.encode_comms({'SMN': 0})
        self.send_comms(self.sock, comm)

    def SMX_comm(self, value):
        """Acts on SMX command"""
        if value > self.camera.min_saturation:
            self.camera.max_saturation = value
            comm = self.encode_comms({'SMX': 1})
        else:
            comm = self.encode_comms({'SMX': 0})
        self.send_comms(self.sock, comm)

    def TYP_comm(self, value):
        """Acts on TYP command, requesting this type of image from the camera"""
        self.camera.capture_q.put({'type': value})

    def STP_comm(self, value):
        """Acts on STP command by adding a stop command dictionary to the camera's capture queue"""
        if value:
            self.camera.capture_q.put({'exit_cont': True})

    def STT_comm(self, value):
        """Acts on STP command by adding a stop command dictionary to the camera's capture queue"""
        if value:
            self.camera.capture_q.put({'start_cont': True})


class PiSocketSpecComms(SocketClient):
    """Subclass of :class: SocketClient for specific use on spectrometer end handling comms

    Parameters
    ----------
    spectrometer: Spectrometer
        Reference to spectrometer object for controlling attributes and acquisition settings"""

    def __init__(self, host_ip, port, spectrometer=None):
        super().__init__(host_ip, port)

        self.spectrometer = spectrometer  # Spectrometer object for interface/control

    def SSS_comm(self, value):
        """Acts on SSS command

        Parameters
        ----------
        value: int
            Value to set spectrometer integration time to
        """
        if not self.spectrometer.auto_int:
            try:
                self.spectrometer.int_time = value
                comm = self.encode_comms({'SSS': 1})
            except ValueError:
                comm = self.encode_comms({'SSS': 0})
        else:
            comm = self.encode_comms({'SSS': 0})

        self.send_comms(self.sock, comm)

    def FRS_comm(self, value):
        """Acts on FRS command

        Parameters
        ----------
        value: float
            Value to set spectrometer framerate to
        """
        try:
            self.spectrometer.framerate = value
            comm = self.encode_comms({'FRS': 1})
        except:
            comm = self.encode_comms({'FRS': 0})
        finally:
            self.send_comms(self.sock, comm)

    def WMN_comm(self, value):
        """Acts on WMN command

        Parameters
        ----------
        value: int
            Value to set Wavelength minimum to"""
        # Check the new value is less than the maximum in the saturation range window
        if value < self.spectrometer.saturation_range[1]:
            self.spectrometer.saturation_range[0] = value
            comm = self.encode_comms({'WMN': 1})
        else:
            comm = self.encode_comms({'WMN': 0})

        # Return communication to say whether the work has been done or not
        self.send_comms(self.sock, comm)

    def WMX_comm(self, value):
        """Acts on WMX command"""
        # Check the new value is more than the minimum in the saturation range window
        if value > self.spectrometer.saturation_range[0]:
            self.spectrometer.saturation_range[1] = value
            comm = self.encode_comms({'WMX': 1})
        else:
            comm = self.encode_comms({'WMX': 0})

        # Return communication to say whether the work has been done or not
        self.send_comms(self.sock, comm)

    def SXS_comm(self, value):
        """Acts on SXS command"""
        # Try to set spectrometer max saturation value. If we encounter any kind of error, return error value
        try:
            self.spectrometer.max_saturation = value
            comm = self.encode_comms({'SXS': 1})
        except:
            comm = self.encode_comms({'SXS': 0})
        finally:
            self.send_comms(self.sock, comm)

    def AUT_comm(self, value):
        """Acts on AUT command"""
        if value:
            self.spectrometer.auto_ss = True
            comm = self.encode_comms({'AUT': True})
        else:
            self.spectrometer.auto_ss = False
            comm = self.encode_comms({'AUT': False})
        self.send_comms(self.sock, comm)

    def STP_comm(self, value):
        """Acts on STP command by adding a stop command dictionary to the camera's capture queue"""
        if value:
            self.spectrometer.capture_q.put({'exit_cont': True})

    def STT_comm(self, value):
        """Acts on STP command by adding a stop command dictionary to the camera's capture queue"""
        if value:
            self.spectrometer.capture_q.put({'start_cont': True})



class SocketServer(SocketMeths):
    """Object for setup of a host socket for network communication using the low-level socket interface

    Parameters
    ----------
    host_ip: str
        IP address of server
    port: int
        Communication port
    """
    def __init__(self, host_ip, port):
        super().__init__()

        self.host_ip = host_ip              # IP address of host
        self.port = port                    # Communication port
        self.server_addr = (host_ip, port)  # Server address
        self.connections = []               # List holding connections
        self.num_conns = 0                  # Number of connections
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
        self.num_conns += 1

    def get_connection(self, conn_num=None, ip=None):
        """Returns connection defined by conn_num. Wrapper for connections, to make access more intuitive.
        If ip is provided then function will use this instead to return connection
        conn_num: int
            Number in connection list
        ip: str
            IP address to find specific connection"""

        # Search for ip address in connections list
        if isinstance(ip, str):
            for i in range(self.num_conns):
                # Check if ip address is in the addr tuple. If it is, we set conn_num to this connection
                if ip in self.connections[i][1]:
                    conn_num = i

            # If we fail to find the ip in any of the connection, we return None
            if conn_num is None:
                return None

        # If we aren't given an ip to search, we use connection number provided, first checking it is within limits of
        # number of connections we have
        elif isinstance(conn_num, int):
            if conn_num > self.num_conns:
                raise IndexError('Connection number is greater than the number of connections to this socket')

        else:
            return None

        # Return the requested connection, which is the first item in a tuple at a defined index in the connection list
        return self.connections[conn_num][0]

    def recv_data(self, connection=None):
        """Receives and decodes header, then receives the rest of message

        Parameters
        ----------
        connection
            Socket connection in standard form from Socket.accept()
        """
        # Receive header and decode
        header = connection.recv(self.header_size)
        header_txt = header.decode()
        bytes_to_recv = int(header_txt.split('=')[1])

        # Receive the rest of the incoming message (needs a while loop to ensure all data is received
        # as packets can come incomplete)
        data_buff = bytearray()
        while len(data_buff) < bytes_to_recv:
            data_buff += connection.recv(bytes_to_recv - len(data_buff))

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

    def recv_img(self, connection=None):
        """Receives image from PiSocketCam socket"""
        # Default to first connection if not provided with a specific socket connection
        if connection is None:
            connection = self.connections[0][0]

        # Receive image data
        data_buff = self.recv_data(connection)

        # Extract filename from the data
        [filename, data] = self.extract_data(data_buff)

        # Reshape data into image array
        img = np.frombuffer(data, dtype='uint16').reshape(self.camera.pix_num_y, self.camera.pix_num_x)

        return img, filename

    def recv_spectrum(self, connection=None):
        """Receives spectrum from PiSocketSpec"""
        # Default to first connection if not provided with a specific socket connection
        if connection is None:
            connection = self.connections[0][0]

        data_buff = self.recv_data(connection)

        # Extract filename
        [filename, data] = self.extract_data(data_buff)

        # Extract wavelengths and spectrum
        data = np.frombuffer(data, dtype='uint16')
        wavelengths = data[:self.spectrometer.pix_num]
        spectrum = data[self.spectrometer.pix_num:]

        return wavelengths, spectrum, filename

    def send_img_cmd(self, cmd, connection=None):
        """Sends imaging command to pi client

        Parameters
        ----------
        cmd: dict
            Dictionary of commands to be sent. Dictionary must contain all expected keys in SendRecvSpecs.pack_info
        connection
            Socket connection in standard form from Socket.accept()
        """
        # Default to first connection if not provided with a specific socket connection
        if connection is None:
            connection = self.connections[0][0]

        # Unpack data from dictionary to list
        data = []
        for i in len(self.pack_info):
            try:
                data[i] = cmd[self.pack_info[i]]
            except KeyError as e:
                raise Exception('Dictionary provided did not contain all keys in SendRecvSpecs.pack_info') from e

        # Pack data using struct
        packed_data = self.pack_fmt.pack(*data)

        # Send data over connection
        connection.send(packed_data)


def recv_save_imgs(sock, connection, event=None):
    """ Continually loops through receiving and saving images, until stopped

    Parameters
    ----------
    sock: SocketSever
        Object which contains receive commands and has been connected to client
    connection:
        Socket connection for Pi
    event: threading.Event
        Event which will close function when set
    """
    while not event.is_set():

        # Receive image from pi client
        img, filename = sock.recv_img(connection)

        # Save image
        save_img(img, FileLocator.IMG_SPEC_PATH + filename)


def recv_save_spectra(sock, connection, event=None):
    """ Continually loops through receiving and saving spectra, until stopped by threading event

    Parameters
    ----------
    sock: SocketSever
        Object which contains receive commands and has been connected to client
    connection:
        Socket connection for Pi
    event: threading.Event
        Event which will close function when set
    """
    while not event.is_set():

        # Receive image from pi client
        wavelengths, spectrum, filename = sock.recv_spectrum(connection)

        # Save image
        save_spectrum(wavelengths, spectrum, FileLocator.IMG_SPEC_PATH + filename)


def recv_comms(sock, connection, mess_q=None, close_q=None):
    """ Continually loops through receiving communications and passing them to a queue

        Parameters
        ----------
        sock: SocketSever
            Object which contains receive commands and has been connected to client
        connection:
            Socket connection for Pi
        mess_q: queue.Queue
            Queue to place received socket messages in
        close_q: queue.Queue
            Queue to close function
        """
    while True:
        # Check queue to see if function is being closed
        try:
            mess = close_q.get(block=False)
            if mess:
                return
        except queue.Empty:
            pass

        # Receive socket data (this is a blocking process until a complete message is received)
        message = sock.recv_comms(connection)

        # Decode the message into dictionary
        dec_mess = sock.decode(message)

        # Add message to queue to be processed
        mess_q.put(dec_mess)


