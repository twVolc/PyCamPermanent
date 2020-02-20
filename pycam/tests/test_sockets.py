# -*- coding: utf-8 -*-

"""Testing socket functionality:
> Sending and receiving images
"""
from pycam.networking.sockets import read_network_file, SocketServer, PiSocketCam, CommsFuncs
from pycam.utils import write_file
import threading
import time
import numpy as np
from PIL import Image


class TestSockets:

    def test_io(self):
        """Tests file I/O of socket information (IP address and port)"""
        filename = '.\\test_data\\network.txt'
        sock_data = {'port': 12345, 'ip_address': '255.255.255.255'}

        # Write socket data to file
        write_file(filename, sock_data)

        # Read network file
        ip_addr, port = read_network_file(filename)

        assert ip_addr == sock_data['ip_address'] and port == sock_data['port']

    def open_sockets(self):
        """Open host and client sockets for use in tests
        NOTE: This is not a test itself, hence its name does not start with test"""
        host = '127.0.0.1'  # Localhost
        port = 12345  # Port number

        # Make host socket object
        sock_serv = SocketServer(host, port)

        # Open socket to listen for connection
        sock_serv.open_socket()

        # Accept connections (thread to continue wit code)
        t_1 = threading.Thread(target=sock_serv.acc_connection, args=())
        t_1.start()

        # Make client socket
        sock_cli = PiSocketCam(host, port)

        # Connect client to socket
        sock_cli.connect_socket()

        t_1.join()

        return sock_serv, sock_cli

    def test_create_sockets(self):
        """Tests initial socket creation and basic send function"""
        # Create sockets
        sock_serv, sock_cli = self.open_sockets()

        # Send data over through socket
        message = b'test'
        sock_serv.connections[0][0].sendall(message)

        # Recv data
        data = sock_cli.sock.recv(1024)

        # Close sockets
        sock_cli.close_socket()
        sock_serv.close_socket()

        assert data == message

    def test_send_recv_img(self):
        """Tests send and receive funcionality of PiSockets for image and associated information"""
        # Path to image for testing
        img_path = '.\\test_data\\2019-09-18T074335_fltrA_1ag_999904ss_Plume.png'

        # Open sockets
        sock_serv, sock_cli = self.open_sockets()

        # Read in image file
        sock_cli.camera.image = np.uint16(Image.open(img_path))
        sock_cli.camera.filename = img_path

        # Set timer for test
        time_start = time.time()

        # Send image over socket
        sock_cli.send_img(sock_cli.camera.filename, sock_cli.camera.image)

        # Receive image on server
        img_recv, filename = sock_serv.recv_img()

        print('Time taken to send and receive image: {:.6f}'.format(time.time() - time_start))

        sock_cli.close_socket()
        sock_serv.close_socket()

        # Compare the sent and received images
        assert img_path == filename
        assert np.array_equal(sock_cli.camera.image, img_recv)

    def test_send_recv_spec(self):
        """Tests send and receive funcionality of PiSockets for spectrum and associated information"""
        pass

    def test_get_connection(self):
        """Tests get connection"""
        sock_serv, sock_cli = self.open_sockets()

        # Attempt the get_connection() function
        conn = sock_serv.get_connection(ip='127.0.0.1')

        # Close socket to finish
        sock_serv.close_socket()

        assert conn == sock_serv.connections[0][0]

    def test_all_comms_cmds(self):
        """Tests that all comms in the CommsFunc.cmd_dict have and associated method in CommsFun"""
        comms_obj = CommsFuncs()

        # Loop through keys in cmd_dict and check they have the associated method
        for key in comms_obj.cmd_dict:

            # The 'ERR' flag does not require a method, so is not checked
            if key is not 'ERR':
                assert hasattr(comms_obj, key)

    def test_send_recv_comms(self):
        """Tests send and receive functionality of PiSockets for general commands and encoding"""
        sock_serv, sock_cli = self.open_sockets()

        t_1 = time.clock()

        # Define a dictionary with some correct messages and some incorrect
        cmd_dict = {'SSC': 300, 'SMN': 0.5, 'SMN': 0.91, 'SSS': 70000, 'EXT': False}

        # Define expected dictionary outcome (note, because a dictionary is used, defining 'SMN' twice overwrites it,
        # so only the second definition is passed through to be encoded and sent. this prevents multiple commands of the
        # same key being sent.
        comms_exp = {'SSC': 300, 'ERR': ['SMN', 'SSS'], 'EXT': False}

        # Encode message to be sent
        comms = sock_serv.encode_comms(cmd_dict)

        # Get connection
        conn = sock_serv.get_connection(ip='127.0.0.1')

        # Send data over connection
        sock_serv.send_comms(conn, comms)

        # Receive data
        cmd = sock_cli.recv_comms(sock_cli.sock)

        # Decode data
        comms_out = sock_cli.decode_comms(cmd)

        t_2 = time.clock() - t_1
        print('Time for sending and encoding/decoding message: {:.6f}s'.format(t_2))

        assert comms_out == comms_exp

        # Perform same test but sending from the client to the server, to ensure it works both ways.
        sock_cli.send_comms(sock_cli.sock, comms)

        cmd = sock_serv.recv_comms(conn)

        comms_out = sock_serv.decode_comms(cmd)

        # Close socket to finish
        sock_serv.close_socket()

        assert comms_out == comms_exp