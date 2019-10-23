# -*- coding: utf-8 -*-

"""Testing socket functionality:
> Sending and receiving images
"""
from pycam.sockets import SocketClient, SocketServer, PiSocketCam
import threading
import time
import cv2
import numpy as np
from PIL import Image


class TestSockets:

    def open_sockets(self):
        """Open host and client sockets for use in tests"""
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