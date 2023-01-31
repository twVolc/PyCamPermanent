# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/pi/')

from pycam.utils import read_file
from pycam.setupclasses import FileLocator, CameraSpecs, SpecSpecs
from pycam.directory_watcher import can_watch_directories, create_dir_watcher
from pycam.utils import get_img_time, get_spec_time
import os
import time

import dropbox
from dropbox.exceptions import AuthError
from dropbox import DropboxOAuth2FlowNoRedirect


class DropboxIO:
    """
    Class for watching directory for new data, then uploading that data to Dropbox
    Connection is via PKCE - https://github.com/dropbox/dropbox-sdk-python/blob/main/example/oauth/commandline-oauth-pkce.py
    """
    def __init__(self, refresh_token_from_file=True, root_folder=None, watch_folder=None, recursive=True,
                 delete_on_upload=False):
        self.recursive = recursive
        self.delete_on_upload = delete_on_upload      # If True, the file is deleted from the local machine after upload

        # Access token for dropbox
        # self.access_token = self.get_access_token_from_file()
        self.app_key = self.get_app_key_from_file()

        # Get refresh token for long-term connection
        if refresh_token_from_file:
            self.refresh_token = self.get_refresh_token_from_file()
        else:
            self.refresh_token = self.get_refresh_token()

        # Connect to dropbox
        self.dbx = self.dropbox_connect()

        if root_folder is None:
            self.root_folder = self.get_root_folder_from_file()
        else:
            self.root_folder = root_folder

        # Setup directory watcher
        if watch_folder is not None:
            self.set_watch_folder(watch_folder, start=False)
        else:
            self.watcher = None

        self.cam_specs = CameraSpecs()
        self.spec_specs = SpecSpecs()

    @staticmethod
    def get_root_folder_from_file():
        file_contents = read_file(FileLocator.DROPBOX_ACCESS_TOKEN)
        folder = file_contents['root_folder']
        return '/' + folder

    @staticmethod
    def get_access_token_from_file():
        file_contents = read_file(FileLocator.DROPBOX_ACCESS_TOKEN)
        access_token = file_contents['access_token']
        return access_token

    @staticmethod
    def get_app_key_from_file():
        file_contents = read_file(FileLocator.DROPBOX_ACCESS_TOKEN)
        app_key = file_contents['app_key']
        return app_key

    @staticmethod
    def get_refresh_token_from_file():
        file_contents = read_file(FileLocator.DROPBOX_ACCESS_TOKEN)
        refresh_token = file_contents['refresh_token']
        return refresh_token

    def get_refresh_token(self):
        """Gets refresh token for prolonged use of app"""
        auth_flow = DropboxOAuth2FlowNoRedirect(self.app_key, use_pkce=True, token_access_type='offline')
        authorize_url = auth_flow.start()
        print("1. Go to: " + authorize_url)
        print("2. Click \"Allow\" (you might have to log in first).")
        print("3. Copy the authorization code.")
        auth_code = input("Enter the authorization code here: ").strip()

        try:
            oauth_result = auth_flow.finish(auth_code)
        except Exception as e:
            print('Error: %s' % (e,))
            exit(1)

        with dropbox.Dropbox(oauth2_refresh_token=oauth_result.refresh_token, app_key=self.app_key) as dbx:
            dbx.users_get_current_account()
            print("Successfully set up client!")

        return oauth_result.refresh_token

    def dropbox_connect(self):
        """Create a connection to Dropbox."""
        try:
            dbx = dropbox.Dropbox(oauth2_refresh_token=self.refresh_token, app_key=self.app_key)
        except AuthError as e:
            print('Error connecting to Dropbox with access token: ' + str(e))
            dbx = None
        return dbx

    def upload_file(self, file_path, filename, folder=None, delete=False):
        """
        Uploads file to folder. If no folder is provided we use the current folder_id
        :param file_path:        Full path to file to be uploaded
        :param filename:        Filename to be uploaded
        :param folder:
        :param delete:  bool    If True, the file is deleted on the local machine after being uploaded
        :return:
        """
        full_path = os.path.join(file_path, filename)
        if not os.path.exists(full_path):
            print('DropboxIO: File does not exist: {}'.format(full_path))

        if folder is not None:
            dropbox_file_path = folder + '/' + filename
        else:
            dropbox_file_path ='/' +  filename

        with open(full_path, "rb") as f:
            meta = self.dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode("overwrite"))

        print('Uploaded file: {}'.format(filename))

        if delete:
            os.remove(full_path)

        return meta

    def set_watch_folder(self, watch_folder, start=True):
        """
        Sets up the watch folder and starts watching to upload new data
        :return:
        """
        self.watch_folder = watch_folder
        if os.path.exists(watch_folder):
            print('Setting up directory watcher: {}'.format(watch_folder))
            self.watcher = create_dir_watcher(watch_folder, self.recursive, self.directory_watch_handler)
            if start:
                self.watcher.start()
        else:
            print('Unrecognised watcher directory: {}'.format(watch_folder))
            self.watcher = None

    def upload_existing_files(self, timeout=1, delete_on_upload=False):
        """
        Uploads pre-existing files in folder
        :return:
        """
        # Get all existing files
        files = os.listdir(self.watch_folder)
        files.sort()

        for filename in files:
            file, ext = os.path.splitext(filename)
            if ext in [self.cam_specs.file_ext, self.spec_specs.file_ext]:

                # Check no lock file exists
                time_1 = time.time()
                while os.path.exists(file + '.lock') and time.time() - time_1 < timeout:
                    pass

                # Upload file
                self.upload_file(self.watch_folder, filename, folder=self.root_folder, delete=delete_on_upload)

    def directory_watch_handler(self, pathname, t):
        """Controls the watching of a directory"""
        directory = os.path.dirname(pathname)
        filename = os.path.basename(pathname)

        # Check there is no coexisting lock file
        file, ext = os.path.splitext(filename)
        if ext in [self.cam_specs.file_ext, self.spec_specs.file_ext]:
            while os.path.exists(file + '.lock'):
                pass
        else:
            return

        # Upload file to correct date directory
        self.upload_file(directory, filename, folder=self.root_folder, delete=self.delete_on_upload)



if __name__ == '__main__':

    # upload_file_list = ['C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\pycam\\tests\\test_data\\2019-09-18T074335_fltrA_1ag_999904ss_Plume.png',
    #                     'C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\pycam\\tests\\test_data\\temp_io_test.txt']

    upload_file_list = ['/home/pi/pycam/Images/2022-10-17T144122_fltrB_1ag_99980ss_Test.png']

    dbx = DropboxIO()

    #Get refresh token
    # dbx.get_refresh_token()


    # Upload files
    for upload_file in upload_file_list:
        pathname = os.path.dirname(upload_file)
        filename = os.path.basename(upload_file)
        # gdrive.upload_file(file_path=pathname, filename=filename, folder='test')
        # res = dbx.dbx.files_list_folder('/Apps')
        dbx.upload_file(file_path=pathname, filename=filename, folder='/Lascar')





