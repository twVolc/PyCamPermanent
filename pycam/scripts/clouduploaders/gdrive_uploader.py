# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/pi/')

from pycam.utils import read_file
from pycam.setupclasses import FileLocator, CameraSpecs, SpecSpecs
from pycam.directory_watcher import can_watch_directories, create_dir_watcher
from pycam.utils import get_img_time, get_spec_time
import os
import time

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class GoogleDriveUploader:
    "Class for watching directory for new data, then uploading that data to Google Drive"
    def __init__(self, root_folder_id=None, watch_folder=None):
        self.gauth = GoogleAuth()
        self.drive = GoogleDrive(self.gauth)
        self.watch_folder = watch_folder

        if root_folder_id is None:
            self.root_folder_id = self.get_folder_id_from_file()
        else:
            self.root_folder_id = root_folder_id
        self.folder_id = self.root_folder_id

        # Setup directory watcher
        if os.path.exists(self.watch_folder):
            self.watcher = create_dir_watcher(self.watch_folder, False, self.directory_watch_handler)
        else:
            self.watcher = None

        self.cam_specs = CameraSpecs()
        self.spec_specs = SpecSpecs()

    @staticmethod
    def get_folder_id_from_file():
        file_contents = read_file(FileLocator.GOOGLE_DRIVE_PARENT_FOLDER)
        folder_id = file_contents['folder_id']
        return folder_id

    def create_folder(self, folder_name, set_id=True):
        """
        Add folder to current folder in G-drive
        set_id  bool    If True, we set this class's folder id to this new folder
        """
        # Check if folder already exists
        folder = self.get_folder(folder_name)
        if folder is not None:
            print("Folder '{}' already exists within parent '{}'".format(folder_name, self.root_folder_id))
            return

        file_metadata = {
            'title': folder_name,
            'parents': [{'id': self.root_folder_id}], #parent folder
            'mimeType': 'application/vnd.google-apps.folder'
        }

        folder = self.drive.CreateFile(file_metadata)
        folder.Upload()

        if set_id:
            self.folder_id = folder['id']
        return folder

    def get_folder(self, folder_name):
        """
        Lists folder in parent folder and identifies if folder exists. If it does, it makes that folder the current
        folder for this object
        :param folder_name:
        :return:
        """
        # List all folders in drive
        # mimeType='application/vnd.google-apps.folder' and
        f = self.drive.ListFile({"q": "'{}' in parents and trashed=false".format(self.root_folder_id)}).GetList()

        # Set original return folder to None (if we don't find the folder, None is returned
        folder_return = None
        for folder in f:
            # If we find the folder with the correct title we return it
            if folder['title'] == folder_name:
                folder_return = folder['id']
                break

        return folder_return

    def upload_file(self, file_path, filename, folder=None):
        """
        Uploads file to folder. If no folder is provided we use the current folder_id
        :param file:        Full path to file to be uploaded
        :param file:        Filename to be uploaded
        :param folder:
        :return:
        """
        if folder is not None:
            self.create_folder(folder)
            gfile = self.drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": self.folder_id}],
                                           'title': filename})
        else:
            gfile = self.drive.CreateFile({'parents': [{'id': self.folder_id}],
                                           'title': filename})

        gfile.SetContentFile(os.path.join(file_path, filename))
        gfile.Upload()
        print('Uploaded file: {}'.format(filename))

    def directory_watch_handler(self, pathname, t):
        """Controls the watching of a directory"""
        directory = os.path.dirname(pathname)
        filename = os.path.basename(pathname)

        ext = os.path.splitext(filename)

        if ext == self.cam_specs.file_ext:
            file_time = get_img_time(filename, date_loc=self.cam_specs.file_datestr)
        elif ext == self.spec_specs.file_ext:
            file_time = get_spec_time(filename, date_loc=self.spec_specs.file_datestr)
        else:
            print('Did not recognise data type with extension: {}'.format(ext))
            return

        # Format time to date directory string
        date_str = file_time.strftime('%Y-%m-%d')

        # Upload file to correct date directory
        self.upload_file(directory, filename, folder=date_str)






if __name__ == '__main__':

    upload_file_list = ['C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\pycam\\tests\\test_data\\2019-09-18T074335_fltrA_1ag_999904ss_Plume.png',
                        'C:\\Users\\tw9616\\Documents\\PostDoc\\Permanent Camera\\PyCamPermanent\\pycam\\tests\\test_data\\temp_io_test.txt']

    gdrive = GoogleDriveUploader()
    for upload_file in upload_file_list:
        pathname = os.path.dirname(upload_file)
        filename = os.path.basename(upload_file)
        gdrive.upload_file(file_path=pathname, filename=filename, folder='test')




    # file_contents = read_file(FileLocator.GOOGLE_DRIVE_FOLDER)
    # folder_id = file_contents['folder_id']
    #
    # gauth = GoogleAuth()
    # drive = GoogleDrive(gauth)


    # for upload_file in upload_file_list:
    #     t1 = time.time()
    #     filename = upload_file.split('\\')[-1]
    #
    #     gfile = drive.CreateFile({'parents': [{'id': folder_id}],
    #                               'title': filename})
    #     # Read file and set it as the content of this instance.
    #
    #     print('Uploading file: {}\n'.format(filename))
    #     gfile.SetContentFile(upload_file)
    #     print('File contents set')
    #     gfile.Upload() # Upload the file.
    #     print('File uploaded. Time taken: {}'.format(time.time() - t1))



