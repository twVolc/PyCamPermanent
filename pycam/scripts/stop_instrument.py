# -*- coding: utf-8 -*-

"""
This script stops the pycam software on the instrument, to be used by crontab to schedule turning the instrument off
at night.
"""

# Update python path so that pycam module can be found
import sys
sys.path.append('/home/pi/')

from pycam.utils import read_file
from pycam.setupclasses import FileLocator, ConfigInfo
import subprocess
import os


def close_pycam():
    """Closes pycam by setting up a socket and telling the program to shutdown"""
    pass


# Read configuration file which contains important information for various things
config = read_file(FileLocator.CONFIG)

# Start_script
start_script = config[ConfigInfo.master_script]
start_script_name = os.path.split(start_script)[-1]

# If pycam is running we stop the script
try:
    proc = subprocess.Popen(['ps axg'], stdout=subprocess.PIPE, shell=True)
    stdout_value = proc.communicate()[0]
    stdout_str = stdout_value.decode("utf-8")
    stdout_lines = stdout_str.split('\n')
    for line in stdout_lines:
        if start_script_name in line:
            close_pycam()
            print('Pycam shutdown')
            sys.exit()

    # If we get to the end without finding the running script we write a warning to the log file
    with open(FileLocator.ERROR_LOG, 'w', newline='\n') as f:
        f.write('ERROR IN STOP SCRIPT: Warning, pycam script was not running when stop_instrument.py commenced\n')

except BaseException as e:
    with open(FileLocator.ERROR_LOG, 'w', newline='\n') as f:
        f.write('ERROR IN STOP SCRIPT: {}\n'.format(e))

