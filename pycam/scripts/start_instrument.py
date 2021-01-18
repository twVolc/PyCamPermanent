# -*- coding: utf-8 -*-

"""
This script starts the pycam software on the instrument, but it first checks to ensure that it isn't already running, to
ensure we don't duplicate the program, which could lead to unexpected behaviour.
*** This is the recommended way that pycam should be started ***
"""

# Update python path so that pycam module can be found
import sys
sys.path.append('/home/pi/')

from pycam.utils import read_file
from pycam.setupclasses import FileLocator, ConfigInfo
import subprocess
import os

# Read configuration file which contains important information for various things
config = read_file(FileLocator.CONFIG)

# Start_script
start_script = config[ConfigInfo.master_script]
start_script_name = os.path.split(start_script)[-1]

try:
    proc = subprocess.Popen(['ps axg'], stdout=subprocess.PIPE, shell=True)
    stdout_value = proc.communicate()[0]
    stdout_str = stdout_value.decode("utf-8")
    stdout_lines = stdout_str.split('\n')
    for line in stdout_lines:
        if start_script_name in line:
            print('{} is already running as process {}'.format(start_script_name, line.split()[0]))
            sys.exit()

    start_script = 'python3 ' + start_script + ' &'
    subprocess.Popen([start_script], shell=True)
    print('{} started on instrument'.format(start_script_name))
except BaseException as e:
    print(e)

