# -*- coding: utf-8 -*-

"""File for running wittypi schedule updater"""

import subprocess

witty_script = '/home/pi/wittypi/runScript.sh'
try:
    proc = subprocess.Popen(witty_script, stdout=subprocess.PIPE, shell=True)
    stdout_value = proc.communicate()[0]
    print('stdout:', repr(stdout_value))
except BaseException as e:
    print(e)
