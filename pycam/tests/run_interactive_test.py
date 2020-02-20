# -*- coding: utf-8 -*-

"""To be run on a raspberry pi
This script tests the interactive_capture() method of <object Camera>"""

from pycam.controllers import Camera
import threading

# Create camera object
cam = Camera()

# Start interactive mode
capt_thread = threading.Thread(target=cam.interactive_capture, args=())
capt_thread.start()

# Send capture command
command = {'ss': 1000, 'type': 'Dark', 'exit': False}
cam.capture_q.put(command)
print('Put command in queue')

# Wait for image to be taken
filename, image = cam.img_q.get(block=True)

# Save image
cam.save_current_image(filename)

# Close interactive imager
command['exit'] = True
cam.capture_q.put(command)
