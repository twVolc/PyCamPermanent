PyCamPermanent

Software for permanent installation PiCam

Bug report:
1. Disconnecting the ext_comm more than once leads to and index error in close_connection (from masterpi)
2. If camera disconnects from masterpi an Index Error is thrown, due to receiving from a connection which doesn't exist possibly?

TODO ideas:
1. A comm command should be introduced to cause the camera to send it all of its current settings, e.g. framerates and shutter speeds.
The external comm can then use this message to update its GUI with the correct current settings
2. Have a connection section (probably in settings at the top toolbar) to input IP and port and then connect to the camera. 
Turn a light green in the main window when we are connected to a camera


Requirements for GUI:
> numpy

> opencv

> ttkthemes (pip install ttkthemes)

> matplotlib 