PyCamPermanent

Software for permanent installation PiCam

To install:
> ./doas/cfg.py needs to be modified for absolute pathnames on local machine
> ./conf/processing_setting_defaults.txt needs to be modified to load clear sky images for specific location

Bug report:
1. Disconnecting the ext_comm more than once leads to and index error in close_connection (from masterpi)
2. If camera disconnects from masterpi an Index Error is thrown, due to receiving from a connection which doesn't exist possibly?

Dev notes:
1. If I have an issue with sockets not performing properly, i have recently changed SocketServer.close_socket() to include a 
try/except clause for shutdown. Previously this contained the shutdown with the clauses. I'm not sure if catching this error
will change functionality anywhere, but the error was preventing tests from passing and I think throwing an OSError saying the
socket is no longer connected is not a useful functionality

TODO ideas:
1. A comm command should be introduced to cause the camera to send all of its current settings, e.g. framerates and shutter speeds.
The external comm can then use this message to update its GUI with the correct current settings
2. Have a connection section (probably in settings at the top toolbar) to input IP and port and then connect to the camera. 
Turn a light green in the main window when we are connected to a camera
3. Calibration of spectrometer is in an optional window to save constant GUI work. But this means that the reference
spectra aren't automatically loaded on start-up. I may need to do this somehow - but perhaps this will be done as part
of the processing thread which will need to be running constantly.


Requirements for GUI:
> In pycam_gui.py must update os.environ[PROJ_LIB] to point to correct place. Only necessary if there are issues with importing basemap
>
> pyplis. pyplis.custom_image_import needs to be updated to contain load_picam_png() function
>
> seabreeze (conda install -c conda-forge seabreeze) for pi only
>
> pyserial
>
> numpy
>
> opencv
>
> ttkthemes (pip install ttkthemes)
>
> matplotlib 
>
> astropy
>
> scipy
>
> scikit-image