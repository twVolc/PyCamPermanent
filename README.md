PyCamPermanent

###Check that I don't add too many dependencies to utils and setupclasses - wittypi runs the remote_pi_on/off scripts as root, so root needs to be able to import all modules otherwise it will fail###

Software for permanent installation PiCam

To install:
> ./doas/cfg.py needs to be modified for absolute pathnames on local machine
> ./conf/processing_setting_defaults.txt needs to be modified to load clear sky images for specific location
> sudo apt install exfat-fuse exfat-utils for SSD compatibility on pi

Hardware setup:
> Connect GPIO 23 (physical pin 16) on masterpi to GPIO 3 (physical pin 5) on slave pi. This allows off/on functionality through wittypi start-up/shutdown scripts

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
2. I am going to have to make the processing a thread, to stop the GUI from freezing up. This means that thread can't be in charge 
of calling the draw() command on plots. Use the SpecScan technique of intermittently updating the plots, and flag draw=False
in the update_plot() calls from the processing thread. Can use this technique only when in the processing thread, and then 
kill the intermittent plot updater when processing is stopped


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
> 
> If using iFit for spectrometer analysis must clone iFit repo from https://github.com/benjaminesse/iFit. Clone to pycam/doas folder
> tqdm (for ifit)