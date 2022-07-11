PyCamPermanent

###Check that I don't add too many dependencies to utils and setupclasses - wittypi runs the remote_pi_on/off scripts as root, so root needs to be able to import all modules otherwise it will fail###

Software for permanent installation PiCam

To install:
> ./conf/cam_specs.txt needs to be set to the correct band on the pis
>
> ./conf/processing_setting_defaults.txt needs to be modified to load clear sky images for specific location
> 
> sudo apt install exfat-fuse exfat-utils for SSD compatibility on pi

Hardware setup:
> Connect GPIO 23 (physical pin 16) on masterpi to GPIO 3 (physical pin 5) on slave pi. This allows off/on functionality through wittypi start-up/shutdown scripts
> If using a 128GB microSD must expand filesystem after copying disk image. sudo raspi-config > advanced options > expand filesystem. All space should then be available to pi

Bug report:
1. Disconnecting the ext_comm more than once leads to and index error in close_connection (from masterpi)
   - FIXED
2. If camera disconnects from masterpi an Index Error is thrown, due to receiving from a connection which doesn't exist possibly?
3. Every now and then I get a Socket already in use error when starting masterpi, this makes it fail and then the camera continually tries to
connect. Need to add a loop to the SocketServer somewhere, so that if I get this error I just keep trying to make the socket until it isn't in use. 
   - I've added a loop to opening sockets in pycam_masterpi.py. This didn't solve the issue
   - I need to work out how to free up the socket again, or change socket (perhaps jump to another port?) FIXED 
4. Get broken pipe error on port 12345 after starting manual acquisition then disconnecting from the insturment and trying to reconnect.
Need to deal with broken pipe in some way - set up server and client to reconnect rather than needing to restart entire system.
   - FIXED generally. But still don't have a clever way of reconnecting if pipes ever do break
5. Acquiring spectra manually throws quite a few errors, possibly due to plotting? 
6. Acquiring test images eventually throws an error related to attempting to perform calibration FOV search on image stack
   - Need to get rid of this when just taking Test images. Or work out why it's failing anyway, as it may fail at other
   times in the same way too.

Dev notes:
1. If I have an issue with sockets not performing properly, i have recently changed SocketServer.close_socket() to include a 
try/except clause for shutdown. Previously this contained the shutdown with the clauses. I'm not sure if catching this error
will change functionality anywhere, but the error was preventing tests from passing and I think throwing an OSError saying the
socket is no longer connected is not a useful functionality

NOTE SSA, SSB, SSS will not change if Auto shutter speed is enabled. May need to send 2 rounds of commands, one to shut 
off Auto SS and the second to set shutter speed - if done in one round, if the SSA command is applied first, it will 
fail as AutoSS still won't have been turned off yet.

TODO ideas:



Requirements for GUI:
> In pycam_gui.py must update os.environ[PROJ_LIB] to point to correct place. Only necessary if there are issues with importing basemap
>
> pyplis
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
> If using iFit for spectrometer analysis must clone iFit repo from https://github.com/benjaminesse/iFit. 
> Clone to pycam/ folder. I also cloned the light_dilution branch to ifit_ld. If this remaisn saved in pycam then no installation is needed for new users 
> tqdm (for ifit)
> shapely (for light diluiton correction of ifit)



CHANGING IP ADDRESSES
IP addresses are read in from config files. A few locations need to be changed (on both Pis!):
> ./pycam/conf/config.txt
> ./pycam/conf/network_transfer.txt
> ./pycam/conf/network_comm.txt
> ./pycam/conf/network_ext.txt
In config.txt you need to change the host_ip (master) and pi_ip (slave) IPs.
You then need to make changes on the Pi operating system itself, since we have set it up to have a static IP address
May find online help useful for setting up the static IPs https://elinux.org/RPi_Setting_up_a_static_IP_in_Debian
Pi 1 (Witty Pi):
> /etc/network/interfaces - line 18: change to desired host address.
> May need to the change lines 20 and 21 in above file: network and broadcast to reflect the new IP. Keep the endings the same, just change the first 2 numbers
> /etc/ntp.conf - line 32: Change "restrict" IP address to desired new Pi 2 Address. 
> May then need to change the broadcast line lower down to contain the same first 3 numbers (then keep 255 I think)
Pi 2:
> /etc/network/interfaces - line 18: Change to desired address. 
> Again, then probably need to change lines 20 and 21 of above file: network and broadcast.
> /etc/ntp.conf - line 7: change "server" to Pi 1 IP
You may then need to change your laptop computer's IP to a static IP in the same subnet, otherwise connection will fail.