PyCamPermanent

Software for permanent installation PiCam


Bug report:
1. Disconnecting the ext_comm more than once leads to and index error in close_connection (from masterpi)
2. If camera disconnects from masterpi and Index Error is thrown, due to receiving from a connection which doesn't exist possibly?