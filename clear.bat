del pcl\_pcl.cpp
del pcl\_pcl.pyd
del pcl\_pcl_172.cpp
del pcl\_pcl_172.pyd
del pcl\_pcl_180.cpp
del pcl\_pcl_180.pyd
del pcl\pcl_registration_160.cpp
del pcl\pcl_registration_160.pyd
del pcl\pcl_registration_172.cpp
del pcl\pcl_registration_172.pyd
del pcl\pcl_registration_180.cpp
del pcl\pcl_registration_180.pyd
del pcl\pcl_visualization.cpp
del pcl\pcl_visualization.pyd
del pcl\pcl_grabber.cpp
del pcl\pcl_grabber.pyd
rd /s /q build
rd /s /q python_pcl.egg-info
pip uninstall python-pcl -y