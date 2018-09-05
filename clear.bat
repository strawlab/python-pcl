del pcl\_pcl*.cpp
del pcl\_pcl*.pyd
del pcl\pcl_registration_*.cpp
del pcl\pcl_registration_*.pyd
del pcl\pcl_visualization*.cpp
del pcl\pcl_visualization*.pyd
del pcl\pcl_grabber*.cpp
del pcl\pcl_grabber*.pyd
rd /s /q build
rd /s /q python_pcl.egg-info
pip uninstall python-pcl -y