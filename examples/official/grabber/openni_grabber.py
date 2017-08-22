# -*- coding: utf-8 -*-
# The OpenNI Grabber Framework in PCL
# http://pointclouds.org/documentation/tutorials/openni_grabber.php

import pcl
import pcl.pcl_grabber
import pcl.pcl_visualization

def class SimpleOpenNIViewer:
    SimpleOpenNIViewer()
        viewer = pcl.pcl_visualization.CloudViewer(b'PCL OpenNI Viewer')
    
    def void cloud_cb_ (pcl.PointCloud_Ptr_t cloud):
        if !viewer.wasStopped() == True:
            viewer.showCloud (cloud)

    def void run ():
        interface = new pcl.OpenNIGrabber()
        
        interface.RegisterCallback (cloud_cb_)
        interface.Start ()
        
        while !viewer.wasStopped() == True:
            # boost::this_thread::sleep (boost::posix_time::seconds (1));
            time.sleep(1)

        interface.Stop()


if __name__ == '__main__':
v = SimpleOpenNIViewer()
v.run()

