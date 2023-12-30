
# -*- coding: utf-8 -*-
# 
# http://ros-robot.blogspot.jp/2011/08/point-cloud-librarykinect.html
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_grabber as pclgrb

cimport eigen as eigen3

from boost_shared_ptr cimport shared_ptr

# callback
from cython.operator cimport dereference as deref
import sys
# referenced from
# http://stackoverflow.com/questions/5242051/cython-implementing-callbacks

ctypedef double (*method_type)(void *param, void *user_data)

cdef extern from "grabber_callback.hpp":
    cdef cppclass cpp_backend:
        cpp_backend(method_type method, void *callback_func)
        double callback_python(void *parameter)

cdef double scaffold(void *parameter, void *callback_func):
    return (<object>callback_func)(<object>parameter)


cdef class SimpleNIGrabber:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in SimpleNIGrabber(pc).
    """
    cdef cpp_backend *thisptr

    def __cinit__(self, pycallback_func):
        self.thisptr = new cpp_backend(scaffold, <void*>pycallback_func)

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    cpdef double callback(self, parameter):
        return self.thisptr.callback_python(<void*>parameter)

    def run():
        cdef pclgrb.Grabber interface = pclgrb.OpenNIGrabber()
        # boost::function<void (const PointCloud_PointXYZRGB_t)> f = 
        #     boost::bind (&SimpleOpenNIViewer::cloud_cb_, this, _1)
        # interface.registerCallback (f)
        interface.start ()

        while (!viewer.wasStopped())
            sleep (1)
        end
        interface.stop ()

