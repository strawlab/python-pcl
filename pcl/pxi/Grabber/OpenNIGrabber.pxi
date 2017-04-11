
# -*- coding: utf-8 -*-
# 
# http://ros-robot.blogspot.jp/2011/08/point-cloud-librarykinect.html
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_grabber as pcl_grb

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


cdef class OpenNIGrabber:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in SimpleNIGrabber(pc).
    """
    # cdef cpp_backend *thisptr
    cdef pclfil.ONIGrabber *me

    def __cinit__(self, string file_name, bool repeat, bool stream):
        self.me = new pcl_grb.ONIGrabber(file_name, repeat, stream)

    def __dealloc__(self):
        del self.me

    cpdef double callback(self, parameter):
        return self.thisptr.callback_python(<void*>parameter)

    def start():
        self.start ()

    def stop():
        self.stop ()

    # string 
    def getName ()
        return self.getName ()

    # bool 
    def isRunning ()
        return self.isRunning ()

    # return float 
    def getFramesPerSecond ()
        return self.getFramesPerSecond ()

