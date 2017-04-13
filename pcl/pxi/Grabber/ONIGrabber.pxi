
# -*- coding: utf-8 -*-
# 
# http://ros-robot.blogspot.jp/2011/08/point-cloud-librarykinect.html
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_grabber as pcl_grb

cimport eigen as eigen3
cimport _bind_defs as _bind

from boost_shared_ptr cimport shared_ptr

# callback
from cython.operator cimport dereference as deref
import sys
# referenced from
# http://stackoverflow.com/questions/5242051/cython-implementing-callbacks

cdef class ONIGrabber:
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

    def RegisterCallback (self, func):
        cdef _bind.arg _1
        cdef _bind.function[_bind.callback_t] callback = _bind.bind[_bind.callback_t](func, _1)
        self.me.register_callback(callback)

    def Start(self):
        self.start ()

    def Stop(self):
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

