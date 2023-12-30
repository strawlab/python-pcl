# -*- coding: utf-8 -*-
# http://ros-robot.blogspot.jp/2011/08/point-cloud-librarykinect.html
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_grabber as pcl_grb

cimport eigen as eigen3
cimport _bind_defs as _bind

from boost_shared_ptr cimport shared_ptr


cdef class ONIGrabber_:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in ONIGrabber(pc).
    """
    cdef pcl_grb.ONIGrabber *me

    def __cinit__(self, string file_name, bool repeat, bool stream):
        self.me = new pcl_grb.ONIGrabber(file_name, repeat, stream)

    def __dealloc__(self):
        del self.me

    # def RegisterCallback (self, func):
    #     cdef _bind.arg _1
    #     cdef _bind.function[_bind.callback_t] callback = _bind.bind[_bind.callback_t](<_bind.callback_t> func, _1)
    #     self.me.register_callback(<_bind.function[callback_t]> callback)

    def Start(self):
        self.me.start ()

    def Stop(self):
        self.me.stop ()

    # string 
    # def getName ():
    #     return self.me.getName ()

    # bool 
    # def isRunning ():
    #     return self.me.isRunning ()

    # return float 
    # def getFramesPerSecond ():
    #     return self.me.getFramesPerSecond ()

