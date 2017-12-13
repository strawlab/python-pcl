# -*- coding: utf-8 -*-
# http://ros-robot.blogspot.jp/2011/08/point-cloud-librarykinect.html
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_grabber as pcl_grb

cimport eigen as eigen3
cimport _bind_defs as _bind

from boost_shared_ptr cimport shared_ptr

cdef void some_callback(void* some_ptr):
    print('Hello from some_callback (Cython) !')
    # print 'some_ptr: ' + some_ptr

cdef class OpenNIGrabber_:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in OpenNIGrabber().
    """
    cdef pcl_grb.OpenNIGrabber *me

    # cdef void some_callback(self, void* some_ptr):
    #     print('Hello from some_callback (Cython) !')
    #     # print 'some_ptr: ' + some_ptr

    def __cinit__(self, device_id, depth_mode, image_mode):
        self.me = new pcl_grb.OpenNIGrabber(device_id, depth_mode, image_mode)

    def __dealloc__(self):
        del self.me

    def RegisterCallback (self, func):
        cdef _bind.arg _1
        # cdef _bind.function[_bind.callback_t] callback = _bind.bind[_bind.callback_t](<_bind.callback_t> func, _1)
        # NG(Cannot assign type 'void (OpenNIGrabber_, void *)' to 'callback_t')
        # cdef _bind.function[_bind.callback_t] callback = _bind.bind[_bind.callback_t](self.some_callback, _1)
        cdef _bind.function[_bind.callback_t] callback = _bind.bind[_bind.callback_t](some_callback, _1)
        # self.me.register_callback(callback)
        self.me.registerCallback[_bind.callback_t](callback)
        # (<pcl_grb.Grabber*>self.me).registerCallback[_bind.callback_t](callback)

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

