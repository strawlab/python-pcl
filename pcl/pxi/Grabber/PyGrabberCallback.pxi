# -*- coding: utf-8 -*-
cimport pcl_grabber as pcl_grb
from ../../grabber_callback cimport PyLibCallBack
from ../../grabber_callback cimport callback

cdef class PyGrabberCallback:
    cdef PyLibCallBack* thisptr

    def __cinit__(self, method):
        # 'callback' :: The pattern/converter method to fire a Python 
        #               object method from C typed infos
        # 'method'   :: The effective method passed by the Python user 
        self.thisptr = new PyLibCallBack(callback, <void*>method)

    def __dealloc__(self):
       if self.thisptr:
           del self.thisptr

    cpdef double execute(self, parameter):
        # 'parameter' :: The parameter to be passed to the 'method'
        return self.thisptr.cy_execute(<void*>parameter)