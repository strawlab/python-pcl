# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_grabber as pcl_grb

cdef class PyGrabberNode:
    cdef double d_prop

    # def __cinit__(self):
    #     self.thisptr = new PyLibCallBack(callback, <void*>method)

    # def __dealloc__(self):
    #    if self.thisptr:
    #        del self.thisptr

    def Test(self):
        print('PyGrabberNode - Test')
        d_prop = 10.0


