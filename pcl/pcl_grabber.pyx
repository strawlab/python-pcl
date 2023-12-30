# -*- coding: utf-8 -*-
# cython: embedsignature=True
#
# Copyright 2014 Netherlands eScience Center

from libcpp cimport bool

cimport numpy as np

cimport _pcl
cimport pcl_defs as cpp
cimport pcl_grabber as pcl_grb
from boost_shared_ptr cimport shared_ptr
# https://groups.google.com/forum/#!topic/cython-users/Eeqp4NkbAAA
cimport _bind_defs 

# from grabber_callback cimport PyLibCallBack
# from grabber_callback cimport callback


### Enum Setting ###
# pcl_visualization_defs.pxd
# cdef enum RenderingProperties:
# Re: [Cython] resolving name conflict -- does not work for enums !? 
# https://www.mail-archive.com/cython-dev@codespeak.net/msg02494.html
# PCLVISUALIZER_POINT_SIZE = pcl_grb.PCL_VISUALIZER_POINT_SIZE
# PCLVISUALIZER_OPACITY = pcl_grb.PCL_VISUALIZER_OPACITY
# PCLVISUALIZER_LINE_WIDTH = pcl_grb.PCL_VISUALIZER_LINE_WIDTH
# PCLVISUALIZER_FONT_SIZE = pcl_grb.PCL_VISUALIZER_FONT_SIZE
# PCLVISUALIZER_COLOR = pcl_grb.PCL_VISUALIZER_COLOR
# PCLVISUALIZER_REPRESENTATION = pcl_grb.PCL_VISUALIZER_REPRESENTATION
# PCLVISUALIZER_IMMEDIATE_RENDERING = pcl_grb.PCL_VISUALIZER_IMMEDIATE_RENDERING
### Enum Setting(define Class InternalType) ###

# CallbackTest
# include "pxi/Grabber/PyGrabberCallback.pxi"
# include "pxi/Grabber/PyGrabberNode.pxi"

# -*- coding: utf-8 -*-
cimport pcl_grabber as pcl_grb

# cdef class PyGrabberCallback:
#     cdef PyLibCallBack* thisptr
# 
#     def __cinit__(self, method):
#         # 'callback' :: The pattern/converter method to fire a Python 
#         #               object method from C typed infos
#         # 'method'   :: The effective method passed by the Python user 
#         self.thisptr = new PyLibCallBack(callback, <void*>method)
# 
#     def __dealloc__(self):
#        if self.thisptr:
#            del self.thisptr
# 
#     cpdef double execute(self, parameter):
#         # 'parameter' :: The parameter to be passed to the 'method'
#         return self.thisptr.cy_execute(<void*>parameter)
# 
# cdef class PyGrabberNode:
#     cdef double d_prop
# 
#     # def __cinit__(self):
#     #     self.thisptr = new PyLibCallBack(callback, <void*>method)
# 
#     # def __dealloc__(self):
#     #    if self.thisptr:
#     #        del self.thisptr
# 
#     def Test(self):
#         print('PyGrabberNode - Test')
#         d_prop = 10.0


# Grabber
# pcl 1.8.0 no use
# include "pxi/Grabber/ONIGrabber.pxi"
include "pxi/Grabber/OpenNIGrabber.pxi"

