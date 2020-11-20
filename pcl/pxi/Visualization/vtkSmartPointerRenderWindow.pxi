# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport cython
cimport pcl_visualization
from pcl_visualization import vtkSmartPointerRenderWindow
cimport pcl_visualization_defs as pcl_vis
# cimport vtk_defs as vtk

from libcpp.string cimport string

from boost_shared_ptr cimport shared_ptr
from boost_shared_ptr cimport sp_assign


cdef class vtkSmartPointerRenderWindow:
    """
    """
    def __cinit__(self):
        pass

    def GetPointer(self):
        # import ctypes
        # build ok./not convert vtk objects
        # return id(<size_t>self.thisptr().GetPointer())
        # return ctypes.addressof(id(<size_t>self.thisptr().GetPointer()))
        # build ok./not convert vtk objects
        # import vtk
        # return vtk.vtkRenderWindow(<size_t>self.thisptr().GetPointer())
        pass


