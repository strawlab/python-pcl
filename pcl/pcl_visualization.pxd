# -*- coding: utf-8 -*-
# Header for pcl_visualization.pyx functionality that needs sharing with other modules.

cimport pcl_visualization_defs as pcl_vis
cimport vtk_defs as vtk

from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

# class override(PointCloud)
cdef class PointCloudColorHandleringCustom:
    cdef pcl_vis.PointCloudColorHandlerCustom_Ptr_t thisptr_shared     # PointCloudColorHandlerCustom[PointXYZ]
    
    # cdef inline PointCloudColorHandlerCustom[cpp.PointXYZ] *thisptr(self) nogil:
    # pcl_visualization_defs
    cdef inline pcl_vis.PointCloudColorHandlerCustom[cpp.PointXYZ] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloudColorHandlerCustom<PointXYZ>.
        return self.thisptr_shared.get()



cdef class PointCloudGeometryHandleringCustom:
    cdef pcl_vis.PointCloudGeometryHandlerCustom_Ptr_t thisptr_shared     # PointCloudGeometryHandlerCustom[PointXYZ]
    
    # cdef inline PointCloudGeometryHandlerCustom[cpp.PointXYZ] *thisptr(self) nogil:
    # pcl_visualization_defs
    cdef inline pcl_vis.PointCloudGeometryHandlerCustom[cpp.PointXYZ] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloudGeometryHandlerCustom<PointXYZ>.
        return self.thisptr_shared.get()


cdef class vtkSmartPointerRenderWindow:
    # cdef vtk.vtkRenderWindow_Ptr_t thisptr_shared     # vtkRenderWindow
    cdef vtk.vtkSmartPointer[vtk.vtkRenderWindow] thisptr_shared
    
    # cdef inline vtk.vtkRenderWindow *thisptr(self) nogil:
    cdef inline vtk.vtkSmartPointer[vtk.vtkRenderWindow] thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying vtkRenderWindow.
        return self.thisptr_shared

