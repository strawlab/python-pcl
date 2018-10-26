# -*- coding: utf-8 -*-
# Header for pcl_visualization.pyx functionality that needs sharing with other modules.

cimport pcl_visualization_defs as vis

from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

# class override(PointCloud)
cdef class PointCloudColorHandleringCustom:
    cdef vis.PointCloudColorHandlerCustom_Ptr_t thisptr_shared     # PointCloudColorHandlerCustom[PointXYZ]
    
    # cdef inline PointCloudColorHandlerCustom[cpp.PointXYZ] *thisptr(self) nogil:
    # pcl_visualization_defs
    cdef inline vis.PointCloudColorHandlerCustom[cpp.PointXYZ] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloudColorHandlerCustom<PointXYZ>.
        return self.thisptr_shared.get()



cdef class PointCloudGeometryHandleringCustom:
    cdef vis.PointCloudGeometryHandlerCustom_Ptr_t thisptr_shared     # PointCloudGeometryHandlerCustom[PointXYZ]
    
    # cdef inline PointCloudGeometryHandlerCustom[cpp.PointXYZ] *thisptr(self) nogil:
    # pcl_visualization_defs
    cdef inline vis.PointCloudGeometryHandlerCustom[cpp.PointXYZ] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloudGeometryHandlerCustom<PointXYZ>.
        return self.thisptr_shared.get()



