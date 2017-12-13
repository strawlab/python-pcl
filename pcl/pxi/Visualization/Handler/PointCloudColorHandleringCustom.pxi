# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_visualization
from _pcl cimport PointCloud_PointWithViewpoint
from _pcl cimport RangeImages
cimport pcl_defs as cpp
cimport numpy as cnp


cimport pcl_visualization_defs as pcl_vis
from boost_shared_ptr cimport sp_assign

cdef class PointCloudColorHandleringCustom:
    """
    """
    # cdef pcl_vis.PointCloudColorHandlerCustom_t *me

    def __cinit__(self):
        pass

    # void sp_assign[T](shared_ptr[T] &p, T *value)
    def __cinit__(self, _pcl.PointCloud pc, int r, int g, int b):
        sp_assign(self.thisptr_shared, new pcl_vis.PointCloudColorHandlerCustom[cpp.PointXYZ](pc.thisptr_shared, r, g, b))
        pass

    # def __cinit__(self, _pcl.RangeImages rangeImage, int r, int g, int b):
    #     sp_assign(self.thisptr_shared, new pcl_vis.PointCloudColorHandlerCustom[cpp.PointWithViewpoint](rangeImage.thisptr_shared, r, g, b))
    #     pass

    def __dealloc__(self):
        # del self.me
        pass



