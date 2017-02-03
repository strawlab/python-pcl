# -*- coding: utf-8 -*-
cimport _pcl
from _pcl cimport PointCloudWrapper_PointWithViewpoint
# NG
# from _pcl cimport RangeImage
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_range_image as pcl_r_img

cimport pcl_visualization as pcl_vis
from boost_shared_ptr cimport sp_assign

cdef class PointCloudColorHandleringCustom:
    """
    """
    cdef pcl_vis.PointCloudColorHandlerCustom_t *me
    
    def __cinit__(self):
        print('__cinit__')
        pass
    
    def __dealloc__(self):
        print('__dealloc__')
        # del self.me
        pass


