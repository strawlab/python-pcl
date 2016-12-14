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

cdef class PointCloudColorHandlerCustoms:
    """
    
    """
    # cdef pcl_vis.PointCloudColorHandlerCustom_t *me
    cdef pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t *me
    
    # def __cinit__(self, PointCloudWrapper_PointWithViewpoint pc, int r = 0, int g = 0, int b = 0):
    # NG
    # def __cinit__(self, RangeImage pc, int r = 0, int g = 0, int b = 0):
    def __cinit__(self, int r = 0, int g = 0, int b = 0):
    # NG
    # def __cinit__(self, pcl_r_img.RangeImage pc, int r = 0, int g = 0, int b = 0):
        # self.me = new pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t()
        # self.me = <pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t*> new pcl_vis.PointCloudColorHandlerCustom(pc.thisptr_shared, r, g, b)
        # self.me = <pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t*> new pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t(pc.thisptr_shared, r, g, b)
        # self.me = <> new pcl_vis.PointCloudColorHandlerCustom[cpp.PointWithRange] ()
        # self.me = <pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t*>new pcl_vis.PointCloudColorHandlerCustom(pc.thisptr()[0], r, g, b)
        # NG
        # self.me = new pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t()
        # ??
        # PointWithRange_t point = new cpp.PointWithRange_t()
        # self.me = <pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t*> new pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t(pc.thisptr_shared, r, g, b)
        # self.me = <pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t*> new pcl_vis.PointCloudColorHandlerCustom(<const cpp.PointCloud[cpp.PointWithRange]>pc.thisptr()[0], r, g, b)
        # self.me = <pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t*> new pcl_vis.PointCloudColorHandlerCustom(<const cpp.PointCloud[cpp.PointWithRange]>pc.me, r, g, b)
        # NG
        # self.me = <pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t*> new pcl_vis.PointCloudColorHandlerCustom(<cpp.PointCloud[cpp.PointWithRange]>pc.point, r, g, b)
        print('__cinit__')
    
    
    def __dealloc__(self):
        print('__dealloc__')
        # del self.me


