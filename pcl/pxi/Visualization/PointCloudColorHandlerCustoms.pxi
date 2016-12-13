# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization as pcl_vis
from boost_shared_ptr cimport sp_assign

cdef class PointCloudColorHandlerCustoms:
    """
    
    """
    # cdef pcl_vis.PointCloudColorHandlerCustom_t *me
    cdef PointCloudColorHandlerCustom_PointWithRange_t *me

    def __cinit__(self, int r, int g, int b):
    # def __cinit__(self, int r, int g, int b, PointCloudWrapper_PointWithViewpoint pc):
        # self.me = new pcl_vis.PointCloudColorHandlerCustom()
        # self.me = <pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t*> new pcl_vis.PointCloudColorHandlerCustom(pc.thisptr_shared, r, g, b)
        # self.me = <pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t*> new pcl_vis.PointCloudColorHandlerCustom_PointWithRange_t(pc.thisptr_shared, r, g, b)
        # self.me = <> new pcl_vis.PointCloudColorHandlerCustom[cpp.PointWithRange] ()
        # self.me = new pcl_vis.PointCloudColorHandlerCustom(pc.thisptr()[0], r, g, b)
        print('__cinit__')

    def __dealloc__(self):
        del self.me

    # cdef inline pcl_vis.PointCloudColorHandlerCustom_t *thisptr(self) nogil:
    #     # Shortcut to get raw pointer to underlying PointCloudColorHandlerCustom
    #     return self.thisptr_shared.get()

    # def __repr__(self):
    #     return "<PointCloudColorHandlerCustom of %d points>" % self.vertices.size()

