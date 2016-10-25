# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization as pclvis
from boost_shared_ptr cimport sp_assign

cdef class PointCloudColorHandlerCustoms:
    """
    
    """
    cdef pclvis.PointCloudColorHandlerCustom_t *me

    def __cinit__(self, int r, int g, int b, PointCloud pc not None):
        # self.me = new pclvis.PointCloudColorHandlerCustom()
        # self.me = new pclvis.PointCloudColorHandlerCustom(pc.thisptr_shared, r, g, b)
        self.me = <pclvis.PointCloudColorHandlerCustom_t*> new pclvis.PointCloudColorHandlerCustom_t(pc.thisptr_shared, r, g, b)
        # self.me = new pclvis.PointCloudColorHandlerCustom(pc.thisptr()[0], r, g, b)
        # self.me = <pcloct.OctreePointCloud_t*> new pcloct.OctreePointCloudSearch_t(resolution)

    def __dealloc__(self):
        del self.me

    # cdef inline pclvis.PointCloudColorHandlerCustom_t *thisptr(self) nogil:
    #     # Shortcut to get raw pointer to underlying PointCloudColorHandlerCustom
    #     return self.thisptr_shared.get()

    # def __repr__(self):
    #     return "<PointCloudColorHandlerCustom of %d points>" % self.vertices.size()

