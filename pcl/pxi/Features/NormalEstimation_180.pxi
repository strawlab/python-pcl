# -*- coding: utf-8 -*-
cimport _pcl
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features_180 as pclftr


cdef class NormalEstimation:
    """
    NormalEstimation class for 
    """
    cdef pclftr.NormalEstimation_t *me

    def __cinit__(self):
        self.me = new pclftr.NormalEstimation_t()
        # sp_assign(self.thisptr_shared, new pclftr.NormalEstimation[cpp.PointXYZ, cpp.Normal]())

    def __dealloc__(self):
        del self.me

    def set_SearchMethod(self, _pcl.KdTree kdtree):
        self.me.setSearchMethod(kdtree.thisptr_shared)

    def set_RadiusSearch(self, double param):
        self.me.setRadiusSearch(param)

    def set_KSearch (self, int param):
        self.me.setKSearch (param)

    def compute(self):
        normals = PointCloud_Normal()
        sp_assign(normals.thisptr_shared, new cpp.PointCloud[cpp.Normal]())
        cdef cpp.PointCloud_Normal_t *cNormal = <cpp.PointCloud_Normal_t*>normals.thisptr()
        (<pclftr.Feature_t*>self.me).compute(deref(cNormal))
        return normals

