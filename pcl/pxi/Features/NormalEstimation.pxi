# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features as pclftr


cdef class NormalEstimation:
    """
    NormalEstimation class for 
    """
    cdef pclftr.NormalEstimation_t *me

    def __cinit__(self):
        self.me = new pclftr.NormalEstimation_t()

    def __dealloc__(self):
        del self.me

    def set_SearchMethod(self, KdTree kdtree):
        self.me.setSearchMethod(kdtree.thisptr_shared)

    def set_RadiusSearch(self, double param):
        self.me.setRadiusSearch(param)

    def set_KSearch (self, int param):
        self.me.setKSearch (param)

    def compute(self):
        normal = PointCloud_Normal()
        cdef cpp.PointCloud_Normal_t *cNormal = <cpp.PointCloud_Normal_t*>normal.thisptr()
        self.me.compute (deref(cNormal))
        return normal

