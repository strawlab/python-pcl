# -*- coding: utf-8 -*-
cimport _pcl
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features as pclftr


cdef class VFHEstimation:
    """
    VFHEstimation class for 
    """
    cdef pclftr.VFHEstimation_t *me

    def __cinit__(self):
        self.me = new pclftr.VFHEstimation_t()

    def __dealloc__(self):
        del self.me

    def set_SearchMethod(self, _pcl.KdTree kdtree):
        self.me.setSearchMethod(kdtree.thisptr_shared)

    def set_KSearch (self, int param):
        self.me.setKSearch (param)

    # use PointCloud[VFHSignature308]
    # def compute(self):
    #     normal = PointCloud_Normal()
    #     cdef cpp.PointCloud_Normal_t *cNormal = <cpp.PointCloud_Normal_t*>normal.thisptr()
    #     self.me.compute (deref(cNormal))
    #     return normal

