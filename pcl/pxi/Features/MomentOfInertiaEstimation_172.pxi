# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features_172 as pclftr


cdef class MomentOfInertiaEstimation:
    """
    MomentOfInertiaEstimation class for 
    """
    cdef pclftr.MomentOfInertiaEstimation_t *me

    def __cinit__(self):
        self.me = new pclftr.MomentOfInertiaEstimation_t()

    def __dealloc__(self):
        del self.me

    def set_SearchMethod(self, KdTree kdtree):
        self.me.setSearchMethod(kdtree.thisptr_shared)

    def set_KSearch (self, int param):
        self.me.setKSearch (param)

    # def compute(self):
    #     cloud_normals = PointCloud[Normal]
    #     self.me.compute (*cloud_normals)
    #     return cloud_normals

