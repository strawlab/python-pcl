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

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclftr.NormalEstimation_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

#   def compute(self, *normals):

