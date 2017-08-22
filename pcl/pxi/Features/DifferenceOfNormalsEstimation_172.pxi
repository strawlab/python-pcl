# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features_172 as pcl_ftr

from boost_shared_ptr cimport sp_assign

cdef class DifferenceOfNormalsEstimation:
    """
    DifferenceOfNormalsEstimation class for 
    """
    cdef pcl_ftr.DifferenceOfNormalsEstimation_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pcl_ftr.DifferenceOfNormalsEstimation_t()
        self.me.setInputCloud(pc.thisptr_shared)


