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

