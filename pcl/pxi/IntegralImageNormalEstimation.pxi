# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features as pclftr


cdef class IntegralImageNormalEstimation:
    """
    IntegralImageNormalEstimation class for 
    """
    cdef pclftr.IntegralImageNormalEstimation_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclftr.IntegralImageNormalEstimation_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

# use minipcl 
#    def set_NormalEstimation_Method (self):
#        cdef pclftr.NormalEstimationMethod method = <pclftr.NormalEstimationMethod>1
#        self.me.setNormalEstimationMethod(method)

    def set_MaxDepthChange_Factor(self, double param):
        self.me.setMaxDepthChangeFactor(param)

    def set_NormalSmoothingSize(self, double param):
        self.me.setNormalSmoothingSize(param)

#   def compute(self, *normals):

