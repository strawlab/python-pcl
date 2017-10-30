# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features_172 as pcl_ftr

from boost_shared_ptr cimport sp_assign

cdef extern from "minipcl.h":
    void mpcl_features_NormalEstimationMethod_AVERAGE_3D_GRADIENT(pcl_ftr.IntegralImageNormalEstimation_t ) except +
    void mpcl_features_NormalEstimationMethod_COVARIANCE_MATRIX(pcl_ftr.IntegralImageNormalEstimation_t ) except +
    void mpcl_features_NormalEstimationMethod_AVERAGE_DEPTH_CHANGE(pcl_ftr.IntegralImageNormalEstimation_t ) except +
    void mpcl_features_NormalEstimationMethod_SIMPLE_3D_GRADIENT(pcl_ftr.IntegralImageNormalEstimation_t ) except +
    void mpcl_features_NormalEstimationMethod_compute(pcl_ftr.IntegralImageNormalEstimation_t, cpp.PointCloud_Normal_t ) except +

cdef class IntegralImageNormalEstimation:
    """
    IntegralImageNormalEstimation class for 
    """
    # cdef pcl_ftr.IntegralImageNormalEstimation_t *me

    def __cinit__(self, PointCloud pc not None):
        sp_assign(self.thisptr_shared, new pcl_ftr.IntegralImageNormalEstimation[cpp.PointXYZ, cpp.Normal]())
        # NG : Reference Count 
        self.thisptr().setInputCloud(pc.thisptr_shared)
        # self.me = new pcl_ftr.IntegralImageNormalEstimation_t()
        # self.me.setInputCloud(pc.thisptr_shared)
        # pass

    def set_NormalEstimation_Method_AVERAGE_3D_GRADIENT (self):
       mpcl_features_NormalEstimationMethod_AVERAGE_3D_GRADIENT(<pcl_ftr.IntegralImageNormalEstimation_t> deref(self.thisptr()))

    def set_NormalEstimation_Method_COVARIANCE_MATRIX (self):
       mpcl_features_NormalEstimationMethod_COVARIANCE_MATRIX(<pcl_ftr.IntegralImageNormalEstimation_t> deref(self.thisptr()))

    def set_NormalEstimation_Method_AVERAGE_DEPTH_CHANGE (self):
       mpcl_features_NormalEstimationMethod_AVERAGE_DEPTH_CHANGE(<pcl_ftr.IntegralImageNormalEstimation_t> deref(self.thisptr()))

    def set_NormalEstimation_Method_SIMPLE_3D_GRADIENT (self):
       mpcl_features_NormalEstimationMethod_SIMPLE_3D_GRADIENT(<pcl_ftr.IntegralImageNormalEstimation_t> deref(self.thisptr()))

    # enum Set NG
    # def set_NormalEstimation_Method (self):
    #    self.thisptr().setNormalEstimationMethod(pcl_ftr.NormalEstimationMethod2.ESTIMATIONMETHOD_COVARIANCE_MATRIX)

    def set_MaxDepthChange_Factor(self, double param):
        self.thisptr().setMaxDepthChangeFactor(param)

    def set_NormalSmoothingSize(self, double param):
        self.thisptr().setNormalSmoothingSize(param)

    def compute(self):
        normal = PointCloud_Normal()
        sp_assign(normal.thisptr_shared, new cpp.PointCloud[cpp.Normal]())
        cdef cpp.PointCloud_Normal_t *cNormal = <cpp.PointCloud_Normal_t*>normal.thisptr()
        (<pclftr.Feature_t*>self.thisptr()).compute(deref(cNormal))
        return normal


