# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features as pcl_ftr

from boost_shared_ptr cimport sp_assign

cdef extern from "minipcl.h":
    void mpcl_features_NormalEstimationMethod_AVERAGE_3D_GRADIENT(pcl_ftr.IntegralImageNormalEstimation_t ) except +
    void mpcl_features_NormalEstimationMethod_COVARIANCE_MATRIX(pcl_ftr.IntegralImageNormalEstimation_t ) except +
    void mpcl_features_NormalEstimationMethod_AVERAGE_DEPTH_CHANGE(pcl_ftr.IntegralImageNormalEstimation_t ) except +
    void mpcl_features_NormalEstimationMethod_SIMPLE_3D_GRADIENT(pcl_ftr.IntegralImageNormalEstimation_t ) except +

cdef class IntegralImageNormalEstimation:
    """
    IntegralImageNormalEstimation class for 
    """
    # cdef pcl_ftr.IntegralImageNormalEstimation_t *me

    def __cinit__(self, PointCloud pc not None):
        print ('load table_scene_mug_stereo_textured.pcd')
        sp_assign(self.thisptr_shared, new pcl_ftr.IntegralImageNormalEstimation[cpp.PointXYZ, cpp.Normal]())
        self.thisptr().setInputCloud(pc.thisptr_shared)
        print ('load table_scene_mug_stereo_textured.pcd')
        # pass
        # self.me = new pcl_ftr.IntegralImageNormalEstimation_t()
        # self.me.setInputCloud(pc.thisptr_shared)

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

#     cdef cpp.PointNormalCloud_t normals
#       mpcl_compute_normals(<cpp.PointCloud[cpp.PointXYZ]> deref(self.thisptr()), ksearch, searchRadius, normals)
#       seg = SegmentationNormal()
#       cdef pclseg.SACSegmentationNormal_t *cseg = <pclseg.SACSegmentationNormal_t *>seg.me
#       cseg.setInputCloud(self.thisptr_shared)
#       cseg.setInputNormals (normals.makeShared())
#       return normals.makeShared()
#     def compute(self):
#         normalEstimation = pcl_ftr.NormalEstimation()
#         cdef pcl_ftr.NormalEstimation_t *cNormalEstimation = <pcl_ftr.NormalEstimation_t *>normalEstimation.me
#         cNormalEstimation.setInputCloud(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> self.thisptr_shared)
#         self.thisptr().compute (*cNormalEstimation)
#         return normalEstimation


