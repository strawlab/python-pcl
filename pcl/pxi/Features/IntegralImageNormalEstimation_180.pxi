# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features_180 as pcl_ftr

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

    def compute(self, PointCloud pc not None):
        # cdef PointCloud_PointNormal normal = PointCloud_PointNormal()
        # normal = PointCloud_PointNormal()
        normal = PointCloud_Normal()
        # NG : No Python object
        # normal = PointCloud_Normal(pc)
        cdef cpp.PointCloud_Normal_t *cPointCloudNormal = <cpp.PointCloud_Normal_t*>normal.thisptr()
        print ('3')
        # print (str(self.thisptr().size))
        
        # compute function based Features class
        # NG 
        # self.thisptr().compute (cPointCloudNormal.makeShared())
        # self.thisptr().compute (cPointCloudNormal.makeShared().get())
        # from cython cimport address
        # self.thisptr().compute (cython.address(cPointCloudNormal.makeShared().get()))
        # self.thisptr().compute (<cpp.PointCloud[Normal]> deref(cPointCloudNormal.makeShared().get()))
        # NG : (Exception)
        # self.thisptr().compute (deref(cPointCloudNormal.makeShared().get()))
        self.thisptr().compute (deref(cPointCloudNormal))
        print ('4')
        return normal

    def compute2(self, PointCloud pc not None):
        normal = PointCloud_Normal()
        cdef cpp.PointCloud_Normal_t *cPointCloudNormal = <cpp.PointCloud_Normal_t*>normal.thisptr()
        print ('3')
        # OK
        cdef cpp.PointCloud_Normal_t normals
        mpcl_features_NormalEstimationMethod_compute(<pcl_ftr.IntegralImageNormalEstimation_t> deref(self.thisptr()), normals)
        print ('3a')
        # Copy?
        cPointCloudNormal = normals.makeShared().get()
        print ('3b')
        
        # NG : Normal Pointer Nothing?
        # mpcl_features_NormalEstimationMethod_compute(<pcl_ftr.IntegralImageNormalEstimation_t> deref(self.thisptr()), deref(cPointCloudNormal.makeShared().get()))
        # mpcl_features_NormalEstimationMethod_compute(<pcl_ftr.IntegralImageNormalEstimation_t> deref(self.thisptr()), cPointCloudNormal.makeShared().get())
        # NG : Normal Pointer Nothing?
        # mpcl_features_NormalEstimationMethod_compute(<pcl_ftr.IntegralImageNormalEstimation_t> deref(self.thisptr()), deref(cPointCloudNormal))
        print ('4')
        return normal

