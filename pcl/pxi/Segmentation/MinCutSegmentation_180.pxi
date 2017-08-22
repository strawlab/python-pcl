# -*- coding: utf-8 -*-
cimport pcl_segmentation_172 as pclseg
cimport pcl_defs as cpp

cdef class MinCutSegmentation:
    """
    MinCutSegmentation class for Sample Consensus methods and models
    """
    cdef pclseg.SACSegmentation_t *me
    def __cinit__(self):
        self.me = new pclseg.SACSegmentation_t()
    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())], \
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)
    def set_model_type(self, pclseg.SacModel m):
        self.me.setModelType(m)
    def set_method_type(self, int m):
        self.me.setMethodType (m)
    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)

cdef class Segmentation_PointXYZI:
    """
    Segmentation class for Sample Consensus methods and models
    """
    cdef pclseg.SACSegmentation_PointXYZI_t *me
    def __cinit__(self):
        self.me = new pclseg.SACSegmentation_PointXYZI_t()
    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())], \
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)
    def set_model_type(self, pclseg.SacModel m):
        self.me.setModelType(m)
    def set_method_type(self, int m):
        self.me.setMethodType (m)
    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)


cdef class Segmentation_PointXYZRGB:
    """
    Segmentation class for Sample Consensus methods and models
    """
    cdef pclseg.SACSegmentation_PointXYZRGB_t *me
    def __cinit__(self):
        self.me = new pclseg.SACSegmentation_PointXYZRGB_t()
    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())], \
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)
    def set_model_type(self, pclseg.SacModel m):
        self.me.setModelType(m)
    def set_method_type(self, int m):
        self.me.setMethodType (m)
    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)


cdef class Segmentation_PointXYZRGBA:
    """
    Segmentation class for Sample Consensus methods and models
    """
    cdef pclseg.SACSegmentation_PointXYZRGBA_t *me
    def __cinit__(self):
        self.me = new pclseg.SACSegmentation_PointXYZRGBA_t()
    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())], \
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)
    def set_model_type(self, pclseg.SacModel m):
        self.me.setModelType(m)
    def set_method_type(self, int m):
        self.me.setMethodType (m)
    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)

