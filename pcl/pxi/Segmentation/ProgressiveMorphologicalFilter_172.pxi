# -*- coding: utf-8 -*-
cimport pcl_segmentation_172 as pclseg
cimport pcl_defs as cpp

cdef class ProgressiveMorphologicalFilter:
    """
    ProgressiveMorphologicalFilter class for Sample Consensus methods and models
    """
    cdef pclseg.ProgressiveMorphologicalFilter_t *me
    def __cinit__(self):
        self.me = new pclseg.ProgressiveMorphologicalFilter_t()

    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())], \
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_InputCloud(self, PointCloud cloud):
        self.me.setInputCloud(b)

    def set_MaxWindowSize(self, size):
        self.me.setMaxWindowSize(m)

    def set_Slope(self, float param):
        self.me.setSlope (param)

    def set_InitialDistance(self, float d):
        self.me.setInitialDistance (d)

    def set_MaxDistance(self, float d):
        self.me.setMaxDistance (d)

    def extract(self):
        self.me.extract ()




