# -*- coding: utf-8 -*-
cimport pcl_segmentation as pclseg
cimport pcl_defs as cpp

cdef class EuclideanClusterExtraction:
    """
    Segmentation class for EuclideanClusterExtraction
    """
    cdef pclseg.EuclideanClusterExtraction_t *me
    def __cinit__(self):
        self.me = new pclseg.EuclideanClusterExtraction_t()
    def __dealloc__(self):
        del self.me
    
    # def segment(self):
    #     cdef cpp.PointIndices ind
    #     cdef cpp.ModelCoefficients coeffs
    #     
    #     self.me.segment (ind, coeffs)
    #     return [ind.indices[i] for i in range(ind.indices.size())], \
    #            [coeffs.values[i] for i in range(coeffs.values.size())]
    
    def set_ClusterTolerance(self, double b):
        self.me.setClusterTolerance(b)
    
    def set_MinClusterSize(self, int min):
        self.me.setMinClusterSize(min)
    
    def setMaxClusterSize(self, int max):
        self.me.setMaxClusterSize(max)
    
    def set_SearchMethod(self, KdTree kdtree):
        self.me.setSearchMethod(kdtree.thisptr_shared)
    
    # def set_Search_Method(self, KdTreeFLANN kdtree):
    #    # self.me.setSearchMethod(kdtree.thisptr())
    #    self.me.setSearchMethod(kdtree.thisptr_shared)
    
    # def set_
    #   self.me.setInputCloud (cloud_filtered)
    
    # def Extract(self):
    #     self.me.extract (cluster_indices)

