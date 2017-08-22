# -*- coding: utf-8 -*-
cimport pcl_segmentation_172 as pclseg
cimport pcl_defs as cpp

cdef class ConditionalEuclideanClustering:
    """
    ConditionalEuclideanClustering class for Sample Consensus methods and models
    """
    cdef pclseg.ConditionalEuclideanClustering_t *me
    def __cinit__(self):
        self.me = new pclseg.ConditionalEuclideanClustering_t()

    def __dealloc__(self):
        del self.me

