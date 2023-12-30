# -*- coding: utf-8 -*-
cimport pcl_segmentation_172 as pcl_seg
cimport pcl_defs as cpp

cdef class ConditionalEuclideanClustering:
    """
    ConditionalEuclideanClustering class for Sample Consensus methods and models
    """
    cdef pcl_seg.ConditionalEuclideanClustering_t *me
    def __cinit__(self):
        self.me = new pcl_seg.ConditionalEuclideanClustering_t()

    def __dealloc__(self):
        del self.me

