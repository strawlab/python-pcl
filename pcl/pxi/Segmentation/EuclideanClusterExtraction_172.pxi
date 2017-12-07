# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_segmentation_172 as pclseg
cimport pcl_defs as cpp
from libcpp.vector cimport vector

cdef class EuclideanClusterExtraction:
    """
    Segmentation class for EuclideanClusterExtraction
    """
    cdef pclseg.EuclideanClusterExtraction_t *me
    def __cinit__(self):
        self.me = new pclseg.EuclideanClusterExtraction_t()
    def __dealloc__(self):
        del self.me
    
    def set_ClusterTolerance(self, double b):
        self.me.setClusterTolerance(b)
    
    def set_MinClusterSize(self, int min):
        self.me.setMinClusterSize(min)
    
    def set_MaxClusterSize(self, int max):
        self.me.setMaxClusterSize(max)
    
    def set_SearchMethod(self, _pcl.KdTree kdtree):
        self.me.setSearchMethod(kdtree.thisptr_shared)
    
    # def set_Search_Method(self, _pcl.KdTreeFLANN kdtree):
    #    # self.me.setSearchMethod(kdtree.thisptr())
    #    self.me.setSearchMethod(kdtree.thisptr_shared)
    
    # def set_
    #   self.me.setInputCloud (cloud_filtered)
    def Extract(self):
        cdef vector[cpp.PointIndices] inds
        self.me.extract (inds)
        # NG(not use Python)
        # return inds
        # return 2-dimension Array?
        # return [inds[0].indices[i] for i in range(inds[0].indices.size())]
        cdef vector[vector[int]] result
        cdef vector[int] dim
        
        # for j, ind in enumerate(inds):
        # for ind in inds.iterator:
        #    for i in range(ind):
        #        dim.push_back(ind.indices[i])
        #    result.push_back(dim)
        # return result
        
        # use Iterator
        # http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
        # http://stackoverflow.com/questions/29200592/how-to-iterate-throught-c-sets-in-cython
        # itertools?
        # http://qiita.com/tomotaka_ito/items/35f3eb108f587022fa09
        # https://docs.python.org/2/library/itertools.html
        # set numpy?
        # http://kesin.hatenablog.com/entry/20120314/1331689014
        cdef vector[cpp.PointIndices].iterator it = inds.begin()
        while it != inds.end():
            idx = deref(it)
            # for i in range(it.indices.size()):
            for i in range(idx.indices.size()):
                dim.push_back(idx.indices[i])
            result.push_back(dim)
            inc(it)
            dim.clear()
        
        return result

