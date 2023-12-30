# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_segmentation_172 as pclseg
cimport pcl_defs as cpp
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cdef class RegionGrowing:
    """
    Segmentation class for RegionGrowing
    """
    cdef pclseg.RegionGrowing_t *me
    def __cinit__(self):
        self.me = new pclseg.RegionGrowing_t()
    def __dealloc__(self):
        del self.me
    
    def get_MinClusterSize(self):
        return self.me.getMinClusterSize()

    def set_MinClusterSize(self, int min):
        self.me.setMinClusterSize(min)
    
    def get_MaxClusterSize(self):
        return self.me.getMaxClusterSize()
    
    def set_MaxClusterSize(self, int max):
        self.me.setMaxClusterSize(max)

    def get_SmoothModeFlag(self):
        return self.me.getSmoothModeFlag()
    
    def set_SmoothModeFlag(self, bool value):
        self.me.setSmoothModeFlag(value)
    
    def get_CurvatureTestFlag(self):
        return self.me.getCurvatureTestFlag()
    
    def set_CurvatureTestFlag(self, bool value):
        self.me.setCurvatureTestFlag(value)
    
    def get_ResidualTestFlag(self):
        return self.me.getResidualTestFlag()
    
    def set_ResidualTestFlag(self, bool value):
        self.me.setResidualTestFlag(value)
    
    def get_SmoothnessThreshold(self):
        return self.me.getSmoothnessThreshold()
    
    def set_SmoothnessThreshold(self, float theta):
        self.me.setSmoothnessThreshold(theta)
    
    def get_ResidualThreshold(self):
        return self.me.getResidualThreshold()
    
    def set_ResidualThreshold(self, float residual):
        self.me.setResidualThreshold(residual)
    
    def get_CurvatureThreshold(self):
        return self.me.getCurvatureThreshold()
    
    def set_CurvatureThreshold(self, float curvature):
        self.me.setCurvatureThreshold(curvature)
    
    def get_NumberOfNeighbours(self):
        return self.me.getNumberOfNeighbours()
    
    def set_NumberOfNeighbours(self, int neighbour_number):
        self.me.setNumberOfNeighbours(neighbour_number)
    
    # def get_SearchMethod(self):
    #    pass

    def set_SearchMethod(self, _pcl.KdTree kdtree):
        self.me.setSearchMethod(kdtree.thisptr_shared)
    
    # def get_InputNormals(self):
    #    pass

    def set_InputNormals(self, _pcl.PointCloud_Normal normals):
        self.me.setInputNormals(normals.thisptr_shared)
    
    def Extract(self):
        cdef vector[cpp.PointIndices] inds
        self.me.extract(inds)

        cdef vector[vector[int]] result
        cdef vector[int] dim

        cdef vector[cpp.PointIndices].iterator it = inds.begin()
        while it != inds.end():
            idx = deref(it)
            for i in range(idx.indices.size()):
                dim.push_back(idx.indices[i])
            result.push_back(dim)
            inc(it)
            dim.clear()
        
        return result
    
    def get_SegmentFromPoint(self, int index):
        cdef cpp.PointIndices cluster
        self.me.getSegmentFromPoint(index, cluster)
        cdef vector[int] result
        for i in range(cluster.indices.size()):
            result.push_back(cluster.indices[i])
        return result

    # def get_ColoredCloud(self):
    #     pass

    # def get_ColoredCloudRGBA(self):
    #    pass

    
    