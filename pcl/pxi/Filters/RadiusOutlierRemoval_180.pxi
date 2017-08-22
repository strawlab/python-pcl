# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_filters_180 as pclfil
cimport pcl_segmentation_180 as pclseg

cdef class RadiusOutlierRemoval:
    """
    RadiusOutlierRemoval class for ...
    """
    cdef pclfil.RadiusOutlierRemoval_t *me
    def __cinit__(self):
        self.me = new pclfil.RadiusOutlierRemoval_t()
    def __dealloc__(self):
        del self.me

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        # Cython 0.25.2 NG(0.24.1 OK)
        # self.me.filter(pc.thisptr()[0])
        # self.me.filter(<cpp.PointCloud[cpp.PointXYZ]> pc.thisptr()[0])
        # Cython 0.24.1 NG(0.25.2 NG)
        # self.me.filter(<vector[int]> pc)
        # pcl 1.7.2
        self.me.filter(<vector[int]&> pc)
        return pc

    def set_radius_search(self, double radius):
        self.me.setRadiusSearch(radius)
    def get_radius_search(self):
        return self.me.getRadiusSearch()
    def set_MinNeighborsInRadius(self, int min_pts):
        self.me.setMinNeighborsInRadius (min_pts)
    def get_MinNeighborsInRadius(self):
        return self.me.getMinNeighborsInRadius ()

