# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_filters as pclfil

cimport eigen as eigen3

from boost_shared_ptr cimport shared_ptr

cdef class ConditionalRemoval:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in ConditionalRemoval(pc).
    """
    cdef pclfil.ConditionalRemoval_t *me

    def __cinit__(self, PointCloud pc not None, ConditionAnd cond):
        self.me = new pclfil.ConditionalRemoval_t(cond)
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def set_KeepOrganized(self, flag):
        self.me.setKeepOrganized(flag)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc
