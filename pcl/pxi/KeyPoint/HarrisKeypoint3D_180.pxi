# -*- coding: utf-8 -*-
from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp
cimport pcl_keypoints_180 as keypt

# boost
from boost_shared_ptr cimport shared_ptr


###############################################################################
# Types
###############################################################################

### base class ###

cdef class HarrisKeypoint3D:
    """
    HarrisKeypoint3D class for 
    """
    cdef keypt.HarrisKeypoint3D_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new keypt.HarrisKeypoint3D_t()
        self.me.setInputCloud(pc.thisptr_shared)
        # pass

    def __dealloc__(self):
        del self.me

    def set_NonMaxSupression(self, bool param):
        self.me.setNonMaxSupression (param)

    def set_Radius(self, float param):
        self.me.setRadius (param)

    def set_RadiusSearch(self, double param):
        self.me.setRadiusSearch (param)

    def compute(self):
        keypoints = PointCloud_PointXYZI()
        sp_assign(keypoints.thisptr_shared, new cpp.PointCloud[cpp.PointXYZI]())
        cdef cpp.PointCloud_PointXYZI_t *ckeypoints = <cpp.PointCloud_PointXYZI_t*>keypoints.thisptr()
        self.me.compute (<cpp.PointCloud[cpp.PointXYZI]&> deref(ckeypoints))
        return keypoints

