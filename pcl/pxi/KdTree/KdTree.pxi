# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_kdtree as pclkdt

cdef class KdTree:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTree(pc).
    """
    cdef cpp.KdTree_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclkdt.KdTree_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me


cdef class KdTree_PointXYZI:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTree(pc).
    """
    cdef cpp.KdTree_PointXYZI_t *me

    def __cinit__(self, PointCloud_PointXYZI pc not None):
        self.me = new cpp.KdTree_PointXYZI_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

cdef class KdTree_PointXYZRGB:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTree(pc).
    """
    cdef cpp.KdTree_PointXYZRGB_t *me

    def __cinit__(self, PointCloud_PointXYZRGB pc not None):
        self.me = new cpp.KdTree_PointXYZRGB_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

cdef class KdTree_PointXYZRGBA:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTree(pc).
    """
    cdef cpp.KdTree_PointXYZRGBA_t *me

    def __cinit__(self, PointCloud_PointXYZRGBA pc not None):
        self.me = new cpp.KdTree_PointXYZRGBA_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me


