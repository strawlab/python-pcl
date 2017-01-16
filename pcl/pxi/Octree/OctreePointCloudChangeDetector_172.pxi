# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_octree as pcloct
# cimport pcl_octree_172 as pcloct

cdef class OctreePointCloudChangeDetector(OctreePointCloud):
    """
    Octree pointcloud ChangeDetector
    """

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = <pcloct.OctreePointCloud_t*> new pcloct.OctreePointCloudChangeDetector_t(resolution)

cdef class OctreePointCloudChangeDetector_PointXYZI(OctreePointCloud_PointXYZI):
    """
    Octree pointcloud ChangeDetector
    """

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = <pcloct.OctreePointCloud_PointXYZI_t*> new pcloct.OctreePointCloudChangeDetector_PointXYZI_t(resolution)

cdef class OctreePointCloudChangeDetector_PointXYZRGB(OctreePointCloud_PointXYZRGB):
    """
    Octree pointcloud ChangeDetector
    """

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = <pcloct.OctreePointCloud_PointXYZRGB_t*> new pcloct.OctreePointCloudChangeDetector_PointXYZRGB_t(resolution)

cdef class OctreePointCloudChangeDetector_PointXYZRGBA(OctreePointCloud_PointXYZRGBA):
    """
    Octree pointcloud ChangeDetector
    """

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = <pcloct.OctreePointCloud_PointXYZRGBA_t*> new pcloct.OctreePointCloudChangeDetector_PointXYZRGBA_t(resolution)

