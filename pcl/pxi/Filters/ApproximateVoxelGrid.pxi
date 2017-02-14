# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_filters as pclfil

cdef class ApproximateVoxelGrid:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef pclfil.ApproximateVoxelGrid_t *me
    def __cinit__(self):
        self.me = new pclfil.ApproximateVoxelGrid_t()
    def __dealloc__(self):
        del self.me

    def set_leaf_size (self, float x, float y, float z):
        """
        Set the voxel grid leaf size.
        """
        self.me.setLeafSize(x,y,z)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc

cdef class ApproximateVoxelGrid_PointXYZI:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef pclfil.ApproximateVoxelGrid_PointXYZI_t *me
    def __cinit__(self):
        self.me = new pclfil.ApproximateVoxelGrid_PointXYZI_t()
    def __dealloc__(self):
        del self.me

    def set_leaf_size (self, float x, float y, float z):
        """
        Set the voxel grid leaf size.
        """
        self.me.setLeafSize(x,y,z)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZI pc = PointCloud_PointXYZI()
        self.me.filter(pc.thisptr()[0])
        return pc

cdef class ApproximateVoxelGrid_PointXYZRGB:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef pclfil.ApproximateVoxelGrid_PointXYZRGB_t *me
    def __cinit__(self):
        self.me = new pclfil.ApproximateVoxelGrid_PointXYZRGB_t()
    def __dealloc__(self):
        del self.me

    def set_leaf_size (self, float x, float y, float z):
        """
        Set the voxel grid leaf size.
        """
        self.me.setLeafSize(x,y,z)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZRGB pc = PointCloud_PointXYZRGB()
        self.me.filter(pc.thisptr()[0])
        return pc

cdef class ApproximateVoxelGrid_PointXYZRGBA:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef pclfil.ApproximateVoxelGrid_PointXYZRGBA_t *me
    def __cinit__(self):
        self.me = new pclfil.ApproximateVoxelGrid_PointXYZRGBA_t()
    def __dealloc__(self):
        del self.me

    def set_leaf_size (self, float x, float y, float z):
        """
        Set the voxel grid leaf size.
        """
        self.me.setLeafSize(x,y,z)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZRGBA pc = PointCloud_PointXYZRGBA()
        self.me.filter(pc.thisptr()[0])
        return pc
