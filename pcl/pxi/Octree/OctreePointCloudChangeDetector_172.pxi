# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_octree_172 as pcloct

cdef class OctreePointCloudChangeDetector(OctreePointCloud):
    """
    Octree pointcloud ChangeDetector
    """
    # override OctreePointCloud(OctreePointCloud_t)
    # cdef pcloct.OctreePointCloudChangeDetector_t *me
    cdef pcloct.OctreePointCloudChangeDetector_t *me2

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me2 = <pcloct.OctreePointCloudChangeDetector_t*> new pcloct.OctreePointCloudChangeDetector_t(resolution)
        self.me = <pcloct.OctreePointCloud_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        return newPointIdxVector

    # use Octree2BufBase class function
    # def switchBuffers (self):
    #     self.me.switchBuffers()


cdef class OctreePointCloudChangeDetector_PointXYZI(OctreePointCloud_PointXYZI):
    """
    Octree pointcloud ChangeDetector
    """
    # override OctreePointCloud_PointXYZI
    # cdef pcloct.OctreePointCloudChangeDetector_PointXYZI_t *me
    cdef pcloct.OctreePointCloudChangeDetector_PointXYZI_t *me2

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me2 = <pcloct.OctreePointCloudChangeDetector_PointXYZI_t*> new pcloct.OctreePointCloudChangeDetector_PointXYZI_t(resolution)
        self.me = <pcloct.OctreePointCloud_PointXYZI_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        return newPointIdxVector

    # use Octree2BufBase class function
    # def switchBuffers (self):
    #     self.me.switchBuffers()


cdef class OctreePointCloudChangeDetector_PointXYZRGB(OctreePointCloud_PointXYZRGB):
    """
    Octree pointcloud ChangeDetector
    """
    # override OctreePointCloud_PointXYZRGB
    # cdef pcloct.OctreePointCloudChangeDetector_PointXYZRGB_t *me
    cdef pcloct.OctreePointCloudChangeDetector_PointXYZRGB_t *me2

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me2 = <pcloct.OctreePointCloudChangeDetector_PointXYZRGB_t*> new pcloct.OctreePointCloudChangeDetector_PointXYZRGB_t(resolution)
        self.me = <pcloct.OctreePointCloud_PointXYZRGB_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        return newPointIdxVector

    # use Octree2BufBase class function
    # def switchBuffers (self):
    #     self.me.switchBuffers()


cdef class OctreePointCloudChangeDetector_PointXYZRGBA(OctreePointCloud_PointXYZRGBA):
    """
    Octree pointcloud ChangeDetector
    """
    # override OctreePointCloud_PointXYZRGBA
    # cdef pcloct.OctreePointCloudChangeDetector_PointXYZRGBA_t *me
    cdef pcloct.OctreePointCloudChangeDetector_PointXYZRGBA_t *me2

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me2 = <pcloct.OctreePointCloudChangeDetector_PointXYZRGBA_t*> new pcloct.OctreePointCloudChangeDetector_PointXYZRGBA_t(resolution)
        self.me = <pcloct.OctreePointCloud_PointXYZRGBA_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        return newPointIdxVector

    # use Octree2BufBase class function
    # def switchBuffers (self):
    #     self.me.switchBuffers()


