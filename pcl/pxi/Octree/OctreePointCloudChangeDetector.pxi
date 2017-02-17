# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_octree as pcloct

cdef class OctreePointCloudChangeDetector(OctreePointCloud2Buf):
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
        self.me = <pcloct.OctreePointCloud2Buf_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        # for i, l in enumerate(newPointIdxVector):
        #      p = idx.getptr(self.thisptr(), <int> i)
        #      p.x, p.y, p.z = l
        # 
        # for i in pyindices:
        #     ind.indices.push_back(i)
        # 
        # cdef cpp.PointIndices ind
        # self.me.segment (ind, coeffs)
        # return [ind.indices[i] for i in range(ind.indices.size())]
        return newPointIdxVector

    # use Octree2BufBase class function
    def switchBuffers (self):
        cdef pcloct.Octree2BufBase_t* buf
        buf = <pcloct.Octree2BufBase_t*>self.me2
        buf.switchBuffers()

    def add_points_from_input_cloud(self):
        """
        Add points from input point cloud to octree.
        """
        self.me2.addPointsFromInputCloud()


cdef class OctreePointCloudChangeDetector_PointXYZI(OctreePointCloud2Buf_PointXYZI):
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
        self.me = <pcloct.OctreePointCloud2Buf_PointXYZI_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        return newPointIdxVector

    # use Octree2BufBase class function
    # def switchBuffers (self):
    #     self.me.switchBuffers()


cdef class OctreePointCloudChangeDetector_PointXYZRGB(OctreePointCloud2Buf_PointXYZRGB):
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
        self.me = <pcloct.OctreePointCloud2Buf_PointXYZRGB_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        return newPointIdxVector

    # use Octree2BufBase class function
    # def switchBuffers (self):
    #     self.me2.switchBuffers()


cdef class OctreePointCloudChangeDetector_PointXYZRGBA(OctreePointCloud2Buf_PointXYZRGBA):
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
        self.me = <pcloct.OctreePointCloud2Buf_PointXYZRGBA_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        return newPointIdxVector

    # use Octree2BufBase class function
    # def switchBuffers (self):
    #     self.me.switchBuffers()


