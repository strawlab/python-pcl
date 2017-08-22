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
        self.me2 = NULL
        self.me = NULL
        if resolution <= 0.:
            raise ValueError("Expected resolution > 0., got %r" % resolution)

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

    # base OctreePointCloud2Buf
    def define_bounding_box(self):
        """
        Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
        """
        self.me2.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me2.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def add_points_from_input_cloud(self):
        """
        Add points from input point cloud to octree.
        """
        self.me2.addPointsFromInputCloud()

    def is_voxel_occupied_at_point(self, point):
        """
        Check if voxel at given point coordinates exist.
        """
        return self.me2.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    def get_occupied_voxel_centers(self):
        """
        Get list of centers of all occupied voxels.
        """
        cdef eig.AlignedPointTVector_t points_v
        cdef int num = self.me2.getOccupiedVoxelCenters (points_v)
        return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    def delete_voxel_at_point(self, point):
        """
        Delete leaf node / voxel at given point.
        """
        self.me2.deleteVoxelAtPoint(to_point_t(point))


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
        self.me2 = NULL
        self.me = NULL
        if resolution <= 0.:
            raise ValueError("Expected resolution > 0., got %r" % resolution)

        self.me2 = <pcloct.OctreePointCloudChangeDetector_PointXYZI_t*> new pcloct.OctreePointCloudChangeDetector_PointXYZI_t(resolution)
        self.me = <pcloct.OctreePointCloud2Buf_PointXYZI_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        return newPointIdxVector

    # use Octree2BufBase class function
    # def switchBuffers (self):
    #     self.me.switchBuffers()

    # base OctreePointCloud2Buf
    def define_bounding_box(self):
        """
        Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
        """
        self.me2.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me2.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def add_points_from_input_cloud(self):
        """
        Add points from input point cloud to octree.
        """
        self.me2.addPointsFromInputCloud()

    def is_voxel_occupied_at_point(self, point):
        """
        Check if voxel at given point coordinates exist.
        """
        return self.me2.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    def get_occupied_voxel_centers(self):
        """
        Get list of centers of all occupied voxels.
        """
        cdef eig.AlignedPointTVector_PointXYZI_t points_v
        cdef int num = self.me2.getOccupiedVoxelCenters (points_v)
        return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    def delete_voxel_at_point(self, point):
        """
        Delete leaf node / voxel at given point.
        """
        self.me2.deleteVoxelAtPoint(to_point2_t(point))


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
        self.me2 = NULL
        self.me = NULL
        if resolution <= 0.:
            raise ValueError("Expected resolution > 0., got %r" % resolution)

        self.me2 = <pcloct.OctreePointCloudChangeDetector_PointXYZRGB_t*> new pcloct.OctreePointCloudChangeDetector_PointXYZRGB_t(resolution)
        self.me = <pcloct.OctreePointCloud2Buf_PointXYZRGB_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        return newPointIdxVector

    # use Octree2BufBase class function
    # def switchBuffers (self):
    #     self.me2.switchBuffers()

    # base OctreePointCloud2Buf
    def define_bounding_box(self):
        """
        Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
        """
        self.me2.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me2.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def add_points_from_input_cloud(self):
        """
        Add points from input point cloud to octree.
        """
        self.me2.addPointsFromInputCloud()

    def is_voxel_occupied_at_point(self, point):
        """
        Check if voxel at given point coordinates exist.
        """
        return self.me2.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    def get_occupied_voxel_centers(self):
        """
        Get list of centers of all occupied voxels.
        """
        cdef eig.AlignedPointTVector_PointXYZRGB_t points_v
        cdef int num = self.me2.getOccupiedVoxelCenters (points_v)
        return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    def delete_voxel_at_point(self, point):
        """
        Delete leaf node / voxel at given point.
        """
        self.me2.deleteVoxelAtPoint(to_point3_t(point))


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
        self.me2 = NULL
        self.me = NULL
        if resolution <= 0.:
            raise ValueError("Expected resolution > 0., got %r" % resolution)

        self.me2 = <pcloct.OctreePointCloudChangeDetector_PointXYZRGBA_t*> new pcloct.OctreePointCloudChangeDetector_PointXYZRGBA_t(resolution)
        self.me = <pcloct.OctreePointCloud2Buf_PointXYZRGBA_t*> self.me2

    def get_PointIndicesFromNewVoxels (self):
        cdef vector[int] newPointIdxVector
        self.me2.getPointIndicesFromNewVoxels (newPointIdxVector, 0)
        return newPointIdxVector

    # use Octree2BufBase class function
    # def switchBuffers (self):
    #     self.me.switchBuffers()

    # base OctreePointCloud2Buf
    def define_bounding_box(self):
        """
        Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
        """
        self.me2.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me2.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def add_points_from_input_cloud(self):
        """
        Add points from input point cloud to octree.
        """
        self.me2.addPointsFromInputCloud()

    def is_voxel_occupied_at_point(self, point):
        """
        Check if voxel at given point coordinates exist.
        """
        return self.me2.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    def get_occupied_voxel_centers(self):
        """
        Get list of centers of all occupied voxels.
        """
        cdef eig.AlignedPointTVector_PointXYZRGBA_t points_v
        cdef int num = self.me2.getOccupiedVoxelCenters (points_v)
        return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    def delete_voxel_at_point(self, point):
        """
        Delete leaf node / voxel at given point.
        """
        self.me2.deleteVoxelAtPoint(to_point4_t(point))


