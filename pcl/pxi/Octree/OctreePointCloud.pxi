# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_octree as pcl_oct

cimport eigen as eig

cdef class OctreePointCloud:
    """
    Octree pointcloud
    """
    cdef pcl_oct.OctreePointCloud_t *me

    # def __cinit__(self, double resolution):
    #     self.me = NULL
    #     if resolution <= 0.:
    #         raise ValueError("Expected resolution > 0., got %r" % resolution)

    # NG(BUild Error)
    # def __init__(self, double resolution):
    #     """
    #     Constructs octree pointcloud with given resolution at lowest octree level
    #     """ 
    #     cdef double param = 0
    #     self.me = new pcl_oct.OctreePointCloud_t(0)
    #     # self.me = new pcl_oct.OctreePointCloud_t(resolution)
    #     # self.me = new pcl_oct.OctreePointCloud_t()

    # def __dealloc__(self):
    #     del self.me
    #     self.me = NULL      # just to be sure

    def set_input_cloud(self, PointCloud pc):
        """
        Provide a pointer to the input data set.
        """
        self.me.setInputCloud(pc.thisptr_shared)

    # def define_bounding_box(self):
    #     """
    #     Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
    #     """
    #     self.me.defineBoundingBox()
        
    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    # def add_points_from_input_cloud(self):
    #     """
    #     Add points from input point cloud to octree.
    #     """
    #     self.me.addPointsFromInputCloud()

    def delete_tree(self):
        """
        Delete the octree structure and its leaf nodes.
        """
        self.me.deleteTree()

    # def is_voxel_occupied_at_point(self, point):
    #     """
    #     Check if voxel at given point coordinates exist.
    #     """
    #     return self.me.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    # def get_occupied_voxel_centers(self):
    #     """
    #     Get list of centers of all occupied voxels.
    #     """
    #     cdef eig.AlignedPointTVector_t points_v
    #     cdef int num = self.me.getOccupiedVoxelCenters (points_v)
    #     # cdef int num = self.me.getOccupiedVoxelCenters (<eig.AlignedPointTVector_t> points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters(self.me, points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters(deref(self.me), points_v)
    #     return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    # def delete_voxel_at_point(self, point):
    #     """
    #     Delete leaf node / voxel at given point.
    #     """
    #     self.me.deleteVoxelAtPoint(to_point_t(point))
    #     # mpcl_deleteVoxelAtPoint(self.me, to_point_t(point))
    #     # mpcl_deleteVoxelAtPoint(deref(self.me), to_point_t(point))


cdef class OctreePointCloud_PointXYZI:
    """
    Octree pointcloud
    """
    cdef pcl_oct.OctreePointCloud_PointXYZI_t *me

    # def __cinit__(self, double resolution):
    #     self.me = NULL
    #     if resolution <= 0.:
    #         raise ValueError("Expected resolution > 0., got %r" % resolution)

    # NG(BUild Error)
    # def __init__(self, double resolution):
    #     """
    #     Constructs octree pointcloud with given resolution at lowest octree level
    #     """ 
    #     cdef double param = 0
    #     # self.me = new pcl_oct.OctreePointCloud_PointXYZI_t(param)
    #     # self.me = new pcl_oct.OctreePointCloud_PointXYZI_t(resolution)
    #     # self.me = new pcl_oct.OctreePointCloud_PointXYZI_t()

    # def __dealloc__(self):
    #     del self.me
    #     self.me = NULL      # just to be sure

    def set_input_cloud(self, PointCloud_PointXYZI pc):
        """
        Provide a pointer to the input data set.
        """
        self.me.setInputCloud(pc.thisptr_shared)

    # def define_bounding_box(self):
    #     """
    #     Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
    #     """
    #     self.me.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    # def add_points_from_input_cloud(self):
    #     """
    #     Add points from input point cloud to octree.
    #     """
    #     self.me.addPointsFromInputCloud()

    def delete_tree(self):
        """
        Delete the octree structure and its leaf nodes.
        """
        self.me.deleteTree()

    # def is_voxel_occupied_at_point(self, point):
    #     """
    #     Check if voxel at given point coordinates exist.
    #     """
    #     return self.me.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    # def get_occupied_voxel_centers(self):
    #     """
    #     Get list of centers of all occupied voxels.
    #     """
    #     cdef eig.AlignedPointTVector_PointXYZI_t points_v
    #     cdef int num = self.me.getOccupiedVoxelCenters (points_v)
    #     # cdef int num = self.me.getOccupiedVoxelCenters (<eig.AlignedPointTVector_PointXYZI_t> points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters(self.me, points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters_PointXYZI(deref(self.me), points_v)
    #     return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    # def delete_voxel_at_point(self, point):
    #     """
    #     Delete leaf node / voxel at given point.
    #     """
    #     # NG (use minipcl?)
    #     self.me.deleteVoxelAtPoint(to_point2_t(point))
    #     # mpcl_deleteVoxelAtPoint(self.me, to_point2_t(point))
    #     # mpcl_deleteVoxelAtPoint_PointXYZI(deref(self.me), to_point2_t(point))


cdef class OctreePointCloud_PointXYZRGB:
    """
    Octree pointcloud
    """
    cdef pcl_oct.OctreePointCloud_PointXYZRGB_t *me

    # def __cinit__(self, double resolution):
    #     self.me = NULL
    #     if resolution <= 0.:
    #         raise ValueError("Expected resolution > 0., got %r" % resolution)

    # NG(BUild Error)
    # def __init__(self, double resolution):
    #     """
    #     Constructs octree pointcloud with given resolution at lowest octree level
    #     """ 
    #     cdef double param = 0
    #     self.me = new pcl_oct.OctreePointCloud_PointXYZRGB_t(param)
    #     # self.me = new pcl_oct.OctreePointCloud_PointXYZRGB_t(resolution)
    #     # self.me = new pcl_oct.OctreePointCloud_PointXYZRGB_t()

    # def __dealloc__(self):
    #     del self.me
    #     self.me = NULL      # just to be sure

    def set_input_cloud(self, PointCloud_PointXYZRGB pc not None):
        """
        Provide a pointer to the input data set.
        """
        self.me.setInputCloud(pc.thisptr_shared)

    # def define_bounding_box(self):
    #     """
    #     Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
    #     """
    #     self.me.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    # def add_points_from_input_cloud(self):
    #     """
    #     Add points from input point cloud to octree.
    #     """
    #     self.me.addPointsFromInputCloud()

    def delete_tree(self):
        """
        Delete the octree structure and its leaf nodes.
        """
        self.me.deleteTree()

    # def is_voxel_occupied_at_point(self, point):
    #     """
    #     Check if voxel at given point coordinates exist.
    #     """
    #     return self.me.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    # def get_occupied_voxel_centers(self):
    #     """
    #     Get list of centers of all occupied voxels.
    #     """
    #     cdef eig.AlignedPointTVector_PointXYZRGB_t points_v
    #     cdef int num = self.me.getOccupiedVoxelCenters (points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters(self.me, points_v)
    #     return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    # def delete_voxel_at_point(self, point):
    #     """
    #     Delete leaf node / voxel at given point.
    #     """
    #     # NG (minipcl?)
    #     self.me.deleteVoxelAtPoint(to_point3_t(point))


cdef class OctreePointCloud_PointXYZRGBA:
    """
    Octree pointcloud
    """
    cdef pcl_oct.OctreePointCloud_PointXYZRGBA_t *me

    # def __cinit__(self, double resolution):
    #     self.me = NULL
    #     if resolution <= 0.:
    #         raise ValueError("Expected resolution > 0., got %r" % resolution)

    # NG(BUild Error)
    # def __init__(self, double resolution):
    #     """
    #     Constructs octree pointcloud with given resolution at lowest octree level
    #     """ 
    #     cdef double param = 0
    #     self.me = new pcl_oct.OctreePointCloud_PointXYZRGBA_t(param)
    #     # self.me = new pcl_oct.OctreePointCloud_PointXYZRGBA_t(resolution)
    #     # self.me = new pcl_oct.OctreePointCloud_PointXYZRGBA_t()

    # def __dealloc__(self):
    #     del self.me
    #     self.me = NULL      # just to be sure

    def set_input_cloud(self, PointCloud_PointXYZRGBA pc):
        """
        Provide a pointer to the input data set.
        """
        self.me.setInputCloud(pc.thisptr_shared)

    # def define_bounding_box(self):
    #     """
    #     Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
    #     """
    #     self.me.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    # use NG
    # def add_points_from_input_cloud(self):
    #     """
    #     Add points from input point cloud to octree.
    #     """
    #     self.me.addPointsFromInputCloud()

    def delete_tree(self):
        """
        Delete the octree structure and its leaf nodes.
        """
        self.me.deleteTree()

    # def is_voxel_occupied_at_point(self, point):
    #     """
    #     Check if voxel at given point coordinates exist.
    #     """
    #     return self.me.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    # def get_occupied_voxel_centers(self):
    #     """
    #     Get list of centers of all occupied voxels.
    #     """
    #     cdef eig.AlignedPointTVector_PointXYZRGBA_t points_v
    #     cdef int num = self.me.getOccupiedVoxelCenters (points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters(self.me, points_v)
    #     return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    # def delete_voxel_at_point(self, point):
    #     """
    #     Delete leaf node / voxel at given point.
    #     """
    #     # NG (minipcl?)
    #     self.me.deleteVoxelAtPoint(to_point4_t(point))


