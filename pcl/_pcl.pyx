#cython: embedsignature=True

from collections import Sequence
import numbers
import numpy as np

cimport numpy as cnp

cimport pcl_defs as cpp
cimport pcl_features as features
# cimport pcl_segmentation as segmentation

cimport cython
# from cython.operator import dereference as deref
from cython.operator cimport dereference as deref, preincrement as inc

from cpython cimport Py_buffer

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

from boost_shared_ptr cimport sp_assign

cdef extern from "minipcl.h":
    void mpcl_compute_normals(cpp.PointCloud_t, int ksearch,
                              double searchRadius,
                              cpp.PointNormalCloud_t) except +
    void mpcl_sacnormal_set_axis(cpp.SACSegmentationNormal_t,
                                 double ax, double ay, double az) except +
    void mpcl_extract(cpp.PointCloudPtr_t, cpp.PointCloud_t *,
                      cpp.PointIndices_t *, bool) except +

SAC_RANSAC = cpp.SAC_RANSAC
SAC_LMEDS = cpp.SAC_LMEDS
SAC_MSAC = cpp.SAC_MSAC
SAC_RRANSAC = cpp.SAC_RRANSAC
SAC_RMSAC = cpp.SAC_RMSAC
SAC_MLESAC = cpp.SAC_MLESAC
SAC_PROSAC = cpp.SAC_PROSAC

SACMODEL_PLANE = cpp.SACMODEL_PLANE
SACMODEL_LINE = cpp.SACMODEL_LINE
SACMODEL_CIRCLE2D = cpp.SACMODEL_CIRCLE2D
SACMODEL_CIRCLE3D = cpp.SACMODEL_CIRCLE3D
SACMODEL_SPHERE = cpp.SACMODEL_SPHERE
SACMODEL_CYLINDER = cpp.SACMODEL_CYLINDER
SACMODEL_CONE = cpp.SACMODEL_CONE
SACMODEL_TORUS = cpp.SACMODEL_TORUS
SACMODEL_PARALLEL_LINE = cpp.SACMODEL_PARALLEL_LINE
SACMODEL_PERPENDICULAR_PLANE = cpp.SACMODEL_PERPENDICULAR_PLANE
SACMODEL_PARALLEL_LINES = cpp.SACMODEL_PARALLEL_LINES
SACMODEL_NORMAL_PLANE = cpp.SACMODEL_NORMAL_PLANE 
SACMODEL_NORMAL_SPHERE = cpp.SACMODEL_NORMAL_SPHERE
SACMODEL_REGISTRATION = cpp.SACMODEL_REGISTRATION
SACMODEL_PARALLEL_PLANE = cpp.SACMODEL_PARALLEL_PLANE
SACMODEL_NORMAL_PARALLEL_PLANE = cpp.SACMODEL_NORMAL_PARALLEL_PLANE
SACMODEL_STICK = cpp.SACMODEL_STICK


cnp.import_array()

include "PointCloudWrapper_PointXYZ.pxi"
# include "PointCloudWrapper_PointXYZRGBA.pxi"

# Empirically determine strides, for buffer support.
# XXX Is there a more elegant way to get these?
cdef Py_ssize_t _strides[2]
cdef PointCloud _pc_tmp = PointCloud(np.array([[1, 2, 3],
                                               [4, 5, 6]], dtype=np.float32))
cdef cpp.PointCloud[cpp.PointXYZ] *p = _pc_tmp.thisptr()
_strides[0] = (  <Py_ssize_t><void *>cpp.getptr(p, 1)
               - <Py_ssize_t><void *>cpp.getptr(p, 0))
_strides[1] = (  <Py_ssize_t><void *>&(cpp.getptr(p, 0).y)
               - <Py_ssize_t><void *>&(cpp.getptr(p, 0).x))
_pc_tmp = None



cdef cpp.PointXYZ to_point_t(point):
    cdef cpp.PointXYZ p
    p.x = point[0]
    p.y = point[1]
    p.z = point[2]
    return p

cdef class OctreePointCloud:
    """
    Octree pointcloud
    """
    cdef cpp.OctreePointCloud_t *me

    def __cinit__(self, double resolution):
        self.me = NULL
        if resolution <= 0.:
            raise ValueError("Expected resolution > 0., got %r" % resolution)

    def __init__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = new cpp.OctreePointCloud_t(resolution)

    def __dealloc__(self):
        del self.me
        self.me = NULL      # just to be sure

    def set_input_cloud(self, PointCloud pc):
        """
        Provide a pointer to the input data set.
        """
        self.me.setInputCloud(pc.thisptr_shared)

    def define_bounding_box(self):
        """
        Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
        """
        self.me.defineBoundingBox()
        
    def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
        """
        Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
        """
        self.me.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def add_points_from_input_cloud(self):
        """
        Add points from input point cloud to octree.
        """
        self.me.addPointsFromInputCloud()

    def delete_tree(self):
        """
        Delete the octree structure and its leaf nodes.
        """
        self.me.deleteTree()

    def is_voxel_occupied_at_point(self, point):
        """
        Check if voxel at given point coordinates exist.
        """
        return self.me.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    def get_occupied_voxel_centers(self):
        """
        Get list of centers of all occupied voxels.
        """
        cdef cpp.AlignedPointTVector_t points_v
        cdef int num = self.me.getOccupiedVoxelCenters (points_v)
        return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    def delete_voxel_at_point(self, point):
        """
        Delete leaf node / voxel at given point.
        """
        self.me.deleteVoxelAtPoint(to_point_t(point))

cdef class OctreePointCloudSearch(OctreePointCloud):
    """
    Octree pointcloud search
    """
    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = <cpp.OctreePointCloud_t*> new cpp.OctreePointCloudSearch_t(resolution)
 
    def radius_search (self, point, double radius, unsigned int max_nn = 0):
        """
        Search for all neighbors of query point that are within a given radius.

        Returns: (k_indices, k_sqr_distances)
        """
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        if max_nn > 0:
            k_indices.resize(max_nn)
            k_sqr_distances.resize(max_nn)
        cdef int k = (<cpp.OctreePointCloudSearch_t*>self.me).radiusSearch(to_point_t(point), radius, k_indices, k_sqr_distances, max_nn)
        cdef cnp.ndarray[float] np_k_sqr_distances = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] np_k_indices = np.zeros(k, dtype=np.int32)
        for i in range(k):
            np_k_sqr_distances[i] = k_sqr_distances[i]
            np_k_indices[i] = k_indices[i]
        return np_k_indices, np_k_sqr_distances

