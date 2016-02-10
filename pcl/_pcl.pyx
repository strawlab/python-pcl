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

include "Segmentation.pxi"
include "SegmentationNormal.pxi"

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

cdef class StatisticalOutlierRemovalFilter:
    """
    Filter class uses point neighborhood statistics to filter outlier data.
    """
    cdef cpp.StatisticalOutlierRemoval_t *me
    def __cinit__(self):
        self.me = new cpp.StatisticalOutlierRemoval_t()
    def __dealloc__(self):
        del self.me

    property mean_k:
        def __get__(self):
            return self.me.getMeanK()
        def __set__(self, int k):
            self.me.setMeanK(k)

    property negative:
        def __get__(self):
            return self.me.getNegative()
        def __set__(self, bool neg):
            self.me.setNegative(neg)

    property stddev_mul_thresh:
        def __get__(self):
            return self.me.getStddevMulThresh()
        def __set__(self, double thresh):
            self.me.setStddevMulThresh(thresh)

    def set_mean_k(self, int k):
        """
        Set the number of points (k) to use for mean distance estimation. 
        """
        self.me.setMeanK(k)

    def set_std_dev_mul_thresh(self, double std_mul):
        """
        Set the standard deviation multiplier threshold.
        """
        self.me.setStddevMulThresh(std_mul)

    def set_negative(self, bool negative):
        """
        Set whether the indices should be returned, or all points except the indices. 
        """
        self.me.setNegative(negative)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc

cdef class MovingLeastSquares:
    """
    Smoothing class which is an implementation of the MLS (Moving Least Squares)
    algorithm for data smoothing and improved normal estimation.
    """
    cdef cpp.MovingLeastSquares_t *me
    def __cinit__(self):
        self.me = new cpp.MovingLeastSquares_t()
    def __dealloc__(self):
        del self.me

    def set_search_radius(self, double radius):
        """
        Set the sphere radius that is to be used for determining the k-nearest neighbors used for fitting. 
        """
        self.me.setSearchRadius (radius)

    def set_polynomial_order(self, bool order):
        """
        Set the order of the polynomial to be fit. 
        """
        self.me.setPolynomialOrder(order)

    def set_polynomial_fit(self, bint fit):
        """
        Sets whether the surface and normal are approximated using a polynomial,
        or only via tangent estimation.
        """
        self.me.setPolynomialFit(fit)

    def process(self):
        """
        Apply the smoothing according to the previously set values and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.process(pc.thisptr()[0])
        return pc

cdef class VoxelGridFilter:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef cpp.VoxelGrid_t *me
    def __cinit__(self):
        self.me = new cpp.VoxelGrid_t()
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

cdef class PassThroughFilter:
    """
    Passes points in a cloud based on constraints for one particular field of the point type
    """
    cdef cpp.PassThrough_t *me
    def __cinit__(self):
        self.me = new cpp.PassThrough_t()
    def __dealloc__(self):
        del self.me

    def set_filter_field_name(self, field_name):
        cdef bytes fname_ascii
        if isinstance(field_name, unicode):
            fname_ascii = field_name.encode("ascii")
        elif not isinstance(field_name, bytes):
            raise TypeError("field_name should be a string, got %r"
                            % field_name)
        else:
            fname_ascii = field_name
        self.me.setFilterFieldName(string(fname_ascii))

    def set_filter_limits(self, float filter_min, float filter_max):
        self.me.setFilterLimits (filter_min, filter_max)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc

cdef class KdTree:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTree(pc).
    """
    cdef cpp.KdTree_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new cpp.KdTree_t()


        self.me.setInputCloud(pc.thisptr_shared)


cdef class KdTreeFLANN:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTreeFLANN(pc).
    """
    cdef cpp.KdTreeFLANN_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new cpp.KdTreeFLANN_t()

        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def nearest_k_search_for_cloud(self, PointCloud pc not None, int k=1):
        """
        Find the k nearest neighbours and squared distances for all points
        in the pointcloud. Results are in ndarrays, size (pc.size, k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.npy_intp n_points = pc.size
        cdef cnp.ndarray[float, ndim=2] sqdist = np.zeros((n_points, k),
                                                          dtype=np.float32)
        cdef cnp.ndarray[int, ndim=2] ind = np.zeros((n_points, k),
                                                     dtype=np.int32)

        for i in range(n_points):
            self._nearest_k(pc, i, k, ind[i], sqdist[i])
        return ind, sqdist

    def nearest_k_search_for_point(self, PointCloud pc not None, int index,
                                   int k=1):
        """
        Find the k nearest neighbours and squared distances for the point
        at pc[index]. Results are in ndarrays, size (k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.ndarray[float] sqdist = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] ind = np.zeros(k, dtype=np.int32)

        self._nearest_k(pc, index, k, ind, sqdist)
        return ind, sqdist

    @cython.boundscheck(False)
    cdef void _nearest_k(self, PointCloud pc, int index, int k,
                         cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
                         cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
                        ) except +:
        # k nearest neighbors query for a single point.
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        k_indices.resize(k)
        k_sqr_distances.resize(k)
        self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices,
                               k_sqr_distances)

        for i in range(k):
            sqdist[i] = k_sqr_distances[i]
            ind[i] = k_indices[i]

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

