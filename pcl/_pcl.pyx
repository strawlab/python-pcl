#cython: embedsignature=True

from collections import Sequence
import numbers
import numpy as np

cimport numpy as cnp

cimport pcl_defs as cpp

cimport cython
from cython.operator import dereference as deref
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "minipcl.h":
    void mpcl_compute_normals(cpp.PointCloud_t, int ksearch, double searchRadius, cpp.PointNormalCloud_t)
    void mpcl_sacnormal_set_axis(cpp.SACSegmentationNormal_t, double ax, double ay, double az)
    void mpcl_extract(cpp.PointCloud_t, cpp.PointCloud_t, cpp.PointIndices_t *, bool)

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
#SACMODEL_NORMAL_SPHERE = cpp.SACMODEL_NORMAL_SPHERE
SACMODEL_REGISTRATION = cpp.SACMODEL_REGISTRATION
SACMODEL_PARALLEL_PLANE = cpp.SACMODEL_PARALLEL_PLANE
SACMODEL_NORMAL_PARALLEL_PLANE = cpp.SACMODEL_NORMAL_PARALLEL_PLANE
SACMODEL_STICK = cpp.SACMODEL_STICK


cnp.import_array()


cdef class Segmentation:
    """
    Segmentation class for Sample Consensus methods and models
    """
    cdef cpp.SACSegmentation_t *me
    def __cinit__(self):
        self.me = new cpp.SACSegmentation_t()
    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())],\
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)
    def set_model_type(self, cpp.SacModel m):
        self.me.setModelType(m)
    def set_method_type(self, int m):
        self.me.setMethodType (m)
    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)

#yeah, I can't be bothered making this inherit from SACSegmentation, I forget the rules
#for how this works in cython templated extension types anyway
cdef class SegmentationNormal:
    """
    Segmentation class for Sample Consensus methods and models that require the
    use of surface normals for estimation.

    Due to Cython limitations this should derive from pcl.Segmentation, but
    is currently unable to do so.
    """
    cdef cpp.SACSegmentationNormal_t *me
    def __cinit__(self):
        self.me = new cpp.SACSegmentationNormal_t()
    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())],\
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)
    def set_model_type(self, cpp.SacModel m):
        self.me.setModelType(m)
    def set_method_type(self, int m):
        self.me.setMethodType (m)
    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)
    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients (b)
    def set_normal_distance_weight(self, float f):
        self.me.setNormalDistanceWeight (f)
    def set_max_iterations(self, int i):
        self.me.setMaxIterations (i)
    def set_radius_limits(self, float f1, float f2):
        self.me.setRadiusLimits (f1, f2)
    def set_eps_angle(self, double ea):
        self.me.setEpsAngle (ea)
    def set_axis(self, double ax, double ay, double az):
        mpcl_sacnormal_set_axis(deref(self.me),ax,ay,az)

cdef class PointCloud:
    """Represents a cloud of points in 3-d space.

    A point cloud can be initialized from either a NumPy ndarray of shape
    (n_points, 3), from a list of triples, or from an integer n to create an
    "empty" cloud of n points.

    To load a point cloud from disk, use pcl.load.
    """
    cdef cpp.PointCloud[cpp.PointXYZ] *thisptr

    def __cinit__(self, init=None):
        self.thisptr = new cpp.PointCloud[cpp.PointXYZ]()

        if init is None:
            return
        elif isinstance(init, (numbers.Integral, np.integer)):
            self.resize(init)
        elif isinstance(init, np.ndarray):
            self.from_array(init)
        elif isinstance(init, Sequence):
            self.from_list(init)
        else:
            raise TypeError("Can't initialize a PointCloud from a %s"
                            % type(init))

    def __dealloc__(self):
        del self.thisptr
    property width:
        """ property containing the width of the point cloud """
        def __get__(self): return self.thisptr.width
    property height:
        """ property containing the height of the point cloud """
        def __get__(self): return self.thisptr.height
    property size:
        """ property containing the number of points in the point cloud """
        def __get__(self): return self.thisptr.size()
    property is_dense:
        """ property containing whether the cloud is dense or not """
        def __get__(self): return self.thisptr.is_dense

    @cython.boundscheck(False)
    def from_array(self, cnp.ndarray[cnp.float32_t, ndim=2] arr not None):
        """
        Fill this object from a 2D numpy array (float32)
        """
        assert arr.shape[1] == 3

        cdef cnp.npy_intp npts = arr.shape[0]
        self.resize(npts)
        self.thisptr.width = npts
        self.thisptr.height = 1

        cdef cpp.PointXYZ *p
        for i in range(npts):
            p = &self.thisptr.at(i)
            p.x, p.y, p.z = arr[i, 0], arr[i, 1], arr[i, 2]

    @cython.boundscheck(False)
    def to_array(self):
        """
        Return this object as a 2D numpy array (float32)
        """
        cdef float x,y,z
        cdef cnp.npy_intp n = self.thisptr.size()
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result
        cdef cpp.PointXYZ *p

        result = np.empty((n, 3), dtype=np.float32)

        for i in range(n):
            p = &self.thisptr.at(i)
            result[i, 0] = p.x
            result[i, 1] = p.y
            result[i, 2] = p.z
        return result

    def from_list(self, _list):
        """
        Fill this pointcloud from a list of 3-tuples
        """
        cdef Py_ssize_t npts = len(_list)
        cdef cpp.PointXYZ *p

        self.resize(npts)
        self.thisptr.width = npts
        self.thisptr.height = 1
        for i,l in enumerate(_list):
            p = &self.thisptr.at(i)
            p.x, p.y, p.z = l

    def to_list(self):
        """
        Return this object as a list of 3-tuples
        """
        return self.to_array().tolist()

    def resize(self, cnp.npy_intp x):
        self.thisptr.resize(x)

    def get_point(self, cnp.npy_intp row, cnp.npy_intp col):
        """
        Return a point (3-tuple) at the given row/column
        """
        cdef cpp.PointXYZ *p = &self.thisptr.at(row, col)
        return p.x, p.y, p.z

    def __getitem__(self, cnp.npy_intp idx):
        cdef cpp.PointXYZ *p = &self.thisptr.at(idx)
        return p.x, p.y, p.z

    def from_file(self, char *f):
        """
        Fill this pointcloud from a file (a local path).
        Only pcd files supported currently.

        Deprecated; use pcl.load instead.
        """
        return self._from_pcd_file(f)

    def _from_pcd_file(self, const char *s):
        cdef int ok = 0
        with nogil:
            ok = cpp.loadPCDFile(string(s), deref(self.thisptr))
        return ok

    def _from_ply_file(self, const char *s):
        cdef int ok = 0
        with nogil:
            ok = cpp.loadPLYFile(string(s), deref(self.thisptr))
        return ok

    def to_file(self, char *f, bool ascii=True):
        """
        Save this pointcloud to a local file.
        Only saving to binary or ascii pcd is supported
        """
        cdef bool binary = not ascii
        cdef int ok = 0
        cdef string s = string(f)
        if f.endswith(".pcd"):
            with nogil:
                ok = cpp.savePCDFile(s, deref(self.thisptr), binary)
        else:
            raise ValueError("Incorrect file extension (must be .pcd)")
        return ok

    def make_segmenter(self):
        """
        Return a pcl.Segmentation object with this object set as the input-cloud
        """
        seg = Segmentation()
        cdef cpp.SACSegmentation_t *cseg = <cpp.SACSegmentation_t *>seg.me
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>self.thisptr
        cseg.setInputCloud(ccloud.makeShared())
        return seg

    def make_segmenter_normals(self, int ksearch=-1, double searchRadius=-1.0):
        """
        Return a pcl.SegmentationNormal object with this object set as the input-cloud
        """
        cdef cpp.PointNormalCloud_t normals
        mpcl_compute_normals(deref(self.thisptr), ksearch, searchRadius, normals)

        seg = SegmentationNormal()
        cdef cpp.SACSegmentationNormal_t *cseg = <cpp.SACSegmentationNormal_t *>seg.me
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>self.thisptr
        cseg.setInputCloud(ccloud.makeShared())
        cseg.setInputNormals (normals.makeShared());

        return seg

    def make_statistical_outlier_filter(self):
        """
        Return a pcl.StatisticalOutlierRemovalFilter object with this object set as the input-cloud
        """
        fil = StatisticalOutlierRemovalFilter()
        cdef cpp.StatisticalOutlierRemoval_t *cfil = <cpp.StatisticalOutlierRemoval_t *>fil.me
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>self.thisptr
        cfil.setInputCloud(ccloud.makeShared())
        return fil

    def make_voxel_grid_filter(self):
        """
        Return a pcl.VoxelGridFilter object with this object set as the input-cloud
        """
        fil = VoxelGridFilter()
        cdef cpp.VoxelGrid_t *cfil = <cpp.VoxelGrid_t *>fil.me
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>self.thisptr
        cfil.setInputCloud(ccloud.makeShared())
        return fil

    def make_passthrough_filter(self):
        """
        Return a pcl.PassThroughFilter object with this object set as the input-cloud
        """
        fil = PassThroughFilter()
        cdef cpp.PassThrough_t *cfil = <cpp.PassThrough_t *>fil.me
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>self.thisptr
        cfil.setInputCloud(ccloud.makeShared())
        return fil

    def make_moving_least_squares(self):
        """
        Return a pcl.MovingLeastSquares object with this object set as the input-cloud
        """
        mls = MovingLeastSquares()

        cdef cpp.MovingLeastSquares_t *cmls = <cpp.MovingLeastSquares_t *>mls.me
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>self.thisptr
        cmls.setInputCloud(ccloud.makeShared())
        return mls

    def make_kdtree_flann(self):
        """
        Return a pcl.kdTreeFLANN object with this object set as the input-cloud
        """
        kdtree = KdTreeFLANN()
        cdef cpp.KdTreeFLANN_t *ckdtree = <cpp.KdTreeFLANN_t *>kdtree.me
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>self.thisptr
        ckdtree.setInputCloud(ccloud.makeShared())
        return kdtree

    def make_octree(self, double resolution):
        """
        Return a pcl.octree object with this object set as the input-cloud
        """
        octree = OctreePointCloud(resolution)
        octree.set_input_cloud(self)
        return octree

    def extract(self, pyindices, bool negative=False):
        """
        Given a list of indices of points in the pointcloud, return a 
        new pointcloud containing only those points.
        """
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>self.thisptr
        cdef cpp.PointCloud_t *out = new cpp.PointCloud_t()
        cdef cpp.PointIndices_t *ind = new cpp.PointIndices_t()

        for i in pyindices:
            ind.indices.push_back(i)

        mpcl_extract(deref(ccloud), deref(out), ind, negative)

        cdef PointCloud pycloud = PointCloud()
        del pycloud.thisptr
        pycloud.thisptr = out

        return pycloud

cdef class StatisticalOutlierRemovalFilter:
    """
    Filter class uses point neighborhood statistics to filter outlier data.
    """
    cdef cpp.StatisticalOutlierRemoval_t *me
    def __cinit__(self):
        self.me = new cpp.StatisticalOutlierRemoval_t()
    def __dealloc__(self):
        del self.me

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
        pc = PointCloud()
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>pc.thisptr
        self.me.filter(deref(ccloud))
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
        pc = PointCloud()
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>pc.thisptr
        self.me.process(deref(ccloud))
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
        pc = PointCloud()
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>pc.thisptr
        self.me.filter(deref(ccloud))
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

    def set_filter_field_name(self, char *field_name):
        cdef string s = string(field_name)
        self.me.setFilterFieldName (s)

    def set_filter_limits(self, float filter_min, float filter_max):
        self.me.setFilterLimits (filter_min, filter_max)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        pc = PointCloud()
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>pc.thisptr
        self.me.filter(deref(ccloud))
        return pc

cdef class KdTreeFLANN:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    this pointcloud
    """
    cdef cpp.KdTreeFLANN_t *me
    def __cinit__(self):
        self.me = new cpp.KdTreeFLANN_t()
    def __dealloc__(self):
        del self.me

    def nearest_k_search_for_cloud(self, PointCloud pc, int k=1):
        """
        Find the k nearest neighbours and squared distances for all points
        in the pointcloud. Results are in ndarrays, size (pc.size, k)
        Returns: (k_indices, k_sqr_distances)
        """
        n_points = pc.size
        cdef cnp.ndarray[float, ndim=2] np_k_sqr_distances = np.zeros(
            (n_points, k), dtype=np.float32)
        cdef cnp.ndarray[int, ndim=2] np_k_indices= np.zeros(
            (n_points, k), dtype=np.int32)
        for i in range(pc.size):
            k_i, k_sqr_d = self.nearest_k_search_for_point(pc, i, k)
            np_k_indices[i,:] = k_i
            np_k_sqr_distances[i,:] = k_sqr_d
        return np_k_indices, np_k_sqr_distances

    def nearest_k_search_for_point(self, PointCloud pc, int index, int k=1):
        """
        Find the k nearest neighbours and squared distances for the point
        at pc[index]. Results are in ndarrays, size (k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>pc.thisptr
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        k_indices.resize(k)
        k_sqr_distances.resize(k)
        self.me.nearestKSearch(deref(ccloud), index, k, k_indices, k_sqr_distances)
        cdef cnp.ndarray[float] np_k_sqr_distances = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] np_k_indices= np.zeros(k, dtype=np.int32)
        for i in range(k):
            np_k_sqr_distances[i] = k_sqr_distances[i]
            np_k_indices[i] = k_indices[i]
        return np_k_indices, np_k_sqr_distances

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
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = new cpp.OctreePointCloud_t(resolution)
    
    def __dealloc__(self):
        del self.me

    def set_input_cloud(self, PointCloud pc):
        """
        Provide a pointer to the input data set.
        """
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>pc.thisptr
        self.me.setInputCloud(ccloud.makeShared())

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
 
    def __dealloc__(self):
        del self.me

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

