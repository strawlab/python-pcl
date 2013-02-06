#cython: embedsignature=True

import numpy as np

cimport numpy as cnp

cimport pcl_defs as cpp

from cython.operator import dereference as deref
from libcpp.string cimport string
from libcpp cimport bool

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
    """
    Represents a class of points, supporting the PointXYZ type.
    """
    cdef cpp.PointCloud[cpp.PointXYZ] *thisptr
    def __cinit__(self):
        self.thisptr = new cpp.PointCloud[cpp.PointXYZ]()
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

    def from_array(self, cnp.ndarray[cnp.float32_t, ndim=2] arr not None):
        """
        Fill this object from a 2D numpy array (float32)
        """
        assert arr.shape[1] == 3

        cdef int npts = arr.shape[0]
        self.resize(npts)
        self.thisptr.width = npts
        self.thisptr.height = 1

        cdef i = 0
        while i < npts:
            self.thisptr.at(i).x = arr[i,0]
            self.thisptr.at(i).y = arr[i,1]
            self.thisptr.at(i).z = arr[i,2]
            i += 1

    def to_array(self):
        """
        Return this object as a 2D numpy array (float32)
        """
        #FIXME: this could be done more efficinetly, i'm sure
        return np.array(self.to_list(), dtype=np.float32)

    def from_list(self, _list):
        """
        Fill this pointcloud from a list of 3-tuples
        """
        assert len(_list)
        assert len(_list[0]) == 3

        cdef npts = len(_list)
        self.resize(npts)
        self.thisptr.width = npts
        self.thisptr.height = 1
        for i,l in enumerate(_list):
            self.thisptr.at(i).x = l[0]
            self.thisptr.at(i).y = l[1]
            self.thisptr.at(i).z = l[2]

    def to_list(self):
        """
        Return this object as a list of 3-tuples
        """
        cdef int i
        cdef float x,y,z
        cdef int n = self.thisptr.size()

        result = []

        i = 0
        while i < n:
            #not efficient, oh well...
            x = self.thisptr.at(i).x
            y = self.thisptr.at(i).y
            z = self.thisptr.at(i).z
            result.append((x,y,z))
            i = i + 1
        return result

    def resize(self, int x):
        self.thisptr.resize(x)

    def get_point(self, int row, int col):
        """
        Return a point (3-tuple) at the given row/column
        """
        #grr.... the following doesnt compile to valid
        #cython.. so just take the perf hit
        #cdef PointXYZ &p = self.thisptr.at(x,y)
        cdef x = self.thisptr.at(row,col).x
        cdef y = self.thisptr.at(row,col).y
        cdef z = self.thisptr.at(row,col).z
        return x,y,z
        
    def __getitem__(self, int idx):
        cdef x = self.thisptr.at(idx).x
        cdef y = self.thisptr.at(idx).y
        cdef z = self.thisptr.at(idx).z
        return x,y,z

    def from_file(self, char *f):
        """
        Fill this pointcloud from a file (a local path).
        Only pcd files supported currently.
        """
        cdef int ok = 0
        cdef string s = string(f)
        if f.endswith(".pcd"):
            ok = cpp.loadPCDFile(s, deref(self.thisptr))
        else:
            raise ValueError("Incorrect file extension (must be .pcd)")
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

    def make_moving_least_squares(self):
        """
        Return a pcl.MovingLeastSquares object with this object set as the input-cloud
        """
        mls = MovingLeastSquares()

        cdef cpp.MovingLeastSquares_t *cmls = <cpp.MovingLeastSquares_t *>mls.me
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>self.thisptr
        cmls.setInputCloud(ccloud.makeShared())
        return mls

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

    def set_polynomial_fit(self, int fit):
        """
        Sets whether the surface and normal are approximated using a polynomial,
        or only via tangent estimation.
        """
        self.me.setPolynomialFit(fit)

    def reconstruct(self):
        """
        Apply the smoothing according to the previously set values and return
        a new pointcloud
        """
        pc = PointCloud()
        cdef cpp.PointCloud_t *ccloud = <cpp.PointCloud_t *>pc.thisptr
        self.me.reconstruct(deref(ccloud))
        return pc


