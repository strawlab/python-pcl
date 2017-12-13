# -*- coding: utf-8 -*-
# main
cimport pcl_defs as cpp
import numpy as np
cimport numpy as cnp

cnp.import_array()

# parts
cimport pcl_features as pclftr
cimport pcl_filters as pclfil
cimport pcl_io as pclio
cimport pcl_kdtree as pclkdt
cimport pcl_octree as pcloct
cimport pcl_sample_consensus as pcl_sc
# cimport pcl_search as pcl_sch
cimport pcl_segmentation as pclseg
cimport pcl_surface as pclsf
cimport pcl_range_image as pcl_r_img

from libcpp cimport bool
cimport indexing as idx

from boost_shared_ptr cimport sp_assign

cdef extern from "minipcl.h":
    void mpcl_compute_normals(cpp.PointCloud_t, int ksearch,
                              double searchRadius,
                              cpp.PointCloud_Normal_t) except +
    void mpcl_extract(cpp.PointCloudPtr_t, cpp.PointCloud_t *,
                              cpp.PointIndices_t *, bool) except +
    ## void mpcl_extract_HarrisKeypoint3D(cpp.PointCloudPtr_t, cpp.PointCloud_PointXYZ *) except +
    # void mpcl_extract_HarrisKeypoint3D(cpp.PointCloudPtr_t, cpp.PointCloud_t *) except +


cdef extern from "ProjectInliers.h":
    void mpcl_ProjectInliers_setModelCoefficients(pclfil.ProjectInliers_t);

# Empirically determine strides, for buffer support.
# XXX Is there a more elegant way to get these?
cdef Py_ssize_t _strides[2]
cdef PointCloud2 _pc_tmp = PointCloud(np.array([[1, 2, 3],
                                               [4, 5, 6]], dtype=np.float32))

cdef cpp.PointCloud2[cpp.PointXYZ] *p = _pc_tmp.thisptr()
_strides[0] = (  <Py_ssize_t><void *>idx.getptr(p, 1)
               - <Py_ssize_t><void *>idx.getptr(p, 0))
_strides[1] = (  <Py_ssize_t><void *>&(idx.getptr(p, 0).y)
               - <Py_ssize_t><void *>&(idx.getptr(p, 0).x))
_pc_tmp = None

cdef class PointCloud2:
    """Represents a cloud of points in 3-d space.

    A point cloud can be initialized from either a NumPy ndarray of shape
    (n_points, 3), from a list of triples, or from an integer n to create an
    "empty" cloud of n points.

    To load a point cloud from disk, use pcl.load.
    """
    def __cinit__(self, init=None):
        cdef PointCloud2 other
        
        self._view_count = 0
        
        # TODO: NG --> import pcl --> pyd Error(python shapedptr/C++ shard ptr collusion?)
        # sp_assign(<cpp.shared_ptr[cpp.PointCloud2[cpp.PointXYZ]]> self.thisptr_shared, new cpp.PointCloud2[cpp.PointXYZ]())
        sp_assign(self.thisptr_shared, new cpp.PointCloud2[cpp.PointXYZ]())
        
        if init is None:
            return
        elif isinstance(init, (numbers.Integral, np.integer)):
            self.resize(init)
        elif isinstance(init, np.ndarray):
            self.from_array(init)
        elif isinstance(init, Sequence):
            self.from_list(init)
        elif isinstance(init, type(self)):
            other = init
            self.thisptr()[0] = other.thisptr()[0]
        else:
            raise TypeError("Can't initialize a PointCloud2 from a %s"
                            % type(init))

    property width:
        """ property containing the width of the point cloud """
        def __get__(self): return self.thisptr().width
    property height:
        """ property containing the height of the point cloud """
        def __get__(self): return self.thisptr().height
    property size:
        """ property containing the number of points in the point cloud """
        def __get__(self): return self.thisptr().size()
    property is_dense:
        """ property containing whether the cloud is dense or not """
        def __get__(self): return self.thisptr().is_dense

    def __repr__(self):
        return "<PointCloud2 of %d points>" % self.size

    # Buffer protocol support. Taking a view locks the PointCloud2 for
    # resizing, because that can move it around in memory.
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # TODO parse flags
        cdef Py_ssize_t npoints = self.thisptr().size()
        
        if self._view_count == 0:
            self._shape[0] = npoints
            self._shape[1] = 3
        self._view_count += 1

        buffer.buf = <char *>&(idx.getptr_at(self.thisptr(), 0).x)
        buffer.format = 'f'
        buffer.internal = NULL
        buffer.itemsize = sizeof(float)
        buffer.len = npoints * 3 * sizeof(float)
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self._shape
        buffer.strides = _strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        self._view_count -= 1

    # Pickle support. XXX this copies the entire PointCloud2; it would be nice
    # to have an asarray member that returns a view, or even better, implement
    # the buffer protocol (https://docs.python.org/c-api/buffer.html).
    def __reduce__(self):
        return type(self), (self.to_array(),)

    property sensor_origin:
        def __get__(self):
            cdef cpp.Vector4f origin = self.thisptr().sensor_origin_
            cdef float *data = origin.data()
            return np.array([data[0], data[1], data[2], data[3]],
                            dtype=np.float32)

    property sensor_orientation:
        def __get__(self):
            # NumPy doesn't have a quaternion type, so we return a 4-vector.
            cdef cpp.Quaternionf o = self.thisptr().sensor_orientation_
            return np.array([o.w(), o.x(), o.y(), o.z()])

    # cdef inline PointCloud2[PointXYZ] *thisptr(self) nogil:
    #     # Shortcut to get raw pointer to underlying PointCloud2
    #     return self.thisptr_shared.get()

    @cython.boundscheck(False)
    def from_array(self, cnp.ndarray[cnp.float32_t, ndim=2] arr not None):
        """
        Fill this object from a 2D numpy array (float32)
        """
        assert arr.shape[1] == 3
        
        cdef cnp.npy_intp npts = arr.shape[0]
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1
        
        cdef cpp.PointXYZ *p
        for i in range(npts):
            p = idx.getptr(self.thisptr(), i)
            p.x, p.y, p.z = arr[i, 0], arr[i, 1], arr[i, 2]

    @cython.boundscheck(False)
    def to_array(self):
        """
        Return this object as a 2D numpy array (float32)
        """
        cdef float x,y,z
        cdef cnp.npy_intp n = self.thisptr().size()
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result
        cdef cpp.PointXYZ *p
        
        result = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            p = idx.getptr(self.thisptr(), i)
            result[i, 0] = p.x
            result[i, 1] = p.y
            result[i, 2] = p.z
        
        return result

    def from_list(self, _list):
        """
        Fill this PointCloud2 from a list of 3-tuples
        """
        cdef Py_ssize_t npts = len(_list)
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1
        cdef cpp.PointXYZ* p
        # OK
        # p = idx.getptr(self.thisptr(), 1)
        # enumerate ? -> i -> type unknown
        for i, l in enumerate(_list):
             p = idx.getptr(self.thisptr(), <int> i)
             p.x, p.y, p.z = l

    def to_list(self):
        """
        Return this object as a list of 3-tuples
        """
        return self.to_array().tolist()

    def resize(self, cnp.npy_intp x):
        if self._view_count > 0:
            raise ValueError("can't resize PointCloud2 while there are"
                             " arrays/memoryviews referencing it")
        self.thisptr().resize(x)

    def get_point(self, cnp.npy_intp row, cnp.npy_intp col):
        """
        Return a point (3-tuple) at the given row/column
        """
        cdef cpp.PointXYZ *p = idx.getptr_at2(self.thisptr(), row, col)
        return p.x, p.y, p.z

    def __getitem__(self, cnp.npy_intp nmidx):
        cdef cpp.PointXYZ *p = idx.getptr_at(self.thisptr(), nmidx)
        return p.x, p.y, p.z

    def from_file(self, char *f):
        """
        Fill this PointCloud2 from a file (a local path).
        Only pcd files supported currently.
        
        Deprecated; use pcl.load instead.
        """
        return self._from_pcd_file(f)

    def _from_pcd_file(self, const char *s):
        cdef int error = 0
        with nogil:
            # NG
            # error = pclio.loadPCDFile(string(s), <cpp.PointCloud2[cpp.PointXYZ]> deref(self.thisptr()))
            error = pclio.loadPCDFile(string(s), deref(self.thisptr()))
        return error

    def _from_ply_file(self, const char *s):
        cdef int ok = 0
        with nogil:
            # NG
            # ok = pclio.loadPLYFile(string(s), <cpp.PointCloud2[cpp.PointXYZ]> deref(self.thisptr()))
            ok = pclio.loadPLYFile(string(s), deref(self.thisptr()))
        return ok

    def to_file(self, const char *fname, bool ascii=True):
        """Save PointCloud2 to a file in PCD format.

        Deprecated: use pcl.save instead.
        """
        return self._to_pcd_file(fname, not ascii)

    def _to_pcd_file(self, const char *f, bool binary=False):
        cdef int error = 0
        cdef string s = string(f)
        with nogil:
            # NG
            # error = pclio.savePCDFile(s, <cpp.PointCloud2[cpp.PointXYZ]> deref(self.thisptr()), binary)
            # OK
            error = pclio.savePCDFile(s, deref(self.thisptr()), binary)
            # pclio.PointCloud2[cpp.PointXYZ] *p = self.thisptr()
            # error = pclio.savePCDFile(s, p, binary)
        return error

    def _to_ply_file(self, const char *f, bool binary=False):
        cdef int error = 0
        cdef string s = string(f)
        with nogil:
            # NG
            # error = pclio.savePLYFile(s, <cpp.PointCloud2[cpp.PointXYZ]> deref(self.thisptr()), binary)
            error = pclio.savePLYFile(s, deref(self.thisptr()), binary)
        return error

    def make_segmenter(self):
        """
        Return a pcl.Segmentation object with this object set as the input-cloud
        """
        seg = Segmentation()
        cdef pclseg.SACSegmentation_t *cseg = <pclseg.SACSegmentation_t *>seg.me
        cseg.setInputCloud(self.thisptr_shared)
        return seg

    def make_segmenter_normals(self, int ksearch=-1, double searchRadius=-1.0):
        """
        Return a pcl.SegmentationNormal object with this object set as the input-cloud
        """
        cdef cpp.PointCloud_Normal_t normals
        mpcl_compute_normals(<cpp.PointCloud2[cpp.PointXYZ]> deref(self.thisptr()), ksearch, searchRadius, normals)
        # p = self.thisptr()
        # mpcl_compute_normals(deref(p), ksearch, searchRadius, normals)
        seg = SegmentationNormal()
        cdef pclseg.SACSegmentationNormal_t *cseg = <pclseg.SACSegmentationNormal_t *>seg.me
        cseg.setInputCloud(self.thisptr_shared)
        cseg.setInputNormals (normals.makeShared());
        return seg

    def make_statistical_outlier_filter(self):
        """
        Return a pcl.StatisticalOutlierRemovalFilter object with this object set as the input-cloud
        """
        fil = StatisticalOutlierRemovalFilter()
        cdef pclfil.StatisticalOutlierRemoval_t *cfil = <pclfil.StatisticalOutlierRemoval_t *>fil.me
        cfil.setInputCloud(<cpp.shared_ptr[cpp.PointCloud2[cpp.PointXYZ]]> self.thisptr_shared)
        return fil

    def make_voxel_grid_filter(self):
        """
        Return a pcl.VoxelGridFilter object with this object set as the input-cloud
        """
        fil = VoxelGridFilter()
        cdef pclfil.VoxelGrid_t *cfil = <pclfil.VoxelGrid_t *>fil.me
        cfil.setInputCloud(<cpp.shared_ptr[cpp.PointCloud2[cpp.PointXYZ]]> self.thisptr_shared)
        return fil

    def make_passthrough_filter(self):
        """
        Return a pcl.PassThroughFilter object with this object set as the input-cloud
        """
        fil = PassThroughFilter()
        cdef pclfil.PassThrough_t *cfil = <pclfil.PassThrough_t *>fil.me
        cfil.setInputCloud(<cpp.shared_ptr[cpp.PointCloud2[cpp.PointXYZ]]> self.thisptr_shared)
        return fil

    def make_moving_least_squares(self):
        """
        Return a pcl.MovingLeastSquares object with this object as input cloud.
        """
        mls = MovingLeastSquares()
        cdef pclsf.MovingLeastSquares_t *cmls = <pclsf.MovingLeastSquares_t *>mls.me
        cmls.setInputCloud(<cpp.shared_ptr[cpp.PointCloud2[cpp.PointXYZ]]> self.thisptr_shared)
        return mls

    def make_kdtree(self):
        """
        Return a pcl.kdTree object with this object set as the input-cloud
        
        Deprecated: use the pcl.KdTree constructor on this cloud.
        """
        return KdTree(self)

    def make_kdtree_flann(self):
        """
        Return a pcl.kdTreeFLANN object with this object set as the input-cloud
        Deprecated: use the pcl.KdTreeFLANN constructor on this cloud.
        """
        return KdTreeFLANN(self)

    def make_octree(self, double resolution):
        """
        Return a pcl.octree object with this object set as the input-cloud
        """
        octree = OctreePointCloud(resolution)
        octree.set_input_cloud(self)
        return octree

    def make_octreeSearch(self, double resolution):
        """
        Return a pcl.make_octreeSearch object with this object set as the input-cloud
        """
        octreeSearch = OctreePointCloudSearch(resolution)
        octreeSearch.set_input_cloud(self)
        return octreeSearch

# pcl 1.6.0 use ok
# cpl 1.7.2, 1.8.0 use ng(octree_pointcloud_changedetector.h(->octree_pointcloud.h) include headerfile comment octree2buf_base.h)
#    def make_octreeChangeDetector(self, double resolution):
#        """
#        Return a pcl.make_octreeSearch object with this object set as the input-cloud
#        """
#        octreeChangeDetector = OctreePointCloudChangeDetector(resolution)
#        octreeChangeDetector.set_input_cloud(self)
#        return octreeChangeDetector


    def make_crophull(self):
        """
        Return a pcl.CropHull object with this object set as the input-cloud

        Deprecated: use the pcl.Vertices constructor on this cloud.
        """
        return CropHull(self)
        
    def make_cropbox(self):
        """
        Return a pcl.CropBox object with this object set as the input-cloud

        Deprecated: use the pcl.Vertices constructor on this cloud.
        """
        return CropBox(self)

    def make_IntegralImageNormalEstimation(self):
        """
        Return a pcl.IntegralImageNormalEstimation object with this object set as the input-cloud

        Deprecated: use the pcl.Vertices constructor on this cloud.
        """
        return IntegralImageNormalEstimation(self)

    def extract(self, pyindices, bool negative=False):
        """
        Given a list of indices of points in the PointCloud2, return a 
        new PointCloud2 containing only those points.
        """
        cdef PointCloud2 result
        cdef cpp.PointIndices_t *ind = new cpp.PointIndices_t()
        
        for i in pyindices:
            ind.indices.push_back(i)
        
        result = PointCloud2()
        # result = ExtractIndices()
        # (<cpp.PointCloud2[cpp.PointXYZ]> deref(self.thisptr())
        mpcl_extract(self.thisptr_shared, result.thisptr(), ind, negative)
        # XXX are we leaking memory here? del ind causes a double free...
        
        return result

    def make_ProjectInliers(self):
        """
        Return a pclfil.ProjectInliers object with this object set as the input-cloud
        """
        # proj = ProjectInliers()
        # cdef pclfil.ProjectInliers_t *cproj = <pclfil.ProjectInliers_t *>proj.me
        # cproj.setInputCloud(self.thisptr_shared)
        # return proj
        # # cdef pclfil.ProjectInliers_t* projInliers
        # # mpcl_ProjectInliers_setModelCoefficients(projInliers)
        # mpcl_ProjectInliers_setModelCoefficients(deref(projInliers))
        # # proj = ProjectInliers()
        # cdef pclfil.ProjectInliers_t *cproj = <pclfil.ProjectInliers_t *>projInliers
        # cproj.setInputCloud(self.thisptr_shared)
        # return proj
        # # NG
        # cdef pclfil.ProjectInliers_t* projInliers
        # # mpcl_ProjectInliers_setModelCoefficients(projInliers)
        # mpcl_ProjectInliers_setModelCoefficients(deref(projInliers))
        # projInliers.setInputCloud(self.thisptr_shared)
        # proj = ProjectInliers()
        # proj.me = projInliers
        # return proj
        proj = ProjectInliers()
        cdef pclfil.ProjectInliers_t *cproj = <pclfil.ProjectInliers_t *>proj.me
        # mpcl_ProjectInliers_setModelCoefficients(cproj)
        mpcl_ProjectInliers_setModelCoefficients(deref(cproj))
        cproj.setInputCloud(<cpp.shared_ptr[cpp.PointCloud2[cpp.PointXYZ]]> self.thisptr_shared)
        return proj

    def make_RadiusOutlierRemoval(self):
        """
        Return a pclfil.RadiusOutlierRemoval object with this object set as the input-cloud
        """
        fil = RadiusOutlierRemoval()
        cdef pclfil.RadiusOutlierRemoval_t *cfil = <pclfil.RadiusOutlierRemoval_t *>fil.me
        cfil.setInputCloud(<cpp.shared_ptr[cpp.PointCloud2[cpp.PointXYZ]]> self.thisptr_shared)
        return fil

    def make_ConditionAnd(self):
        """
        Return a pcl.ConditionAnd object with this object set as the input-cloud
        """
        condAnd = ConditionAnd()
        cdef pclfil.ConditionAnd_t *cCondAnd = <pclfil.ConditionAnd_t *>condAnd.me
        return condAnd

    def make_ConditionalRemoval(self, ConditionAnd range_conf):
        """
        Return a pcl.ConditionalRemoval object with this object set as the input-cloud
        """
        condRemoval = ConditionalRemoval(range_conf)
        cdef pclfil.ConditionalRemoval_t *cCondRemoval = <pclfil.ConditionalRemoval_t *>condRemoval.me
        cCondRemoval.setInputCloud(<cpp.shared_ptr[cpp.PointCloud2[cpp.PointXYZ]]> self.thisptr_shared)
        return condRemoval

    def make_ConcaveHull(self):
        """
        Return a pcl.ConditionalRemoval object with this object set as the input-cloud
        """
        concaveHull = ConcaveHull()
        cdef pclsf.ConcaveHull_t *cConcaveHull = <pclsf.ConcaveHull_t *>concaveHull.me
        cConcaveHull.setInputCloud(<cpp.shared_ptr[cpp.PointCloud2[cpp.PointXYZ]]> self.thisptr_shared)
        return concaveHull


    def make_HarrisKeypoint3D(self):
        """
        Return a pcl.PointCloud2 object with this object set as the input-cloud
        """
        cdef PointCloud2 result

        result = PointCloud_PointXYZI()
        # # harris = HarrisKeypoint3D()
        # mpcl_extract_HarrisKeypoint3D(self.thisptr_shared, result.thisptr())
        # # mpcl_extract_HarrisKeypoint3D(self.thisptr_shared, result.thisptr_shared)
        # # cdef keypt.HarrisKeypoint3DPtr_t *cseg = <pclseg.SACSegmentationNormal_t *>harris.me
        # # charris.setInputCloud(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> self.thisptr_shared)
        # # charris.setNonMaxSupression (true)
        # # charris.setRadius (1.0)
        # # charris.setRadiusSearch (searchRadius)
        # # charris.compare(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> result.thisptr())

        return result

    def make_NormalEstimation(self):
        normalEstimation = NormalEstimation()
        cdef pclftr.NormalEstimation_t *cNormalEstimation = <pclftr.NormalEstimation_t *>normalEstimation.me
        cNormalEstimation.setInputCloud(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> self.thisptr_shared)
        return normalEstimation

    def make_VFHEstimation(self):
        vfhEstimation = VFHEstimation()
        cdef pclftr.VFHEstimation_t *cVFHEstimation = <pclftr.VFHEstimation_t *>vfhEstimation.me
        cVFHEstimation.setInputCloud(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> self.thisptr_shared)
        return vfhEstimation

    def make_RangeImage(self):
        rangeImages = RangeImages(self)
        # cdef pcl_r_img.RangeImage_t *cRangeImage = <pcl_r_img.RangeImage_t *>rangeImages.me
        return rangeImages

    def make_EuclideanClusterExtraction(self):
        euclideanclusterextraction = EuclideanClusterExtraction(self)
        cdef pclseg.EuclideanClusterExtraction_t *cEuclideanClusterExtraction = <pclseg.EuclideanClusterExtraction_t *>euclideanclusterextraction.me
        cEuclideanClusterExtraction.setInputCloud(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> self.thisptr_shared)
        return euclideanclusterextraction

    # registration - icp?
    # def make_IterativeClosestPoint():
    #     iterativeClosestPoint = IterativeClosestPoint(self)
    #     cdef pclseg.IterativeClosestPoint *cEuclideanClusterExtraction = <pclseg.IterativeClosestPoint *>euclideanclusterextraction.me
    #     
    #     cEuclideanClusterExtraction.setInputCloud(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> self.thisptr_shared)
    #     # icp.setInputCloud(cloud_in);
    #     # icp.setInputTarget(cloud_out);
    #     return euclideanclusterextraction

###


### include ###
include "Segmentation/Segmentation.pxi"
include "Segmentation/SegmentationNormal.pxi"
include "Segmentation/EuclideanClusterExtraction.pxi"
include "Filters/StatisticalOutlierRemovalFilter.pxi"
include "Filters/VoxelGridFilter.pxi"
include "Filters/PassThroughFilter.pxi"
include "Surface/MovingLeastSquares.pxi"
# include "KdTree/KdTree.pxi"
include "KdTree/KdTree_FLANN.pxi"
# Octree
include "Octree/OctreePointCloud.pxi"
include "Octree/OctreePointCloudSearch.pxi"
include "Vertices.pxi"
include "Filters/CropHull.pxi"
include "Filters/CropBox.pxi"
include "Filters/ProjectInliers.pxi"
include "Filters/RadiusOutlierRemoval.pxi"
include "Filters/ConditionAnd.pxi"
include "Filters/ConditionalRemoval.pxi"
include "Surface/ConcaveHull.pxi"
include "Common/RangeImage/RangeImages.pxi"

# include "Visualization/PointCloudColorHandlerCustoms.pxi"

# Features
include "Features/NormalEstimation.pxi"
include "Features/VFHEstimation.pxi"
include "Features/IntegralImageNormalEstimation.pxi"

# keyPoint
# include "KeyPoint/UniformSampling.pxi"
include "KeyPoint/HarrisKeypoint3D.pxi"

