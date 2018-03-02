# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
import numpy as np
cimport numpy as cnp

cnp.import_array()

# parts
cimport pcl_features_180 as pclftr
cimport pcl_filters_180 as pclfil
cimport pcl_io_180 as pclio
cimport pcl_kdtree_180 as pclkdt
# cimport pcl_octree_180 as pcloct
# cimport pcl_sample_consensus_180 as pcl_sc
# cimport pcl_search_180 as pcl_sch
cimport pcl_segmentation_180 as pclseg
cimport pcl_surface_180 as pclsf

from libcpp cimport bool
cimport indexing as idx
from boost_shared_ptr cimport sp_assign
from _pcl cimport PointCloud_PointXYZI

cdef extern from "minipcl.h":
    void mpcl_compute_normals_PointXYZI(cpp.PointCloud_PointXYZI_t, int ksearch,
                              double searchRadius,
                              cpp.PointCloud_Normal_t) except +
    void mpcl_sacnormal_set_axis_PointXYZI(pclseg.SACSegmentationNormal_PointXYZI_t,
                              double ax, double ay, double az) except +
    void mpcl_extract_PointXYZI(cpp.PointCloud_PointXYZI_Ptr_t, cpp.PointCloud_PointXYZI_t *,
                              cpp.PointIndices_t *, bool) except +

# Empirically determine strides, for buffer support.
# XXX Is there a more elegant way to get these?
cdef Py_ssize_t _strides_xyzi_4[2]
cdef PointCloud_PointXYZI _pc_xyzi_tmp4 = PointCloud_PointXYZI(np.array([[1, 2, 3, 0],
                                                                    [4, 5, 6, 0]], dtype=np.float32))
cdef cpp.PointCloud[cpp.PointXYZI] *p_xyzi_4 = _pc_xyzi_tmp4.thisptr()
_strides_xyzi_4[0] = (  <Py_ssize_t><void *>idx.getptr(p_xyzi_4, 1)
               - <Py_ssize_t><void *>idx.getptr(p_xyzi_4, 0))
_strides_xyzi_4[1] = (  <Py_ssize_t><void *>&(idx.getptr(p_xyzi_4, 0).y)
               - <Py_ssize_t><void *>&(idx.getptr(p_xyzi_4, 0).x))
_pc_xyzi_tmp4 = None

cdef class PointCloud_PointXYZI:
    """Represents a cloud of points in 4-d space.

    A point cloud can be initialized from either a NumPy ndarray of shape
    (n_points, 4), from a list of triples, or from an integer n to create an
    "empty" cloud of n points.

    To load a point cloud from disk, use pcl.load.
    """
    def __cinit__(self, init=None):
        cdef PointCloud_PointXYZI other
        
        self._view_count = 0
        
        # TODO: NG --> import pcl --> pyd Error(python shapedptr/C++ shard ptr collusion?)
        # sp_assign(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZI]]> self.thisptr_shared, new cpp.PointCloud[cpp.PointXYZI]())
        sp_assign(self.thisptr_shared, new cpp.PointCloud[cpp.PointXYZI]())
        
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
            raise TypeError("Can't initialize a PointCloud from a %s"
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
        return "<PointCloud of %d points>" % self.size

    # Buffer protocol support. Taking a view locks the pointcloud for
    # resizing, because that can move it around in memory.
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # TODO parse flags
        cdef Py_ssize_t npoints = self.thisptr().size()
        
        if self._view_count == 0:
            self._shape[0] = npoints
            self._shape[1] = 4
        self._view_count += 1

        buffer.buf = <char *>&(idx.getptr_at(self.thisptr(), 0).x)
        buffer.format = 'f'
        buffer.internal = NULL
        buffer.itemsize = sizeof(float)
        buffer.len = npoints * 4 * sizeof(float)
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self._shape
        buffer.strides = _strides_xyzi_4
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        self._view_count -= 1

    # Pickle support. XXX this copies the entire pointcloud; it would be nice
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

        def __set__(self, cnp.ndarray[cnp.float32_t, ndim=1] new_origin):
            self.thisptr().sensor_origin_ = cpp.Vector4f(
                    new_origin[0],
                    new_origin[1],
                    new_origin[2],
                    0.0)

    property sensor_orientation:
        def __get__(self):
            # NumPy doesn't have a quaternion type, so we return a 4-vector.
            cdef cpp.Quaternionf o = self.thisptr().sensor_orientation_
            return np.array([o.w(), o.x(), o.y(), o.z()], dtype=np.float32)
        
        def __set__(self, cnp.ndarray[cnp.float32_t, ndim=1] new_orient):
            self.thisptr().sensor_orientation_ = cpp.Quaternionf(
                    new_orient[0],
                    new_orient[1],
                    new_orient[2],
                    new_orient[3])

    @cython.boundscheck(False)
    def from_array(self, cnp.ndarray[cnp.float32_t, ndim=2] arr not None):
        """
        Fill this object from a 2D numpy array (float32)
        """
        assert arr.shape[1] == 4
        
        cdef cnp.npy_intp npts = arr.shape[0]
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1
        
        cdef cpp.PointXYZI *p
        for i in range(npts):
            p = idx.getptr(self.thisptr(), i)
            p.x, p.y, p.z, p.intensity = arr[i, 0], arr[i, 1], arr[i, 2], arr[i, 3]

    @cython.boundscheck(False)
    def to_array(self):
        """
        Return this object as a 2D numpy array (float32)
        """
        cdef float x,y,z
        cdef cnp.npy_intp n = self.thisptr().size()
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result
        cdef cpp.PointXYZI *p
        
        result = np.empty((n, 4), dtype=np.float32)
        for i in range(n):
            p = idx.getptr(self.thisptr(), i)
            result[i, 0] = p.x
            result[i, 1] = p.y
            result[i, 2] = p.z
            result[i, 3] = p.intensity
        return result

    @cython.boundscheck(False)
    def from_list(self, _list):
        """
        Fill this pointcloud from a list of 4-tuples
        """
        cdef Py_ssize_t npts = len(_list)
        cdef cpp.PointXYZI* p
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1
        # OK
        # p = idx.getptr(self.thisptr(), 1)
        # enumerate ? -> i -> type unknown
        for i, l in enumerate(_list):
             p = idx.getptr(self.thisptr(), <int> i)
             p.x, p.y, p.z, p.intensity = l

    def to_list(self):
        """
        Return this object as a list of 4-tuples
        """
        return self.to_array().tolist()

    def resize(self, cnp.npy_intp x):
        if self._view_count > 0:
            raise ValueError("can't resize PointCloud while there are"
                             " arrays/memoryviews referencing it")
        self.thisptr().resize(x)

    def get_point(self, cnp.npy_intp row, cnp.npy_intp col):
        """
        Return a point (4-tuple) at the given row/column
        """
        cdef cpp.PointXYZI *p = idx.getptr_at2(self.thisptr(), row, col)
        return p.x, p.y, p.z, p.intensity

    def __getitem__(self, cnp.npy_intp nmidx):
        cdef cpp.PointXYZI *p = idx.getptr_at(self.thisptr(), nmidx)
        return p.x, p.y, p.z, p.intensity

    def from_file(self, char *f):
        """
        Fill this pointcloud from a file (a local path).
        Only pcd files supported currently.
        
        Deprecated; use pcl.load instead.
        """
        return self._from_pcd_file(f)

    def _from_pcd_file(self, const char *s):
        cdef int error = 0
        with nogil:
            # NG
            # error = pclio.loadPCDFile [cpp.PointXYZI](string(s), <cpp.PointCloud[cpp.PointXYZI]> deref(self.thisptr()))
            error = pclio.loadPCDFile [cpp.PointXYZI](string(s), deref(self.thisptr()))
        return error

    def _from_ply_file(self, const char *s):
        cdef int ok = 0
        with nogil:
            # NG
            # ok = pclio.loadPLYFile [cpp.PointXYZI](string(s), <cpp.PointCloud[cpp.PointXYZI(self.thisptr()))
            ok = pclio.loadPLYFile [cpp.PointXYZI](string(s), deref(self.thisptr()))
        return ok

    def _from_obj_file(self, const char *s):
        cdef int ok = 0
        with nogil:
            # NG
            # ok = pclio.loadOBJFile [cpp.PointXYZI](string(s), <cpp.PointCloud[cpp.PointXYZI]> deref(self.thisptr()))
            ok = pclio.loadOBJFile [cpp.PointXYZI](string(s), deref(self.thisptr()))
        return ok

    def to_file(self, const char *fname, bool ascii=True):
        """Save pointcloud to a file in PCD format.

        Deprecated: use pcl.save instead.
        """
        return self._to_pcd_file(fname, not ascii)

    def _to_pcd_file(self, const char *f, bool binary=False):
        cdef int error = 0
        cdef string s = string(f)
        with nogil:
            # NG
            # error = pclio.savePCDFile [cpp.PointXYZI](s, <cpp.PointCloud[cpp.PointXYZI]> deref(self.thisptr()), binary)
            # OK
            error = pclio.savePCDFile [cpp.PointXYZI](s, deref(self.thisptr()), binary)
            # pclio.PointCloud[cpp.PointXYZI] *p = self.thisptr()
            # error = pclio.savePCDFile [cpp.PointXYZI](s, p, binary)
        return error

    def _to_ply_file(self, const char *f, bool binary=False):
        cdef int error = 0
        cdef string s = string(f)
        with nogil:
            # NG
            # error = pclio.savePLYFile [cpp.PointXYZI](s, <cpp.PointCloud[cpp.PointXYZI]> deref(self.thisptr()), binary)
            error = pclio.savePLYFile [cpp.PointXYZI](s, deref(self.thisptr()), binary)
        return error

    def make_segmenter(self):
        """
        Return a pcl.Segmentation object with this object set as the input-cloud
        """
        seg = Segmentation_PointXYZI()
        cdef pclseg.SACSegmentation_PointXYZI_t *cseg = <pclseg.SACSegmentation_PointXYZI_t *>seg.me
        cseg.setInputCloud(self.thisptr_shared)
        return seg

    def make_segmenter_normals(self, int ksearch=-1, double searchRadius=-1.0):
        """
        Return a pcl.SegmentationNormal object with this object set as the input-cloud
        """
        cdef cpp.PointCloud_Normal_t normals
        mpcl_compute_normals_PointXYZI(<cpp.PointCloud[cpp.PointXYZI]> deref(self.thisptr()), ksearch, searchRadius, normals)
        # p = self.thisptr()
        # mpcl_compute_normals(deref(p), ksearch, searchRadius, normals)
        seg = Segmentation_PointXYZI_Normal()
        cdef pclseg.SACSegmentationFromNormals_PointXYZI_t *cseg = <pclseg.SACSegmentationFromNormals_PointXYZI_t *>seg.me
        cseg.setInputCloud(self.thisptr_shared)
        cseg.setInputNormals (normals.makeShared());
        return seg

    def make_statistical_outlier_filter(self):
        """
        Return a pcl.StatisticalOutlierRemovalFilter object with this object set as the input-cloud
        """
        fil = StatisticalOutlierRemovalFilter_PointXYZI()
        cdef pclfil.StatisticalOutlierRemoval_PointXYZI_t *cfil = <pclfil.StatisticalOutlierRemoval_PointXYZI_t *>fil.me
        cfil.setInputCloud(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZI]]> self.thisptr_shared)
        return fil

    def make_voxel_grid_filter(self):
        """
        Return a pcl.VoxelGridFilter object with this object set as the input-cloud
        """
        fil = VoxelGridFilter_PointXYZI()
        cdef pclfil.VoxelGrid_PointXYZI_t *cfil = <pclfil.VoxelGrid_PointXYZI_t *>fil.me
        cfil.setInputCloud(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZI]]> self.thisptr_shared)
        return fil

    def make_passthrough_filter(self):
        """
        Return a pcl.PassThroughFilter object with this object set as the input-cloud
        """
        fil = PassThroughFilter_PointXYZI()
        cdef pclfil.PassThrough_PointXYZI_t *cfil = <pclfil.PassThrough_PointXYZI_t *>fil.me
        cfil.setInputCloud(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZI]]> self.thisptr_shared)
        return fil

# Error(PointXYZI use?)
#     def make_moving_least_squares(self):
#         """
#         Return a pcl.MovingLeastSquares object with this object as input cloud.
#         """
#         mls = MovingLeastSquares_PointXYZI()
#         cdef pclsf.MovingLeastSquares_PointXYZI_t *cmls = <pclsf.MovingLeastSquares_PointXYZI_t *>mls.me
#         cmls.setInputCloud(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZI]]> self.thisptr_shared)
#         return mls
# 
    def make_kdtree_flann(self):
        """
        Return a pcl.kdTreeFLANN object with this object set as the input-cloud

        Deprecated: use the pcl.KdTreeFLANN constructor on this cloud.
        """
        return KdTreeFLANN_PointXYZI(self)

#     def make_octree(self, double resolution):
#         """
#         Return a pcl.octree object with this object set as the input-cloud
#         """
#         octree = OctreePointCloud_PointXYZI(resolution)
#         octree.set_input_cloud(self)
#         return octree
# 
#     def make_crophull(self):
#         """
#         Return a pcl.CropHull object with this object set as the input-cloud
# 
#         Deprecated: use the pcl.Vertices constructor on this cloud.
#         """
#         return CropHull(self)
#         
#     def make_cropbox(self):
#         """
#         Return a pcl.CropBox object with this object set as the input-cloud
# 
#         Deprecated: use the pcl.Vertices constructor on this cloud.
#         """
#         return CropBox(self)

    def extract(self, pyindices, bool negative=False):
        """
        Given a list of indices of points in the pointcloud, return a 
        new pointcloud containing only those points.
        """
        cdef PointCloud_PointXYZI result
        cdef cpp.PointIndices_t *ind = new cpp.PointIndices_t()
        
        for i in pyindices:
            ind.indices.push_back(i)
        
        result = PointCloud_PointXYZI()
        # (<cpp.PointCloud[cpp.PointXYZI]> deref(self.thisptr())
        mpcl_extract_PointXYZI(self.thisptr_shared, result.thisptr(), ind, negative)
        # XXX are we leaking memory here? del ind causes a double free...
        
        return result
###

