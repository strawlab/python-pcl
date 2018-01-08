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
    ## void mpcl_extract_HarrisKeypoint3D(cpp.PointCloudPtr_t, cpp.PointCloud_ReferenceFrame *) except +
    # void mpcl_extract_HarrisKeypoint3D(cpp.PointCloudPtr_t, cpp.PointCloud_t *) except +


cdef extern from "ProjectInliers.h":
    void mpcl_ProjectInliers_setModelCoefficients(pclfil.ProjectInliers_t) except +

# Empirically determine strides, for buffer support.
# XXX Is there a more elegant way to get these?
cdef Py_ssize_t _strides[2]
cdef PointCloud _pc_tmp = PointCloud(np.array([[1, 2, 3],
                                               [4, 5, 6]], dtype=np.float32))

cdef cpp.PointCloud[cpp.ReferenceFrame] *p = _pc_tmp.thisptr()
_strides[0] = (  <Py_ssize_t><void *>idx.getptr(p, 1)
               - <Py_ssize_t><void *>idx.getptr(p, 0))
_strides[1] = (  <Py_ssize_t><void *>&(idx.getptr(p, 0).y)
               - <Py_ssize_t><void *>&(idx.getptr(p, 0).x))
_pc_tmp = None

cdef class PointCloud_ReferenceFrame:
    """Represents a cloud of points in 3-d space.

    A point cloud can be initialized from either a NumPy ndarray of shape
    (n_points, 3), from a list of triples, or from an integer n to create an
    "empty" cloud of n points.

    To load a point cloud from disk, use pcl.load.
    """
    def __cinit__(self, init=None):
        cdef PointCloud other
        
        self._view_count = 0
        
        # TODO: NG --> import pcl --> pyd Error(python shapedptr/C++ shard ptr collusion?)
        # sp_assign(<cpp.shared_ptr[cpp.PointCloud[cpp.ReferenceFrame]]> self.thisptr_shared, new cpp.PointCloud[cpp.ReferenceFrame]())
        sp_assign(self.thisptr_shared, new cpp.PointCloud[cpp.ReferenceFrame]())
        
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

    # cdef inline PointCloud[ReferenceFrame] *thisptr(self) nogil:
    #     # Shortcut to get raw pointer to underlying PointCloud
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
        
        cdef cpp.ReferenceFrame *p
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
        cdef cpp.ReferenceFrame *p
        
        result = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            p = idx.getptr(self.thisptr(), i)
            result[i, 0] = p.x
            result[i, 1] = p.y
            result[i, 2] = p.z
        
        return result

    def from_list(self, _list):
        """
        Fill this pointcloud from a list of 3-tuples
        """
        cdef Py_ssize_t npts = len(_list)
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1
        cdef cpp.ReferenceFrame* p
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
            raise ValueError("can't resize PointCloud while there are"
                             " arrays/memoryviews referencing it")
        self.thisptr().resize(x)

    def get_point(self, cnp.npy_intp row, cnp.npy_intp col):
        """
        Return a point (3-tuple) at the given row/column
        """
        cdef cpp.ReferenceFrame *p = idx.getptr_at2(self.thisptr(), row, col)
        return p.x, p.y, p.z

    def __getitem__(self, cnp.npy_intp nmidx):
        cdef cpp.ReferenceFrame *p = idx.getptr_at(self.thisptr(), nmidx)
        return p.x, p.y, p.z

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
            # error = pclio.loadPCDFile(string(s), <cpp.PointCloud[cpp.ReferenceFrame]> deref(self.thisptr()))
            error = pclio.loadPCDFile(string(s), deref(self.thisptr()))
        return error

    def _from_ply_file(self, const char *s):
        cdef int ok = 0
        with nogil:
            # NG
            # ok = pclio.loadPLYFile(string(s), <cpp.PointCloud[cpp.ReferenceFrame]> deref(self.thisptr()))
            ok = pclio.loadPLYFile(string(s), deref(self.thisptr()))
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
            # error = pclio.savePCDFile(s, <cpp.PointCloud[cpp.ReferenceFrame]> deref(self.thisptr()), binary)
            # OK
            error = pclio.savePCDFile(s, deref(self.thisptr()), binary)
            # pclio.PointCloud[cpp.ReferenceFrame] *p = self.thisptr()
            # error = pclio.savePCDFile(s, p, binary)
        return error

    def _to_ply_file(self, const char *f, bool binary=False):
        cdef int error = 0
        cdef string s = string(f)
        with nogil:
            # NG
            # error = pclio.savePLYFile(s, <cpp.PointCloud[cpp.ReferenceFrame]> deref(self.thisptr()), binary)
            error = pclio.savePLYFile(s, deref(self.thisptr()), binary)
        return error

