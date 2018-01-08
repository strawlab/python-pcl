# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
import numpy as np
cimport numpy as cnp

cnp.import_array()

# 
cimport pcl_PCLPointCloud2_180 as pcl_pc2

# parts
cimport pcl_common_180 as pcl_cmn
cimport pcl_features_180 as pclftr
cimport pcl_filters_180 as pclfil
cimport pcl_io_180 as pclio
cimport pcl_kdtree_180 as pclkdt
cimport pcl_octree_180 as pcloct
cimport pcl_sample_consensus_180 as pcl_sc
# cimport pcl_search_180 as pcl_sch
cimport pcl_segmentation_180 as pclseg
cimport pcl_surface_180 as pclsf
cimport pcl_range_image_180 as pcl_r_img
cimport pcl_registration_180 as pcl_reg

from libcpp cimport bool
cimport indexing as idx

from boost_shared_ptr cimport sp_assign

cdef extern from "ProjectInliers.h":
    void mpcl_ProjectInliers_setModelCoefficients(pclfil.ProjectInliers_t) except +

# Empirically determine strides, for buffer support.
# XXX Is there a more elegant way to get these?
# cdef Py_ssize_t _strides_pointcloud2[2]
# cdef PointCloud2 _pc_tmp_pointcloud2 = PointCloud2(np.array([[1, 2, 3],
#                                                [4, 5, 6]], dtype=np.float32))
# 
# cdef cpp.PointCloud[pcl_pc2.PCLPointCloud2] *p_pointcloud2 = _pc_tmp_pointcloud2.thisptr()
# _strides_pointcloud2[0] = (  <Py_ssize_t><void *>idx.getptr(p_pointcloud2, 1)
#                - <Py_ssize_t><void *>idx.getptr(p_pointcloud2, 0))
# _strides_pointcloud2[1] = (  <Py_ssize_t><void *>&(idx.getptr(p_pointcloud2, 0).y)
#                - <Py_ssize_t><void *>&(idx.getptr(p_pointcloud2, 0).x))
# _pc_tmp_pointcloud2 = None

cdef class PCLPointCloud2:
    """Represents a cloud of points in 3-d space.

    A point cloud can be initialized from either a NumPy ndarray of shape
    (n_points, 3), from a list of triples, or from an integer n to create an
    "empty" cloud of n points.

    To load a point cloud from disk, use pcl.load.
    """
    cdef pcl_pc2.PointCloud_PCLPointCloud2Ptr_t thisptr_shared     # XYZ
    
    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count
    
    cdef inline cpp.PointCloud[pcl_pc2.PCLPointCloud2] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloud<PCLPointCloud2>.
        return self.thisptr_shared.get()

    def __cinit__(self, init=None):
        cdef PCLPointCloud2 other
        
        self._view_count = 0
        
        # TODO: NG --> import pcl --> pyd Error(python shapedptr/C++ shard ptr collusion?)
        sp_assign(self.thisptr_shared, new cpp.PointCloud[pcl_pc2.PCLPointCloud2]())
        
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

        # buffer.buf = <char *>&(idx.getptr_at(self.thisptr(), 0).x)
        buffer.format = 'f'
        buffer.internal = NULL
        buffer.itemsize = sizeof(float)
        buffer.len = npoints * 3 * sizeof(float)
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self._shape
        # buffer.strides = _strides
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
        assert arr.shape[1] == 3
        
        cdef cnp.npy_intp npts = arr.shape[0]
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1
        
        cdef pcl_pc2.PCLPointCloud2 *p
        for i in range(npts):
            p = idx.getptr(self.thisptr(), i)
            # p.x, p.y, p.z = arr[i, 0], arr[i, 1], arr[i, 2]
            # bit shift(4byte separate 1byte)
            # = arr[i, 0]
            # p.data.push_back()
            # = arr[i, 0]
            # p.data.push_back()
            # = arr[i, 0]
            # p.data.push_back()
            # = arr[i, 0]
            # p.data.push_back()
            # = arr[i, 1]
            # p.data.push_back()
            # = arr[i, 1]
            # p.data.push_back()
            # = arr[i, 1]
            # p.data.push_back()
            # = arr[i, 1]
            # p.data.push_back()
            # = arr[i, 2]
            # p.data.push_back()
            # = arr[i, 2]
            # p.data.push_back()
            # = arr[i, 2]
            # p.data.push_back()
            # = arr[i, 2]
            # p.data.push_back()

    @cython.boundscheck(False)
    def to_array(self):
        """
        Return this object as a 2D numpy array (float32)
        """
        cdef float x,y,z
        cdef cnp.npy_intp n = self.thisptr().size()
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result
        cdef pcl_pc2.PCLPointCloud2 *p
        
        result = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            pass
            # p = idx.getptr(self.thisptr(), i)
            # bit shift
            result[i, 0] = p.data[i * 12 + 0 * 4 + 0] + p.data[i * 12 + 0 * 4 + 1] + p.data[i * 12 + 0 * 4 + 2] + p.data[i * 12 + 0 * 4 + 3]
            result[i, 1] = p.data[i * 12 + 1 * 4 + 0] + p.data[i * 12 + 1 * 4 + 1] + p.data[i * 12 + 1 * 4 + 2] + p.data[i * 12 + 1 * 4 + 3]
            result[i, 2] = p.data[i * 12 + 2 * 4 + 0] + p.data[i * 12 + 2 * 4 + 1] + p.data[i * 12 + 2 * 4 + 2] + p.data[i * 12 + 2 * 4 + 3]
        
        return result

    def from_list(self, _list):
        """
        Fill this pointcloud from a list of 3-tuples
        """
        cdef Py_ssize_t npts = len(_list)
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1
        cdef pcl_pc2.PCLPointCloud2* p
        # OK
        # p = idx.getptr(self.thisptr(), 1)
        # enumerate ? -> i -> type unknown
        for i, l in enumerate(_list):
            pass
            # p = idx.getptr(self.thisptr(), <int> i)
            # p.x, p.y, p.z = l

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
        # cdef pcl_pc2.PCLPointCloud2 *p = idx.getptr_at2(self.thisptr(), row, col)
        # return p.x, p.y, p.z
        return 0.0, 0.0, 0.0

    def __getitem__(self, cnp.npy_intp nmidx):
        # cdef pcl_pc2.PCLPointCloud2 *p = idx.getptr_at(self.thisptr(), nmidx)
        # return p.x, p.y, p.z
        return 0.0, 0.0, 0.0

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
            # error = pclio.loadPCDFile(string(s), <cpp.PointCloud[pcl_pc2.PCLPointCloud2]> deref(self.thisptr()))
            # error = pclio.loadPCDFile(string(s), deref(self.thisptr()))
            pass
        
        return error

    def _from_ply_file(self, const char *s):
        cdef int ok = 0
        with nogil:
            # NG
            # ok = pclio.loadPLYFile(string(s), <cpp.PointCloud[pcl_pc2.PCLPointCloud2]> deref(self.thisptr()))
            # ok = pclio.loadPLYFile(string(s), deref(self.thisptr()))
            pass
        
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
            # OK
            # error = pclio.savePCDFile(s, deref(self.thisptr()), binary)
            pass
        
        return error

    def _to_ply_file(self, const char *f, bool binary=False):
        cdef int error = 0
        cdef string s = string(f)
        with nogil:
            # error = pclio.savePLYFile(s, deref(self.thisptr()), binary)
            pass
        
        return error

    # def copyPointCloud(self, vector[int] indices):
    #     cloud_out = PointCloud()
    #     # NG : Function Override Error
    #     # pcl_cmn.copyPointCloud_Indices [pcl_pc2.PCLPointCloud2](self.thisptr_shared, <vector[int]> indices, <cpp.shared_ptr[cpp.PointCloud[pcl_pc2.PCLPointCloud2]]> cloud_out.thisptr_shared)
    #     # pcl_cmn.copyPointCloud_Indices [pcl_pc2.PCLPointCloud2](self.thisptr_shared.get(), <vector[int]> indices, cloud_out.thisptr_shared.get())
    #     # pcl_cmn.copyPointCloud_Indices [pcl_pc2.PCLPointCloud2](self.thisptr_shared.get(), <const vector[int]> &indices, deref(cloud_out.thisptr_shared.get()))
    #     pcl_cmn.copyPointCloud_Indices [pcl_pc2.PCLPointCloud2](<const shared_ptr[PointCloud[PointCloud2]]> self.thisptr_shared, <const vector[int]> &indices, deref(cloud_out.thisptr_shared))
    #     
    #     return cloud_out


###

