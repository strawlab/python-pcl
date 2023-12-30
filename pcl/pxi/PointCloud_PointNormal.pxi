# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
import numpy as np
cimport numpy as cnp

cnp.import_array()

from libcpp cimport bool
cimport indexing as idx
from boost_shared_ptr cimport sp_assign
from _pcl cimport PointCloud_PointNormal

cdef class PointCloud_PointNormal:
    """
    Represents a cloud of points in 4-d space.
    A point cloud can be initialized from either a NumPy ndarray of shape
    (n_points, 4), from a list of triples, or from an integer n to create an
    "empty" cloud of n points.
    To load a point cloud from disk, use pcl.load.
    """
    def __cinit__(self, init=None):
        cdef PointCloud_PointNormal other
        
        self._view_count = 0
        
        # TODO: NG --> import pcl --> pyd Error(python shapedptr/C++ shard ptr collusion?)
        # sp_assign(<cpp.shared_ptr[cpp.PointCloud[cpp.PointNormal]]> self.thisptr_shared, new cpp.PointCloud[cpp.PointNormal]())
        sp_assign(self.thisptr_shared, new cpp.PointCloud[cpp.PointNormal]())
        
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

    def __releasebuffer__(self, Py_buffer *buffer):
        self._view_count -= 1

    # Pickle support. XXX this copies the entire pointcloud; it would be nice
    # to have an asarray member that returns a view, or even better, implement
    # the buffer protocol (https://docs.python.org/c-api/buffer.html).
    def __reduce__(self):
        return type(self), (self.to_array(),)

    @cython.boundscheck(False)
    def from_array(self, cnp.ndarray[cnp.float32_t, ndim=2] arr not None):
        """
        Fill this object from a 2D numpy array (float32)
        """
        assert arr.shape[1] == 7
        
        cdef cnp.npy_intp npts = arr.shape[0]
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1
        
        cdef cpp.PointNormal *p
        for i in range(npts):
            p = idx.getptr(self.thisptr(), i)
            p.x, p.y, p.z, p.normal_x, p.normal_y, p.normal_z, p.curvature = arr[i, 0], arr[i, 1], arr[i, 2], arr[i, 3], arr[i, 4], arr[i, 5], arr[i, 6]

    @cython.boundscheck(False)
    def to_array(self):
        """
        Return this object as a 2D numpy array (float32)
        """
        cdef float x,y,z
        cdef cnp.npy_intp n = self.thisptr().size()
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result
        cdef cpp.PointNormal *p
        
        result = np.empty((n, 7), dtype=np.float32)
        for i in range(n):
            p = idx.getptr(self.thisptr(), i)
            result[i, 0] = p.x
            result[i, 1] = p.y
            result[i, 2] = p.z
            result[i, 3] = p.normal_x
            result[i, 4] = p.normal_y
            result[i, 5] = p.normal_z
            result[i, 6] = p.curvature
        return result

    @cython.boundscheck(False)
    def from_list(self, _list):
        """
        Fill this pointcloud from a list of 4-tuples
        """
        cdef Py_ssize_t npts = len(_list)
        cdef cpp.PointNormal* p
        self.resize(npts)
        self.thisptr().width = npts
        self.thisptr().height = 1
        # OK
        # p = idx.getptr(self.thisptr(), 1)
        # enumerate ? -> i -> type unknown
        for i, l in enumerate(_list):
             p = idx.getptr(self.thisptr(), <int> i)
             p.x, p.y, p.z, p.normal_x, p.normal_y, p.normal_z, p.curvature = l

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
        cdef cpp.PointNormal *p = idx.getptr_at2(self.thisptr(), row, col)
        return p.x, p.y, p.z, p.normal_x, p.normal_y, p.normal_z, p.curvature

    def __getitem__(self, cnp.npy_intp nmidx):
        cdef cpp.PointNormal *p = idx.getptr_at(self.thisptr(), nmidx)
        return p.x, p.y, p.z, p.normal_x, p.normal_y, p.normal_z, p.curvature

    # def extract(self, pyindices, bool negative=False):
    #     """
    #     Given a list of indices of points in the pointcloud, return a 
    #     new pointcloud containing only those points.
    #     """
    #     cdef PointCloud_PointNormal result
    #     cdef cpp.PointIndices_t *ind = new cpp.PointIndices_t()
    #     
    #     for i in pyindices:
    #         ind.indices.push_back(i)
    #     
    #     result = PointCloud_PointNormal()
    #     # (<cpp.PointCloud[cpp.PointNormal]> deref(self.thisptr())
    #     mpcl_extract_Normal(self.thisptr_shared, result.thisptr(), ind, negative)
    #     # XXX are we leaking memory here? del ind causes a double free...
    #     
    #     return result

###

