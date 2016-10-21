# -*- coding: utf-8 -*-
# main
cimport pcl_defs as cpp
cimport numpy as cnp

cdef class Vertices:
    """
    """
    # cdef cpp.Vertices *me

    def __cinit__(self, init=None):
        # cdef cpp.Vertices vertices
        self._view_count = 0
        
        # self.me = new cpp.Vertices()
        # sp_assign(<cpp.shared_ptr[cpp.Vertices]> self.thisptr_shared, new cpp.Vertices())
        sp_assign(self.thisptr_shared, new cpp.Vertices())
        
        if init is None:
            return
        # elif isinstance(init, (numbers.Integral, np.integer)):
        #      self.resize(init)
        # elif isinstance(init, np.ndarray):
        #      self.from_array(init)
        # elif isinstance(init, Sequence):
        #      self.from_list(init)
        # elif isinstance(init, type(self)):
        #      other = init
        #      self.thisptr()[0] = other.thisptr()[0]
        else:
            raise TypeError("Can't initialize a PointCloud from a %s" % type(init))

    # property vertices:
    #     """ property containing the vertices of the Vertices """
    #     def __get__(self): return self.thisptr().vertices

    def __repr__(self):
        return "<Vertices of %d points>" % self.vertices.size()

    @cython.boundscheck(False)
    def from_array(self, cnp.ndarray[cnp.int_t, ndim=1] arr not None):
        """
        Fill this object from a 2D numpy array (float32)
        """
        cdef cnp.npy_intp npts = arr.shape[0]
        
        # cdef cpp.Vertices *p
        for i in range(npts):
            self.thisptr().vertices.push_back(arr[i])

    @cython.boundscheck(False)
    def to_array(self):
        """
        Return this object as a 2D numpy array (float32)
        """
        cdef float index
        cdef cnp.npy_intp n = self.thisptr().vertices.size()
        cdef cnp.ndarray[cnp.int, ndim=1, mode="c"] result
        cdef cpp.Vertices *p
        
        result = np.empty((n, 1), dtype=np.float32)
        for i in range(n):
            result[i, 0] = self.thisptr().vertices.at(i)
        
        return result

    def from_list(self, _list):
        """
        Fill this pointcloud from a list of 3-tuples
        """
        cdef Py_ssize_t npts = len(_list)
        self.resize(npts)
        # self.thisptr().width = npts
        # self.thisptr().height = 1
        cdef cpp.Vertices* p
        # enumerate ? -> i -> type unknown
        # for i, l in enumerate(_list):
        #      p = idx.getptr(self.thisptr(), <int> i)
        #      p.x, p.y, p.z = l

    def to_list(self):
        """
        Return this object as a list of 3-tuples
        """
        return self.to_array().tolist()

    def resize(self, cnp.npy_intp x):
        if self._view_count > 0:
            raise ValueError("can't resize Vertices while there are"
                             " arrays/memoryviews referencing it")
        self.thisptr().vertices.resize(x)
###
