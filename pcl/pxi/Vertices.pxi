
# main
cimport pcl_defs as cpp

cdef class Vertices:
    """
    """
    cdef cpp.shared_ptr[cpp.Vertices] thisptr_shared
    cdef cpp.Vertices *me

    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count

    def __cinit__(self, init=None):
        # cdef cpp.Vertices vertices
        self._view_count = 0
        
        self.me = new cpp.Vertices()
        # sp_assign(<cpp.shared_ptr[cpp.Vertices]> self.thisptr_shared, new cpp.Vertices())
        # sp_assign(<cpp.shared_ptr[cpp.Vertices]> self.thisptr_shared, new cpp.Vertices())
        
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

    property vertices:
        """ property containing the vertices of the Vertices """
        def __get__(self): return self.me.vertices

    def __repr__(self):
        return "<PointCloud of %d points>" % self.size

#     @cython.boundscheck(False)
#     def from_array(self, cnp.ndarray[cnp.float32_t, ndim=2] arr not None):
#         """
#         Fill this object from a 2D numpy array (float32)
#         """
#         assert arr.shape[1] == 3
# 
#         cdef cnp.npy_intp npts = arr.shape[0]
#         self.resize(npts)
#         self.thisptr().width = npts
#         self.thisptr().height = 1
# 
#         cdef cpp.PointXYZ *p
#         for i in range(npts):
#             p = idx.getptr(self.thisptr(), i)
#             p.x, p.y, p.z = arr[i, 0], arr[i, 1], arr[i, 2]
# 
#     @cython.boundscheck(False)
#     def to_array(self):
#         """
#         Return this object as a 2D numpy array (float32)
#         """
#         cdef float x,y,z
#         cdef cnp.npy_intp n = self.thisptr().size()
#         cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result
#         cdef cpp.PointXYZ *p
# 
#         result = np.empty((n, 3), dtype=np.float32)
#         for i in range(n):
#             p = idx.getptr(self.thisptr(), i)
#             result[i, 0] = p.x
#             result[i, 1] = p.y
#             result[i, 2] = p.z
# 
#         return result
# 
#     def from_list(self, _list):
#         """
#         Fill this pointcloud from a list of 3-tuples
#         """
#         cdef Py_ssize_t npts = len(_list)
#         self.resize(npts)
#         self.thisptr().width = npts
#         self.thisptr().height = 1
#         cdef cpp.PointXYZ* p
#         # OK
#         # p = idx.getptr(self.thisptr(), 1)
#         # enumerate ? -> i -> type unknown
#         for i, l in enumerate(_list):
#              p = idx.getptr(self.thisptr(), <int> i)
#              p.x, p.y, p.z = l
# 
#     def to_list(self):
#         """
#         Return this object as a list of 3-tuples
#         """
#         return self.to_array().tolist()
# 
#     def resize(self, cnp.npy_intp x):
#         if self._view_count > 0:
#             raise ValueError("can't resize PointCloud while there are"
#                              " arrays/memoryviews referencing it")
#         self.thisptr().resize(x)
###
