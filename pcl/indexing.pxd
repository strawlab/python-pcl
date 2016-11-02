# -*- coding: utf-8 -*-
from libc.stddef cimport size_t
cimport pcl_defs as cpp

###############################################################################
# Types
###############################################################################

cdef extern from "indexing.hpp" nogil:
    # Use these instead of operator[] or at.
    cdef cpp.PointXYZ *getptr(cpp.PointCloud[cpp.PointXYZ] *, size_t)
    cdef cpp.PointXYZ *getptr_at(cpp.PointCloud[cpp.PointXYZ] *, size_t) except +
    cdef cpp.PointXYZ *getptr_at2(cpp.PointCloud[cpp.PointXYZ] *, int, int) except +
    cdef cpp.PointXYZI *getptr(cpp.PointCloud[cpp.PointXYZI] *, size_t)
    cdef cpp.PointXYZI *getptr_at(cpp.PointCloud[cpp.PointXYZI] *, size_t) except +
    cdef cpp.PointXYZI *getptr_at2(cpp.PointCloud[cpp.PointXYZI] *, int, int) except +
    cdef cpp.PointXYZRGB *getptr(cpp.PointCloud[cpp.PointXYZRGB] *, size_t)
    cdef cpp.PointXYZRGB *getptr_at(cpp.PointCloud[cpp.PointXYZRGB] *, size_t) except +
    cdef cpp.PointXYZRGB *getptr_at2(cpp.PointCloud[cpp.PointXYZRGB] *, int, int) except +
    cdef cpp.PointXYZRGBA *getptr(cpp.PointCloud[cpp.PointXYZRGBA] *, size_t)
    cdef cpp.PointXYZRGBA *getptr_at(cpp.PointCloud[cpp.PointXYZRGBA] *, size_t) except +
    cdef cpp.PointXYZRGBA *getptr_at2(cpp.PointCloud[cpp.PointXYZRGBA] *, int, int) except +
    # 
    cdef cpp.PointWithViewpoint *getptr(cpp.PointCloud[cpp.PointWithViewpoint] *, size_t)
    cdef cpp.PointWithViewpoint *getptr_at(cpp.PointCloud[cpp.PointWithViewpoint] *, size_t) except +
    cdef cpp.PointWithViewpoint *getptr_at2(cpp.PointCloud[cpp.PointWithViewpoint] *, int, int) except +
    
#     # T *getptr(PointCloud[T] *, size_t)
#     # T *getptr_at(PointCloud[T] *, size_t) except +
#     # T *getptr_at(PointCloud[T] *, int, int) except +
#     # cpdef cpp.PointCloudTypes *getptr(cpp.PointCloud[cpp.PointCloudTypes] *, size_t)
#     # cpdef cpp.PointCloudTypes *getptr_at(cpp.PointCloud[cpp.PointCloudTypes] *, size_t) except +
#     # cpdef cpp.PointCloudTypes *getptr_at(cpp.PointCloud[cpp.PointCloudTypes] *, int, int) except +

# cdef extern from "indexing.hpp" nogil:
#     cdef cppclass getptr[T]:
#         T *getptr(PointCloud[T] *, size_t)
#     cdef cppclass getptr_at[T]:
#         T *getptr_at(PointCloud[T] *, size_t) except +
#         T *getptr_at(PointCloud[T] *, int, int) except +

#cdef extern from "indexing_assign.h" nogil:
#     #void sp_assign(shared_ptr[cpp.PointCloud[cpp.PointXYZ]] &t, cpp.PointCloud[cpp.PointXYZ] *value)
#     #void sp_assign[T](shared_ptr[T] &p, T *value)


###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################

