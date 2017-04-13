# -*- coding: utf-8 -*-
from libc.stddef cimport size_t
cimport pcl_defs as cpp
# cimport pcl_PointCloud2_160 as cpp2

###############################################################################
# Types
###############################################################################

cdef extern from "indexing.hpp" nogil:
# cdef extern from "indexing.hpp":
    # Use these instead of operator[] or at.
    PointCloudType *getptr [PointCloudType](cpp.PointCloud[PointCloudType] *, size_t)
    PointCloudType *getptr_at [PointCloudType](cpp.PointCloud[PointCloudType] *, size_t) except +
    PointCloudType *getptr_at2 [PointCloudType](cpp.PointCloud[PointCloudType] *, int, int) except +


#cdef extern from "indexing_assign.h" nogil:
#     #void sp_assign(shared_ptr[cpp.PointCloud[cpp.PointXYZ]] &t, cpp.PointCloud[cpp.PointXYZ] *value)
#     #void sp_assign[T](shared_ptr[T] &p, T *value)


###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################

