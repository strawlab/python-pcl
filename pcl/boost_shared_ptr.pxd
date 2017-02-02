# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
from libcpp cimport bool

###############################################################################
# Types
###############################################################################

# cdef extern from "boost/smart_ptr/shared_ptr.hpp" namespace "boost" nogil:
cdef extern from "boost/shared_ptr.hpp" namespace "boost" nogil:
    cdef cppclass shared_ptr[T]:
        shared_ptr()
        shared_ptr(T*)
        # shared_ptr(T*, T*)
        # shared_ptr(T*, T*, T*)
        # shared_ptr(weak_ptr[T])
        # shared_ptr(weak_ptr[T], boost::detail::sp_nothrow_tag)
        
        T* get()
        bool unique()
        long use_count()
        void swap(shared_ptr[T])
        void reset(T*)

cdef extern from "boost_shared_ptr_assign.h" nogil:
     # void sp_assign(shared_ptr[cpp.PointCloud[cpp.PointXYZ]] &t, cpp.PointCloud[cpp.PointXYZ] *value)
     void sp_assign[T](shared_ptr[T] &p, T *value)

###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################
