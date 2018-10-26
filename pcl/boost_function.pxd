# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
from libcpp cimport bool

###############################################################################
# Types
###############################################################################

# cdef extern from "boost/function/function.hpp" namespace "boost" nogil:
cdef extern from "boost/function.hpp" namespace "boost" nogil:
    cdef cppclass function[T]:
        function()
        function(T*)
        
        T* get()
        bool unique()
        long use_count()
        void swap(shared_ptr[T])
        void reset(T*)

###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################
