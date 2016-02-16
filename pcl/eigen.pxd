from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cimport pcl_defs as cpp
from boost_shared_ptr cimport shared_ptr
from vector cimport vector as vector2

###############################################################################
# Types
###############################################################################

cdef extern from "Eigen/Eigen" namespace "Eigen" nogil:
    cdef cppclass Matrix4f:
        float *data()
    cdef cppclass Vector4f:
        float *data()
    cdef cppclass Quaternionf:
        float w()
        float x()
        float y()
        float z()
    cdef cppclass aligned_allocator[T]:
        pass

ctypedef aligned_allocator[cpp.PointXYZ] aligned_allocator_t 
ctypedef vector2[cpp.PointXYZ, aligned_allocator_t] AlignedPointTVector_t

