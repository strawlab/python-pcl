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
    cdef cppclass Matrix3f:
        float *data()
    cdef cppclass Vector4f:
        float *data()
    cdef cppclass Vector3f:
        float *data()
    cdef cppclass Vector3i:
        int *data()
    cdef cppclass Quaternionf:
        float w()
        float x()
        float y()
        float z()
    cdef cppclass Affine3f:
        float *data()
    cdef cppclass aligned_allocator[T]:
        pass

# VectorXf

ctypedef aligned_allocator[cpp.PointXYZ] aligned_allocator_t 
ctypedef aligned_allocator[cpp.PointXYZI] aligned_allocator_PointXYZI_t 
ctypedef aligned_allocator[cpp.PointXYZRGB] aligned_allocator_PointXYZRGB_t 
ctypedef aligned_allocator[cpp.PointXYZRGBA] aligned_allocator_PointXYZRGBA_t 
ctypedef vector2[cpp.PointXYZ, aligned_allocator_t] AlignedPointTVector_t
ctypedef vector2[cpp.PointXYZI, aligned_allocator_PointXYZI_t] AlignedPointTVector_PointXYZI_t
ctypedef vector2[cpp.PointXYZRGB, aligned_allocator_PointXYZRGB_t] AlignedPointTVector_PointXYZRGB_t
ctypedef vector2[cpp.PointXYZRGBA, aligned_allocator_PointXYZRGBA_t] AlignedPointTVector_PointXYZRGBA_t

# http://stackoverflow.com/questions/29913524/set-coefficient-element-of-eigenmatrix3d-in-cython
# cdef extern from "eigen3/Eigen/Dense" namespace "Eigen":
#     # I'm also unsure if you want a Matrix3d or a Vector3d
#     # so I assumed matrix
#     cdef cppclass Matrix3d:
#         Matrix3d() except +
#         double& element "operator()"(int row,int col)
# 
# NG
# cdef extern from "Eigen/Dense" namespace "Eigen":
#     cdef cppclass Vector3d:
#         Matrix3d() except +
#         double coeff(int row, int col)

# Cython C++ wrapper operator() overloading error
# http://stackoverflow.com/questions/18690005/cython-c-wrapper-operator-overloading-error

# Cython/Python/C++ - Inheritance: Passing Derived Class as Argument to Function expecting base class
# http://stackoverflow.com/questions/28573479/cython-python-c-inheritance-passing-derived-class-as-argument-to-function-e