# -*- coding: utf-8 -*-
from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cimport pcl_defs as cpp
from boost_shared_ptr cimport shared_ptr
from vector cimport vector as vector2

# Cython C++ wrapper operator() overloading error
# http://stackoverflow.com/questions/18690005/cython-c-wrapper-operator-overloading-error

# Cython/Python/C++ - Inheritance: Passing Derived Class as Argument to Function expecting base class
# http://stackoverflow.com/questions/28573479/cython-python-c-inheritance-passing-derived-class-as-argument-to-function-e
###

###############################################################################
# Types
###############################################################################

# Array
# cdef extern from "Eigen/Array" namespace "Eigen" nogil:
###

# Cholesky
# cdef extern from "Eigen/Cholesky" namespace "Eigen" nogil:
###

# Core?
# cdef extern from "Eigen/Core" namespace "Eigen" nogil:
###

# Dense
# http://stackoverflow.com/questions/29913524/set-coefficient-element-of-eigenmatrix3d-in-cython
cdef extern from "Eigen/Dense" namespace "Eigen":
    # I'm also unsure if you want a Matrix3d or a Vector3d so I assumed matrix
    cdef cppclass Matrix3d:
        Matrix3d() except +
        # NG
        # double coeff(int row, int col)
        double& element "operator()"(int row, int col)


###

# Eigen
cdef extern from "Eigen/Eigen" namespace "Eigen" nogil:
    cdef cppclass Matrix4f:
        Matrix4f() except +
        float *data()
        float& element "operator()"(int row, int col)
    cdef cppclass Matrix3f:
        Matrix3f() except +
        float coeff(int row, int col)
        float *data()
        float& element "operator()"(int row, int col)
    cdef cppclass Vector4f:
        Vector4f() except +
        Vector4f(float c0, float c1, float c2, float c3) except + 
        float *data()
        float& element "operator()"(int row, int col)
    cdef cppclass Vector3f:
        Vector3f() except +
        Vector3f(float c0, float c1, float c2) except + 
        float *data()
        float& element "operator()"(int row, int col)
    cdef cppclass Vector3i:
        Vector3i() except +
        int *data()
        int& element "operator()"(int row, int col)
    cdef cppclass Vector3d: 
        Vector3d() except + 
        Vector3d(double c0, double c1, double c2) except + 
        double coeff(int row, int col)
        double& element "operator()"(int row, int col)
    cdef cppclass Quaternionf:
        Quaternionf()
        Quaternionf(float, float, float, float)
        float w()
        float x()
        float y()
        float z()
    cdef cppclass Affine3f:
        Affine3f() except +
        float *data()
    cdef cppclass aligned_allocator[T]:
        pass


###

# VectorXf

ctypedef aligned_allocator[cpp.PointXYZ] aligned_allocator_t 
ctypedef aligned_allocator[cpp.PointXYZI] aligned_allocator_PointXYZI_t 
ctypedef aligned_allocator[cpp.PointXYZRGB] aligned_allocator_PointXYZRGB_t 
ctypedef aligned_allocator[cpp.PointXYZRGBA] aligned_allocator_PointXYZRGBA_t 
ctypedef vector2[cpp.PointXYZ, aligned_allocator_t] AlignedPointTVector_t
ctypedef vector2[cpp.PointXYZI, aligned_allocator_PointXYZI_t] AlignedPointTVector_PointXYZI_t
ctypedef vector2[cpp.PointXYZRGB, aligned_allocator_PointXYZRGB_t] AlignedPointTVector_PointXYZRGB_t
ctypedef vector2[cpp.PointXYZRGBA, aligned_allocator_PointXYZRGBA_t] AlignedPointTVector_PointXYZRGBA_t

# Eigen2Support?
# cdef extern from "Eigen/Eigen2Support" namespace "Eigen" nogil:

# Eigenvalues
# cdef extern from "Eigen/Eigenvalues" namespace "Eigen" nogil:

# Geometry
cdef extern from "Eigen/Geometry" namespace "Eigen" nogil:
    cdef cppclass Translation2f:
        Translation2f() except +
        Translation2f(float a, float b) except +
    cdef cppclass Translation2d:
        Translation2d() except +
        Translation2d(double a, double b) except +
    cdef cppclass Translation3f:
        Translation3f() except +
        Translation3f(float a, float b, float c) except +
        float *data()
    cdef cppclass Translation3d:
        Translation3d() except +
        Translation3d(double a, double b, double c) except +
        double *data()

# Householder
# cdef extern from "Eigen/Householder" namespace "Eigen" nogil:

# Jacobi
# cdef extern from "Eigen/Jacobi" namespace "Eigen" nogil:

# LeastSquares
# cdef extern from "Eigen/LeastSquares" namespace "Eigen" nogil:

# LU
# cdef extern from "Eigen/LU" namespace "Eigen" nogil:

# QR
# cdef extern from "Eigen/QR" namespace "Eigen" nogil:

# QtAlignedMalloc
# cdef extern from "Eigen/QtAlignedMalloc" namespace "Eigen" nogil:

# Sparse
# cdef extern from "Eigen/Sparse" namespace "Eigen" nogil:

# StdDeque
# cdef extern from "Eigen/StdDeque" namespace "Eigen" nogil:

# StdList
# cdef extern from "Eigen/StdList" namespace "Eigen" nogil:

# StdVector
# cdef extern from "Eigen/StdVector" namespace "Eigen" nogil:

# SVD
# cdef extern from "Eigen/SVD" namespace "Eigen" nogil:

###

