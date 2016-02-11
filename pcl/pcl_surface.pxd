from libcpp cimport bool
from libcpp.vector cimport vector

# main
cimport pcl_defs as cpp
from boost_shared_ptr cimport shared_ptr

cdef extern from "pcl/surface/mls.h" namespace "pcl":
    cdef cppclass MovingLeastSquares[I,O]:
        MovingLeastSquares()
        void setInputCloud (shared_ptr[cpp.PointCloud[I]])
        void setSearchRadius (double)
        void setPolynomialOrder(bool)
        void setPolynomialFit(int)
        void process(cpp.PointCloud[O] &) except +

ctypedef MovingLeastSquares[cpp.PointXYZ, cpp.PointXYZ] MovingLeastSquares_t
ctypedef MovingLeastSquares[cpp.PointXYZRGBA, cpp.PointXYZRGBA] MovingLeastSquares2_t