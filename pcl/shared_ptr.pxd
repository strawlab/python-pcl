cimport pcl_defs as cpp
from libcpp cimport bool

cdef extern from "boost/smart_ptr/shared_ptr.hpp" namespace "boost" nogil:
    cdef cppclass shared_ptr[T]:
        shared_ptr()
        shared_ptr(T*)
        T* get()
        bool unique()
        long use_count()
        void swap(shared_ptr[T])
        void reset(T*)

cdef extern from "shared_ptr_assign.h" nogil:
    #void sp_assign(shared_ptr[cpp.PointCloud[cpp.PointXYZ]] &t, cpp.PointCloud[cpp.PointXYZ] *value)
    void sp_assign[T](shared_ptr[T] &p, T *value)
