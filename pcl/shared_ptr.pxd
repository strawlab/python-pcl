from libcpp cimport bool

cdef extern from "boost/smart_ptr/shared_ptr.hpp" namespace "boost":
    cdef cppclass shared_ptr[T]:
        shared_ptr()
        shared_ptr(T*)
        T* get()
        bool unique()
        long use_count()
        void swap(shared_ptr[T])
        void reset(T*)
