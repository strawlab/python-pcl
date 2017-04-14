# cimport pcl_defs as cpp

cdef extern from "boost/function.hpp" namespace "boost":
  cdef cppclass function[T]:
    pass

cdef extern from "boost/bind.hpp" namespace "boost":
  cdef struct arg:
    pass
  cdef function[T] bind[T](T callback, arg _1)

cdef extern from "boost/signals2.hpp" namespace "boost::signals2":
  cdef cppclass connection:
    pass

# 
ctypedef void callback_t(void*)
# ctypedef void callback2_t(cpp.PointCloud_Ptr_t)

cdef extern from "bind.h":
  cdef connection register_callback(function[callback_t] callback)
