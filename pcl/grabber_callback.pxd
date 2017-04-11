# -*- coding: utf-8 -*-
# cimport pcl_defs as cpp
from libcpp cimport bool

ctypedef double (*Method)(void *param, void *user_data)

cdef extern from "grabber_callback.h" namespace "grabber_callback":
    cdef cppclass PyLibCallBack:
        PyLibCallBack(Method method, void *user_data)
        double cy_execute(void *parameter)

# The pattern/converter method to be used for translating C typed prototype to a Python object call
cdef inline double callback(void *parameter, void *method):
    return (<object>method)(<object>parameter)
