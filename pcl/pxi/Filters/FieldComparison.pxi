# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.string cimport string

cimport pcl_defs as cpp
cimport pcl_filters as pclfil

from pcl_filters cimport CompareOp2
from boost_shared_ptr cimport shared_ptr

cdef class FieldComparison:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in ConditionAnd(pc).
    """
    cdef pclfil.FieldComparison_t *me

    def __cinit__(self, field_name, CompareOp2 op, double thresh):
        cdef bytes fname_ascii
        if isinstance(field_name, unicode):
            fname_ascii = field_name.encode("ascii")
        elif not isinstance(field_name, bytes):
            raise TypeError("field_name should be a string, got %r"
                            % field_name)
        else:
            fname_ascii = field_name

        self.me = new pclfil.FieldComparison_t(field_name, op, thresh)

    def __dealloc__(self):
        del self.me



