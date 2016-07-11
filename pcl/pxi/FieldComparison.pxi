from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_filters as pclfil

from pcl_filters cimport CompareOp
from boost_shared_ptr cimport shared_ptr

cdef class FieldComparison:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in ConditionAnd(pc).
    """
    cdef pclfil.FieldComparison_t *me

    def __cinit__(self, string axis, CompareOp op, double param):
        self.me = new pclfil.FieldComparison_t(axis, op, param)

    def __dealloc__(self):
        del self.me



