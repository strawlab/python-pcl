from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.string cimport string

cimport pcl_defs as cpp
cimport pcl_filters as pclfil

from boost_shared_ptr cimport shared_ptr

cdef class FieldComparison:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in ConditionAnd(pc).
    """
    cdef pclfil.FieldComparison_t *me

    def __cinit__(self, string field_name, CompareOp2 op, double thresh):
        self.me = new pclfil.FieldComparison_t(field_name, op, thresh)

    def __dealloc__(self):
        del self.me



