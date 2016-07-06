from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_filters as pclfil

from boost_shared_ptr cimport shared_ptr

cdef class ConditionAnd:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in ConditionAnd(pc).
    """
    cdef pclfil.ConditionAnd_t *me

    def __cinit__(self):
        self.me = new pclfil.ConditionAnd_t()

    def __dealloc__(self):
        del self.me

    def add_Comparison(fieldCompressPtr):
        self.me.addComparison(fieldCompressPtr)


