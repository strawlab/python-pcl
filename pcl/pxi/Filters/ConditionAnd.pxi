# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool
# from libcpp.string cimport string

cimport pcl_defs as cpp
cimport pcl_filters as pcl_fil

from pcl_filters cimport CompareOp2
from boost_shared_ptr cimport shared_ptr
from boost_shared_ptr cimport sp_assign

# cdef class ConditionAnd(ConditionBase):
cdef class ConditionAnd:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in ConditionAnd(pc).
    """
    cdef pcl_fil.ConditionAnd_t *me

    def __cinit__(self):
        self.me = new pcl_fil.ConditionAnd_t()

    def __dealloc__(self):
        del self.me

    # def add_Comparison(self, comparison):
    #     self.me.addComparison(comparison.this_ptr())

    def add_Comparison2(self, field_name, CompareOp2 compOp, double thresh):
        cdef bytes fname_ascii
        if isinstance(field_name, unicode):
            fname_ascii = field_name.encode("ascii")
        elif not isinstance(field_name, bytes):
            raise TypeError("field_name should be a string, got %r" % field_name)
        else:
            fname_ascii = field_name
        
        cdef pcl_fil.FieldComparisonConstPtr_t fieldComp = <pcl_fil.FieldComparisonConstPtr_t> new pcl_fil.FieldComparison_t(string(fname_ascii), compOp, thresh)
        
        # Cython 0.25.2 NG
        # self.me.addComparison(<shared_ptr[const pcl_fil.ComparisonBase[cpp.PointXYZ]]> fieldComp)
        self.me.addComparison(<shared_ptr[const pcl_fil.ComparisonBase[cpp.PointXYZ]]> fieldComp)
        
        # NG
        # sp_assign( self.fieldCompPtr, new pcl_fil.FieldComparison_t(string(fname_ascii), compOp, thresh) )
        # self.me.addComparison(<shared_ptr[const pcl_fil.ComparisonBase[cpp.PointXYZ]]> self.fieldCompPtr)

