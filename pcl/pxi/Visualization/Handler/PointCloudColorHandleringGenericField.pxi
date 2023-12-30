# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization_defs as pcl_vis
from boost_shared_ptr cimport sp_assign

cdef class PointCloudColorHandleringGenericField:
    """
    """
    cdef pcl_vis.PointCloudColorHandlerGenericField_t *me
    
    def __cinit__(self):
        print('__cinit__')
        pass
    
    def __dealloc__(self):
        print('__dealloc__')
        # del self.me
        pass



