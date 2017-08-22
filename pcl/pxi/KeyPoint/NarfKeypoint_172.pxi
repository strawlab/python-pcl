# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_keypoints_172 as pclkp

cdef class NarfKeypoint:
    """
    """
    cdef pclkp.NarfKeypoint_t *me

    def __cinit__(self, RangeImageBorderExtractor pc not None):
        self.me = <pclkp.NarfKeypoint>new pclkp.NarfKeypoint(pc, -1.0)

    def __dealloc__(self):
        del self.me

