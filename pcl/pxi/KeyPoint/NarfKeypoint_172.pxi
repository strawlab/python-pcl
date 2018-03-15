# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_keypoints_172 as pcl_kp

cdef class NarfKeypoint:
    """
    """
    cdef pcl_kp.NarfKeypoint_t *me

    def __cinit__(self, RangeImageBorderExtractor pc not None):
        self.me = <pcl_kp.NarfKeypoint>new pcl_kp.NarfKeypoint(pc, -1.0)

    def __dealloc__(self):
        del self.me

