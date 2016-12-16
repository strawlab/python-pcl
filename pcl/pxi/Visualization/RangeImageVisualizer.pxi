# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization as pclvis
from boost_shared_ptr cimport sp_assign

cdef class RangeImageVisualization:
    """
    RangeImageVisualization
    """
    cdef pclvis.RangeImageVisualizer *me
    def __cinit__(self):
        # print('__cinit__')
        self.me = new pclvis.RangeImageVisualizer()
    
    def __dealloc__(self):
        # print('__dealloc__')
        del self.me
    
    # def ShowRangeImage(self, RangeImage range_image):
    #    self.me.showRangeImage(range_image)
    
    # def MarkPoint(self, int ind_width, ind_width, int point, int width):
    #    self.me.markPoint(ind_width, keypoint_indices.points[i], range_image.width)
    
    def SpinOnce (self, int time = 1, bool force_redraw = True):
        self.me.spinOnce(time, force_redraw)
    

