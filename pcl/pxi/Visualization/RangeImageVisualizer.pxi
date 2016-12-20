# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization as pcl_vis
from boost_shared_ptr cimport sp_assign
# from pcl_range_image cimport RangeImage


cdef class RangeImageVisualization:
    """
    RangeImageVisualization
    """
    cdef pcl_vis.RangeImageVisualizer *me
    def __cinit__(self):
        # print('__cinit__')
        self.me = new pcl_vis.RangeImageVisualizer()
    
    def __dealloc__(self):
        # print('__dealloc__')
        del self.me
    
    # -std::numeric_limits<float>::infinity ()
    #  std::numeric_limits<float>::infinity ()
    def ShowRangeImage (self, _pcl.RangeImages range_image, float min_value = -99999.0, float max_value = 99999.0, bool grayscale = False):
         # self.me.showRangeImage(range_image, min_value, max_value, grayscale)
         # self.me.showRangeImage(range_image.thisptr(), min_value, max_value, grayscale)
         cdef pcl_r_img.RangeImage_t user
         user = <pcl_r_img.RangeImage_t> range_image.thisptr()[0]
         self.me.showRangeImage(user, min_value, max_value, grayscale)
    
    # def MarkPoint(self, int ind_width, int point, int width):
    #   self.me.markPoint(ind_width, point, width)
    
    def SpinOnce (self, int time = 1, bool force_redraw = True):
        self.me.spinOnce(time, force_redraw)
    

