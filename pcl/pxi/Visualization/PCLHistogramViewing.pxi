# -*- coding: utf-8 -*-
# main
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization as pclvis
from boost_shared_ptr cimport sp_assign

cdef class PCLHistogramViewing:
    """
    """
    cdef pclvis.PCLHistogramVisualizerPtr_t thisptr_shared

    cdef Py_ssize_t _view_count

    def __cinit__(self, init=None):
        self._view_count = 0
        
        sp_assign(self.thisptr_shared, new pclvis.PCLHistogramVisualizer())
        
        if init is None:
            return
        else:
            raise TypeError("Can't initialize a PCLHistogramViewing from a %s" % type(init))

    cdef inline pclvis.PCLHistogramVisualizer *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PCLHistogramVisualizer
        return self.thisptr_shared.get()

    def SpinOnce(self, int time = 1, bool force_redraw = False):
        # self.thisptr().spinOnce(time, force_redraw)
        self.thisptr().spinOnce()

    def Spin (self):
        self.thisptr().spin()

    # NG - msg::PointCloud2 batting?
    # def SetBackgroundColor (self, double r, double g, double b, int viewport = 0):
    #     self.thisptr().setBackgroundColor(r, g, b, viewport)
    # 
    # def AddFeatureHistogram (self, _pcl.PointCloud cloud, int hsize, string cloudname, int win_width = 640, int win_height = 200):
    #     self.thisptr().addFeatureHistogram(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> cloud.thisptr_shared, <int>hsize, <string>cloudname, <int>win_width, <int>win_height)
    #     # self.thisptr().addFeatureHistogram(cloud.thisptr_shared, hsize)
    #     # self.thisptr().addFeatureHistogram(cloud.thisptr(), hsize, cloudname, win_width, win_height)
    #     # # self.thisptr().addFeatureHistogram(<cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> cloud.thisptr_shared, hsize, cloudname, win_width, win_height)

