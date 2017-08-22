# -*- coding: utf-8 -*-
# main
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization_defs as pclvis
from boost_shared_ptr cimport sp_assign

cdef class PCLHistogramViewing:
    """
    """
    cdef pclvis.PCLHistogramVisualizerPtr_t thisptr_shared

    def __cinit__(self):
        sp_assign(self.thisptr_shared, new pclvis.PCLHistogramVisualizer())

    cdef inline pclvis.PCLHistogramVisualizer *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PCLHistogramVisualizer
        return self.thisptr_shared.get()

    def SpinOnce(self, int time = 1, bool force_redraw = False):
        self.thisptr().spinOnce()

    # self.thisptr().addFeatureHistogram[PointT](shared_ptr[cpp.PointCloud[PointT]] &cloud, int hsize, string cloudname, int win_width, int win_height)
    def AddFeatureHistogram(self, _pcl.PointCloud cloud, int hsize, cloudname, int win_width = 640, int win_height = 200):
        # self.thisptr().addFeatureHistogram[cpp.PointXYZ](<cpp.PointCloudPtr_t> cloud.thisptr_shared, hsize, cloudname, win_width, win_height);
        # self.thisptr().addFeatureHistogram[cpp.PointXYZ](<cpp.PointCloudPtr_t> cloud.thisptr_shared, <string> cloudname, 0, "test", win_width, win_height)
        # NG(build ok use NG)
        # visualization/impl/histogram_visualizer.hpp
        # xy[1] = cloud.points[0].histogram[d]
        # 
        # self.thisptr().addFeatureHistogram[cpp.PointXYZ](<const cpp.PointCloud[cpp.PointXYZ]&> cloud.thisptr_shared, hsize, cloudname, win_width, win_height)
        pass

    # def AddFeatureHistogram(self, _pcl.PointCloud_PointXYZRGB cloud, int hsize, cloudname):
    #     # self.thisptr().addFeatureHistogram[PointT](shared_ptr[cpp.PointCloud[PointT]] &cloud, int hsize, cloudname, int win_width, int win_height)
    #     self.thisptr().addFeatureHistogram(<cpp.PointCloudPtr_t> cloud.thisptr_shared, hsize, <string> cloudname, 640, 200)

    def Spin (self):
        self.thisptr().spin()

