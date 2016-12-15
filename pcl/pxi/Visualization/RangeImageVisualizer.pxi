# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization as pclvis
from boost_shared_ptr cimport sp_assign

cdef class RangeImageVisualizer:
    """
    RangeImageVisualizer
    """
    cdef pclvis.CloudViewerPtr_t thisptr_shared
    # cdef pclvis.CloudViewer *me

    cdef Py_ssize_t _view_count

    def __cinit__(self):
        self._view_count = 0
        
        # self.me = new pclvis.CloudViewer()
        # sp_assign(<cpp.shared_ptr[pclvis.CloudViewer]> self.thisptr_shared, new pclvis.CloudViewer('cloud'))
        sp_assign(self.thisptr_shared, new pclvis.CloudViewer('cloud'))
        

    cdef inline pclvis.CloudViewer *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying CloudViewer
        return self.thisptr_shared.get()

    # def __repr__(self):
    #     return "<CloudViewer of %d points>" % self.vertices.size()

    def ShowMonochromeCloud(self, _pcl.PointCloud pc, string cloudname='cloud'):
        # cdef cpp.PointCloudPtr_t tmpPoint
        # tmpPoint = <cpp.PointCloudPtr_t> pc
        # sp_assign(tmpPoint, pc)
        # NG
        # self.thisptr().showCloud(pc, cloudname)
        # self.thisptr().showCloud(deref(pc), cloudname)
        self.thisptr().showCloud(pc.thisptr_shared, cloudname)

    def ShowGrayCloud(self, _pcl.PointCloud_PointXYZI pc, string cloudname='cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, cloudname)

    def ShowColorCloud(self, _pcl.PointCloud_PointXYZRGB pc, string cloudname='cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, cloudname)

    def ShowColorACloud(self, _pcl.PointCloud_PointXYZRGBA pc, string cloudname='cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, cloudname)

    def WasStopped(self, int millis_to_wait = 1):
        self.thisptr().wasStopped(millis_to_wait)

    # def SpinOnce(self, int millis_to_wait = 1):
    #     self.thisptr().spinOnce (millis_to_wait)

    # def OffScreenRendering(bool)
    # 
