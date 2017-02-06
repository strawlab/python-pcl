# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization as pclvis
from boost_shared_ptr cimport sp_assign

cdef class CloudViewing:
    """
    """
    # cdef pclvis.CloudViewer *me
    cdef pclvis.CloudViewerPtr_t thisptr_shared

    def __cinit__(self):
        # self.me = new pclvis.CloudViewer()
        # sp_assign(<cpp.shared_ptr[pclvis.CloudViewer]> self.thisptr_shared, new pclvis.CloudViewer('cloud'))
        sp_assign(self.thisptr_shared, new pclvis.CloudViewer('cloud'))

    cdef inline pclvis.CloudViewer *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying CloudViewer
        return self.thisptr_shared.get()

    def ShowMonochromeCloud(self, _pcl.PointCloud pc, string cloudname='cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, cloudname)

    def ShowGrayCloud(self, _pcl.PointCloud_PointXYZI pc, string cloudname='cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, cloudname)

    def ShowColorCloud(self, _pcl.PointCloud_PointXYZRGB pc, string cloudname='cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, cloudname)

    def ShowColorACloud(self, _pcl.PointCloud_PointXYZRGBA pc, string cloudname='cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, cloudname)

    def WasStopped(self, int millis_to_wait = 1):
        return self.thisptr().wasStopped(millis_to_wait)

    # def SpinOnce(self, int millis_to_wait = 1):
    #     self.thisptr().spinOnce (millis_to_wait)

    # def OffScreenRendering(bool)
    # 