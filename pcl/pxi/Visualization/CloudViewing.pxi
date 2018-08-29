# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization_defs as pcl_vis
from boost_shared_ptr cimport sp_assign

cdef class CloudViewing:
    """
    """
    # cdef pcl_vis.CloudViewer *me
    cdef pcl_vis.CloudViewerPtr_t thisptr_shared

    def __cinit__(self):
        # self.me = new pcl_vis.CloudViewer()
        # sp_assign(<cpp.shared_ptr[pcl_vis.CloudViewer]> self.thisptr_shared, new pcl_vis.CloudViewer('cloud'))
        sp_assign(self.thisptr_shared, new pcl_vis.CloudViewer('cloud'))

    cdef inline pcl_vis.CloudViewer *thisptr(self):
        # Shortcut to get raw pointer to underlying CloudViewer
        return self.thisptr_shared.get()

    def ShowMonochromeCloud(self, _pcl.PointCloud pc, cloudname=b'cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, <string> cloudname)

    def ShowGrayCloud(self, _pcl.PointCloud_PointXYZI pc, cloudname=b'cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, <string> cloudname)

    def ShowColorCloud(self, _pcl.PointCloud_PointXYZRGB pc, cloudname=b'cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, <string> cloudname)

    def ShowColorACloud(self, _pcl.PointCloud_PointXYZRGBA pc, cloudname=b'cloud'):
        self.thisptr().showCloud(pc.thisptr_shared, <string> cloudname)

    def WasStopped(self, int millis_to_wait = 1):
        return self.thisptr().wasStopped(millis_to_wait)

    # def SpinOnce(self, int millis_to_wait = 1):
    #     self.thisptr().spinOnce (millis_to_wait)

    # def OffScreenRendering(bool flag)
    #     self.thisptr().offScreenRendering (flag)
