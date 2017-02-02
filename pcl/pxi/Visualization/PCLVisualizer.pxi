# -*- coding: utf-8 -*-
# main
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization as pclvis
from boost_shared_ptr cimport sp_assign


cdef class PCLVisualizer:
    """
    """
    # cdef pclvis.PCLVisualizerPtr_t thisptr_shared
    cdef pclvis.PCLVisualizer *me
    cdef Py_ssize_t _view_count
    
    # def __cinit__(self, init=None):
    #     self._view_count = 0
    #     print ('test1')
    #     sp_assign(self.thisptr_shared, new pclvis.PCLVisualizer())
    #     print ('test2')
    #     if init is None:
    #         return
    #     else:
    #         raise TypeError("Can't initialize a HistogramVisualizer from a %s" % type(init))

    # cdef inline pclvis.PCLVisualizer *thisptr(self) nogil:
    #     # Shortcut to get raw pointer to underlying PCLVisualizer
    #     return self.thisptr_shared.get()

    def __cinit__(self):
        self.me = new pclvis.PCLVisualizer()

    def __dealloc__(self):
        del self.me

    def AddCoordinateSystem(self, double scale = 1.0, int viewpoint = 0):
        self.me.addCoordinateSystem(scale, viewpoint)

    def AddCoordinateSystem(self, double scale, float x, float y, float z, int viewpoint = 0):
        self.me.addCoordinateSystem(scale, x, y, z, viewpoint)

    def WasStopped(self):
         self.me.wasStopped()

    def ResetStoppedFlag(self):
         self.me.resetStoppedFlag()

    def SpinOnce(self, int millis_to_wait = 1, bool force_redraw = False):
        self.me.spinOnce (millis_to_wait, force_redraw)

    def SetBackgroundColor (self, int r, int g, int b):
        self.me.setBackgroundColor(r, g, b, 0)

    # def AddPointCloud(self, RangeImage rangeImage, PointCloudColorHandlerCustoms custom, string name):
    # def AddPointCloud(self, RangeImage rangeImage, PointCloudColorHandlerCustoms custom, string name):
    #     self.me.addPointCloud (rangeImage.me, custom.thisptr(), name)

    def AddPointCloud (self, _pcl.PointCloud cloud, string id = "cloud", int viewport = 0):
        self.me.addPointCloud(cloud.thisptr_shared, id, viewport)

    # Add a Point Cloud (templated) to screen.
    # def AddPointCloud (self, _pcl.PointCloud cloud, PointCloudColorHandlerCustom custom, string id = "cloud", int viewport = 0):
    #     self.me.addPointCloud(cloud.thisptr_shared, custom.thisptr_shared, id, viewport)

    def AddPointCloudNormals(self, _pcl.PointCloud cloud, _pcl.PointNormalCloud normal):
          self.me.addPointCloudNormals(cloud.thisptr_shared, normal.thisptr_shared, 10, 0.05, 'normals', 0)

    def SetPointCloudRenderingProperties(self, int propType, int propValue, string propName = 'cloud'):
        self.me.setPointCloudRenderingProperties (propType, propValue, propName, 0)

    def InitCameraParameters(self):
        self.me.initCameraParameters()

