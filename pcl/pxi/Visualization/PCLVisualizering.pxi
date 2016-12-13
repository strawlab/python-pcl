# -*- coding: utf-8 -*-
# main
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization as pclvis
from boost_shared_ptr cimport sp_assign


cdef class PCLVisualizering:
    """
    """
    cdef pclvis.PCLVisualizerPtr_t thisptr_shared

    cdef Py_ssize_t _view_count

    def __cinit__(self, init=None):
        self._view_count = 0
        
        print ('test1')
        sp_assign(self.thisptr_shared, new pclvis.PCLVisualizer())
        print ('test2')
        
        if init is None:
            return
        else:
            raise TypeError("Can't initialize a HistogramVisualizer from a %s" % type(init))

    cdef inline pclvis.PCLVisualizer *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PCLVisualizer
        return self.thisptr_shared.get()

    # def AddPointCloud (self, _pcl.PointCloud cloud):
    #    self.thisptr().addPointCloud(cloud.thisptr_shared)

    # def AddPointCloud (self, _pcl.PointCloud cloud, PointCloudColorHandlerCustom custom):
    #     self.thisptr().addPointCloud(cloud.thisptr_shared)

    # def AddPointCloudNormals(self, _pcl.PointCloud cloud, _pcl.PointNormalCloud normal):
    #     self.thisptr().addPointCloudNormals(cloud.thisptr_shared, normal.thisptr_shared, 10, 0.05, 'normals', 0)

    # def AddCoordinateSystem(self, param):
    #     self.thisptr().addCoordinateSystem(param)

    # def initCameraParameters(self):
    #     self.thisptr().initCameraParameters()

    # def WasStopped(self, millis_to_wait = 1):
    #     self.thisptr().wasStopped(millis_to_wait)

    # def SpinOnce(self, millis_to_wait = 1):
    #     self.thisptr().spinOnce (millis_to_wait)

    def SetBackgroundColor (self, int r, int g, int b):
        self.me.setBackgroundColor(r, g, b)

    # def AddPointCloud(self, pcl._pcl.RangeImage rangeImage, PointCloudColorHandlerCustoms custom, string name):
    # def AddPointCloud(self, RangeImage rangeImage, PointCloudColorHandlerCustoms custom, string name):
    #     self.me.addPointCloud (rangeImage.thisptr(), custom.thisptr(), name)

    def SetPointCloudRenderingProperties(self, int propType, int propValue, string propName):
        self.me.setPointCloudRenderingProperties (propType, propValue, propName)

    def InitCametaParamters(self):
        self.me.initCameraParameters()

