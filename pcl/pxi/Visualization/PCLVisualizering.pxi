# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport pcl_visualization as pclvis
from boost_shared_ptr cimport sp_assign


cdef class PCLVisualizering:
    """
    """
    cdef pclvis.PCLVisualizerPtr_t thisptr_shared
    
    def __cinit__(self):
        sp_assign(self.thisptr_shared, new pclvis.PCLVisualizer('visual', True))

    cdef inline pclvis.PCLVisualizer *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PCLVisualizer
        return self.thisptr_shared.get()

    def AddCoordinateSystem(self, double scale = 1.0, int viewpoint = 0):
        self.thisptr().addCoordinateSystem(scale, viewpoint)

    def AddCoordinateSystem(self, double scale, float x, float y, float z, int viewpoint = 0):
        self.thisptr().addCoordinateSystem(scale, x, y, z, viewpoint)

    def WasStopped(self):
        self.thisptr().wasStopped()

    def ResetStoppedFlag(self):
        self.thisptr().resetStoppedFlag()

    def SpinOnce(self, int millis_to_wait = 1, bool force_redraw = False):
        self.thisptr().spinOnce (millis_to_wait, force_redraw)

    def SetBackgroundColor (self, int r, int g, int b):
        self.thisptr().setBackgroundColor(r, g, b, 0)

    def AddPointCloud (self, _pcl.PointCloud cloud, string id = "cloud", int viewport = 0):
        self.thisptr().addPointCloud(<cpp.PointCloudPtr_t> cloud.thisptr_shared, id, viewport)

    # <const shared_ptr[PointCloudColorHandler[PointT]]> 
    def AddPointCloud_ColorHandler(self, _pcl.PointCloud cloud, PointCloudColorHandleringCustom color_handler, string id):
        self.thisptr().addPointCloud(<cpp.PointCloudPtr_t> cloud.thisptr_shared, color_handler.thisptr_shared, id, 0)

    # def AddPointCloudNormals(self, _pcl.PointCloud cloud, _pcl.PointCloud_Normal normal):
    #     self.thisptr().addPointCloudNormals(<cpp.PointCloudPtr_t> cloud.thisptr_shared, <cpp.PointCloud_Normal_Ptr_t> normal.thisptr_shared, 10, 0.05, 'normals', 0)

    def SetPointCloudRenderingProperties(self, int propType, int propValue, string propName = 'cloud'):
        self.thisptr().setPointCloudRenderingProperties (propType, propValue, propName, 0)

    def InitCameraParameters(self):
        self.thisptr().initCameraParameters()


