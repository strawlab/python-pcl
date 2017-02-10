# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport cython
cimport pcl_visualization

cimport pcl_visualization_defs as pclvis
from boost_shared_ptr cimport shared_ptr
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

    # def AddPointCloud (self, _pcl.PointCloud cloud, str id, int viewport):
    def AddPointCloud (self, _pcl.PointCloud cloud, string id = 'cloud', int viewport = 0):
        # NG : Cython 25.0.2
        # self.thisptr().addPointCloud(<cpp.PointCloudPtr_t> cloud.thisptr_shared, id, viewport)
        # self.thisptr().addPointCloud(<cpp.PointCloudPtr_t> cloud.thisptr_shared, <string> id, <int> viewport)
        # [cpp.PointCloudPtr_t, string, int]
        # self.thisptr().addPointCloud[cpp.pointXYZ](cloud.thisptr_shared, id, viewport)
        # overloaded function setting
        # self.thisptr().addPointCloud[cpp.PointXYZ](cloud.thisptr_shared, id, viewport)
        # self.thisptr().addPointCloud[cpp.PointXYZ](<const shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> cython.address(cloud.thisptr_shared), <const string> id, <int> viewport)
        # self.thisptr().addPointCloud[cpp.PointXYZ](<const shared_ptr[const cpp.PointCloud[cpp.PointXYZ]]> cloud.thisptr_shared, <string> id, <int> viewport)
        self.thisptr().addPointCloud(cloud.thisptr_shared, id, viewport)

    # <const shared_ptr[PointCloudColorHandler[PointT]]> 
    def AddPointCloud_ColorHandler(self, _pcl.PointCloud cloud, pcl_visualization.PointCloudColorHandleringCustom color_handler, string id = 'cloud', int viewport = 0):
    # def AddPointCloud_ColorHandler(self, _pcl.PointCloud cloud, pcl_visualization.PointCloudColorHandleringCustom color_handler, str id = 'cloud', int viewport = 0):
        # NG : Base Class
        # self.thisptr().addPointCloud[cpp.PointXYZ](cloud.thisptr_shared, <const pclvis.PointCloudColorHandler[cpp.PointXYZ]> deref(color_handler.thisptr_shared.get()), id, viewport)
        # OK? : Inheritance Class(PointCloudColorHandler)
        # self.thisptr().addPointCloud[cpp.PointXYZ](cloud.thisptr_shared, <const pclvis.PointCloudColorHandlerCustom[cpp.PointXYZ]> deref(color_handler.thisptr_shared.get()), id, viewport)
        self.thisptr().addPointCloud[cpp.PointXYZ](cloud.thisptr_shared, <const pclvis.PointCloudColorHandlerCustom[cpp.PointXYZ]> deref(color_handler.thisptr_shared.get()), id, viewport)
        pass

    # <const shared_ptr[PointCloudGeometryHandler[PointT]]> 
    # def AddPointCloud_GeometryHandler(self, _pcl.PointCloud cloud, pcl_visualization.PointCloudGeometryHandleringCustom color_handler, str id = 'cloud', int viewport = 0):
        # NG
        # self.thisptr().addPointCloud[cpp.PointXYZ](<cpp.PointCloudPtr_t> cloud.thisptr_shared, color_handler.thisptr_shared, id, 0)
        # self.thisptr().addPointCloud[cpp.PointXYZ](<cpp.PointCloudPtr_t> cloud.thisptr_shared, deref(color_handler.thisptr_shared), id, 0)
        # self.thisptr().addPointCloud[cpp.PointXYZ](<cpp.PointCloudPtr_t> cloud.thisptr_shared, <const pcl_vis.PointCloudColorHandlerCustom[cpp.PointXYZ]&> color_handler.thisptr_shared, id, 0)
        # self.thisptr().addPointCloud(<cpp.PointCloudPtr_t> cloud.thisptr_shared, color_handler.thisptr_shared, id)
        # self.thisptr().addPointCloud[cpp.PointCloudPtr_t, pclvis.PointCloudColorHandlerCustom_Ptr_t, string, int](cloud.thisptr_shared, color_handler.thisptr_shared, id, viewport)
        # self.thisptr().addPointCloud(<cpp.PointCloudPtr_t> cloud.thisptr_shared, <pclvis.PointCloudColorHandlerCustom_Ptr_t> color_handler.thisptr_shared.super(), <string> id, <int> viewport)
        # overloaded
        # self.thisptr().addPointCloud[cpp.PointXYZ](<const shared_ptr[cpp.PointCloud[cpp.PointXYZ]]*> deref(cloud.thisptr_shared), <pclvis.PointCloudColorHandlerCustom_Ptr_t> color_handler.thisptr_shared, <const string> id, <int> viewport)
        # self.thisptr().addPointCloud[cpp.PointXYZ](<const shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> deref(cloud.thisptr_shared), <pclvis.PointCloudColorHandlerCustom_Ptr_t> color_handler.thisptr_shared, <const string> id, <int> viewport)
        # self.thisptr().addPointCloud[cpp.PointXYZ](deref(cloud.thisptr_shared), color_handler.thisptr_shared, id, viewport)
        # self.thisptr().addPointCloud[cpp.PointXYZ](deref(cloud.thisptr_shared), color_handler.thisptr_shared.get(), id, viewport)
        # self.thisptr().addPointCloud[cpp.PointXYZ](cloud.thisptr_shared, <pclvis.PointCloudColorHandlerCustom_Ptr_t> color_handler.thisptr_shared.get(), id, viewport)
        # self.thisptr().addPointCloud[cpp.PointXYZ](cloud.thisptr_shared, <const shared_ptr[pclvis.PointCloudColorHandler[cpp.PointXYZ]]> color_handler.thisptr_shared, id, viewport)
        # pass

    # def AddPointCloudNormals(self, _pcl.PointCloud cloud, _pcl.PointCloud_Normal normal):
    #     # self.thisptr().addPointCloudNormals(<cpp.PointCloudPtr_t> cloud.thisptr_shared, <cpp.PointCloud_Normal_Ptr_t> normal.thisptr_shared, 100, 0.02, 'cloud', 0)
    #     self.thisptr().addPointCloudNormals[cpp.PointXYZ, cpp.Normal](<cpp.PointCloudPtr_t> cloud.thisptr_shared, <cpp.PointCloud_Normal_Ptr_t> normal.thisptr_shared, 100, 0.02, 'cloud', 0)
    def AddPointCloudNormals(self, _pcl.PointCloud cloud, _pcl.PointCloud_Normal normal, int level = 100, double scale = 0.02, const string &id = 'cloud', int viewport = 0):
        self.thisptr().addPointCloudNormals[cpp.PointXYZ, cpp.Normal](<cpp.PointCloudPtr_t> cloud.thisptr_shared, <cpp.PointCloud_Normal_Ptr_t> normal.thisptr_shared, level, scale, id, viewport)

    # def updatePointCloud(self, _pcl.PointCloud cloud, string id = 'cloud'):
    #     flag = self.thisptr().updatePointCloud[cpp.PointXYZ](<cpp.PointCloudPtr_t> cloud.thisptr_shared, id)
    #     return flag

    def SetPointCloudRenderingProperties(self, int propType, int propValue, string propName = 'cloud'):
        self.thisptr().setPointCloudRenderingProperties (propType, propValue, propName, 0)

    def InitCameraParameters(self):
        self.thisptr().initCameraParameters()


