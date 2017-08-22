# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport numpy as cnp

cimport cython
cimport pcl_visualization

cimport pcl_visualization_defs as pclvis

from libcpp.string cimport string

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

    def SetFullScreen(self, bool mode):
        self.thisptr().setFullScreen(mode)

    def SetWindowBorders(self, bool mode):
        self.thisptr().setWindowBorders(mode)

    def Spin(self):
        self.thisptr().spin()

    def SpinOnce(self, int millis_to_wait = 1, bool force_redraw = False):
        self.thisptr().spinOnce (millis_to_wait, force_redraw)

    def AddCoordinateSystem(self, double scale = 1.0, int viewpoint = 0):
        self.thisptr().addCoordinateSystem(scale, viewpoint)

    def AddCoordinateSystem(self, double scale, float x, float y, float z, int viewpoint = 0):
        self.thisptr().addCoordinateSystem(scale, x, y, z, viewpoint)

    # void addCoordinateSystem (double scale, const eigen3.Affine3f& t, int viewport)

    # return bool
    def removeCoordinateSystem (self, int viewport):
        return self.thisptr().removeCoordinateSystem (viewport)

    # return bool
    def RemovePointCloud(self, string id, int viewport):
        return self.thisptr().removePointCloud (id, viewport)

    def RemovePolygonMesh(self, string id, int viewport):
        return self.thisptr().removePolygonMesh (id, viewport)

    def RemoveShape(self, string id, int viewport):
        return self.thisptr().removeShape (id, viewport)

    def RemoveText3D(self, string id, int viewport):
        return self.thisptr().removeText3D (id, viewport)

    def RemoveAllPointClouds(self, int viewport):
        return self.thisptr().removeAllPointClouds (viewport)

    def RemoveAllShapes(self, int viewport):
        return self.thisptr().removeAllShapes (viewport)

    def SetBackgroundColor (self, int r, int g, int b):
        self.thisptr().setBackgroundColor(r, g, b, 0)

    # return bool
    def AddText (self, string text, int xpos, int ypos, id, int viewport):
        return self.thisptr().addText (text, xpos, ypos, <string> id, viewport)

    # return bool
    def AddText (self, string text, int xpos, int ypos, double r, double g, double b, id, int viewport):
        return self.thisptr().addText (text, xpos, ypos, r, g, b, <string> id, viewport)

    # return bool
    def AddText (self, string text, int xpos, int ypos, int fontsize, double r, double g, double b, id, int viewport):
        return self.thisptr().addText (text, xpos, ypos, fontsize, r, g, b, <string> id, viewport)

    # return bool
    # def UpdateText (self, string text, int xpos, int ypos, const string &id):
    def UpdateText (self, string text, int xpos, int ypos, id):
        return self.thisptr().updateText (text, xpos, ypos, <string> id)

    # return bool
    # def UpdateText (self, string text, int xpos, int ypos, double r, double g, double b, const string &id):
    def UpdateText (self, string text, int xpos, int ypos, double r, double g, double b, id):
        return self.thisptr().updateText (text, xpos, ypos,  r,  g,  b, <string> id)

    # return bool
    # def UpdateText (self, string text, int xpos, int ypos, int fontsize, double r, double g, double b, const string &id):
    def UpdateText (self, string text, int xpos, int ypos, int fontsize, double r, double g, double b, id):
        return self.thisptr().updateText (text, xpos, ypos, fontsize, r, g, b, <string> id)

    # bool updateShapePose (const string &id, const eigen3.Affine3f& pose)
    
    # return bool
    # def AddText3D[PointT](const string &text, const PointT &position, double textScale, double r, double g, double b, const string &id, int viewport)
    #     return self.thisptr().AddText3D[PointT](const string &text, const PointT &position, double textScale, double r, double g, double b, const string &id, int viewport)

    # bool addPointCloudNormals [PointNT](cpp.PointCloud[PointNT] cloud, int level, double scale, string id, int viewport)
    # bool addPointCloudNormals [PointT, PointNT] (const shared_ptr[cpp.PointCloud[PointT]] &cloud, const shared_ptr[cpp.PointCloud[PointNT]] &normals, int level, double scale, const string &id, int viewport)

    # bool updatePointCloud[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, string &id)
    # bool updatePointCloud[PointT](const shared_ptr[cpp.PointCloud[PointT]] &cloud, const PointCloudGeometryHandler[PointT] &geometry_handler, string &id)

    # def updatePointCloud(self, _pcl.PointCloud cloud, string id = 'cloud'):
    #     flag = self.thisptr().updatePointCloud[cpp.PointXYZ](<cpp.PointCloudPtr_t> cloud.thisptr_shared, id)
    #     return flag

    # def AddPointCloud (self, _pcl.PointCloud cloud, string id = 'cloud', int viewport = 0):
    # call (ex. id=b'range image')
    def AddPointCloud (self, _pcl.PointCloud cloud, id = b'cloud', int viewport = 0):
        self.thisptr().addPointCloud(cloud.thisptr_shared, <string> id, viewport)
        pass

    # <const shared_ptr[PointCloudColorHandler[PointT]]> 
    # def AddPointCloud_ColorHandler(self, _pcl.PointCloud cloud, pcl_visualization.PointCloudColorHandleringCustom color_handler, string id = 'cloud', int viewport = 0):
    def AddPointCloud_ColorHandler(self, _pcl.PointCloud cloud, pcl_visualization.PointCloudColorHandleringCustom color_handler, id = b'cloud', viewport = 0):
        # NG : Base Class
        # self.thisptr().addPointCloud[cpp.PointXYZ](cloud.thisptr_shared, <const pclvis.PointCloudColorHandler[cpp.PointXYZ]> deref(color_handler.thisptr_shared.get()), id, viewport)
        # OK? : Inheritance Class(PointCloudColorHandler)
        # self.thisptr().addPointCloud[cpp.PointXYZ](cloud.thisptr_shared, <const pclvis.PointCloudColorHandlerCustom[cpp.PointXYZ]> deref(color_handler.thisptr_shared.get()), id, viewport)
        self.thisptr().addPointCloud[cpp.PointXYZ](cloud.thisptr_shared, <const pclvis.PointCloudColorHandlerCustom[cpp.PointXYZ]> deref(color_handler.thisptr_shared.get()), <string> id, viewport)
        pass

    def AddPointCloud_ColorHandler(self, _pcl.RangeImages cloud, pcl_visualization.PointCloudColorHandleringCustom color_handler, id = b'cloud', int viewport = 0):
        # self.thisptr().addPointCloud[cpp.PointWithRange](cloud.thisptr_shared, <const pclvis.PointCloudColorHandlerCustom[cpp.PointXYZ]> deref(color_handler.thisptr_shared.get()), id, viewport)
        pass

    # <const shared_ptr[PointCloudGeometryHandler[PointT]]> 
    # def AddPointCloud_GeometryHandler(self, _pcl.PointCloud cloud, pcl_visualization.PointCloudGeometryHandleringCustom geometry_handler, id = b'cloud', int viewport = 0):
    #     # overloaded
    #     self.thisptr().addPointCloud[cpp.PointXYZ](cloud.thisptr_shared, <const pclvis.PointCloudGeometryHandlerCustom[cpp.PointXYZ]> deref(geometry_handler.thisptr_shared.get()), <string> id, viewport)
    #     # pass

    def AddPointCloudNormals(self, _pcl.PointCloud cloud, _pcl.PointCloud_Normal normal, int level = 100, double scale = 0.02, id = b'cloud', int viewport = 0):
        self.thisptr().addPointCloudNormals[cpp.PointXYZ, cpp.Normal](<cpp.PointCloudPtr_t> cloud.thisptr_shared, <cpp.PointCloud_Normal_Ptr_t> normal.thisptr_shared, level, scale, <string> id, viewport)

    def SetPointCloudRenderingProperties(self, int propType, int propValue, propName = b'cloud'):
        self.thisptr().setPointCloudRenderingProperties (propType, propValue, <string> propName, 0)

    def InitCameraParameters(self):
        self.thisptr().initCameraParameters()

    # return bool
    def WasStopped(self):
        return self.thisptr().wasStopped()

    def ResetStoppedFlag(self):
        self.thisptr().resetStoppedFlag()

    def Close(self):
        self.thisptr().close ()

    # def AddCube(self, double min_x, double max_x, double min_y, double max_y, double min_z, double max_z, double r, double g, double b, string name):
    def AddCube(self, double min_x, double max_x, double min_y, double max_y, double min_z, double max_z, double r, double g, double b, name):
        self.thisptr().addCube(min_x,  max_x,  min_y,  max_y,  min_z,  max_z, r, g, b, name, 0)

    # def AddLine(self, _pcl.PointCloud center, _pcl.PointCloud axis, double x, double y, double z, id = b'minor eigen vector')
    #     # pcl::PointXYZ
    #     self.thisptr().addLine(center, z_axis, 0.0, 0.0, 1.0, id)

    def AddCone(self):
        # self.thisptr().addCone()
        pass

    def AddCircle(self):
        # self.thisptr().addCone()
        pass

    def AddPlane(self):
        # self.thisptr().addPlane()
        pass

    def AddLine(self):
        # self.thisptr().addLine()
        pass

    def AddSphere(self):
        # self.thisptr().addSphere()
        pass

    def AddCylinder(self):
        # self.thisptr().addCylinder()
        pass

    def AddCircle(self):
        # self.thisptr().addCone()
        pass
