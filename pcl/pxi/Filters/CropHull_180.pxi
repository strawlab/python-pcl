# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_filters_180 as pclfil

from boost_shared_ptr cimport shared_ptr

cdef class CropHull:
    """
    Must be constructed from the reference point cloud, which is copied, 
    so changed to pc are not reflected in CropHull(pc).
    """
    cdef pclfil.CropHull_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclfil.CropHull_t()
        (<cpp.PCLBase_t*>self.me).setInputCloud (pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def set_InputCloud(self, PointCloud pc not None):
        (<cpp.PCLBase_t*>self.me).setInputCloud (pc.thisptr_shared)

    # def SetParameter(self, shared_ptr[cpp.PointCloud[cpp.PointXYZ]] points, cpp.Vertices vt):
    def SetParameter(self, PointCloud points, Vertices vt):
        cdef const vector[cpp.Vertices] tmp_vertices
        # NG
        # tmp_vertices.push_back(deref(vt))
        # tmp_vertices.push_back<cpp.Vertices>(vt)
        # tmp_vertices.push_back[cpp.Vertices](vt)
        # tmp_vertices.push_back(vt)
        # tmp_vertices.push_back(deref(vt.thisptr()))
        self.me.setHullIndices(tmp_vertices)
        # self.me.setHullCloud(<shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> points)
        # convert <PointCloud> to <shared_ptr[cpp.PointCloud[cpp.PointXYZ]]>
        self.me.setHullCloud(points.thisptr_shared)
        self.me.setDim(<int> 2)
        self.me.setCropOutside(<bool> False)

    def Filtering(self, PointCloud outputCloud):
        # Cython 0.25.2 NG(0.24.1 OK)
        # self.me.filter(deref(outputCloud.thisptr()))
        # self.me.filter(<cpp.PointCloud[cpp.PointXYZ]> outputCloud.thisptr()[0])
        # Cython 0.24.1 NG(0.25.2 OK)
        # self.me.filter(<vector[int]> outputCloud)
        # self.me.filter(<vector[int]&> outputCloud)
        self.me.c_filter(outputCloud.thisptr()[0])
        print("filter: outputCloud size = " + str(outputCloud.size))
        return outputCloud

    def filter(self):
        cdef PointCloud pc = PointCloud()
        self.me.c_filter(pc.thisptr()[0])
        print("filter: pc size = " + str(pc.size))
        return pc


