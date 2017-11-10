# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_filters as pclfil

cimport eigen as eigen3

from boost_shared_ptr cimport shared_ptr

cdef class CropBox:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in CropBox(pc).
    """
    cdef pclfil.CropBox_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclfil.CropBox_t()
        (<cpp.PCLBase_t*>self.me).setInputCloud (pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def set_InputCloud(self, PointCloud pc not None):
        (<cpp.PCLBase_t*>self.me).setInputCloud (pc.thisptr_shared)

    def set_Translation(self, tx, ty, tz):
        # Convert Eigen::Vector3f
        cdef eigen3.Vector3f origin
        cdef float *data = origin.data()
        data[0] = tx
        data[1] = ty
        data[2] = tz
        self.me.setTranslation(origin)

    # def set_Rotation(self, cnp.ndarray[ndim=3, dtype=int, mode='c'] ind):
    def set_Rotation(self, rx, ry, rz):
        # Convert Eigen::Vector3f
        cdef eigen3.Vector3f origin
        cdef float *data = origin.data()
        data[0] = rx
        data[1] = ry
        data[2] = rz
        self.me.setRotation(origin)

    def set_Min(self, minx, miny, minz, mins):
        # Convert Eigen::Vector4f
        cdef eigen3.Vector4f originMin
        cdef float *dataMin = originMin.data()
        dataMin[0] = minx
        dataMin[1] = miny
        dataMin[2] = minz
        dataMin[3] = mins
        self.me.setMin(originMin)

    def set_Max(self, maxx, maxy, maxz, maxs):
        cdef eigen3.Vector4f originMax;
        cdef float *dataMax = originMax.data()
        dataMax[0] = maxx
        dataMax[1] = maxy
        dataMax[2] = maxz
        dataMax[3] = maxs
        self.me.setMax(originMax)

    def set_MinMax(self, minx, miny, minz, mins, maxx, maxy, maxz, maxs):
        # Convert Eigen::Vector4f
        cdef eigen3.Vector4f originMin
        cdef float *dataMin = originMin.data()
        dataMin[0] = minx
        dataMin[1] = miny
        dataMin[2] = minz
        dataMin[3] = mins
        self.me.setMin(originMin)
        
        cdef eigen3.Vector4f originMax;
        cdef float *dataMax = originMax.data()
        dataMax[0] = maxx
        dataMax[1] = maxy
        dataMax[2] = maxz
        dataMax[3] = maxs
        self.me.setMax(originMax)

    def set_Negative(self, bool flag):
        self.me.setNegative(flag)

    # bool
    def get_Negative (self):
        self.me.getNegative ()

    def get_RemovedIndices(self):
        self.me.getRemovedIndices()

    def filter(self):
        cdef PointCloud pc = PointCloud()
        # Cython 0.25.2 NG(0.24.1 OK)
        # self.me.filter(deref(pc.thisptr()))
        # self.me.filter(<cpp.PointCloud[cpp.PointXYZ]> pc.thisptr()[0])
        # Cython 0.24.1 NG(0.25.2 OK)
        # self.me.filter(<vector[int]&> pc)
        self.me.c_filter(pc.thisptr()[0])
        return pc


