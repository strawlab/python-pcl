# -*- coding: utf-8 -*-
cimport pcl_surface as pcl_srf
cimport pcl_defs as cpp

cdef class ConcaveHull:
    """
    ConcaveHull (alpha shapes) using libqhull library.
    """
    cdef pcl_srf.ConcaveHull_t *me
    def __cinit__(self):
        self.me = new pcl_srf.ConcaveHull_t()
    def __dealloc__(self):
        del self.me

    def reconstruct(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.reconstruct(pc.thisptr()[0])
        return pc

    def set_Alpha(self, double d):
        """
        Set the alpha value, which limits the size of the resultant hull segments (the smaller the more detailed the hull). 
        """
        self.me.setAlpha (d)


cdef class ConcaveHull_PointXYZI:
    """
    ConcaveHull class for ...
    """
    cdef pcl_srf.ConcaveHull_PointXYZI_t *me
    def __cinit__(self):
        self.me = new pcl_srf.ConcaveHull_PointXYZI_t()
    def __dealloc__(self):
        del self.me

    def reconstruct(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZI pc = PointCloud_PointXYZI()
        self.me.reconstruct(pc.thisptr()[0])
        return pc

    def set_Alpha(self, double d):
        """
        Set the alpha value, which limits the size of the resultant hull segments (the smaller the more detailed the hull). 
        """
        self.me.setAlpha (d)


cdef class ConcaveHull_PointXYZRGB:
    """
    ConcaveHull class for ...
    """
    cdef pcl_srf.ConcaveHull_PointXYZRGB_t *me
    def __cinit__(self):
        self.me = new pcl_srf.ConcaveHull_PointXYZRGB_t()
    def __dealloc__(self):
        del self.me

    def reconstruct(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZRGB pc = PointCloud_PointXYZRGB()
        self.me.reconstruct(pc.thisptr()[0])
        return pc

    def set_Alpha(self, double d):
        """
        Set the alpha value, which limits the size of the resultant hull segments (the smaller the more detailed the hull). 
        """
        self.me.setAlpha (d)


cdef class ConcaveHull_PointXYZRGBA:
    """
    ConcaveHull class for ...
    """
    cdef pcl_srf.ConcaveHull_PointXYZRGBA_t *me
    def __cinit__(self):
        self.me = new pcl_srf.ConcaveHull_PointXYZRGBA_t()
    def __dealloc__(self):
        del self.me

    def reconstruct(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZRGBA pc = PointCloud_PointXYZRGBA()
        self.me.reconstruct(pc.thisptr()[0])
        return pc

    def set_Alpha(self, double d):
        """
        Set the alpha value, which limits the size of the resultant hull segments (the smaller the more detailed the hull). 
        """
        self.me.setAlpha (d)

