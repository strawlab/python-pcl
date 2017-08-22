# -*- coding: utf-8 -*-
cimport _pcl
cimport pcl_defs as cpp
cimport pcl_surface as pclsf
cimport pcl_kdtree as pclkdt

cdef class MovingLeastSquares:
    """
    Smoothing class which is an implementation of the MLS (Moving Least Squares)
    algorithm for data smoothing and improved normal estimation.
    """
    cdef pclsf.MovingLeastSquares_t *me
    
    def __cinit__(self):
        self.me = new pclsf.MovingLeastSquares_t()
    
    def __dealloc__(self):
        del self.me
    
    def set_search_radius(self, double radius):
        """
        Set the sphere radius that is to be used for determining the k-nearest neighbors used for fitting. 
        """
        self.me.setSearchRadius (radius)
    
    def set_polynomial_order(self, bool order):
        """
        Set the order of the polynomial to be fit. 
        """
        self.me.setPolynomialOrder(order)
    
    def set_polynomial_fit(self, bool fit):
        """
        Sets whether the surface and normal are approximated using a polynomial,
        or only via tangent estimation.
        """
        self.me.setPolynomialFit(fit)
    
    def set_Compute_Normals(self, bool flag):
        self.me.setComputeNormals(flag)
    
    def set_Search_Method(self, _pcl.KdTree kdtree):
       # self.me.setSearchMethod(kdtree.thisptr()[0])
       # self.me.setSearchMethod(kdtree.thisptr())
       self.me.setSearchMethod(kdtree.thisptr_shared)
    
    # def set_Search_Method(self, _pcl.KdTreeFLANN kdtree):
    #    # self.me.setSearchMethod(kdtree.thisptr())
    #    self.me.setSearchMethod(kdtree.thisptr_shared)
    
    def process(self):
        """
        Apply the smoothing according to the previously set values and return
        a new PointCloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.process(pc.thisptr()[0])
        return pc
        # cdef PointCloud_PointNormal pcNormal = PointCloud_PointNormal()
        # self.me.process(pcNormal.thisptr()[0])
        # return pcNormal


# cdef class MovingLeastSquares_PointXYZI:
#     """
#     Smoothing class which is an implementation of the MLS (Moving Least Squares)
#     algorithm for data smoothing and improved normal estimation.
#     """
#     cdef pclsf.MovingLeastSquares_PointXYZI_t *me
#     
#     def __cinit__(self):
#         self.me = new pclsf.MovingLeastSquares_PointXYZI_t()
#     def __dealloc__(self):
#         del self.me
# 
#     def set_search_radius(self, double radius):
#         """
#         Set the sphere radius that is to be used for determining the k-nearest neighbors used for fitting. 
#         """
#         self.me.setSearchRadius (radius)
# 
#     def set_polynomial_order(self, bool order):
#         """
#         Set the order of the polynomial to be fit. 
#         """
#         self.me.setPolynomialOrder(order)
# 
#     def set_polynomial_fit(self, bint fit):
#         """
#         Sets whether the surface and normal are approximated using a polynomial,
#         or only via tangent estimation.
#         """
#         self.me.setPolynomialFit(fit)
# 
#     def process(self):
#         """
#         Apply the smoothing according to the previously set values and return
#         a new pointcloud
#         """
#         cdef PointCloud_PointXYZI pc = PointCloud_PointXYZI()
#         self.me.process(pc.thisptr()[0])
#         return pc


cdef class MovingLeastSquares_PointXYZRGB:
    """
    Smoothing class which is an implementation of the MLS (Moving Least Squares)
    algorithm for data smoothing and improved normal estimation.
    """
    cdef pclsf.MovingLeastSquares_PointXYZRGB_t *me
    
    def __cinit__(self):
        self.me = new pclsf.MovingLeastSquares_PointXYZRGB_t()
    
    def __dealloc__(self):
        del self.me
    
    def set_search_radius(self, double radius):
        """
        Set the sphere radius that is to be used for determining the k-nearest neighbors used for fitting. 
        """
        self.me.setSearchRadius (radius)
    
    def set_polynomial_order(self, bool order):
        """
        Set the order of the polynomial to be fit. 
        """
        self.me.setPolynomialOrder(order)
    
    def set_polynomial_fit(self, bint fit):
        """
        Sets whether the surface and normal are approximated using a polynomial,
        or only via tangent estimation.
        """
        self.me.setPolynomialFit(fit)
    
    def process(self):
        """
        Apply the smoothing according to the previously set values and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZRGB pc = PointCloud_PointXYZRGB()
        self.me.process(pc.thisptr()[0])
        return pc
        # cdef PointCloud_PointNormal pcNormal = PointCloud_PointNormal()
        # self.me.process(pcNormal.thisptr()[0])
        # return pcNormal

cdef class MovingLeastSquares_PointXYZRGBA:
    """
    Smoothing class which is an implementation of the MLS (Moving Least Squares)
    algorithm for data smoothing and improved normal estimation.
    """
    cdef pclsf.MovingLeastSquares_PointXYZRGBA_t *me
    
    def __cinit__(self):
        self.me = new pclsf.MovingLeastSquares_PointXYZRGBA_t()
    
    def __dealloc__(self):
        del self.me
    
    def set_search_radius(self, double radius):
        """
        Set the sphere radius that is to be used for determining the k-nearest neighbors used for fitting. 
        """
        self.me.setSearchRadius (radius)
    
    def set_polynomial_order(self, bool order):
        """
        Set the order of the polynomial to be fit. 
        """
        self.me.setPolynomialOrder(order)
    
    def set_polynomial_fit(self, bint fit):
        """
        Sets whether the surface and normal are approximated using a polynomial,
        or only via tangent estimation.
        """
        self.me.setPolynomialFit(fit)
    
    def process(self):
        """
        Apply the smoothing according to the previously set values and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZRGBA pc = PointCloud_PointXYZRGBA()
        self.me.process(pc.thisptr()[0])
        return pc
        # cdef PointCloud_PointNormal pcNormal = PointCloud_PointNormal()
        # self.me.process(pcNormal.thisptr()[0])
        # return pcNormal

