
cimport pcl_defs as cpp
cimport pcl_surface as pclsf

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
        cdef PointCloud pc = PointCloud()
        self.me.process(pc.thisptr()[0])
        return pc
