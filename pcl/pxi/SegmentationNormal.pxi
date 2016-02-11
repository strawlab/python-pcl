
cimport pcl_segmentation as pclseg
cimport pcl_defs as cpp

cdef extern from "minipcl.h":
    void mpcl_compute_normals(cpp.PointCloud_t, int ksearch,
                              double searchRadius,
                              cpp.PointNormalCloud_t) except +
    void mpcl_sacnormal_set_axis(pclseg.SACSegmentationNormal_t,
                                 double ax, double ay, double az) except +
    void mpcl_extract(cpp.PointCloudPtr_t, cpp.PointCloud_t *,
                      cpp.PointIndices_t *, bool) except +


#yeah, I can't be bothered making this inherit from SACSegmentation, I forget the rules
#for how this works in cython templated extension types anyway
cdef class SegmentationNormal:
    """
    Segmentation class for Sample Consensus methods and models that require the
    use of surface normals for estimation.

    Due to Cython limitations this should derive from pcl.Segmentation, but
    is currently unable to do so.
    """
    cdef pclseg.SACSegmentationNormal_t *me
    def __cinit__(self):
        self.me = new pclseg.SACSegmentationNormal_t()

    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())],\
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)
    def set_model_type(self, pclseg.SacModel m):
        self.me.setModelType(m)
    def set_method_type(self, int m):
        self.me.setMethodType (m)
    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)
    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients (b)
    def set_normal_distance_weight(self, float f):
        self.me.setNormalDistanceWeight (f)
    def set_max_iterations(self, int i):
        self.me.setMaxIterations (i)
    def set_radius_limits(self, float f1, float f2):
        self.me.setRadiusLimits (f1, f2)
    def set_eps_angle(self, double ea):
        self.me.setEpsAngle (ea)
    def set_axis(self, double ax, double ay, double az):
        mpcl_sacnormal_set_axis(deref(self.me),ax,ay,az)
    def set_min_max_opening_angle(self, double min_angle, double max_angle):
        """ Set the minimum and maximum cone opening angles in radians for a cone model.
        """
        self.me.setMinMaxOpeningAngle(min_angle, max_angle)
    def get_min_max_opening_angle(self):
        min_angle = 0.0
        max_angle = 0.0
        self.me.getMinMaxOpeningAngle(min_angle, max_angle)
        return min_angle, max_angle
