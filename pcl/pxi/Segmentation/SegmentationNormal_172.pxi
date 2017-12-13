# -*- coding: utf-8 -*-
cimport pcl_segmentation_172 as pclseg
cimport pcl_defs as cpp
cimport pcl_sample_consensus_172 as pcl_sc

cimport eigen as eigen3

# yeah, I can't be bothered making this inherit from SACSegmentation, I forget the rules
# for how this works in cython templated extension types anyway
cdef class SegmentationNormal:
    """
    Segmentation class for Sample Consensus methods and models that require the
    use of surface normals for estimation.

    Due to Cython limitations this should derive from pcl.Segmentation, but
    is currently unable to do so.
    """
    cdef pclseg.SACSegmentationFromNormals_t *me

    def __cinit__(self):
        self.me = new pclseg.SACSegmentationFromNormals_t()


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


    def set_model_type(self, pcl_sc.SacModel m):
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
        (<pclseg.SACSegmentation_t*>self.me).setEpsAngle (ea)


    def get_eps_angle(self):
        return (<pclseg.SACSegmentation_PointXYZRGB_t*>self.me).getEpsAngle()


    def set_axis(self, double ax1, double ax2, double ax3):
        cdef eigen3.Vector3f* vec = new eigen3.Vector3f(ax1, ax2, ax3)
        (<pclseg.SACSegmentation_t*>self.me).setAxis(deref(vec))


    def get_axis(self):
        vec = (<pclseg.SACSegmentation_t*>self.me).getAxis()
        cdef float *data = vec.data()
        return np.array([data[0], data[1], data[2]], dtype=np.float32)


    def set_min_max_opening_angle(self, double min_angle, double max_angle):
        """
        Set the minimum and maximum cone opening angles in radians for a cone model.
        """
        self.me.setMinMaxOpeningAngle(min_angle, max_angle)


    def get_min_max_opening_angle(self):
        min_angle = 0.0
        max_angle = 0.0
        self.me.getMinMaxOpeningAngle(min_angle, max_angle)
        return min_angle, max_angle


cdef class Segmentation_PointXYZI_Normal:
    """
    Segmentation class for Sample Consensus methods and models that require the
    use of surface normals for estimation.

    Due to Cython limitations this should derive from pcl.Segmentation, but
    is currently unable to do so.
    """
    cdef pclseg.SACSegmentationFromNormals_PointXYZI_t *me
    def __cinit__(self):
        self.me = new pclseg.SACSegmentationFromNormals_PointXYZI_t()

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


    def set_model_type(self, pcl_sc.SacModel m):
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


    def get_eps_angle(self):
        return (<pclseg.SACSegmentation_PointXYZRGB_t*>self.me).getEpsAngle()


    def set_axis(self, double ax1, double ax2, double ax3):
        cdef eigen3.Vector3f* vec = new eigen3.Vector3f(ax1, ax2, ax3)
        (<pclseg.SACSegmentation_PointXYZI_t*>self.me).setAxis(deref(vec))


    def get_axis(self):
        vec = (<pclseg.SACSegmentation_t*>self.me).getAxis()
        cdef float *data = vec.data()
        return np.array([data[0], data[1], data[2]], dtype=np.float32)


    def set_min_max_opening_angle(self, double min_angle, double max_angle):
        """
        Set the minimum and maximum cone opening angles in radians for a cone model.
        """
        self.me.setMinMaxOpeningAngle(min_angle, max_angle)


    def get_min_max_opening_angle(self):
        min_angle = 0.0
        max_angle = 0.0
        self.me.getMinMaxOpeningAngle(min_angle, max_angle)
        return min_angle, max_angle


cdef class Segmentation_PointXYZRGB_Normal:
    """
    Segmentation class for Sample Consensus methods and models that require the
    use of surface normals for estimation.

    Due to Cython limitations this should derive from pcl.Segmentation, but
    is currently unable to do so.
    """
    cdef pclseg.SACSegmentationFromNormals_PointXYZRGB_t *me
    def __cinit__(self):
        self.me = new pclseg.SACSegmentationFromNormals_PointXYZRGB_t()


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


    def set_model_type(self, pcl_sc.SacModel m):
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


    def get_eps_angle(self):
        return (<pclseg.SACSegmentation_PointXYZRGB_t*>self.me).getEpsAngle()


    def set_axis(self, double ax1, double ax2, double ax3):
        cdef eigen3.Vector3f* vec = new eigen3.Vector3f(ax1, ax2, ax3)
        (<pclseg.SACSegmentation_PointXYZRGB_t*>self.me).setAxis(deref(vec))


    def get_axis(self):
        vec = (<pclseg.SACSegmentation_t*>self.me).getAxis()
        cdef float *data = vec.data()
        return np.array([data[0], data[1], data[2]], dtype=np.float32)


    def set_min_max_opening_angle(self, double min_angle, double max_angle):
        """
        Set the minimum and maximum cone opening angles in radians for a cone model.
        """
        self.me.setMinMaxOpeningAngle(min_angle, max_angle)


    def get_min_max_opening_angle(self):
        min_angle = 0.0
        max_angle = 0.0
        self.me.getMinMaxOpeningAngle(min_angle, max_angle)
        return min_angle, max_angle


cdef class Segmentation_PointXYZRGBA_Normal:
    """
    Segmentation class for Sample Consensus methods and models that require the
    use of surface normals for estimation.

    Due to Cython limitations this should derive from pcl.Segmentation, but
    is currently unable to do so.
    """
    cdef pclseg.SACSegmentationFromNormals_PointXYZRGBA_t *me
    def __cinit__(self):
        self.me = new pclseg.SACSegmentationFromNormals_PointXYZRGBA_t()

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


    def set_model_type(self, pcl_sc.SacModel m):
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
        vec = (<pclseg.SACSegmentation_PointXYZRGBA_t*>self.me).setEpsAngle(ea)


    def get_eps_angle(self):
        return (<pclseg.SACSegmentation_PointXYZRGBA_t*>self.me).getEpsAngle()


    def set_axis(self, double ax1, double ax2, double ax3):
        cdef eigen3.Vector3f* vec = new eigen3.Vector3f(ax1, ax2, ax3)
        (<pclseg.SACSegmentation_PointXYZRGBA_t*>self.me).setAxis(deref(vec))


    def get_axis(self):
        vec = (<pclseg.SACSegmentation_PointXYZRGBA_t*>self.me).getAxis()
        cdef float *data = vec.data()
        return np.array([data[0], data[1], data[2]], dtype=np.float32)


    def set_min_max_opening_angle(self, double min_angle, double max_angle):
        """
        Set the minimum and maximum cone opening angles in radians for a cone model.
        """
        self.me.setMinMaxOpeningAngle(min_angle, max_angle)


    def get_min_max_opening_angle(self):
        min_angle = 0.0
        max_angle = 0.0
        self.me.getMinMaxOpeningAngle(min_angle, max_angle)
        return min_angle, max_angle

