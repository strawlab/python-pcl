# -*- coding: utf-8 -*-
cimport pcl_segmentation_172 as pcl_seg
cimport pcl_sample_consensus_172 as pcl_sac
cimport pcl_defs as cpp

cimport eigen as eigen3

cdef class Segmentation:
    """
    Segmentation class for Sample Consensus methods and models
    """
    cdef pcl_seg.SACSegmentation_t *me
    def __cinit__(self):
        self.me = new pcl_seg.SACSegmentation_t()


    def __dealloc__(self):
        del self.me


    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())], \
               [coeffs.values[i] for i in range(coeffs.values.size())]


    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)


    def set_model_type(self, pcl_sac.SacModel m):
        self.me.setModelType(m)


    def set_method_type(self, int m):
        self.me.setMethodType (m)


    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)


    def set_max_iterations(self, int count):
        self.me.setMaxIterations (count)


    def set_eps_angle(self, double ea):
        (<pcl_seg.SACSegmentation_t*>self.me).setEpsAngle (ea)


    def get_eps_angle(self):
        return (<pcl_seg.SACSegmentation_t*>self.me).getEpsAngle()


    def set_axis(self, double ax1, double ax2, double ax3):
        cdef eigen3.Vector3f* vec = new eigen3.Vector3f(ax1, ax2, ax3)
        (<pcl_seg.SACSegmentation_t*>self.me).setAxis(deref(vec))


    def get_axis(self):
        vec = (<pcl_seg.SACSegmentation_t*>self.me).getAxis()
        cdef float *data = vec.data()
        return np.array([data[0], data[1], data[2]], dtype=np.float32)


cdef class Segmentation_PointXYZI:
    """
    Segmentation class for Sample Consensus methods and models
    """
    cdef pcl_seg.SACSegmentation_PointXYZI_t *me
    def __cinit__(self):
        self.me = new pcl_seg.SACSegmentation_PointXYZI_t()


    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())], \
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)


    def set_model_type(self, pcl_sac.SacModel m):
        self.me.setModelType(m)


    def set_method_type(self, int m):
        self.me.setMethodType (m)


    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)


    def set_eps_angle(self, double ea):
        (<pcl_seg.SACSegmentation_PointXYZI_t*>self.me).setEpsAngle (ea)


    def get_eps_angle(self):
        return (<pcl_seg.SACSegmentation_PointXYZI_t*>self.me).getEpsAngle()


    def set_axis(self, double ax1, double ax2, double ax3):
        cdef eigen3.Vector3f* vec = new eigen3.Vector3f(ax1, ax2, ax3)
        (<pcl_seg.SACSegmentation_PointXYZI_t*>self.me).setAxis(deref(vec))


    def get_axis(self):
        vec = (<pcl_seg.SACSegmentation_PointXYZI_t*>self.me).getAxis()
        cdef float *data = vec.data()
        return np.array([data[0], data[1], data[2]], dtype=np.float32)


cdef class Segmentation_PointXYZRGB:
    """
    Segmentation class for Sample Consensus methods and models
    """
    cdef pcl_seg.SACSegmentation_PointXYZRGB_t *me
    def __cinit__(self):
        self.me = new pcl_seg.SACSegmentation_PointXYZRGB_t()


    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())], \
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)


    def set_model_type(self, pcl_sac.SacModel m):
        self.me.setModelType(m)


    def set_method_type(self, int m):
        self.me.setMethodType (m)


    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)


    def set_eps_angle(self, double ea):
        (<pcl_seg.SACSegmentation_PointXYZRGB_t*>self.me).setEpsAngle (ea)


    def get_eps_angle(self):
        return (<pcl_seg.SACSegmentation_PointXYZRGB_t*>self.me).getEpsAngle()


    def set_axis(self, double ax1, double ax2, double ax3):
        cdef eigen3.Vector3f* vec = new eigen3.Vector3f(ax1, ax2, ax3)
        (<pcl_seg.SACSegmentation_PointXYZRGB_t*>self.me).setAxis(deref(vec))


    def get_axis(self):
        vec = (<pcl_seg.SACSegmentation_PointXYZRGB_t*>self.me).getAxis()
        cdef float *data = vec.data()
        return np.array([data[0], data[1], data[2]], dtype=np.float32)


cdef class Segmentation_PointXYZRGBA:
    """
    Segmentation class for Sample Consensus methods and models
    """
    cdef pcl_seg.SACSegmentation_PointXYZRGBA_t *me
    def __cinit__(self):
        self.me = new pcl_seg.SACSegmentation_PointXYZRGBA_t()


    def __dealloc__(self):
        del self.me

    def segment(self):
        cdef cpp.PointIndices ind
        cdef cpp.ModelCoefficients coeffs
        
        self.me.segment (ind, coeffs)
        return [ind.indices[i] for i in range(ind.indices.size())], \
               [coeffs.values[i] for i in range(coeffs.values.size())]

    def set_optimize_coefficients(self, bool b):
        self.me.setOptimizeCoefficients(b)


    def set_model_type(self, pcl_sac.SacModel m):
        self.me.setModelType(m)


    def set_method_type(self, int m):
        self.me.setMethodType (m)


    def set_distance_threshold(self, float d):
        self.me.setDistanceThreshold (d)


    def set_eps_angle(self, double ea):
        (<pcl_seg.SACSegmentation_PointXYZRGBA_t*>self.me).setEpsAngle (ea)


    def get_eps_angle(self):
        return (<pcl_seg.SACSegmentation_PointXYZRGBA_t*>self.me).getEpsAngle()


    def set_axis(self, double ax1, double ax2, double ax3):
        cdef eigen3.Vector3f* vec = new eigen3.Vector3f(ax1, ax2, ax3)
        (<pcl_seg.SACSegmentation_PointXYZRGBA_t*>self.me).setAxis(deref(vec))


    def get_axis(self):
        vec = (<pcl_seg.SACSegmentation_PointXYZRGBA_t*>self.me).getAxis()
        cdef float *data = vec.data()
        return np.array([data[0], data[1], data[2]], dtype=np.float32)

