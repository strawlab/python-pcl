# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features_172 as pclftr


cdef class MomentOfInertiaEstimation:
    """
    MomentOfInertiaEstimation class for 
    """
    cdef pclftr.MomentOfInertiaEstimation_t *me
    # std::vector <float> moment_of_inertia;
    # std::vector <float> eccentricity;
    # pcl::PointXYZ min_point_AABB;
    # pcl::PointXYZ max_point_AABB;
    # pcl::PointXYZ min_point_OBB;
    # pcl::PointXYZ max_point_OBB;
    # pcl::PointXYZ position_OBB;
    # Eigen::Matrix3f rotational_matrix_OBB;
    # float major_value, middle_value, minor_value;
    # Eigen::Vector3f major_vector, middle_vector, minor_vector;
    # Eigen::Vector3f mass_center;

    def __cinit__(self):
        self.me = new pclftr.MomentOfInertiaEstimation_t()

    def __dealloc__(self):
        del self.me

    # feature_extractor.getMomentOfInertia (moment_of_inertia);
    # feature_extractor.getEccentricity (eccentricity);
    # feature_extractor.getAABB (min_point_AABB, max_point_AABB);
    # feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    # feature_extractor.getEigenValues (major_value, middle_value, minor_value);
    # feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
    # feature_extractor.getMassCenter (mass_center);
    def get_MomentOfInertia (self):
        cdef vector[float] moment_of_inertia
        self.me.getMomentOfInertia(moment_of_inertia)
        return moment_of_inertia

    def get_Eccentricity (self):
        cdef vector[float] eccentricity
        self.me.getEccentricity(eccentricity)
        return eccentricity

    @cython.boundscheck(False)
    def get_AABB (self):
        cdef cpp.PointXYZ min_point_AABB
        cdef cpp.PointXYZ max_point_AABB
        self.me.getAABB (min_point_AABB, max_point_AABB)
        # return min_point_AABB, max_point_AABB
        
        cdef cnp.npy_intp n = 1
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] result
        
        result1 = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            result1[i, 0] = min_point_AABB.x
            result1[i, 1] = min_point_AABB.y
            result1[i, 2] = min_point_AABB.z
        
        result2 = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            result2[i, 0] = max_point_AABB.x
            result2[i, 1] = max_point_AABB.y
            result2[i, 2] = max_point_AABB.z
        
        return result1, result2

    @cython.boundscheck(False)
    # def get_OBB (self):
    #   cdef cpp.PointXYZ min_point_OBB
    #   cdef cpp.PointXYZ max_point_OBB
    #   cdef cpp.PointXYZ position_OBB
    #   cdef eigen3.Matrix3f rotational_matrix_OBB
    #   self.me.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB)
    #   return min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB

    def get_EigenValues (self):
        cdef float major_value = 0.0
        cdef float middle_value = 0.0
        cdef float minor_value = 0.0
        self.me.getEigenValues (major_value, middle_value, minor_value)
        return major_value, middle_value, minor_value

    # def get_EigenVectors (self):
    #   # cdef eigen3.Vector3f major_vector
    #   # cdef eigen3.Vector3f middle_vector
    #   # cdef eigen3.Vector3f minor_vector
    #     self.me.getEigenVectors (major_vector, middle_vector, minor_vector)
    #     return major_vector, middle_vector, minor_vector

    # def get_MassCenter (self):
    #       # cdef eigen3.Vector3f mass_center
    #     self.me.getMassCenter (mass_center)
    #     return mass_center

