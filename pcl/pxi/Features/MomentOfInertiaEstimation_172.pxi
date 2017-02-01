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
    cdef vector[float] moment_of_inertia
    cdef vector[float] eccentricity
    cdef cpp.PointXYZ min_point_AABB
    cdef cpp.PointXYZ max_point_AABB
    cdef cpp.PointXYZ min_point_OBB
    cdef cpp.PointXYZ max_point_OBB
    cdef cpp.PointXYZ position_OBB
    # cdef eigen3.Matrix3f rotational_matrix_OBB
    cdef float major_value
    cdef float middle_value
    cdef float minor_value
    # eigen3.Vector3f major_vector
    # eigen3.Vector3f middle_vector
    # eigen3.Vector3f minor_vector
    # eigen3.Vector3f mass_center

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
        self.getMomentOfInertia(moment_of_inertia)
        return moment_of_inertia

    def get_Eccentricity (self):
        self.getEccentricity(eccentricity)
        return eccentricity

    def get_AABB (self):
        self.getAABB (min_point_AABB, max_point_AABB)
        return min_point_AABB, max_point_AABB

    def get_OBB (self):
        self.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB)
        return min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB

    def get_EigenValues (self):
        self.getEigenValues (major_value, middle_value, minor_value)
        return major_value, middle_value, minor_value

    def get_EigenVectors (self):
        self.getEigenVectors (major_vector, middle_vector, minor_vector)
        return major_vector, middle_vector, minor_vector

    def get_MassCenter (self):
        self.getMassCenter (mass_center)
        return mass_center

