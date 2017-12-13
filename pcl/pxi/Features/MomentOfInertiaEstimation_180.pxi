# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_features_180 as pclftr


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

    def set_InputCloud(self, PointCloud pc not None):
        (<cpp.PCLBase_t*>self.me).setInputCloud(pc.thisptr_shared)

    # feature_extractor.getMomentOfInertia (moment_of_inertia);
    # feature_extractor.getEccentricity (eccentricity);
    # feature_extractor.getAABB (min_point_AABB, max_point_AABB);
    # feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    # feature_extractor.getEigenValues (major_value, middle_value, minor_value);
    # feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
    # feature_extractor.getMassCenter (mass_center);
    def compute (self):
        self.me.compute()

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
    def get_OBB (self):
        cdef cpp.PointXYZ min_point_OBB
        cdef cpp.PointXYZ max_point_OBB
        cdef cpp.PointXYZ position_OBB
        cdef eigen3.Matrix3f rotational_matrix_OBB
        self.me.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB)
        # NG : Convert Python object
    #   return min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB
        cdef cnp.npy_intp n = 1
        # cdef cnp.ndarray[cnp.float32_t, ndim=4, mode="c"] result
        result1 = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            result1[i, 0] = min_point_OBB.x
            result1[i, 1] = min_point_OBB.y
            result1[i, 2] = min_point_OBB.z
        pcl_min_point_OBB = PointCloud(result1)
        
        result2 = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            result2[i, 0] = max_point_OBB.x
            result2[i, 1] = max_point_OBB.y
            result2[i, 2] = max_point_OBB.z
        pcl_max_point_OBB = PointCloud(result2)
        
        result3 = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            result3[i, 0] = position_OBB.x
            result3[i, 1] = position_OBB.y
            result3[i, 2] = position_OBB.z
        pcl_position_OBB = PointCloud(result3)
        
        # cdef np.ndarray[np.double_t, ndim=2] np_rotational_matrix_OBB = np.empty((3,3), dtype=np.float64)
        # cdef np.ndarray[np.float32_t, ndim=2] np_rotational_matrix_OBB = np.empty((3,3), dtype=np.float32)
        np_rotational_matrix_OBB = np.empty((3,3), dtype=np.float32)
        # np_rotational_matrix_OBB[0,0] = rotational_matrix_OBB(0,0); np_rotational_matrix_OBB[0,1] = rotational_matrix_OBB(0,1); np_rotational_matrix_OBB[0,2] = rotational_matrix_OBB(0,2);
        # np_rotational_matrix_OBB[1,0] = rotational_matrix_OBB(1,0); np_rotational_matrix_OBB[1,1] = rotational_matrix_OBB(1,1); np_rotational_matrix_OBB[1,2] = rotational_matrix_OBB(1,2);
        # np_rotational_matrix_OBB[2,0] = rotational_matrix_OBB(2,0); np_rotational_matrix_OBB[2,1] = rotational_matrix_OBB(2,1); np_rotational_matrix_OBB[2,2] = rotational_matrix_OBB(2,2);
        np_rotational_matrix_OBB[0, 0] = rotational_matrix_OBB.element(0, 0)
        np_rotational_matrix_OBB[0, 1] = rotational_matrix_OBB.element(0, 1)
        np_rotational_matrix_OBB[0, 2] = rotational_matrix_OBB.element(0, 2)
        np_rotational_matrix_OBB[1, 0] = rotational_matrix_OBB.element(1, 0)
        np_rotational_matrix_OBB[1, 1] = rotational_matrix_OBB.element(1, 1)
        np_rotational_matrix_OBB[1, 2] = rotational_matrix_OBB.element(1, 2)
        np_rotational_matrix_OBB[2, 0] = rotational_matrix_OBB.element(2, 0)
        np_rotational_matrix_OBB[2, 1] = rotational_matrix_OBB.element(2, 1)
        np_rotational_matrix_OBB[2, 2] = rotational_matrix_OBB.element(2, 2)
        return result1, result2, result3, np_rotational_matrix_OBB

    def get_EigenValues (self):
        cdef float major_value = 0.0
        cdef float middle_value = 0.0
        cdef float minor_value = 0.0
        self.me.getEigenValues (major_value, middle_value, minor_value)
        return major_value, middle_value, minor_value

    def get_EigenVectors (self):
        cdef eigen3.Vector3f major_vector
        cdef eigen3.Vector3f middle_vector
        cdef eigen3.Vector3f minor_vector
        self.me.getEigenVectors (major_vector, middle_vector, minor_vector)
        
        # cdef np.ndarray[np.float32_t, ndim=2] np_major_vec = np.empty((1,3), dtype=np.float32)
        np_major_vec = np.empty((1,3), dtype=np.float32)
        np_major_vec[0,0] = major_vector.element(0,0)
        np_major_vec[0,1] = major_vector.element(1,1)
        np_major_vec[0,2] = major_vector.element(2,2)
        
        # cdef np.ndarray[np.float32_t, ndim=2] np_middle_vec = np.empty((1,3), dtype=np.float32)
        np_middle_vec = np.empty((1,3), dtype=np.float32)
        np_middle_vec[0,0] = middle_vector.element(0,0)
        np_middle_vec[0,1] = middle_vector.element(1,1)
        np_middle_vec[0,2] = middle_vector.element(2,2)
        
        # cdef np.ndarray[np.float32_t, ndim=2] np_minor_vec = np.empty((1,3), dtype=np.float32)
        np_minor_vec = np.empty((1,3), dtype=np.float32)
        np_minor_vec[0,0] = minor_vector.element(0,0)
        np_minor_vec[0,1] = minor_vector.element(1,1)
        np_minor_vec[0,2] = minor_vector.element(2,2)
        
        return np_major_vec, np_middle_vec, np_minor_vec

    def get_MassCenter (self):
        cdef eigen3.Vector3f mass_center
        self.me.getMassCenter (mass_center)
        
        # cdef np.ndarray[np.float32_t, ndim=2] np_mass_center_vec = np.empty((1,3), dtype=np.float32)
        np_mass_center_vec = np.empty((1,3), dtype=np.float32)
        np_mass_center_vec[0, 0] = mass_center.element(0, 0)
        # np_mass_center_vec[0, 1] = mass_center.element(0, 1)
        np_mass_center_vec[0, 1] = mass_center.element(1, 1)
        # np_mass_center_vec[0, 2] = mass_center.element(0, 2)
        np_mass_center_vec[0, 2] = mass_center.element(2, 2)
        
        return np_mass_center_vec


