from libcpp cimport bool

# main
cimport pcl_defs as cpp
from boost_shared_ptr cimport shared_ptr


###
# bfgs.h

###

# correspondence_estimation.h

###

# correspondence_estimation_normal_shooting.h

###

# correspondence_rejection.h

###

# correspondence_rejection_distance.h

###

# correspondence_rejection_features.h

###

# correspondence_rejection_median_distance.h

###

# correspondence_rejection_one_to_one.h

###

# correspondence_rejection_sample_consensus.h

###

# correspondence_rejection_surface_normal.h

###

# correspondence_rejection_trimmed.h

###

# correspondence_rejection_var_trimmed.h

###

# correspondence_sorting.h

###

# correspondence_types.h

###

# distances.h

###

# eigen.h

###

# elch.h

###

# exceptions.h

###

# gicp.h

cdef extern from "pcl/registration/gicp.h" namespace "pcl" nogil:
    cdef cppclass GeneralizedIterativeClosestPoint[Source, Target](Registration[Source, Target]):
        GeneralizedIterativeClosestPoint() except +

###

# ia_ransac.h

###

# icp.h
cdef extern from "pcl/registration/icp.h" namespace "pcl" nogil:
    cdef cppclass IterativeClosestPoint[Source, Target](Registration[Source, Target]):
        IterativeClosestPoint() except +
###

# icp_nl.h

cdef extern from "pcl/registration/icp_nl.h" namespace "pcl" nogil:
    cdef cppclass IterativeClosestPointNonLinear[Source, Target](Registration[Source, Target]):
        IterativeClosestPointNonLinear() except +

###

# ppf_registration.h

###

# pyramid_feature_matching.h

###

# registration.h
cdef extern from "pcl/registration/registration.h" namespace "pcl" nogil:
    cdef cppclass Registration[Source, Target]:
        cppclass Matrix4:
            float *data()
        void align(cpp.PointCloud[Source] &) except +
        Matrix4 getFinalTransformation() except +
        double getFitnessScore() except +
        bool hasConverged() except +
        void setInputSource(cpp.PointCloudPtr_t) except +
        void setInputTarget(cpp.PointCloudPtr_t) except +
        void setMaximumIterations(int) except +
###

# transformation_estimation.h

###

# transformation_estimation_lm.h

###

# transformation_estimation_point_to_plane.h

###

# transformation_estimation_point_to_plane_lls.h

###

# transformation_estimation_svd.h

###

# transformation_validation.h

###

# transformation_validation_euclidean.h

###

# transforms.h

###

# warp_point_rigid.h

###

# warp_point_rigid_3d.h

###

# warp_point_rigid_6d.h

###

