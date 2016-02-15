from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from boost_shared_ptr cimport shared_ptr

###############################################################################
# Types
###############################################################################

###
# lmeds.h
# method_types.h
# mlesac.h
# model_types.h
# msac.h
# prosac.h
# ransac.h
# rmsac.h
# rransac.h
# sac.h
# sac_model.h
# sac_model_circle.h
# sac_model_cone.h
# sac_model_cylinder.h
# sac_model_line.h
# sac_model_normal_parallel_plane.h
# sac_model_normal_plane.h
# sac_model_normal_sphere.h
# sac_model_parallel_line.h
# sac_model_parallel_plane.h
# sac_model_perpendicular_plane.h
# sac_model_plane.h
# sac_model_registration.h
# sac_model_sphere.h
# sac_model_stick.h


###############################################################################
# Enum
###############################################################################
cdef extern from "pcl/sample_consensus/model_types.h" namespace "pcl":
    cdef enum SacModel:
        SACMODEL_PLANE
        SACMODEL_LINE
        SACMODEL_CIRCLE2D
        SACMODEL_CIRCLE3D
        SACMODEL_SPHERE
        SACMODEL_CYLINDER
        SACMODEL_CONE
        SACMODEL_TORUS
        SACMODEL_PARALLEL_LINE
        SACMODEL_PERPENDICULAR_PLANE
        SACMODEL_PARALLEL_LINES
        SACMODEL_NORMAL_PLANE
        SACMODEL_NORMAL_SPHERE        # Version 1.6
        SACMODEL_REGISTRATION
        SACMODEL_PARALLEL_PLANE
        SACMODEL_NORMAL_PARALLEL_PLANE
        SACMODEL_STICK

cdef extern from "pcl/sample_consensus/method_types.h" namespace "pcl":
    cdef enum:
        SAC_RANSAC = 0
        SAC_LMEDS = 1
        SAC_MSAC = 2
        SAC_RRANSAC = 3
        SAC_RMSAC = 4
        SAC_MLESAC = 5
        SAC_PROSAC = 6

###############################################################################
# Activation
###############################################################################
