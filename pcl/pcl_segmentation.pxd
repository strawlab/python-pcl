from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# NG
# from libcpp.memory cimport shared_ptr
from boost_shared_ptr cimport shared_ptr

# main
# cimport pcl_defs as cpp
# 
from pcl_defs cimport PointIndices
from pcl_defs cimport ModelCoefficients
from pcl_defs cimport PointCloud
from pcl_defs cimport PointXYZ
from pcl_defs cimport Normal

from pcl_sample_consensus cimport SacModel

from vector cimport vector as vector2

###############################################################################
# Types
###############################################################################

cdef extern from "pcl/segmentation/sac_segmentation.h" namespace "pcl":
    cdef cppclass SACSegmentationFromNormals[T, N]:
        SACSegmentationFromNormals()
        void setOptimizeCoefficients (bool)
        void setModelType (SacModel)
        void setMethodType (int)
        void setNormalDistanceWeight (float)
        void setMaxIterations (int)
        void setDistanceThreshold (float)
        void setRadiusLimits (float, float)
        void setInputCloud (shared_ptr[PointCloud[T]])
        void setInputNormals (shared_ptr[PointCloud[N]])
        void setEpsAngle (double ea)
        void segment (PointIndices, ModelCoefficients)
        void setMinMaxOpeningAngle(double, double)
        void getMinMaxOpeningAngle(double, double)

    cdef cppclass SACSegmentation[T]:
        void setOptimizeCoefficients (bool)
        void setModelType (SacModel)
        void setMethodType (int)
        void setDistanceThreshold (float)
        void setInputCloud (shared_ptr[PointCloud[T]])
        void segment (PointIndices, ModelCoefficients)

ctypedef SACSegmentation[PointXYZ] SACSegmentation_t
ctypedef SACSegmentationFromNormals[PointXYZ,Normal] SACSegmentationNormal_t

###
# comparator.h
# edge_aware_plane_comparator.h
# euclidean_cluster_comparator.h
# euclidean_plane_coefficient_comparator.h
# extract_clusters.h
# extract_labeled_clusters.h
# extract_polygonal_prism_data.h
# organized_connected_component_segmentation.h
# organized_multi_plane_segmentation.h
# planar_polygon_fusion.h
# planar_region.h
# plane_coefficient_comparator.h
# plane_refinement_comparator.h
# region_3d.h
# rgb_plane_coefficient_comparator.h
# sac_segmentation.h
# segment_differences.h


###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################
