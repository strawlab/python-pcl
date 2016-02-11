from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

from boost_shared_ptr cimport shared_ptr

cdef extern from "pcl/filters/statistical_outlier_removal.h" namespace "pcl":
    cdef cppclass StatisticalOutlierRemoval[T]:
        StatisticalOutlierRemoval()
        int getMeanK()
        void setMeanK (int nr_k)
        double getStddevMulThresh()
        void setStddevMulThresh (double std_mul)
        bool getNegative()
        void setNegative (bool negative)
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        void filter(cpp.PointCloud[T] &c)

ctypedef StatisticalOutlierRemoval[cpp.PointXYZ] StatisticalOutlierRemoval_t
ctypedef StatisticalOutlierRemoval[cpp.PointXYZRGBA] StatisticalOutlierRemoval2_t


cdef extern from "pcl/filters/voxel_grid.h" namespace "pcl":
    cdef cppclass VoxelGrid[T]:
        VoxelGrid()
        void setLeafSize (float, float, float)
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        void filter(cpp.PointCloud[T] c)

ctypedef VoxelGrid[cpp.PointXYZ] VoxelGrid_t
ctypedef VoxelGrid[cpp.PointXYZRGBA] VoxelGrid2_t


cdef extern from "pcl/filters/passthrough.h" namespace "pcl":
    cdef cppclass PassThrough[T]:
        PassThrough()
        void setFilterFieldName (string field_name)
        void setFilterLimits (float, float)
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        void filter(cpp.PointCloud[T] c)

ctypedef PassThrough[cpp.PointXYZ] PassThrough_t
ctypedef PassThrough[cpp.PointXYZRGBA] PassThrough2_t

