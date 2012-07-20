from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from shared_ptr cimport shared_ptr

cdef extern from "pcl/point_cloud.h" namespace "pcl":
    cdef cppclass PointCloud[T]:
        PointCloud()
        PointCloud(int, int)
        unsigned int width
        unsigned int height
        bool is_dense
        void resize(int)
        int size()
        T& operator[](int)
        T& at(int)
        T& at(int, int)
        shared_ptr[PointCloud[T]] makeShared()

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointXYZ:
        PointXYZ()
        float x
        float y
        float z
    cdef struct Normal:
        pass

cdef extern from "pcl/features/normal_3d.h" namespace "pcl":
    cdef cppclass NormalEstimation[T, N]:
        NormalEstimation()

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

    cdef cppclass SACSegmentation[T]:
        void setOptimizeCoefficients (bool)
        void setModelType (SacModel)
        void setMethodType (int)
        void setDistanceThreshold (float)
        void setInputCloud (shared_ptr[PointCloud[T]])
        void segment (PointIndices, ModelCoefficients)

ctypedef SACSegmentation[PointXYZ] SACSegmentation_t
ctypedef SACSegmentationFromNormals[PointXYZ,Normal] SACSegmentationNormal_t

cdef extern from "pcl/surface/mls.h" namespace "pcl":
    cdef cppclass MovingLeastSquares[I,N]:
        MovingLeastSquares()
        void setInputCloud (shared_ptr[PointCloud[I]])
        void setSearchRadius (double)
        void setPolynomialOrder(bool)
        void setPolynomialFit(int)
        void reconstruct (PointCloud[I])

ctypedef MovingLeastSquares[PointXYZ,Normal] MovingLeastSquares_t

cdef extern from "pcl/search/kdtree.h" namespace "pcl::search":
    cdef cppclass KdTree[T]:
        KdTree()

cdef extern from "pcl/ModelCoefficients.h" namespace "pcl":
    cdef struct ModelCoefficients:
        vector[float] values

cdef extern from "pcl/PointIndices.h" namespace "pcl":
    cdef struct PointIndices:
        vector[int] indices

cdef extern from "pcl/io/pcd_io.h" namespace "pcl::io":
    int loadPCDFile (string file_name, PointCloud[PointXYZ] cloud)
    int savePCDFile (string file_name, PointCloud[PointXYZ] cloud, bool binary_mode)

#http://dev.pointclouds.org/issues/624
#cdef extern from "pcl/io/ply_io.h" namespace "pcl::io":
#    int loadPLYFile (string file_name, PointCloud[PointXYZ] cloud)
#    int savePLYFile (string file_name, PointCloud[PointXYZ] cloud, bool binary_mode)

cdef extern from "pcl/sample_consensus/model_types.h" namespace "pcl":
    cdef enum SacModel:
        SACMODEL_PLANE
        SACMODEL_LINE
        SACMODEL_CIRCLE2D
        SACMODEL_CIRCLE3D, 
        SACMODEL_SPHERE
        SACMODEL_CYLINDER
        SACMODEL_CONE
        SACMODEL_TORUS, 
        SACMODEL_PARALLEL_LINE
        SACMODEL_PERPENDICULAR_PLANE
        SACMODEL_PARALLEL_LINES
        SACMODEL_NORMAL_PLANE, 
        SACMODEL_NORMAL_SPHERE
        SACMODEL_REGISTRATION
        SACMODEL_PARALLEL_PLANE
        SACMODEL_NORMAL_PARALLEL_PLANE, 
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

ctypedef PointCloud[PointXYZ] PointCloud_t
ctypedef PointCloud[Normal] PointNormalCloud_t
ctypedef shared_ptr[PointCloud[PointXYZ]] PointCloudPtr_t

