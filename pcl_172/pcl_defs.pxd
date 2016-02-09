from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# NG
# from libcpp.memory cimport shared_ptr
from boost_shared_ptr cimport shared_ptr

from vector cimport vector as vector2

###############################################################################
# Types
###############################################################################

cdef extern from "Eigen/Eigen" namespace "Eigen" nogil:
    cdef cppclass Vector4f:
        float *data()
    cdef cppclass Quaternionf:
        float w()
        float x()
        float y()
        float z()
    cdef cppclass aligned_allocator[T]:
        pass

cdef extern from "pcl/point_cloud.h" namespace "pcl":
    cdef cppclass PointCloud[T]:
        PointCloud() except +
        PointCloud(unsigned int, unsigned int) except +
        unsigned int width
        unsigned int height
        bool is_dense
        void resize(size_t) except +
        size_t size()
        #T& operator[](size_t)
        #T& at(size_t) except +
        #T& at(int, int) except +
        shared_ptr[PointCloud[T]] makeShared()

        Quaternionf sensor_orientation_
        Vector4f sensor_origin_

cdef extern from "indexing.hpp":
    # Use these instead of operator[] or at.
    PointXYZ *getptr(PointCloud[PointXYZ] *, size_t)
    PointXYZ *getptr_at(PointCloud[PointXYZ] *, size_t) except +
    PointXYZ *getptr_at(PointCloud[PointXYZ] *, int, int) except +

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointXYZ:
        PointXYZ()
        float x
        float y
        float z
    cdef struct Normal:
        pass
    cdef struct PointXYZRGBA:
        PointXYZRGBA()
        float x
        float y
        float z
        uint32_t rgba
    cdef struct PointXYZRGBL:
        PointXYZRGBA()
        float x
        float y
        float z
        uint32_t rgba
        uint32_t label
    cdef struct PointXYZHSV:
    	float x
        float y
        float z
        float h
        float s
        float v
	cdef struct PointXY:
    	float x
        float y
	cdef struct InterestPoint
	    float x
    	float y
	    float z
    	float strength
    cdef struct PointXYZI
        float x
    	float y
    	float z
    	float intensity
    cdef struct PointXYZL
    	float x
    	float y
    	float z
    	uint32_t label
    cdef struct Label
    	uint32_t, label
	cdef struct Normal:
    	float normal_x
    	float normal_y
    	float normal_z
    	float curvature
	cdef struct Axis:
    	float normal_x
    	float normal_y
    	float normal_z
	cdef struct PointNormal:
	    float x
	    float y
	    float z
	    float normal_x
	    float normal_y
	    float normal_z
	    float curvature
	cdef struct PointXYZRGBNormal:
	    float x
	    float y
	    float z
	    float rgb
	    float normal_x
	    float normal_y
	    float normal_z
	    float curvature

	cdef struct PointXYZINormal:
	    float x
	    float y
	    float z
	    float intensity
	    float normal_x
	    float normal_y
	    float normal_z
	    float curvature
	cdef struct PointWithRange:
	    float x
	    float y
	    float z
	    float range
	cdef struct PointWithViewpoint:
	    float x
	    float y
	    float z
	    float vp_x
	    float vp_y
	    float vp_z
	cdef struct MomentInvariants:
	    float j1
	    float j2
	    float j3
	cdef struct PrincipalRadiiRSD:
	    float r_min
	    float r_max
	cdef struct Boundary:
	    uint8_t boundary_point
	cdef struct PrincipalCurvatures:
	    float principal_curvature_x
	    float principal_curvature_y
	    float principal_curvature_z
	    float pc1
	    float pc2
	cdef struct PFHSignature125:
	    float[125] histogram
	cdef struct PFHRGBSignature250:
	    float[250] histogram
	cdef struct PPFSignature:
	    float f1
	    float f2
	    float f3
	    float f4
	    float alpha_m
	cdef struct PPFRGBSignature:
	    float f1
	    float f2
	    float f3
	    float f4
	    float r_ratio
	    float g_ratio
	    float b_ratio
	    float alpha_m
	cdef struct NormalBasedSignature12:
	    float[12] values
	cdef struct SHOT352:
	    float[352] descriptor
	    float[9] rf
	cdef struct SHOT1344:
	    float[1344] descriptor
	    float[9] rf
	cdef struct FPFHSignature33:
	    float[33] histogram
	cdef struct VFHSignature308:
	    float[308] histogram
	cdef struct ESFSignature640:
	    float[640] histogram
	cdef struct Narf36:
	    float[36] descriptor
	cdef struct GFPFHSignature16:
	    float[16] histogram
	cdef struct IntensityGradient:
	    float gradient_x
	    float gradient_y
	    float gradient_z
	cdef struct PointWithScale:
	    float x
	    float y
	    float z
	    float scale
	cdef struct PointSurfel:
	    float x
	    float y
	    float z
	    float normal_x
	    float normal_y
	    float normal_z
	    uint32_t rgba
	    float radius
	    float confidence
	    float curvature
	cdef struct ReferenceFrame:
	    float[3] x_axis
	    float[3] y_axis
	    float[3] z_axis
	    //float confidence


cdef extern from "pcl/surface/mls.h" namespace "pcl":
    cdef cppclass MovingLeastSquares[I,O]:
        MovingLeastSquares()
        void setInputCloud (shared_ptr[PointCloud[I]])
        void setSearchRadius (double)
        void setPolynomialOrder(bool)
        void setPolynomialFit(int)
        void process(PointCloud[O] &) except +

ctypedef MovingLeastSquares[PointXYZ,PointXYZ] MovingLeastSquares_t

cdef extern from "pcl/search/kdtree.h" namespace "pcl::search":
    cdef cppclass KdTree[T]:
        KdTree()

ctypedef aligned_allocator[PointXYZ] aligned_allocator_t 
ctypedef vector2[PointXYZ, aligned_allocator_t] AlignedPointTVector_t

cdef extern from "pcl/octree/octree_pointcloud.h" namespace "pcl::octree":
    cdef cppclass OctreePointCloud[T]:
        OctreePointCloud(double)
        void setInputCloud (shared_ptr[PointCloud[T]])
        void defineBoundingBox()
        void defineBoundingBox(double, double, double, double, double, double)
        void addPointsFromInputCloud()
        void deleteTree()
        bool isVoxelOccupiedAtPoint(double, double, double)
        int getOccupiedVoxelCenters(AlignedPointTVector_t)	
        void deleteVoxelAtPoint(PointXYZ)

ctypedef OctreePointCloud[PointXYZ] OctreePointCloud_t

cdef extern from "pcl/octree/octree_search.h" namespace "pcl::octree":
    cdef cppclass OctreePointCloudSearch[T]:
        OctreePointCloudSearch(double)
        int radiusSearch (PointXYZ, double, vector[int], vector[float], unsigned int)

ctypedef OctreePointCloudSearch[PointXYZ] OctreePointCloudSearch_t

cdef extern from "pcl/ModelCoefficients.h" namespace "pcl":
    cdef struct ModelCoefficients:
        vector[float] values

cdef extern from "pcl/PointIndices.h" namespace "pcl":
    #FIXME: I made this a cppclass so that it can be allocated using new (cython barfs otherwise), and
    #hence passed to shared_ptr. This is needed because if one passes memory allocated
    #using malloc (which is required if this is a struct) to shared_ptr it aborts with
    #std::bad_alloc
    #
    #I don't know if this is actually a problem. It is cpp and there is no incompatibility in
    #promoting struct to class in this instance
    cdef cppclass PointIndices:
        vector[int] indices

ctypedef PointIndices PointIndices_t
ctypedef shared_ptr[PointIndices] PointIndicesPtr_t

cdef extern from "pcl/io/pcd_io.h" namespace "pcl::io":
    int load(string file_name, PointCloud[PointXYZ] &cloud) nogil except +
    int loadPCDFile(string file_name,
                    PointCloud[PointXYZ] &cloud) nogil except +
    int savePCDFile(string file_name, PointCloud[PointXYZ] &cloud,
                    bool binary_mode) nogil except +

cdef extern from "pcl/io/ply_io.h" namespace "pcl::io":
    int loadPLYFile(string file_name,
                    PointCloud[PointXYZ] &cloud) nogil except +
    int savePLYFile(string file_name, PointCloud[PointXYZ] &cloud,
                    bool binary_mode) nogil except +

#http://dev.pointclouds.org/issues/624
#cdef extern from "pcl/io/ply_io.h" namespace "pcl::io":
#    int loadPLYFile (string file_name, PointCloud[PointXYZ] cloud)
#    int savePLYFile (string file_name, PointCloud[PointXYZ] cloud, bool binary_mode)

ctypedef PointCloud[PointXYZ] PointCloud_t
ctypedef PointCloud[Normal] PointNormalCloud_t
ctypedef shared_ptr[PointCloud[PointXYZ]] PointCloudPtr_t

cdef extern from "pcl/filters/statistical_outlier_removal.h" namespace "pcl":
    cdef cppclass StatisticalOutlierRemoval[T]:
        StatisticalOutlierRemoval()
        int getMeanK()
        void setMeanK (int nr_k)
        double getStddevMulThresh()
        void setStddevMulThresh (double std_mul)
        bool getNegative()
        void setNegative (bool negative)
        void setInputCloud (shared_ptr[PointCloud[T]])
        void filter(PointCloud[T] &c)

ctypedef StatisticalOutlierRemoval[PointXYZ] StatisticalOutlierRemoval_t

cdef extern from "pcl/filters/voxel_grid.h" namespace "pcl":
    cdef cppclass VoxelGrid[T]:
        VoxelGrid()
        void setLeafSize (float, float, float)
        void setInputCloud (shared_ptr[PointCloud[T]])
        void filter(PointCloud[T] c)

ctypedef VoxelGrid[PointXYZ] VoxelGrid_t

cdef extern from "pcl/filters/passthrough.h" namespace "pcl":
    cdef cppclass PassThrough[T]:
        PassThrough()
        void setFilterFieldName (string field_name)
        void setFilterLimits (float, float)
        void setInputCloud (shared_ptr[PointCloud[T]])
        void filter(PointCloud[T] c)

ctypedef PassThrough[PointXYZ] PassThrough_t

cdef extern from "pcl/kdtree/kdtree_flann.h" namespace "pcl":
    cdef cppclass KdTreeFLANN[T]:
        KdTreeFLANN()
        void setInputCloud (shared_ptr[PointCloud[T]])
        int nearestKSearch (PointCloud[T],
          int, int, vector[int], vector[float])

ctypedef KdTreeFLANN[PointXYZ] KdTreeFLANN_t
# ctypedef KdTreeFLANN[PointXYZRGB] KdTreeFLANN_t2

###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################
