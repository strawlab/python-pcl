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

ctypedef aligned_allocator[PointXYZ] aligned_allocator_t 
ctypedef vector2[PointXYZ, aligned_allocator_t] AlignedPointTVector_t

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

# use cython type ?
# ctypedef fused PointCloudTypes:
#     PointXYZ
#     PointXYZRGBA

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointXYZ:
        PointXYZ()
        float x
        float y
        float z
    cdef struct Normal:
        pass

# cdef extern from "pcl/point_types.h" namespace "pcl":
#      cdef struct Normal:
#       float normal_x
#       float normal_y
#       float normal_z
#       float curvature

cdef extern from "pcl/point_types.h" namespace "pcl":
     cdef struct PointXYZRGBA:
         PointXYZRGBA()
         float x
         float y
         float z
         # uint32_t rgba
         # unsigned long rgba
         float rgba

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointXYZRGB:
         PointXYZRGB()
         float x
         float y
         float z
         float rgb

cdef extern from "pcl/point_types.h" namespace "pcl":
     cdef struct PointXYZRGBL:
         PointXYZRGBA()
         float x
         float y
         float z
         # uint32_t rgba
         #unsigned long rgba
         float rgba
         # uint32_t label
         #unsigned long label
         float label

cdef extern from "pcl/point_types.h" namespace "pcl":
     cdef struct PointXYZHSV:
         PointXYZHSV()
         float x
         float y
         float z
         float h
         float s
         float v

cdef extern from "pcl/point_types.h" namespace "pcl":
     cdef struct PointXY:
         PointXY()
         float x
         float y

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct InterestPoint:
        InterestPoint()
        float x
        float y
        float z
        float strength

cdef extern from "pcl/point_types.h" namespace "pcl":
     cdef struct PointXYZI:
        PointXYZI()
        float x
        float y
        float z
        float intensity

cdef extern from "pcl/point_types.h" namespace "pcl":
     cdef struct PointXYZL:
        PointXYZL()
        float x
        float y
        float z
        # unsigned long label
        float label

cdef extern from "pcl/point_types.h" namespace "pcl":
     cdef struct Label:
        Label()
        # uint32_t label
        # unsigned long label
        float label

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct Axis:
        Axis()
        float normal_x
        float normal_y
        float normal_z

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointNormal:
        PointNormal()
        float x
        float y
        float z
        float normal_x
        float normal_y
        float normal_z
        float curvature

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointXYZRGBNormal:
        PointXYZRGBNormal()
        float x
        float y
        float z
        float rgb
        float normal_x
        float normal_y
        float normal_z
        float curvature

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointXYZINormal:
        PointXYZINormal()
        float x
        float y
        float z
        float intensity
        float normal_x
        float normal_y
        float normal_z
        float curvature

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointWithRange:
        PointWithRange()
        float x
        float y
        float z
        float range

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointWithViewpoint:
        PointWithViewpoint()
        float x
        float y
        float z
        float vp_x
        float vp_y
        float vp_z

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct MomentInvariants:
        MomentInvariants()
        float j1
        float j2
        float j3

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PrincipalRadiiRSD:
        PrincipalRadiiRSD()
        float r_min
        float r_max

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct Boundary:
        Boundary()
        # uint8_t boundary_point
        unsigned char boundary_point

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PrincipalCurvatures:
        PrincipalCurvatures()
        float principal_curvature_x
        float principal_curvature_y
        float principal_curvature_z
        float pc1
        float pc2

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PFHSignature125:
        PFHSignature125()
        float[125] histogram

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PFHRGBSignature250:
        PFHRGBSignature250()
        float[250] histogram

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PPFSignature:
        PPFSignature()
        float f1
        float f2
        float f3
        float f4
        float alpha_m

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PPFRGBSignature:
        PPFRGBSignature()
        float f1
        float f2
        float f3
        float f4
        float r_ratio
        float g_ratio
        float b_ratio
        float alpha_m

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct NormalBasedSignature12:
        NormalBasedSignature12()
        float[12] values

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct SHOT352:
        SHOT352()
        float[352] descriptor
        float[9] rf

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct SHOT1344:
        SHOT1344()
        float[1344] descriptor
        float[9] rf

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct FPFHSignature33:
        FPFHSignature33()
        float[33] histogram

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct VFHSignature308:
        VFHSignature308()
        float[308] histogram

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct ESFSignature640:
        ESFSignature640()
        float[640] histogram

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct Narf36:
        Narf36()
        float[36] descriptor

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct GFPFHSignature16:
        GFPFHSignature16()
        float[16] histogram

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct IntensityGradient:
        IntensityGradient()
        float gradient_x
        float gradient_y
        float gradient_z

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointWithScale:
        PointWithScale()
        float x
        float y
        float z
        float scale

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct PointSurfel:
        PointSurfel()
        float x
        float y
        float z
        float normal_x
        float normal_y
        float normal_z
        # uint32_t rgba
        unsigned long rgba
        float radius
        float confidence
        float curvature

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct ReferenceFrame:
        ReferenceFrame()
        float[3] x_axis
        float[3] y_axis
        float[3] z_axis
        # float confidence

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

ctypedef PointCloud[PointXYZ] PointCloud_t
ctypedef PointCloud[PointXYZRGBA] PointCloud2_t
ctypedef PointCloud[PointXYZRGB] PointCloud3_t
ctypedef PointCloud[Normal] PointNormalCloud_t
ctypedef shared_ptr[PointCloud[PointXYZ]] PointCloudPtr_t
ctypedef shared_ptr[PointCloud[PointXYZRGBA]] PointCloudPtr2_t
ctypedef shared_ptr[PointCloud[PointXYZRGB]] PointCloudPtr3_t

###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################
