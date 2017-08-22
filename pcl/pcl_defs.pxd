# -*- coding: utf-8 -*-
from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from boost_shared_ptr cimport shared_ptr

# Eigen
from eigen cimport Vector4f
from eigen cimport Quaternionf

from vector cimport vector as vector2

# Vertices
# ctypedef unsigned int uint32_t

###############################################################################
# Types
###############################################################################

### base class ###

###

### Inheritance class ###

# channel_properties.h
###

# cloud_properties.h
###

# correspondence.h
###

# exceptions.h

###

# for_each_type.h

###

# pcl_config.h

###

# pcl_exports.h

###

# pcl_macros.h

###

# pcl_tests.h

###

# point_representation.h

###

# point_traits.h

###

# point_types_conversion.h

###

# template <>
# class PCL_EXPORTS PCLBase<sensor_msgs::PointCloud2>
#   public:
#     typedef sensor_msgs::PointCloud2 PointCloud2;
#     typedef PointCloud2::Ptr PointCloud2Ptr;
#     typedef PointCloud2::ConstPtr PointCloud2ConstPtr;
#       typedef PointIndices::Ptr PointIndicesPtr;
#       typedef PointIndices::ConstPtr PointIndicesConstPtr;
#       /** \brief Empty constructor. */
#       PCLBase ()
#       /** \brief destructor. */
#       virtual ~PCLBase() 
#       /** \brief Provide a pointer to the input dataset
#         * \param cloud the const boost shared pointer to a PointCloud message
#         */
#       void setInputCloud (const PointCloud2ConstPtr &cloud);
#       /** \brief Get a pointer to the input point cloud dataset. */
#        PointCloud2ConstPtr const getInputCloud ()
#       /** \brief Provide a pointer to the vector of indices that represents the input data.
#         * \param indices a pointer to the vector of indices that represents the input data.
#         */
#        void setIndices (const IndicesPtr &indices)
#       /** \brief Provide a pointer to the vector of indices that represents the input data.
#         * \param indices a pointer to the vector of indices that represents the input data.
#         */
#        void setIndices (const PointIndicesConstPtr &indices)
#       /** \brief Get a pointer to the vector of indices used. */
#        IndicesPtr const getIndices ()
###

# point_cloud.h
cdef extern from "pcl/point_cloud.h" namespace "pcl":
    cdef cppclass PointCloud[T]:
        PointCloud() except +
        PointCloud(unsigned int, unsigned int) except +
        unsigned int width
        unsigned int height
        bool is_dense
        void resize(size_t) except +
        size_t size()
        # NG
        #T& operator[](size_t)
        # ???(No Test)
        #T& "operator[]"(size_t)
        #T& at(size_t) except +
        #T& at(int, int) except +
        shared_ptr[PointCloud[T]] makeShared()
        
        Quaternionf sensor_orientation_
        Vector4f sensor_origin_

###

# point_types.h
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
#     cdef struct Normal:
#         pass

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct Normal:
        float normal_x
        float normal_y
        float normal_z
        float curvature

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

# brief Data type to store extended information about a transition from foreground to backgroundSpecification of the fields for BorderDescription::traits.
# ingroup common
# typedef std::bitset<32> BorderTraits;
# 
# brief Specification of the fields for BorderDescription::traits.
# ingroup common
# 
# enum BorderTrait
# {
#   BORDER_TRAIT__OBSTACLE_BORDER, BORDER_TRAIT__SHADOW_BORDER, BORDER_TRAIT__VEIL_POINT,
#   BORDER_TRAIT__SHADOW_BORDER_TOP, BORDER_TRAIT__SHADOW_BORDER_RIGHT, BORDER_TRAIT__SHADOW_BORDER_BOTTOM,
#   BORDER_TRAIT__SHADOW_BORDER_LEFT, BORDER_TRAIT__OBSTACLE_BORDER_TOP, BORDER_TRAIT__OBSTACLE_BORDER_RIGHT,
#   BORDER_TRAIT__OBSTACLE_BORDER_BOTTOM, BORDER_TRAIT__OBSTACLE_BORDER_LEFT, BORDER_TRAIT__VEIL_POINT_TOP,
#   BORDER_TRAIT__VEIL_POINT_RIGHT, BORDER_TRAIT__VEIL_POINT_BOTTOM, BORDER_TRAIT__VEIL_POINT_LEFT
# };

cdef extern from "pcl/point_types.h" namespace "pcl":
    cdef struct BorderDescription:
        BorderDescription()
        int x
        int y
        int traits
        # BorderTraits traits;
        # //std::vector<const BorderDescription*> neighbors;

# inline std::ostream& operator << (std::ostream& os, const BorderDescription& p)
# {
#   os << "(" << p.x << "," << p.y << ")";
#   return (os);
# }

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
###

# ModelCoefficients.h
cdef extern from "pcl/ModelCoefficients.h" namespace "pcl":
    cdef struct ModelCoefficients:
        vector[float] values

ctypedef ModelCoefficients ModelCoefficients_t
ctypedef shared_ptr[ModelCoefficients] ModelCoefficientsPtr_t

###

# PointIndices.h
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
###

ctypedef PointCloud[PointXYZ] PointCloud_t
ctypedef PointCloud[PointXYZI] PointCloud_PointXYZI_t
ctypedef PointCloud[PointXYZRGB] PointCloud_PointXYZRGB_t
ctypedef PointCloud[PointXYZRGBA] PointCloud_PointXYZRGBA_t
ctypedef PointCloud[VFHSignature308] PointCloud_VFHSignature308_t
ctypedef PointCloud[PointWithViewpoint] PointCloud_PointWithViewpoint_t

ctypedef shared_ptr[PointCloud[PointXYZ]] PointCloudPtr_t
ctypedef shared_ptr[PointCloud[PointXYZI]] PointCloud_PointXYZI_Ptr_t
ctypedef shared_ptr[PointCloud[PointXYZRGB]] PointCloud_PointXYZRGB_Ptr_t
ctypedef shared_ptr[PointCloud[PointXYZRGBA]] PointCloud_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[PointCloud[VFHSignature308]] PointCloud_VFHSignature308_Ptr_t
ctypedef shared_ptr[PointCloud[PointWithViewpoint]] PointCloud_PointWithViewpoint_Ptr_t

ctypedef PointCloud[Normal] PointCloud_Normal_t
ctypedef shared_ptr[PointCloud[Normal]] PointCloud_Normal_Ptr_t

ctypedef PointCloud[PointNormal] PointCloud_PointNormal_t
ctypedef shared_ptr[PointCloud[PointNormal]] PointCloud_PointNormal_Ptr_t

# definitions used everywhere
ctypedef shared_ptr[vector[int]] IndicesPtr_t;
# ctypedef shared_ptr[vector[int]] IndicesConstPtr_t;


# pcl_base.h
# template <typename PointT>
# class PCLBase
cdef extern from "pcl/pcl_base.h" namespace "pcl":
    cdef cppclass PCLBase[PointT]:
        PCLBase ()
        # PCLBase (const PCLBase& base)
        # virtual void setInputCloud (PointCloudPtr_t cloud)
        # void setInputCloud (PointCloudPtr_t cloud)
        void setInputCloud (shared_ptr[PointCloud[PointT]] cloud)
        
        # PointCloudPtr_t getInputCloud ()
        shared_ptr[PointCloud[PointT]] getInputCloud ()
        
        void setIndices (IndicesPtr_t &indices)
        # void setIndices (IndicesConstPtr_t &indices)
        # void setIndices (const PointIndicesPtr_t &indices)
        # void setIndices (size_t row_start, size_t col_start, size_t nb_rows, size_t nb_cols)
        
        # IndicesConstPtr_t getIndices ()
        # # const PointT& operator[] (size_t pos)


ctypedef PCLBase[PointXYZ] PCLBase_t
ctypedef PCLBase[PointXYZI] PCLBase_PointXYZI_t
ctypedef PCLBase[PointXYZRGB] PCLBase_PointXYZRGB_t
ctypedef PCLBase[PointXYZRGBA] PCLBase_PointXYZRGBA_t
ctypedef shared_ptr[PCLBase[PointXYZ]] PCLBasePtr_t
ctypedef shared_ptr[PCLBase[PointXYZI]] PCLBase_PointXYZI_Ptr_t
ctypedef shared_ptr[PCLBase[PointXYZRGB]] PCLBase_PointXYZRGB_Ptr_t
ctypedef shared_ptr[PCLBase[PointXYZRGBA]] PCLBase_PointXYZRGBA_Ptr_t
###

# PolygonMesh.h
# namespace pcl
#  struct PolygonMesh
cdef extern from "pcl/PolygonMesh.h" namespace "pcl":
    cdef cppclass PolygonMesh:
        PolygonMesh()

# ctypedef shared_ptr[PolygonMesh] PolygonMeshPtr;
# ctypedef shared_ptr[PolygonMesh const] PolygonMeshConstPtr;
# inline std::ostream& operator<<(std::ostream& s, const  ::pcl::PolygonMesh &v)
###

# TextureMesh.h
# namespace pcl
# struct TexMaterial
cdef extern from "pcl/TextureMesh.h" namespace "pcl":
    cdef cppclass TexMaterial:
        TexMaterial ()
        # cdef struct RGB
        #     float r
        #     float g
        #     float b
        string tex_name
        string tex_file
        # RGB tex_Ka
        # RGB tex_Kd
        # RGB tex_Ks
        float tex_d
        float tex_Ns
        int tex_illum


###

cdef extern from "pcl/TextureMesh.h" namespace "pcl":
    cdef cppclass TextureMesh:
        TextureMesh ()
        # std_msgs::Header          header
        # sensor_msgs::PointCloud2  cloud
        # vector[vector[Vertices] ]    tex_polygons             // polygon which is mapped with specific texture defined in TexMaterial
        # vector[vector[Eigen::Vector2f] ]  tex_coordinates     // UV coordinates
        # vector[TexMaterial]               tex_materials       // define texture material

# ctypedef shared_ptr[TextureMesh] TextureMeshPtr_t
# ctypedef shared_ptr[TextureMesh const] TextureMeshConstPtr_t
###

# Vertices.h
# namespace pcl
# struct Vertices
cdef extern from "pcl/Vertices.h" namespace "pcl":
    cdef cppclass Vertices:
        Vertices()
        vector[size_t] vertices;
        # ostream& element "operator()"(ostream s, Vertices v)
        # public:
        # ctypedef shared_ptr[Vertices] Ptr
        # ctypedef shared_ptr[Vertices const] ConstPtr


# ctypedef Vertices Vertices_t
ctypedef shared_ptr[Vertices] VerticesPtr_t
# ctypedef shared_ptr[Vertices const] VerticesConstPtr
# inline std::ostream& operator<<(std::ostream& s, const  ::pcl::Vertices & v)
###


###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################

