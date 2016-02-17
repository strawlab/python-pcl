from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# NG
# from libcpp.memory cimport shared_ptr
from boost_shared_ptr cimport shared_ptr

# Eigen
from eigen cimport Vector4f
from eigen cimport Quaternionf

from vector cimport vector as vector2


###############################################################################
# Types
###############################################################################

# template <>
# class PCL_EXPORTS PCLBase<sensor_msgs::PointCloud2>
# {
#   public:
#     typedef sensor_msgs::PointCloud2 PointCloud2;
#     typedef PointCloud2::Ptr PointCloud2Ptr;
#     typedef PointCloud2::ConstPtr PointCloud2ConstPtr;
# 
#       typedef PointIndices::Ptr PointIndicesPtr;
#       typedef PointIndices::ConstPtr PointIndicesConstPtr;
# 
#       /** \brief Empty constructor. */
#       PCLBase () : input_ (), indices_ (), use_indices_ (false), fake_indices_ (false),
#                    field_sizes_ (0), x_idx_ (-1), y_idx_ (-1), z_idx_ (-1),
#                    x_field_name_ ("x"), y_field_name_ ("y"), z_field_name_ ("z")
#       {};
# 
#       /** \brief destructor. */
#       virtual ~PCLBase() 
#       {
#         input_.reset ();
#         indices_.reset ();
#       }
# 
#       /** \brief Provide a pointer to the input dataset
#         * \param cloud the const boost shared pointer to a PointCloud message
#         */
#       void 
#       setInputCloud (const PointCloud2ConstPtr &cloud);
# 
#       /** \brief Get a pointer to the input point cloud dataset. */
#        PointCloud2ConstPtr const 
#       getInputCloud () { return (input_); }
# 
#       /** \brief Provide a pointer to the vector of indices that represents the input data.
#         * \param indices a pointer to the vector of indices that represents the input data.
#         */
#        void
#       setIndices (const IndicesPtr &indices)
#       {
#         indices_ = indices;
#         fake_indices_ = false;
#         use_indices_  = true;
#       }
# 
#       /** \brief Provide a pointer to the vector of indices that represents the input data.
#         * \param indices a pointer to the vector of indices that represents the input data.
#         */
#        void
#       setIndices (const PointIndicesConstPtr &indices)
#       {
#         indices_.reset (new std::vector<int> (indices->indices));
#         fake_indices_ = false;
#         use_indices_  = true;
#       }
# 
#       /** \brief Get a pointer to the vector of indices used. */
#        IndicesPtr const 
#       getIndices () { return (indices_); }
# 
#     protected:
#       /** \brief The input point cloud dataset. */
#       PointCloud2ConstPtr input_;
# 
#       /** \brief A pointer to the vector of point indices to use. */
#       IndicesPtr indices_;
# 
#       /** \brief Set to true if point indices are used. */
#       bool use_indices_;
# 
#       /** \brief If no set of indices are given, we construct a set of fake indices that mimic the input PointCloud. */
#       bool fake_indices_;
# 
#       /** \brief The size of each individual field. */
#       std::vector<int> field_sizes_;
# 
#       /** \brief The x-y-z fields indices. */
#       int x_idx_, y_idx_, z_idx_;
# 
#       /** \brief The desired x-y-z field names. */
#       std::string x_field_name_, y_field_name_, z_field_name_;
# 
#       bool initCompute ();
#       bool deinitCompute ();
#     public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
###

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

# definitions used everywhere
ctypedef shared_ptr[vector[int]] IndicesPtr_t;
# ctypedef shared_ptr[vector[int]] IndicesPtrConst_t;

# pcl_base.h
# template <typename PointT>
# class PCLBase
cdef extern from "pcl/pcl_base.h" namespace "pcl":
    cdef cppclass PCLBase[T]:
        PCLBase ()
        # PCLBase (const PCLBase& base)
        # virtual void setInputCloud (PointCloudPtr_t cloud)
        void setInputCloud (PointCloudPtr_t cloud)
        PointCloudPtr_t getInputCloud ()
        void setIndices (IndicesPtr_t &indices)
        # #  void setIndices (const IndicesConstPtr &indices)
        void setIndices (const PointIndicesPtr_t &indices)
        void setIndices (size_t row_start, size_t col_start, size_t nb_rows, size_t nb_cols)
        IndicesPtr_t const getIndices ()
        # # const PointT& operator[] (size_t pos)
        # # public:
        # # EIGEN_MAKE_ALIGNED_OPERATOR_NEW



###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################
