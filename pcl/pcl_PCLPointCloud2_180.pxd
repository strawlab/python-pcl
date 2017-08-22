from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

# boost
from boost_shared_ptr cimport shared_ptr

###############################################################################
# Types
###############################################################################

# PCLHeader.h
cdef extern from "pcl/PCLHeader.h" namespace "pcl":
    cdef struct PCLHeader:
        PCLHeader ()
        unsigned int seq
        unsigned long stamp
        string frame_id


# inline std::ostream& operator << (std::ostream& out, const PCLHeader &h)

# typedef boost::shared_ptr<PCLHeader> HeaderPtr;
# typedef boost::shared_ptr<PCLHeader const> HeaderConstPtr;
ctypedef shared_ptr[PCLHeader] PCLHeaderPtr_t
ctypedef shared_ptr[const PCLHeader] PCLHeaderConstPtr_t
###

# PCLImage.h
cdef extern from "pcl/PCLImage.h" namespace "pcl":
    cdef struct PCLImage:
        PCLImage ()
        PCLHeader header
        unsigned int height
        unsigned int width
        string encoding
        unsigned char is_bigendian
        unsigned int step;
        vector[unsigned int] data;


# inline std::ostream& operator<<(std::ostream& s, const  ::pcl::PCLImage & v)

ctypedef shared_ptr[PCLImage] PCLImagePtr_t
ctypedef shared_ptr[const PCLImage] PCLImageConstPtr_t
###

# PCLPointField
cdef extern from "pcl/PCLPointField.h" namespace "pcl":
    cdef struct PCLPointField:
        PCLPointField ()
        string name
        unsigned int offset
        unsigned char datatype
        unsigned int count

# inline std::ostream& operator<<(std::ostream& s, const  ::pcl::PCLPointField & v)

ctypedef shared_ptr[PCLPointField] PCLPointFieldPtr_t
ctypedef shared_ptr[const PCLPointField] PCLPointFieldConstPtr_t
###


######
# namespace pcl
cdef extern from "pcl/PCLPointCloud2.h" namespace "pcl":
    cdef struct PCLPointCloud2:
        PointCloud2 ()
        PCLHeader header
        unsigned int height
        unsigned int width
        vector[PCLPointField] fields
        unsigned char is_bigendian
        unsigned int point_step
        unsigned int row_step
        vector[unsigned char] data
        unsigned char is_dense

# inline std::ostream& operator<<(std::ostream& s, const  ::pcl::PCLPointCloud2 &v)

# ctypedef shared_ptr[PCLPointCloud2] PCLPointCloud2Ptr_t
# ctypedef shared_ptr[const PCLPointCloud2] PCLPointCloud2ConstPtr_t
ctypedef cpp.PointCloud[PCLPointCloud2] PointCloud_PCLPointCloud2_t
ctypedef shared_ptr[cpp.PointCloud[PCLPointCloud2]] PointCloud_PCLPointCloud2Ptr_t
###

# pcl/conversions.h
# namespace pcl
# name space detail
# // For converting template point cloud to message.
# template<typename PointT>
# struct FieldAdder
# cdef extern from "pcl/conversions.h" namespace "pcl::detail":
#     cdef struct FieldAdder[PointT]:
#         FieldAdder (vector[PCLPointField]& fields)
#         # template<typename U> void operator()
#         vector[PCLPointField] &fields_
# 
# 
###

# cdef extern from "pcl/conversions.h" namespace "pcl::detail":
#     cdef struct FieldMapper[PointT]:
#         FieldMapper (const vector[PCLPointField] &fields, vector[FieldMapping] &map)
#         # template<typename Tag> void operator ()
#         const vector[PCLPointField] & fields_
#         vector[FieldMapping] & map_
# 
#     inline bool fieldOrdering (const FieldMapping& a, const FieldMapping& b)
###

# pcl/conversions.h
# namespace pcl
# template<typename PointT> void createMapping (const std::vector<pcl::PCLPointField>& msg_fields, MsgFieldMap& field_map)
# cdef extern from "pcl/conversions.h" namespace "pcl":
#     void createMapping [PointT](const vector[PCLPointField]& msg_fields, MsgFieldMap& field_map)
# 
# 
###

# pcl/conversions.h
# namespace pcl
# /** \brief Convert a PCLPointCloud2 binary data blob into a pcl::PointCloud<T> object using a field_map.
# * \param[in] msg the PCLPointCloud2 binary blob
# * \param[out] cloud the resultant pcl::PointCloud<T>
# * \param[in] field_map a MsgFieldMap object
# * \note Use fromPCLPointCloud2 (PCLPointCloud2, PointCloud<T>) directly or create you
# * own MsgFieldMap using:
# * \code
# * MsgFieldMap field_map;
# * createMapping<PointT> (msg.fields, field_map);
# * \endcode
# */
# template <typename PointT> void fromPCLPointCloud2 (const pcl::PCLPointCloud2& msg, pcl::PointCloud<PointT>& cloud, const MsgFieldMap& field_map)
# cdef extern from "pcl/conversions.h" namespace "pcl":
#     void fromPCLPointCloud2 [PointT](const PCLPointCloud2& msg, PointCloud[PointT] & cloud, const MsgFieldMap& field_map)
# 
# 
###

# pcl/conversions.h
# namespace pcl
# /** \brief Convert a PCLPointCloud2 binary data blob into a pcl::PointCloud<T> object.
# * \param[in] msg the PCLPointCloud2 binary blob
# * \param[out] cloud the resultant pcl::PointCloud<T>
# */
# template<typename PointT> void fromPCLPointCloud2 (const pcl::PCLPointCloud2& msg, pcl::PointCloud<PointT>& cloud)
# cdef extern from "pcl/conversions.h" namespace "pcl":
#     void fromPCLPointCloud2 [PointT](const PCLPointCloud2& msg, PointCloud[PointT]& cloud)
# 
# 
###

# pcl/conversions.h
# namespace pcl
# /** \brief Convert a pcl::PointCloud<T> object to a PCLPointCloud2 binary data blob.
# * \param[in] cloud the input pcl::PointCloud<T>
# * \param[out] msg the resultant PCLPointCloud2 binary blob
# */
# template<typename PointT> void toPCLPointCloud2 (const pcl::PointCloud<PointT>& cloud, pcl::PCLPointCloud2& msg)
# cdef extern from "pcl/conversions.h" namespace "pcl":
#     void toPCLPointCloud2 [PointT](const PointCloud[PointT]& cloud, PCLPointCloud2& msg)
# 
# 
###

# pcl/conversions.h
# namespace pcl
# /** \brief Copy the RGB fields of a PointCloud into pcl::PCLImage format
#  * \param[in] cloud the point cloud message
#  * \param[out] msg the resultant pcl::PCLImage
#  * CloudT cloud type, CloudT should be akin to pcl::PointCloud<pcl::PointXYZRGBA>
#  * \note will throw std::runtime_error if there is a problem
#  */
# template<typename CloudT> void toPCLPointCloud2 (const CloudT& cloud, pcl::PCLImage& msg)
# cdef extern from "pcl/conversions.h" namespace "pcl":
#     void toPCLPointCloud2 [CloudT](const CloudT& cloud, PCLImage& msg)
# 
# 
###

# pcl/conversions.h
# namespace pcl
# /** 
# * \brief Copy the RGB fields of a PCLPointCloud2 msg into pcl::PCLImage format
# * \param cloud the point cloud message
# * \param msg the resultant pcl::PCLImage
# * will throw std::runtime_error if there is a problem
# */
# inline void toPCLPointCloud2 (const pcl::PCLPointCloud2& cloud, pcl::PCLImage& msg)
# cdef extern from "pcl/conversions.h" namespace "pcl":
#     void toPCLPointCloud2 (const PCLPointCloud2& cloud, PCLImage& msg)
# 
# 
###

###############################################################################
# Enum
###############################################################################

cdef extern from "pcl/PCLPointField.h" namespace "pcl":
    cdef enum:
        INT8 = 1
        UINT8 = 2
        INT16 = 3
        UINT16 = 4
        INT32 = 5
        UINT32 = 6
        FLOAT32 = 7
        FLOAT64 = 8

###############################################################################
# Activation
###############################################################################

