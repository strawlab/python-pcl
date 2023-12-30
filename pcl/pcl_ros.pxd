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

# conversions.h
# namespace pcl
# namespace detail
# // For converting template point cloud to message.
# template<typename PointT>
# struct FieldAdder
#     {
#       FieldAdder (std::vector<sensor_msgs::PointField>& fields) : fields_ (fields) {};
#       template<typename U> void operator() ()
#       std::vector<sensor_msgs::PointField>& fields_;
#     };
###

# conversions.h
# namespace pcl
# namespace detail
# // For converting message to template point cloud.
# template<typename PointT>
# struct FieldMapper
#     {
#       FieldMapper (const std::vector<sensor_msgs::PointField>& fields, std::vector<FieldMapping>& map) : fields_ (fields), map_ (map)
#       template<typename Tag> void operator () ()
# 
#       const std::vector<sensor_msgs::PointField>& fields_;
#       std::vector<FieldMapping>& map_;
#     };
# 
#     inline bool fieldOrdering (const FieldMapping& a, const FieldMapping& b)
#     {
#       return (a.serialized_offset < b.serialized_offset);
#     }
# 
#   } //namespace detail

# conversions.h
# namespace pcl
# template<typename PointT> void 
# createMapping (const std::vector<sensor_msgs::PointField>& msg_fields, MsgFieldMap& field_map)
#   {
#     // Create initial 1-1 mapping between serialized data segments and struct fields
#     detail::FieldMapper<PointT> mapper (msg_fields, field_map);
#     for_each_type< typename traits::fieldList<PointT>::type > (mapper);
# 
#     // Coalesce adjacent fields into single memcpy's where possible
#     if (field_map.size() > 1)
#     {
#       std::sort(field_map.begin(), field_map.end(), detail::fieldOrdering);
#       MsgFieldMap::iterator i = field_map.begin(), j = i + 1;
#       while (j != field_map.end())
#       {
#         // This check is designed to permit padding between adjacent fields.
#         /// @todo One could construct a pathological case where the struct has a
#         /// field where the serialized data has padding
#         if (j->serialized_offset - i->serialized_offset == j->struct_offset - i->struct_offset)
#         {
#           i->size += (j->struct_offset + j->size) - (i->struct_offset + i->size);
#           j = field_map.erase(j);
#         }
#         else
#         {
#           ++i;
#           ++j;
#         }
#       }
#     }
#   }
###

# conversions.h
# namespace pcl
# /** \brief Convert a PointCloud2 binary data blob into a pcl::PointCloud<T> object using a field_map.
#   * \param[in] msg the PointCloud2 binary blob
#   * \param[out] cloud the resultant pcl::PointCloud<T>
#   * \param[in] field_map a MsgFieldMap object
#   *
#   * \note Use fromROSMsg (PointCloud2, PointCloud<T>) directly or create you
#   * own MsgFieldMap using:
#   *
#   * \code
#   * MsgFieldMap field_map;
#   * createMapping<PointT> (msg.fields, field_map);
#   * \endcode
#   */
# template <typename PointT> void 
# fromROSMsg (const sensor_msgs::PointCloud2& msg, pcl::PointCloud<PointT>& cloud, const MsgFieldMap& field_map)
# {
# }
###

# conversions.h
# namespace pcl
# /** \brief Convert a PointCloud2 binary data blob into a pcl::PointCloud<T> object.
#   * \param[in] msg the PointCloud2 binary blob
#   * \param[out] cloud the resultant pcl::PointCloud<T>
#   */
# template<typename PointT> void 
# fromROSMsg (const sensor_msgs::PointCloud2& msg, pcl::PointCloud<PointT>& cloud)
###

# conversions.h
# namespace pcl
# /** \brief Convert a pcl::PointCloud<T> object to a PointCloud2 binary data blob.
#   * \param[in] cloud the input pcl::PointCloud<T>
#   * \param[out] msg the resultant PointCloud2 binary blob
#   */
# template<typename PointT> void 
# toROSMsg (const pcl::PointCloud<PointT>& cloud, sensor_msgs::PointCloud2& msg)
###

# conversions.h
# namespace pcl
# /** \brief Copy the RGB fields of a PointCloud into sensor_msgs::Image format
#   * \param[in] cloud the point cloud message
#   * \param[out] msg the resultant sensor_msgs::Image
#   * CloudT cloud type, CloudT should be akin to pcl::PointCloud<pcl::PointXYZRGBA>
#   * \note will throw std::runtime_error if there is a problem
#   */
# template<typename CloudT> void
# toROSMsg (const CloudT& cloud, sensor_msgs::Image& msg)
###

# conversions.h
# namespace pcl
# /** \brief Copy the RGB fields of a PointCloud2 msg into sensor_msgs::Image format
#   * \param cloud the point cloud message
#   * \param msg the resultant sensor_msgs::Image
#   * will throw std::runtime_error if there is a problem
#   */
# inline void toROSMsg (const sensor_msgs::PointCloud2& cloud, sensor_msgs::Image& msg)
###


# register_point_struct.h
# #include <pcl/register_point_struct.h>
# // Must be used in global namespace with name fully qualified
# #define POINT_CLOUD_REGISTER_POINT_STRUCT(name, fseq)               \
#   POINT_CLOUD_REGISTER_POINT_STRUCT_I(name,                         \
#     BOOST_PP_CAT(POINT_CLOUD_REGISTER_POINT_STRUCT_X fseq, 0))      \
#   /***/
# 
# #define POINT_CLOUD_REGISTER_POINT_WRAPPER(wrapper, pod)    \
#   BOOST_MPL_ASSERT_MSG(sizeof(wrapper) == sizeof(pod), POINT_WRAPPER_AND_POD_TYPES_HAVE_DIFFERENT_SIZES, (wrapper&, pod&)); \
#   namespace pcl {                                           \
#     namespace traits {                                      \
#       template<> struct POD<wrapper> { typedef pod type; }; \
#     }                                                       \
#   }                                                         \
#   /***/
# 
# // These macros help transform the unusual data structure (type, name, tag)(type, name, tag)...
# // into a proper preprocessor sequence of 3-tuples ((type, name, tag))((type, name, tag))...
# #define POINT_CLOUD_REGISTER_POINT_STRUCT_X(type, name, tag)            \
#   ((type, name, tag)) POINT_CLOUD_REGISTER_POINT_STRUCT_Y
# #define POINT_CLOUD_REGISTER_POINT_STRUCT_Y(type, name, tag)            \
#   ((type, name, tag)) POINT_CLOUD_REGISTER_POINT_STRUCT_X
# #define POINT_CLOUD_REGISTER_POINT_STRUCT_X0
# #define POINT_CLOUD_REGISTER_POINT_STRUCT_Y0
# 
# // Construct type traits given full sequence of (type, name, tag) triples
# //  BOOST_MPL_ASSERT_MSG(boost::is_pod<name>::value,                    
# //                       REGISTERED_POINT_TYPE_MUST_BE_PLAIN_OLD_DATA, (name)); 
# #define POINT_CLOUD_REGISTER_POINT_STRUCT_I(name, seq)                           \
#   namespace pcl                                                                  \
#   {                                                                              \
#     namespace fields                                                             \
#     {                                                                            \
#       BOOST_PP_SEQ_FOR_EACH(POINT_CLOUD_REGISTER_FIELD_TAG, name, seq)           \
#     }                                                                            \
#     namespace traits                                                             \
#     {                                                                            \
#       BOOST_PP_SEQ_FOR_EACH(POINT_CLOUD_REGISTER_FIELD_NAME, name, seq)          \
#       BOOST_PP_SEQ_FOR_EACH(POINT_CLOUD_REGISTER_FIELD_OFFSET, name, seq)        \
#       BOOST_PP_SEQ_FOR_EACH(POINT_CLOUD_REGISTER_FIELD_DATATYPE, name, seq)      \
#       POINT_CLOUD_REGISTER_POINT_FIELD_LIST(name, POINT_CLOUD_EXTRACT_TAGS(seq)) \
#     }                                                                            \
#   }                                                                              \
#   /***/
# 
# #define POINT_CLOUD_REGISTER_FIELD_TAG(r, name, elem)   \
#   struct BOOST_PP_TUPLE_ELEM(3, 2, elem);               \
#   /***/
# 
# #define POINT_CLOUD_REGISTER_FIELD_NAME(r, point, elem)                 \
#   template<int dummy>                                                   \
#   struct name<point, pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem), dummy> \
#   {                                                                     \
#     static const char value[];                                          \
#   };                                                                    \
#                                                                         \
#   template<int dummy>                                                   \
#   const char name<point,                                                \
#                   pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem),         \
#                   dummy>::value[] =                                     \
#     BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(3, 2, elem));                \
#   /***/
# 
# #define POINT_CLOUD_REGISTER_FIELD_OFFSET(r, name, elem)                \
#   template<> struct offset<name, pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem)> \
#   {                                                                     \
#     static const size_t value = offsetof(name, BOOST_PP_TUPLE_ELEM(3, 1, elem)); \
#   };                                                                    \
#   /***/
# 
# // \note: the mpl::identity weirdness is to support array types without requiring the
# // user to wrap them. The basic problem is:
# // typedef float[81] type; // SYNTAX ERROR!
# // typedef float type[81]; // OK, can now use "type" as a synonym for float[81]
# #define POINT_CLOUD_REGISTER_FIELD_DATATYPE(r, name, elem)              \
#   template<> struct datatype<name, pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem)> \
#   {                                                                     \
#     typedef boost::mpl::identity<BOOST_PP_TUPLE_ELEM(3, 0, elem)>::type type; \
#     typedef decomposeArray<type> decomposed;                            \
#     static const uint8_t value = asEnum<decomposed::type>::value;       \
#     static const uint32_t size = decomposed::value;                     \
#   };                                                                    \
#   /***/
# 
# #define POINT_CLOUD_TAG_OP(s, data, elem) pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem)
# 
# #define POINT_CLOUD_EXTRACT_TAGS(seq) BOOST_PP_SEQ_TRANSFORM(POINT_CLOUD_TAG_OP, _, seq)
# 
# #define POINT_CLOUD_REGISTER_POINT_FIELD_LIST(name, seq)        \
#   template<> struct fieldList<name>                             \
#   {                                                             \
#     typedef boost::mpl::vector<BOOST_PP_SEQ_ENUM(seq)> type;    \
#   };                                                            \
#   /***/
# 
# // Disabling barely-used Fusion registration of point types for now.
# #if 0
# #define POINT_CLOUD_EXPAND_TAG_OP(s, data, elem)                \
#   (boost::mpl::identity<BOOST_PP_TUPLE_ELEM(3, 0, elem)>::type, \
#    BOOST_PP_TUPLE_ELEM(3, 1, elem),                             \
#    pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem))                \
#   /***/
# 
# #define POINT_CLOUD_EXPAND_TAGS(seq) BOOST_PP_SEQ_TRANSFORM(POINT_CLOUD_EXPAND_TAG_OP, _, seq)
# 
# #define POINT_CLOUD_REGISTER_WITH_FUSION(name, seq)                     \
#   BOOST_FUSION_ADAPT_ASSOC_STRUCT_I(name, POINT_CLOUD_EXPAND_TAGS(seq)) \
#   /***/
# #endif
# 
# 
###
