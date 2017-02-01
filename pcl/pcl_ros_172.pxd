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

# common\include\pcl\ros\conversions.h
# namespace pcl
# {
    # /** \brief Convert a PCLPointCloud2 binary data blob into a pcl::PointCloud<T> object using a field_map.
    #   * \param[in] msg the PCLPointCloud2 binary blob
    #   * \param[out] cloud the resultant pcl::PointCloud<T>
    #   * \param[in] field_map a MsgFieldMap object
    #   *
    #   * \note Use fromROSMsg (PCLPointCloud2, PointCloud<T>) directly or create you
    #   * own MsgFieldMap using:
    #   *
    #   * \code
    #   * MsgFieldMap field_map;
    #   * createMapping<PointT> (msg.fields, field_map);
    #   * \endcode
    # */
    # template <typename PointT>
    # PCL_DEPRECATED ("pcl::fromROSMsg is deprecated, please use fromPCLPointCloud2 instead.")
    # void fromROSMsg (const pcl::PCLPointCloud2& msg, pcl::PointCloud<PointT>& cloud, const MsgFieldMap& field_map)
    # 
    # /** \brief Convert a PCLPointCloud2 binary data blob into a pcl::PointCloud<T> object.
    #     * \param[in] msg the PCLPointCloud2 binary blob
    #   * \param[out] cloud the resultant pcl::PointCloud<T>
    # */
    # template<typename PointT>
    # PCL_DEPRECATED ("pcl::fromROSMsg is deprecated, please use fromPCLPointCloud2 instead.")
    # void fromROSMsg (const pcl::PCLPointCloud2& msg, pcl::PointCloud<PointT>& cloud)
    # 
    # /** \brief Convert a pcl::PointCloud<T> object to a PCLPointCloud2 binary data blob.
    #   * \param[in] cloud the input pcl::PointCloud<T>
    #   * \param[out] msg the resultant PCLPointCloud2 binary blob
    # */
    # template<typename PointT>
    # PCL_DEPRECATED ("pcl::fromROSMsg is deprecated, please use fromPCLPointCloud2 instead.")
    # void toROSMsg (const pcl::PointCloud<PointT>& cloud, pcl::PCLPointCloud2& msg)
    # 
    # /** \brief Copy the RGB fields of a PointCloud into pcl::PCLImage format
    #   * \param[in] cloud the point cloud message
    #   * \param[out] msg the resultant pcl::PCLImage
    #   * CloudT cloud type, CloudT should be akin to pcl::PointCloud<pcl::PointXYZRGBA>
    #   * \note will throw std::runtime_error if there is a problem
    #  */
    # template<typename CloudT>
    # PCL_DEPRECATED ("pcl::fromROSMsg is deprecated, please use fromPCLPointCloud2 instead.")
    # void toROSMsg (const CloudT& cloud, pcl::PCLImage& msg)
    # 
    # /** \brief Copy the RGB fields of a PCLPointCloud2 msg into pcl::PCLImage format
    #   * \param cloud the point cloud message
    #   * \param msg the resultant pcl::PCLImage
    #   * will throw std::runtime_error if there is a problem
    # */
    # inline void
    # PCL_DEPRECATED ("pcl::fromROSMsg is deprecated, please use fromPCLPointCloud2 instead.")
    # toROSMsg (const pcl::PCLPointCloud2& cloud, pcl::PCLImage& msg)


###


# common\include\pcl\ros\register_point_struct.h
# changed pcl/register_point_struct.h
# include <pcl/register_point_struct.h>
###


