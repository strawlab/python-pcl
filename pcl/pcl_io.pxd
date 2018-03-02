# -*- coding: utf-8 -*-

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

# main
cimport pcl_defs as cpp

# from boost_shared_ptr cimport shared_ptr

cdef extern from "pcl/io/pcd_io.h" namespace "pcl::io":
    # Template 
    int load [PointT](string file_name, cpp.PointCloud[PointT] &cloud) nogil except +
    int loadPCDFile [PointT](string file_name, cpp.PointCloud[PointT] &cloud) nogil except +
    int savePCDFile [PointT](string file_name, cpp.PointCloud[PointT] &cloud, bool binary_mode) nogil except +
    int savePCDFileASCII [PointT](string file_name, cpp.PointCloud[PointT] &cloud) nogil except +
    int savePCDFileBinary [PointT](string &file_name, cpp.PointCloud[PointT] &cloud) nogil except +
    int savePCDFile_Index "savePCDFile" [PointT](string &file_name, 
                     cpp.PointCloud[PointT] &cloud,
                     vector[int] &indices, 
                     bool binary_mode) nogil except +


cdef extern from "pcl/io/ply_io.h" namespace "pcl::io":
    # Template 
    int loadPLYFile [PointT](string file_name,
                             cpp.PointCloud[PointT] &cloud) nogil except +
    int savePLYFile [PointT](string file_name, cpp.PointCloud[PointT] &cloud,
                    bool binary_mode) nogil except +


###

# namespace pcl
# {
#   class PCL_EXPORTS MTLReader
#   {
#     public:
#       /** \brief empty constructor */
#       MTLReader ();
# 
#       /** \brief empty destructor */
#       virtual ~MTLReader() {}
# 
#       /** \brief Read a MTL file given its full path.
#         * \param[in] filename full path to MTL file
#         * \return 0 on success < 0 else.
#         */
#       int
#       read (const std::string& filename);
# 
#       /** \brief Read a MTL file given an OBJ file full path and the MTL file name.
#         * \param[in] obj_file_name full path to OBJ file
#         * \param[in] mtl_file_name MTL file name
#         * \return 0 on success < 0 else.
#         */
#       int
#       read (const std::string& obj_file_name, const std::string& mtl_file_name);
# 
#       std::vector<pcl::TexMaterial>::const_iterator
#       getMaterial (const std::string& material_name) const;
# 
#       /// materials array
#       std::vector<pcl::TexMaterial> materials_;
# 
#     private:
#       /// converts CIE XYZ to RGB
#       inline void
#       cie2rgb (const Eigen::Vector3f& xyz, pcl::TexMaterial::RGB& rgb) const;
#       /// fill a pcl::TexMaterial::RGB from a split line containing CIE x y z values
#       int
#       fillRGBfromXYZ (const std::vector<std::string>& split_line, pcl::TexMaterial::RGB& rgb);
#       /// fill a pcl::TexMaterial::RGB from a split line containing r g b values
#       int
#       fillRGBfromRGB (const std::vector<std::string>& split_line, pcl::TexMaterial::RGB& rgb);
#       /// matrix to convert CIE to RGB
#       Eigen::Matrix3f xyz_to_rgb_matrix_;
# 
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };

# cdef extern from "pcl/io/obj_io.h" namespace "pcl::io":
#     # Template 
#     # /** \brief Load any OBJ file into a templated PointCloud type
#     #   * \param[in] file_name the name of the file to load
#     #   * \param[out] cloud the resultant templated point cloud
#     #   * \ingroup io
#     #   */
#     # template<typename PointT> inline int
#     # loadOBJFile (const std::string &file_name, pcl::PointCloud<PointT> &cloud)
#     int loadOBJFile [PointT](string file_name, cpp.PointCloud[PointT] &cloud) nogil except +
# 
#     # /** \brief Load any OBJ file into a templated PointCloud type.
#     #   * \param[in] file_name the name of the file to load
#     #   * \param[out] cloud the resultant templated point cloud
#     #   * \param[out] origin the sensor acquisition origin, null
#     #   * \param[out] orientation the sensor acquisition orientation, identity
#     #   * \ingroup io
#     #   */
#     # inline int loadOBJFile (const std::string &file_name, pcl::PCLPointCloud2 &cloud, Eigen::Vector4f &origin, Eigen::Quaternionf &orientation)
# 
#     # /** \brief Load an OBJ file into a PCLPointCloud2 blob type.
#     #   * \param[in] file_name the name of the file to load
#     #   * \param[out] cloud the resultant templated point cloud
#     #   * \return 0 on success < 0 on error
#     #   * \ingroup io
#     #   */
#     # inline int loadOBJFile (const std::string &file_name, pcl::PCLPointCloud2 &cloud)
# 
#     # /** \brief Load any OBJ file into a PolygonMesh type.
#     #   * \param[in] file_name the name of the file to load
#     #   * \param[out] mesh the resultant mesh
#     #   * \return 0 on success < 0 on error
#     #   *
#     #   * \ingroup io
#     #   */
#     # inline int
#     # loadOBJFile (const std::string &file_name, pcl::PolygonMesh &mesh)
# 
#     # /** \brief Load any OBJ file into a TextureMesh type.
#     #   * \param[in] file_name the name of the file to load
#     #   * \param[out] mesh the resultant mesh
#     #   * \return 0 on success < 0 on error
#     #   *
#     #   * \ingroup io
#     #   */
#     # inline int
#     # loadOBJFile (const std::string &file_name, pcl::TextureMesh &mesh)
# 
#     # /** \brief Saves a TextureMesh in ascii OBJ format.
#     #   * \param[in] file_name the name of the file to write to disk
#     #   * \param[in] tex_mesh the texture mesh to save
#     #   * \param[in] precision the output ASCII precision
#     #   * \ingroup io
#     #   */
#     # PCL_EXPORTS int
#     # saveOBJFile (const std::string &file_name,
#     #              const pcl::TextureMesh &tex_mesh,
#     #              unsigned precision = 5);
# 
#     # /** \brief Saves a PolygonMesh in ascii PLY format.
#     #   * \param[in] file_name the name of the file to write to disk
#     #   * \param[in] mesh the polygonal mesh to save
#     #   * \param[in] precision the output ASCII precision default 5
#     #   * \ingroup io
#     #   */
#     # PCL_EXPORTS int
#     # saveOBJFile (const std::string &file_name,
#     #              const pcl::PolygonMesh &mesh,
#     #              unsigned precision = 5);


###

#http://dev.pointclouds.org/issues/624
#cdef extern from "pcl/io/ply_io.h" namespace "pcl::io":
#    int loadPLYFile (string file_name, PointCloud[cpp.PointXYZ] cloud)
#    int savePLYFile (string file_name, PointCloud[cpp.PointXYZ] cloud, bool binary_mode)

# cdef extern from "pcl/io/ply_io.h" namespace "pcl::io":

###
# file_io.h
# namespace pcl
# class PCL_EXPORTS FileReader
#     public:
#       /** \brief empty constructor */ 
#       FileReader() {}
#       /** \brief empty destructor */ 
#       virtual ~FileReader() {}
#       /** \brief Read a point cloud data header from a FILE file. 
#         * Load only the meta information (number of points, their types, etc),
#         * and not the points themselves, from a given FILE file. Useful for fast
#         * evaluation of the underlying data structure.
#         * Returns:
#         *  * < 0 (-1) on error
#         *  * > 0 on success
#         * \param[in] file_name the name of the file to load
#         * \param[out] cloud the resultant point cloud dataset (only the header will be filled)
#         * \param[out] origin the sensor acquisition origin (only for > FILE_V7 - null if not present)
#         * \param[out] orientation the sensor acquisition orientation (only for > FILE_V7 - identity if not present)
#         * \param[out] file_version the FILE version of the file (either FILE_V6 or FILE_V7)
#         * \param[out] data_type the type of data (binary data=1, ascii=0, etc)
#         * \param[out] data_idx the offset of cloud data within the file
#         * \param[in] offset the offset in the file where to expect the true header to begin.
#         * One usage example for setting the offset parameter is for reading
#         * data from a TAR "archive containing multiple files: TAR files always
#         * add a 512 byte header in front of the actual file, so set the offset
#         * to the next byte after the header (e.g., 513).
#         */
#       virtual int 
#       readHeader (const std::string &file_name, sensor_msgs::PointCloud2 &cloud, 
#                   Eigen::Vector4f &origin, Eigen::Quaternionf &orientation, 
#                   int &file_version, int &data_type, unsigned int &data_idx, const int offset = 0) = 0;
# 
#       /** \brief Read a point cloud data from a FILE file and store it into a sensor_msgs/PointCloud2.
#         * \param[in] file_name the name of the file containing the actual PointCloud data
#         * \param[out] cloud the resultant PointCloud message read from disk
#         * \param[out] origin the sensor acquisition origin (only for > FILE_V7 - null if not present)
#         * \param[out] orientation the sensor acquisition orientation (only for > FILE_V7 - identity if not present)
#         * \param[out] file_version the FILE version of the file (either FILE_V6 or FILE_V7)
#         * \param[in] offset the offset in the file where to expect the true header to begin.
#         * One usage example for setting the offset parameter is for reading
#         * data from a TAR "archive containing multiple files: TAR files always
#         * add a 512 byte header in front of the actual file, so set the offset
#         * to the next byte after the header (e.g., 513).
#         */
#       virtual int 
#       read (const std::string &file_name, sensor_msgs::PointCloud2 &cloud, 
#             Eigen::Vector4f &origin, Eigen::Quaternionf &orientation, int &file_version, 
#             const int offset = 0) = 0;
# 
#       /** \brief Read a point cloud data from a FILE file (FILE_V6 only!) and store it into a sensor_msgs/PointCloud2.
#         * 
#         * \note This function is provided for backwards compatibility only and
#         * it can only read FILE_V6 files correctly, as sensor_msgs::PointCloud2
#         * does not contain a sensor origin/orientation. Reading any file 
#         * > FILE_V6 will generate a warning. 
#         *
#         * \param[in] file_name the name of the file containing the actual PointCloud data
#         * \param[out] cloud the resultant PointCloud message read from disk
#         *
#         * \param[in] offset the offset in the file where to expect the true header to begin.
#         * One usage example for setting the offset parameter is for reading
#         * data from a TAR "archive containing multiple files: TAR files always
#         * add a 512 byte header in front of the actual file, so set the offset
#         * to the next byte after the header (e.g., 513).
#         */
#       int 
#       read (const std::string &file_name, sensor_msgs::PointCloud2 &cloud, const int offset = 0)
#       {
#         Eigen::Vector4f origin;
#         Eigen::Quaternionf orientation;
#         int file_version;
#         return (read (file_name, cloud, origin, orientation, file_version, offset));
#       }
# 
#       /** \brief Read a point cloud data from any FILE file, and convert it to the given template format.
#         * \param[in] file_name the name of the file containing the actual PointCloud data
#         * \param[out] cloud the resultant PointCloud message read from disk
#         * \param[in] offset the offset in the file where to expect the true header to begin.
#         * One usage example for setting the offset parameter is for reading
#         * data from a TAR "archive containing multiple files: TAR files always
#         * add a 512 byte header in front of the actual file, so set the offset
#         * to the next byte after the header (e.g., 513).
#         */
#       template<typename PointT> inline int
#       read (const std::string &file_name, pcl::PointCloud<PointT> &cloud, const int offset  =0)
#       {
#         sensor_msgs::PointCloud2 blob;
#         int file_version;
#         int res = read (file_name, blob, cloud.sensor_origin_, cloud.sensor_orientation_, 
#                         file_version, offset);
# 
#         // Exit in case of error
#         if (res < 0)
#           return res;
#         pcl::fromROSMsg (blob, cloud);
#         return (0);
#       }
#   };
# 
#   /** \brief Point Cloud Data (FILE) file format writer.
#     * Any (FILE) format file reader should implement its virtual methodes
#     * \author Nizar Sallem
#     * \ingroup io
#     */
#   class PCL_EXPORTS FileWriter
#   {
#     public:
#       /** \brief Empty constructor */ 
#       FileWriter () {}
# 
#       /** \brief Empty destructor */ 
#       virtual ~FileWriter () {}
# 
#       /** \brief Save point cloud data to a FILE file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         * \param[in] binary set to true if the file is to be written in a binary
#         * FILE format, false (default) for ASCII
#         */
#       virtual int
#       write (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud, 
#              const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#              const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#              const bool binary = false) = 0;
# 
#       /** \brief Save point cloud data to a FILE file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message (boost shared pointer)
#         * \param[in] binary set to true if the file is to be written in a binary
#         * FILE format, false (default) for ASCII
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         */
#       inline int
#       write (const std::string &file_name, const sensor_msgs::PointCloud2::ConstPtr &cloud, 
#              const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#              const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#              const bool binary = false)
#       {
#         return (write (file_name, *cloud, origin, orientation, binary));
#       }
# 
#       /** \brief Save point cloud data to a FILE file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the pcl::PointCloud data
#         * \param[in] binary set to true if the file is to be written in a binary
#         * FILE format, false (default) for ASCII
#         */
#       template<typename PointT> inline int
#       write (const std::string &file_name, 
#              const pcl::PointCloud<PointT> &cloud, 
#              const bool binary = false)
#       {
#         Eigen::Vector4f origin = cloud.sensor_origin_;
#         Eigen::Quaternionf orientation = cloud.sensor_orientation_;
# 
#         sensor_msgs::PointCloud2 blob;
#         pcl::toROSMsg (cloud, blob);
# 
#         // Save the data
#         return (write (file_name, blob, origin, orientation, binary));
#       }
#   };
# 
#   /** \brief insers a value of type Type (uchar, char, uint, int, float, double, ...) into a stringstream.
#     *
#     * If the value is NaN, it inserst "nan".
#     *
#     * \param[in] cloud the cloud to copy from
#     * \param[in] point_index the index of the point
#     * \param[in] point_size the size of the point in the cloud
#     * \param[in] field_idx the index of the dimension/field
#     * \param[in] fields_count the current fields count
#     * \param[out] stream the ostringstream to copy into
#     */
#   template <typename Type> inline void
#   copyValueString (const sensor_msgs::PointCloud2 &cloud, 
#                    const unsigned int point_index, 
#                    const int point_size, 
#                    const unsigned int field_idx, 
#                    const unsigned int fields_count, 
#                    std::ostream &stream)
#   {
#     Type value;
#     memcpy (&value, &cloud.data[point_index * point_size + cloud.fields[field_idx].offset + fields_count * sizeof (Type)], sizeof (Type));
#     if (pcl_isnan (value))
#       stream << "nan";
#     else
#       stream << boost::numeric_cast<Type>(value);
#   }
#   template <> inline void
#   copyValueString<int8_t> (const sensor_msgs::PointCloud2 &cloud, 
#                            const unsigned int point_index, 
#                            const int point_size, 
#                            const unsigned int field_idx, 
#                            const unsigned int fields_count, 
#                            std::ostream &stream)
#   {
#     int8_t value;
#     memcpy (&value, &cloud.data[point_index * point_size + cloud.fields[field_idx].offset + fields_count * sizeof (int8_t)], sizeof (int8_t));
#     if (pcl_isnan (value))
#       stream << "nan";
#     else
#       // Numeric cast doesn't give us what we want for int8_t
#       stream << boost::numeric_cast<int>(value);
#   }
#   template <> inline void
#   copyValueString<uint8_t> (const sensor_msgs::PointCloud2 &cloud, 
#                             const unsigned int point_index, 
#                             const int point_size, 
#                             const unsigned int field_idx, 
#                             const unsigned int fields_count, 
#                             std::ostream &stream)
#   {
#     uint8_t value;
#     memcpy (&value, &cloud.data[point_index * point_size + cloud.fields[field_idx].offset + fields_count * sizeof (uint8_t)], sizeof (uint8_t));
#     if (pcl_isnan (value))
#       stream << "nan";
#     else
#       // Numeric cast doesn't give us what we want for uint8_t
#       stream << boost::numeric_cast<int>(value);
#   }
# 
#   /** \brief Check whether a given value of type Type (uchar, char, uint, int, float, double, ...) is finite or not
#     *
#     * \param[in] cloud the cloud that contains the data
#     * \param[in] point_index the index of the point
#     * \param[in] point_size the size of the point in the cloud
#     * \param[in] field_idx the index of the dimension/field
#     * \param[in] fields_count the current fields count
#     *
#     * \return true if the value is finite, false otherwise
#     */
#   template <typename Type> inline bool
#   isValueFinite (const sensor_msgs::PointCloud2 &cloud, 
#                  const unsigned int point_index, 
#                  const int point_size, 
#                  const unsigned int field_idx, 
#                  const unsigned int fields_count)
#   {
#     Type value;
#     memcpy (&value, &cloud.data[point_index * point_size + cloud.fields[field_idx].offset + fields_count * sizeof (Type)], sizeof (Type));
#     if (!pcl_isfinite (value))
#       return (false);
#     return (true);
#   }
# 
#   /** \brief Copy one single value of type T (uchar, char, uint, int, float, double, ...) from a string
#     * 
#     * Uses aoti/atof to do the conversion.
#     * Checks if the st is "nan" and converts it accordingly.
#     *
#     * \param[in] st the string containing the value to convert and copy
#     * \param[out] cloud the cloud to copy it to
#     * \param[in] point_index the index of the point
#     * \param[in] field_idx the index of the dimension/field
#     * \param[in] fields_count the current fields count
#     */
#   template <typename Type> inline void
#   copyStringValue (const std::string &st, sensor_msgs::PointCloud2 &cloud,
#                    unsigned int point_index, unsigned int field_idx, unsigned int fields_count)
#   {
#     Type value;
#     if (st == "nan")
#     {
#       value = std::numeric_limits<Type>::quiet_NaN ();
#       cloud.is_dense = false;
#     }
#     else
#     {
#       std::istringstream is (st);
#       is.imbue (std::locale::classic ());
#       is >> value;
#     }
# 
#     memcpy (&cloud.data[point_index * cloud.point_step + 
#                         cloud.fields[field_idx].offset + 
#                         fields_count * sizeof (Type)], reinterpret_cast<char*> (&value), sizeof (Type));
#   }
# 
#   template <> inline void
#   copyStringValue<int8_t> (const std::string &st, sensor_msgs::PointCloud2 &cloud,
#                            unsigned int point_index, unsigned int field_idx, unsigned int fields_count)
#   {
#     int8_t value;
#     if (st == "nan")
#     {
#       value = static_cast<int8_t> (std::numeric_limits<int>::quiet_NaN ());
#       cloud.is_dense = false;
#     }
#     else
#     {
#       int val;
#       std::istringstream is (st);
#       is.imbue (std::locale::classic ());
#       is >> val;
#       value = static_cast<int8_t> (val);
#     }
# 
#     memcpy (&cloud.data[point_index * cloud.point_step + 
#                         cloud.fields[field_idx].offset + 
#                         fields_count * sizeof (int8_t)], reinterpret_cast<char*> (&value), sizeof (int8_t));
#   }
# 
#   template <> inline void
#   copyStringValue<uint8_t> (const std::string &st, sensor_msgs::PointCloud2 &cloud,
#                            unsigned int point_index, unsigned int field_idx, unsigned int fields_count)
#   {
#     uint8_t value;
#     if (st == "nan")
#     {
#       value = static_cast<uint8_t> (std::numeric_limits<int>::quiet_NaN ());
#       cloud.is_dense = false;
#     }
#     else
#     {
#       int val;
#       std::istringstream is (st);
#       is.imbue (std::locale::classic ());
#       is >> val;
#       value = static_cast<uint8_t> (val);
#     }
# 
#     memcpy (&cloud.data[point_index * cloud.point_step + 
#                         cloud.fields[field_idx].offset + 
#                         fields_count * sizeof (uint8_t)], reinterpret_cast<char*> (&value), sizeof (uint8_t));
###

# io.h
# #include <pcl/common/io.h>
###

# lzf.h
# namespace pcl
#   PCL_EXPORTS unsigned int 
#   lzfCompress (const void *const in_data,  unsigned int in_len,
#                void             *out_data, unsigned int out_len);
#   PCL_EXPORTS unsigned int 
#   lzfDecompress (const void *const in_data,  unsigned int in_len,
#                  void             *out_data, unsigned int out_len);
###

# obj_io.h
# namespace pcl
#   namespace io
#     PCL_EXPORTS int
#     saveOBJFile (const std::string &file_name, 
#                  const pcl::TextureMesh &tex_mesh, 
#                  unsigned precision = 5);
# 
#     PCL_EXPORTS int
#     saveOBJFile (const std::string &file_name, 
#                  const pcl::PolygonMesh &mesh, 
#                  unsigned precision = 5);
# 
###

# pcd_io.h
# namespace pcl
# {
#   /** \brief Point Cloud Data (PCD) file format reader.
#     * \author Radu Bogdan Rusu
#     * \ingroup io
#     */
#   class PCL_EXPORTS PCDReader : public FileReader
#   {
#     public:
#       /** Empty constructor */      
#       PCDReader () : FileReader () {}
#       /** Empty destructor */      
#       ~PCDReader () {}
#       /** \brief Various PCD file versions.
#         *
#         * PCD_V6 represents PCD files with version 0.6, which contain the following fields:
#         *   - lines beginning with # are treated as comments
#         *   - FIELDS ...
#         *   - SIZE ...
#         *   - TYPE ...
#         *   - COUNT ...
#         *   - WIDTH ...
#         *   - HEIGHT ...
#         *   - POINTS ...
#         *   - DATA ascii/binary
#         * 
#         * Everything that follows \b DATA is intepreted as data points and
#         * will be read accordingly.
#         *
#         * PCD_V7 represents PCD files with version 0.7 and has an important
#         * addon: it adds sensor origin/orientation (aka viewpoint) information
#         * to a dataset through the use of a new header field:
#         *   - VIEWPOINT tx ty tz qw qx qy qz
#         */
#       enum
#       {
#         PCD_V6 = 0,
#         PCD_V7 = 1
#       };
# 
#       /** \brief Read a point cloud data header from a PCD file. 
#         *
#         * Load only the meta information (number of points, their types, etc),
#         * and not the points themselves, from a given PCD file. Useful for fast
#         * evaluation of the underlying data structure.
#         *
#         * \attention The PCD data is \b always stored in ROW major format! The
#         * read/write PCD methods will detect column major input and automatically convert it.
#         *
#         * \param[in] file_name the name of the file to load
#         * \param[out] cloud the resultant point cloud dataset (only the header will be filled)
#         * \param[out] origin the sensor acquisition origin (only for > PCD_V7 - null if not present)
#         * \param[out] orientation the sensor acquisition orientation (only for > PCD_V7 - identity if not present)
#         * \param[out] pcd_version the PCD version of the file (i.e., PCD_V6, PCD_V7)
#         * \param[out] data_type the type of data (0 = ASCII, 1 = Binary, 2 = Binary compressed) 
#         * \param[out] data_idx the offset of cloud data within the file
#         * \param[in] offset the offset of where to expect the PCD Header in the
#         * file (optional parameter). One usage example for setting the offset
#         * parameter is for reading data from a TAR "archive containing multiple
#         * PCD files: TAR files always add a 512 byte header in front of the
#         * actual file, so set the offset to the next byte after the header
#         * (e.g., 513).
#         *
#         * \return
#         *  * < 0 (-1) on error
#         *  * == 0 on success
#         */
#       int 
#       readHeader (const std::string &file_name, sensor_msgs::PointCloud2 &cloud, 
#                   Eigen::Vector4f &origin, Eigen::Quaternionf &orientation, int &pcd_version,
#                   int &data_type, unsigned int &data_idx, const int offset = 0);
# 
# 
#       /** \brief Read a point cloud data header from a PCD file. 
#         *
#         * Load only the meta information (number of points, their types, etc),
#         * and not the points themselves, from a given PCD file. Useful for fast
#         * evaluation of the underlying data structure.
#         *
#         * \attention The PCD data is \b always stored in ROW major format! The
#         * read/write PCD methods will detect column major input and automatically convert it.
#         *
#         * \param[in] file_name the name of the file to load
#         * \param[out] cloud the resultant point cloud dataset (only the header will be filled)
#         * \param[in] offset the offset of where to expect the PCD Header in the
#         * file (optional parameter). One usage example for setting the offset
#         * parameter is for reading data from a TAR "archive containing multiple
#         * PCD files: TAR files always add a 512 byte header in front of the
#         * actual file, so set the offset to the next byte after the header
#         * (e.g., 513).
#         *
#         * \return
#         *  * < 0 (-1) on error
#         *  * == 0 on success
#         */
#       int 
#       readHeader (const std::string &file_name, sensor_msgs::PointCloud2 &cloud, const int offset = 0);
# 
#       /** \brief Read a point cloud data header from a PCD file. 
#         *
#         * Load only the meta information (number of points, their types, etc),
#         * and not the points themselves, from a given PCD file. Useful for fast
#         * evaluation of the underlying data structure.
#         *
#         * \attention The PCD data is \b always stored in ROW major format! The
#         * read/write PCD methods will detect column major input and automatically convert it.
#         *
#         * \param[in] file_name the name of the file to load
#         * \param[out] cloud the resultant point cloud dataset (only the properties will be filled)
#         * \param[out] pcd_version the PCD version of the file (either PCD_V6 or PCD_V7)
#         * \param[out] data_type the type of data (0 = ASCII, 1 = Binary, 2 = Binary compressed) 
#         * \param[out] data_idx the offset of cloud data within the file
#         * \param[in] offset the offset of where to expect the PCD Header in the
#         * file (optional parameter). One usage example for setting the offset
#         * parameter is for reading data from a TAR "archive containing multiple
#         * PCD files: TAR files always add a 512 byte header in front of the
#         * actual file, so set the offset to the next byte after the header
#         * (e.g., 513).
#         *
#         * \return
#         *  * < 0 (-1) on error
#         *  * == 0 on success
#         *
#         */
#       int 
#       readHeaderEigen (const std::string &file_name, pcl::PointCloud<Eigen::MatrixXf> &cloud,
#                        int &pcd_version, int &data_type, unsigned int &data_idx, const int offset = 0);
# 
#       /** \brief Read a point cloud data from a PCD file and store it into a sensor_msgs/PointCloud2.
#         * \param[in] file_name the name of the file containing the actual PointCloud data
#         * \param[out] cloud the resultant PointCloud message read from disk
#         * \param[out] origin the sensor acquisition origin (only for > PCD_V7 - null if not present)
#         * \param[out] orientation the sensor acquisition orientation (only for > PCD_V7 - identity if not present)
#         * \param[out] pcd_version the PCD version of the file (either PCD_V6 or PCD_V7)
#         * \param[in] offset the offset of where to expect the PCD Header in the
#         * file (optional parameter). One usage example for setting the offset
#         * parameter is for reading data from a TAR "archive containing multiple
#         * PCD files: TAR files always add a 512 byte header in front of the
#         * actual file, so set the offset to the next byte after the header
#         * (e.g., 513).
#         *
#         * \return
#         *  * < 0 (-1) on error
#         *  * == 0 on success
#         */
#       int 
#       read (const std::string &file_name, sensor_msgs::PointCloud2 &cloud, 
#             Eigen::Vector4f &origin, Eigen::Quaternionf &orientation, int &pcd_version, const int offset = 0);
# 
#       /** \brief Read a point cloud data from a PCD (PCD_V6) and store it into a sensor_msgs/PointCloud2.
#         * 
#         * \note This function is provided for backwards compatibility only and
#         * it can only read PCD_V6 files correctly, as sensor_msgs::PointCloud2
#         * does not contain a sensor origin/orientation. Reading any file 
#         * > PCD_V6 will generate a warning. 
#         *
#         * \param[in] file_name the name of the file containing the actual PointCloud data
#         * \param[out] cloud the resultant PointCloud message read from disk
#         * \param[in] offset the offset of where to expect the PCD Header in the
#         * file (optional parameter). One usage example for setting the offset
#         * parameter is for reading data from a TAR "archive containing multiple
#         * PCD files: TAR files always add a 512 byte header in front of the
#         * actual file, so set the offset to the next byte after the header
#         * (e.g., 513).
#         *
#         * \return
#         *  * < 0 (-1) on error
#         *  * == 0 on success
#         */
#       int 
#       read (const std::string &file_name, sensor_msgs::PointCloud2 &cloud, const int offset = 0);
# 
#       /** \brief Read a point cloud data from any PCD file, and convert it to the given template format.
#         * \param[in] file_name the name of the file containing the actual PointCloud data
#         * \param[out] cloud the resultant PointCloud message read from disk
#         * \param[in] offset the offset of where to expect the PCD Header in the
#         * file (optional parameter). One usage example for setting the offset
#         * parameter is for reading data from a TAR "archive containing multiple
#         * PCD files: TAR files always add a 512 byte header in front of the
#         * actual file, so set the offset to the next byte after the header
#         * (e.g., 513).
#         *
#         * \return
#         *  * < 0 (-1) on error
#         *  * == 0 on success
#         */
#       template<typename PointT> int
#       read (const std::string &file_name, pcl::PointCloud<PointT> &cloud, const int offset = 0)
#       {
#         sensor_msgs::PointCloud2 blob;
#         int pcd_version;
#         int res = read (file_name, blob, cloud.sensor_origin_, cloud.sensor_orientation_, 
#                         pcd_version, offset);
# 
#         // If no error, convert the data
#         if (res == 0)
#           pcl::fromROSMsg (blob, cloud);
#         return (res);
#       }
# 
#       /** \brief Read a point cloud data from any PCD file, and convert it to a pcl::PointCloud<Eigen::MatrixXf> format.
#         * \attention The PCD data is \b always stored in ROW major format! The
#         * read/write PCD methods will detect column major input and automatically convert it.
#         *
#         * \param[in] file_name the name of the file containing the actual PointCloud data
#         * \param[out] cloud the resultant PointCloud message read from disk
#         * \param[in] offset the offset of where to expect the PCD Header in the
#         * file (optional parameter). One usage example for setting the offset
#         * parameter is for reading data from a TAR "archive containing multiple
#         * PCD files: TAR files always add a 512 byte header in front of the
#         * actual file, so set the offset to the next byte after the header
#         * (e.g., 513).
#         *
#         * \return
#         *  * < 0 (-1) on error
#         *  * == 0 on success
#         */
#       int
#       readEigen (const std::string &file_name, pcl::PointCloud<Eigen::MatrixXf> &cloud, const int offset = 0);
#   };
# 
#   /** \brief Point Cloud Data (PCD) file format writer.
#     * \author Radu Bogdan Rusu
#     * \ingroup io
#     */
#   class PCL_EXPORTS PCDWriter : public FileWriter
#   {
#     public:
#       PCDWriter() : FileWriter(), map_synchronization_(false) {}
#       ~PCDWriter() {}
# 
#       /** \brief Set whether mmap() synchornization via msync() is desired before munmap() calls. 
#         * Setting this to true could prevent NFS data loss (see
#         * http://www.pcl-developers.org/PCD-IO-consistency-on-NFS-msync-needed-td4885942.html).
#         * Default: false
#         * \note This option should be used by advanced users only!
#         * \note Please note that using msync() on certain systems can reduce the I/O performance by up to 80%!
#         * \param[in] sync set to true if msync() should be called before munmap()
#         */
#       void
#       setMapSynchronization (bool sync)
#       {
#         map_synchronization_ = sync;
#       }
# 
#       /** \brief Generate the header of a PCD file format
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         */
#       std::string
#       generateHeaderBinary (const sensor_msgs::PointCloud2 &cloud, 
#                             const Eigen::Vector4f &origin, 
#                             const Eigen::Quaternionf &orientation);
# 
#       /** \brief Generate the header of a BINARY_COMPRESSED PCD file format
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         */
#       std::string
#       generateHeaderBinaryCompressed (const sensor_msgs::PointCloud2 &cloud, 
#                                       const Eigen::Vector4f &origin, 
#                                       const Eigen::Quaternionf &orientation);
# 
#       /** \brief Generate the header of a PCD file format
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         */
#       std::string
#       generateHeaderASCII (const sensor_msgs::PointCloud2 &cloud, 
#                            const Eigen::Vector4f &origin, 
#                            const Eigen::Quaternionf &orientation);
# 
#       /** \brief Generate the header of a PCD file format
#         * \param[in] cloud the point cloud data message
#         * \param[in] nr_points if given, use this to fill in WIDTH, HEIGHT (=1), and POINTS in the header
#         * By default, nr_points is set to INTMAX, and the data in the header is used instead.
#         */
#       template <typename PointT> static std::string
#       generateHeader (const pcl::PointCloud<PointT> &cloud, 
#                       const int nr_points = std::numeric_limits<int>::max ());
# 
#       /** \brief Generate the header of a PCD file format
#         * \note This version is specialized for PointCloud<Eigen::MatrixXf> data types. 
#         * \attention The PCD data is \b always stored in ROW major format! The
#         * read/write PCD methods will detect column major input and automatically convert it.
#         *
#         * \param[in] cloud the point cloud data message
#         * \param[in] nr_points if given, use this to fill in WIDTH, HEIGHT (=1), and POINTS in the header
#         * By default, nr_points is set to INTMAX, and the data in the header is used instead.
#         */
#       std::string
#       generateHeaderEigen (const pcl::PointCloud<Eigen::MatrixXf> &cloud, 
#                            const int nr_points = std::numeric_limits<int>::max ());
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points, in ASCII format
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         * \param[in] precision the specified output numeric stream precision (default: 8)
#         *
#         * Caution: PointCloud structures containing an RGB field have
#         * traditionally used packed float values to store RGB data. Storing a
#         * float as ASCII can introduce variations to the smallest bits, and
#         * thus significantly alter the data. This is a known issue, and the fix
#         * involves switching RGB data to be stored as a packed integer in
#         * future versions of PCL.
#         *
#         * As an intermediary solution, precision 8 is used, which guarantees lossless storage for RGB.
#         */
#       int 
#       writeASCII (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud, 
#                   const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#                   const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#                   const int precision = 8);
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points, in BINARY format
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         */
#       int 
#       writeBinary (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud,
#                    const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#                    const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity ());
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points, in BINARY_COMPRESSED format
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         */
#       int 
#       writeBinaryCompressed (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud,
#                              const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#                              const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity ());
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         * \param[in] binary set to true if the file is to be written in a binary
#         * PCD format, false (default) for ASCII
#         *
#         * Caution: PointCloud structures containing an RGB field have
#         * traditionally used packed float values to store RGB data. Storing a
#         * float as ASCII can introduce variations to the smallest bits, and
#         * thus significantly alter the data. This is a known issue, and the fix
#         * involves switching RGB data to be stored as a packed integer in
#         * future versions of PCL.
#         *
#         * As an intermediary solution, precision 8 is used, which guarantees lossless storage for RGB.
#         */
#       inline int
#       write (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud, 
#              const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#              const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#              const bool binary = false)
#       {
#         if (binary)
#           return (writeBinary (file_name, cloud, origin, orientation));
#         else
#           return (writeASCII (file_name, cloud, origin, orientation, 8));
#       }
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message (boost shared pointer)
#         * \param[in] binary set to true if the file is to be written in a binary PCD format, 
#         * false (default) for ASCII
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         *
#         * Caution: PointCloud structures containing an RGB field have
#         * traditionally used packed float values to store RGB data. Storing a
#         * float as ASCII can introduce variations to the smallest bits, and
#         * thus significantly alter the data. This is a known issue, and the fix
#         * involves switching RGB data to be stored as a packed integer in
#         * future versions of PCL.
#         */
#       inline int
#       write (const std::string &file_name, const sensor_msgs::PointCloud2::ConstPtr &cloud, 
#              const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#              const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#              const bool binary = false)
#       {
#         return (write (file_name, *cloud, origin, orientation, binary));
#       }
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points, in BINARY format
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         */
#       template <typename PointT> int 
#       writeBinary (const std::string &file_name, 
#                    const pcl::PointCloud<PointT> &cloud);
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points, in BINARY format
#         * \note This version is specialized for PointCloud<Eigen::MatrixXf> data types. 
#         * \attention The PCD data is \b always stored in ROW major format! The
#         * read/write PCD methods will detect column major input and automatically convert it.
#         *
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data
#         */
#       int 
#       writeBinaryEigen (const std::string &file_name, 
#                         const pcl::PointCloud<Eigen::MatrixXf> &cloud);
# 
#       /** \brief Save point cloud data to a binary comprssed PCD file
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         */
#       template <typename PointT> int 
#       writeBinaryCompressed (const std::string &file_name, 
#                              const pcl::PointCloud<PointT> &cloud);
# 
#       /** \brief Save point cloud data to a binary comprssed PCD file.
#         * \note This version is specialized for PointCloud<Eigen::MatrixXf> data types. 
#         * \attention The PCD data is \b always stored in ROW major format! The
#         * read/write PCD methods will detect column major input and automatically convert it.
#         *
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         */
#       int 
#       writeBinaryCompressedEigen (const std::string &file_name, 
#                                   const pcl::PointCloud<Eigen::MatrixXf> &cloud);
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points, in BINARY format
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] indices the set of point indices that we want written to disk
#         */
#       template <typename PointT> int 
#       writeBinary (const std::string &file_name, 
#                    const pcl::PointCloud<PointT> &cloud, 
#                    const std::vector<int> &indices);
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points, in ASCII format
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] precision the specified output numeric stream precision (default: 8)
#         */
#       template <typename PointT> int 
#       writeASCII (const std::string &file_name, 
#                   const pcl::PointCloud<PointT> &cloud,
#                   const int precision = 8);
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points, in ASCII format
#         * \note This version is specialized for PointCloud<Eigen::MatrixXf> data types. 
#         * \attention The PCD data is \b always stored in ROW major format! The
#         * read/write PCD methods will detect column major input and automatically convert it.
#         *
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] precision the specified output numeric stream precision (default: 8)
#         */
#       int 
#       writeASCIIEigen (const std::string &file_name, 
#                        const pcl::PointCloud<Eigen::MatrixXf> &cloud,
#                        const int precision = 8);
# 
#        /** \brief Save point cloud data to a PCD file containing n-D points, in ASCII format
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] indices the set of point indices that we want written to disk
#         * \param[in] precision the specified output numeric stream precision (default: 8)
#         */
#       template <typename PointT> int 
#       writeASCII (const std::string &file_name, 
#                   const pcl::PointCloud<PointT> &cloud,
#                   const std::vector<int> &indices,
#                   const int precision = 8);
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the pcl::PointCloud data
#         * \param[in] binary set to true if the file is to be written in a binary
#         * PCD format, false (default) for ASCII
#         *
#         * Caution: PointCloud structures containing an RGB field have
#         * traditionally used packed float values to store RGB data. Storing a
#         * float as ASCII can introduce variations to the smallest bits, and
#         * thus significantly alter the data. This is a known issue, and the fix
#         * involves switching RGB data to be stored as a packed integer in
#         * future versions of PCL.
#         */
#       template<typename PointT> inline int
#       write (const std::string &file_name, 
#              const pcl::PointCloud<PointT> &cloud, 
#              const bool binary = false)
#       {
#         if (binary)
#           return (writeBinary<PointT> (file_name, cloud));
#         else
#           return (writeASCII<PointT> (file_name, cloud));
#       }
# 
#       /** \brief Save point cloud data to a PCD file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the pcl::PointCloud data
#         * \param[in] indices the set of point indices that we want written to disk
#         * \param[in] binary set to true if the file is to be written in a binary
#         * PCD format, false (default) for ASCII
#         *
#         * Caution: PointCloud structures containing an RGB field have
#         * traditionally used packed float values to store RGB data. Storing a
#         * float as ASCII can introduce variations to the smallest bits, and
#         * thus significantly alter the data. This is a known issue, and the fix
#         * involves switching RGB data to be stored as a packed integer in
#         * future versions of PCL.
#         */
#       template<typename PointT> inline int
#       write (const std::string &file_name, 
#              const pcl::PointCloud<PointT> &cloud, 
#              const std::vector<int> &indices,
#              bool binary = false)
#       {
#         if (binary)
#           return (writeBinary<PointT> (file_name, cloud, indices));
#         else
#           return (writeASCII<PointT> (file_name, cloud, indices));
#       }
# 
#     private:
#       /** \brief Set to true if msync() should be called before munmap(). Prevents data loss on NFS systems. */
#       bool map_synchronization_;
# 
#       typedef std::pair<std::string, pcl::ChannelProperties> pair_channel_properties;
#       /** \brief Internal structure used to sort the ChannelProperties in the
#         * cloud.channels map based on their offset. 
#         */
#       struct ChannelPropertiesComparator
#       {
#         bool 
#         operator()(const pair_channel_properties &lhs, const pair_channel_properties &rhs) 
#         {
#           return (lhs.second.offset < rhs.second.offset);
#         }
#       };
#   };
# 
#   namespace io
#   {
#     /** \brief Load a PCD v.6 file into a templated PointCloud type.
#       * 
#       * Any PCD files > v.6 will generate a warning as a
#       * sensor_msgs/PointCloud2 message cannot hold the sensor origin.
#       *
#       * \param[in] file_name the name of the file to load
#       * \param[out] cloud the resultant templated point cloud
#       * \ingroup io
#       */
#     inline int 
#     loadPCDFile (const std::string &file_name, sensor_msgs::PointCloud2 &cloud)
#     {
#       pcl::PCDReader p;
#       return (p.read (file_name, cloud));
#     }
# 
#     /** \brief Load any PCD file into a templated PointCloud type.
#       * \param[in] file_name the name of the file to load
#       * \param[out] cloud the resultant templated point cloud
#       * \param[out] origin the sensor acquisition origin (only for > PCD_V7 - null if not present)
#       * \param[out] orientation the sensor acquisition orientation (only for >
#       * PCD_V7 - identity if not present)
#       * \ingroup io
#       */
#     inline int 
#     loadPCDFile (const std::string &file_name, sensor_msgs::PointCloud2 &cloud,
#                  Eigen::Vector4f &origin, Eigen::Quaternionf &orientation)
#     {
#       pcl::PCDReader p;
#       int pcd_version;
#       return (p.read (file_name, cloud, origin, orientation, pcd_version));
#     }
# 
#     /** \brief Load any PCD file into a templated PointCloud type
#       * \param[in] file_name the name of the file to load
#       * \param[out] cloud the resultant templated point cloud
#       * \ingroup io
#       */
#     template<typename PointT> inline int
#     loadPCDFile (const std::string &file_name, pcl::PointCloud<PointT> &cloud)
#     {
#       pcl::PCDReader p;
#       return (p.read (file_name, cloud));
#     }
# 
#     /** \brief Save point cloud data to a PCD file containing n-D points
#       * \param[in] file_name the output file name
#       * \param[in] cloud the point cloud data message
#       * \param[in] origin the sensor acquisition origin
#       * \param[in] orientation the sensor acquisition orientation
#       * \param[in] binary_mode true for binary mode, false (default) for ASCII
#       *
#       * Caution: PointCloud structures containing an RGB field have
#       * traditionally used packed float values to store RGB data. Storing a
#       * float as ASCII can introduce variations to the smallest bits, and
#       * thus significantly alter the data. This is a known issue, and the fix
#       * involves switching RGB data to be stored as a packed integer in
#       * future versions of PCL.
#       * \ingroup io
#       */
#     inline int 
#     savePCDFile (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud, 
#                  const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#                  const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#                  const bool binary_mode = false)
#     {
#       PCDWriter w;
#       return (w.write (file_name, cloud, origin, orientation, binary_mode));
#     }
# 
#     /** \brief Templated version for saving point cloud data to a PCD file
#       * containing a specific given cloud format
#       * \param[in] file_name the output file name
#       * \param[in] cloud the point cloud data message
#       * \param[in] binary_mode true for binary mode, false (default) for ASCII
#       *
#       * Caution: PointCloud structures containing an RGB field have
#       * traditionally used packed float values to store RGB data. Storing a
#       * float as ASCII can introduce variations to the smallest bits, and
#       * thus significantly alter the data. This is a known issue, and the fix
#       * involves switching RGB data to be stored as a packed integer in
#       * future versions of PCL.
#       * \ingroup io
#       */
#     template<typename PointT> inline int
#     savePCDFile (const std::string &file_name, const pcl::PointCloud<PointT> &cloud, bool binary_mode = false)
#     {
#       PCDWriter w;
#       return (w.write<PointT> (file_name, cloud, binary_mode));
#     }
# 
#     /** 
#       * \brief Templated version for saving point cloud data to a PCD file
#       * containing a specific given cloud format.
#       *
#       *      This version is to retain backwards compatibility.
#       * \param[in] file_name the output file name
#       * \param[in] cloud the point cloud data message
#       *
#       * Caution: PointCloud structures containing an RGB field have
#       * traditionally used packed float values to store RGB data. Storing a
#       * float as ASCII can introduce variations to the smallest bits, and
#       * thus significantly alter the data. This is a known issue, and the fix
#       * involves switching RGB data to be stored as a packed integer in
#       * future versions of PCL.
#       * \ingroup io
#       */
#     template<typename PointT> inline int
#     savePCDFileASCII (const std::string &file_name, const pcl::PointCloud<PointT> &cloud)
#     {
#       PCDWriter w;
#       return (w.write<PointT> (file_name, cloud, false));
#     }
# 
#     /** 
#       * \brief Templated version for saving point cloud data to a PCD file
#       * containing a specific given cloud format.
#       *
#       *      This version is to retain backwards compatibility.
#       * \param[in] file_name the output file name
#       * \param[in] cloud the point cloud data message
#       * \ingroup io
#       */
#     template<typename PointT> inline int
#     savePCDFileBinary (const std::string &file_name, const pcl::PointCloud<PointT> &cloud)
#     {
#       PCDWriter w;
#       return (w.write<PointT> (file_name, cloud, true));
#     }
# 
#     /** 
#       * \brief Templated version for saving point cloud data to a PCD file
#       * containing a specific given cloud format
#       *
#       * \param[in] file_name the output file name
#       * \param[in] cloud the point cloud data message
#       * \param[in] indices the set of indices to save
#       * \param[in] binary_mode true for binary mode, false (default) for ASCII
#       *
#       * Caution: PointCloud structures containing an RGB field have
#       * traditionally used packed float values to store RGB data. Storing a
#       * float as ASCII can introduce variations to the smallest bits, and
#       * thus significantly alter the data. This is a known issue, and the fix
#       * involves switching RGB data to be stored as a packed integer in
#       * future versions of PCL.
#       * \ingroup io
#       */
#     template<typename PointT> int
#     savePCDFile (const std::string &file_name, 
#                  const pcl::PointCloud<PointT> &cloud,
#                  const std::vector<int> &indices, 
#                  const bool binary_mode = false)
#     {
#       // Save the data
#       PCDWriter w;
#       return (w.write<PointT> (file_name, cloud, indices, binary_mode));
#     }
#  }
###

# pcl_io_exception.h
# namespace pcl
# {
#   /** \brief Base exception class for I/O operations
#     * \ingroup io
#     */
#   class PCLIOException : public PCLException
#   {
#     public:
#       /** \brief Constructor
#         * \param[in] error_description a string describing the error
#         * \param[in] file_name the name of the file where the exception was caused
#         * \param[in] function_name the name of the method where the exception was caused
#         * \param[in] line_number the number of the line where the exception was caused
#         */
#       PCLIOException (const std::string& error_description,
#                       const std::string& file_name = "",
#                       const std::string& function_name = "",
#                       unsigned line_number = 0)
#       : PCLException (error_description, file_name, function_name, line_number)
#       {
#       }
#   };
# 
#   /** \brief
#     * \param[in] function_name the name of the method where the exception was caused
#     * \param[in] file_name the name of the file where the exception was caused
#     * \param[in] line_number the number of the line where the exception was caused
#     * \param[in] format printf format
#     * \ingroup io
#     */
#   inline void 
#   throwPCLIOException (const char* function_name, const char* file_name, unsigned line_number, 
#                        const char* format, ...)
#   {
#     char msg[1024];
#     va_list args;
#     va_start (args, format);
#     vsprintf (msg, format, args);
# 
#     throw PCLIOException (msg, file_name, function_name, line_number);
#   }
# 
###

# ply_io.h
# namespace pcl
# {
#   /** \brief Point Cloud Data (PLY) file format reader.
#     *
#     * The PLY data format is organized in the following way:
#     * lines beginning with "comment" are treated as comments
#     *   - ply
#     *   - format [ascii|binary_little_endian|binary_big_endian] 1.0
#     *   - element vertex COUNT
#     *   - property float x 
#     *   - property float y 
#     *   - [property float z] 
#     *   - [property float normal_x] 
#     *   - [property float normal_y] 
#     *   - [property float normal_z] 
#     *   - [property uchar red] 
#     *   - [property uchar green] 
#     *   - [property uchar blue] ...
#     *   - ascii/binary point coordinates
#     *   - [element camera 1]
#     *   - [property float view_px] ...
#     *   - [element range_grid COUNT]
#     *   - [property list uchar int vertex_indices]
#     *   - end header
#     *
#     * \author Nizar Sallem
#     * \ingroup io
#     */
#   class PCL_EXPORTS PLYReader : public FileReader
#   {
#     public:
#       enum
#       {
#         PLY_V0 = 0,
#         PLY_V1 = 1
#       };
#       
#       PLYReader ()
#         : FileReader ()
#         , parser_ ()
#         , origin_ (Eigen::Vector4f::Zero ())
#         , orientation_ (Eigen::Matrix3f::Zero ())
#         , cloud_ ()
#         , vertex_count_ (0)
#         , vertex_properties_counter_ (0)
#         , vertex_offset_before_ (0)
#         , range_grid_ (0)
#         , range_count_ (0)
#         , range_grid_vertex_indices_element_index_ (0)
#         , rgb_offset_before_ (0)
#       {}
# 
#       PLYReader (const PLYReader &p)
#         : parser_ ()
#         , origin_ (Eigen::Vector4f::Zero ())
#         , orientation_ (Eigen::Matrix3f::Zero ())
#         , cloud_ ()
#         , vertex_count_ (0)
#         , vertex_properties_counter_ (0)
#         , vertex_offset_before_ (0)
#         , range_grid_ (0)
#         , range_count_ (0)
#         , range_grid_vertex_indices_element_index_ (0)
#         , rgb_offset_before_ (0)
#       {
#         *this = p;
#       }
# 
#       PLYReader& operator = (const PLYReader &p)
# 
#       ~PLYReader () { delete range_grid_; }
#       /** \brief Read a point cloud data header from a PLY file.
#         * Load only the meta information (number of points, their types, etc),
#         * and not the points themselves, from a given PLY file. Useful for fast
#         * evaluation of the underlying data structure.
#         * Returns:
#         *  * < 0 (-1) on error
#         *  * > 0 on success
#         * \param[in] file_name the name of the file to load
#         * \param[out] cloud the resultant point cloud dataset (only the header will be filled)
#         * \param[in] origin the sensor data acquisition origin (translation)
#         * \param[in] orientation the sensor data acquisition origin (rotation)
#         * \param[out] ply_version the PLY version read from the file
#         * \param[out] data_type the type of PLY data stored in the file
#         * \param[out] data_idx the data index
#         * \param[in] offset the offset in the file where to expect the true header to begin.
#         * One usage example for setting the offset parameter is for reading
#         * data from a TAR "archive containing multiple files: TAR files always
#         * add a 512 byte header in front of the actual file, so set the offset
#         * to the next byte after the header (e.g., 513).
#         */
#       int readHeader (const std::string &file_name, sensor_msgs::PointCloud2 &cloud,
#                   Eigen::Vector4f &origin, Eigen::Quaternionf &orientation,
#                   int &ply_version, int &data_type, unsigned int &data_idx, const int offset = 0);
# 
#       /** \brief Read a point cloud data from a PLY file and store it into a sensor_msgs/PointCloud2.
#         * \param[in] file_name the name of the file containing the actual PointCloud data
#         * \param[out] cloud the resultant PointCloud message read from disk
#         * \param[in] origin the sensor data acquisition origin (translation)
#         * \param[in] orientation the sensor data acquisition origin (rotation)
#         * \param[out] ply_version the PLY version read from the file
#         * \param[in] offset the offset in the file where to expect the true header to begin.
#         * One usage example for setting the offset parameter is for reading
#         * data from a TAR "archive containing multiple files: TAR files always
#         * add a 512 byte header in front of the actual file, so set the offset
#         * to the next byte after the header (e.g., 513).
#         */
#       int read (const std::string &file_name, sensor_msgs::PointCloud2 &cloud,
#             Eigen::Vector4f &origin, Eigen::Quaternionf &orientation, int& ply_version, const int offset = 0);
# 
#       /** \brief Read a point cloud data from a PLY file (PLY_V6 only!) and store it into a sensor_msgs/PointCloud2.
#         *
#         * \note This function is provided for backwards compatibility only and
#         * it can only read PLY_V6 files correctly, as sensor_msgs::PointCloud2
#         * does not contain a sensor origin/orientation. Reading any file
#         * > PLY_V6 will generate a warning.
#         *
#         * \param[in] file_name the name of the file containing the actual PointCloud data
#         * \param[out] cloud the resultant PointCloud message read from disk
#         * \param[in] offset the offset in the file where to expect the true header to begin.
#         * One usage example for setting the offset parameter is for reading
#         * data from a TAR "archive containing multiple files: TAR files always
#         * add a 512 byte header in front of the actual file, so set the offset
#         * to the next byte after the header (e.g., 513).
#         */
#       inline int read (const std::string &file_name, sensor_msgs::PointCloud2 &cloud, const int offset = 0)
# 
#       /** \brief Read a point cloud data from any PLY file, and convert it to the given template format.
#         * \param[in] file_name the name of the file containing the actual PointCloud data
#         * \param[out] cloud the resultant PointCloud message read from disk
#         * \param[in] offset the offset in the file where to expect the true header to begin.
#         * One usage example for setting the offset parameter is for reading
#         * data from a TAR "archive containing multiple files: TAR files always
#         * add a 512 byte header in front of the actual file, so set the offset
#         * to the next byte after the header (e.g., 513).
#         */
#       template<typename PointT> inline int
#       read (const std::string &file_name, pcl::PointCloud<PointT> &cloud, const int offset = 0)
#       {
#         sensor_msgs::PointCloud2 blob;
#         int ply_version;
#         int res = read (file_name, blob, cloud.sensor_origin_, cloud.sensor_orientation_,
#                         ply_version, offset);
# 
#         // Exit in case of error
#         if (res < 0)
#           return (res);
#         pcl::fromROSMsg (blob, cloud);
#         return (0);
#       }
#       
#     private:
#       ::pcl::io::ply::ply_parser parser_;
# 
#       bool
#       parse (const std::string& istream_filename);
# 
#       /** \brief Info callback function
#         * \param[in] filename PLY file read
#         * \param[in] line_number line triggering the callback
#         * \param[in] message information message
#         */
#       void 
#       infoCallback (const std::string& filename, std::size_t line_number, const std::string& message)
#       
#       /** \brief Warning callback function
#         * \param[in] filename PLY file read
#         * \param[in] line_number line triggering the callback
#         * \param[in] message warning message
#         */
#       void 
#       warningCallback (const std::string& filename, std::size_t line_number, const std::string& message)
#       
#       /** \brief Error callback function
#         * \param[in] filename PLY file read
#         * \param[in] line_number line triggering the callback
#         * \param[in] message error message
#         */
#       void 
#       errorCallback (const std::string& filename, std::size_t line_number, const std::string& message)
#       
#       /** \brief function called when the keyword element is parsed
#         * \param[in] element_name element name
#         * \param[in] count number of instances
#         */
#       boost::tuple<boost::function<void ()>, boost::function<void ()> > 
#       elementDefinitionCallback (const std::string& element_name, std::size_t count);
#       
#       bool
#       endHeaderCallback ();
# 
#       /** \brief function called when a scalar property is parsed
#         * \param[in] element_name element name to which the property belongs
#         * \param[in] property_name property name
#         */
#       template <typename ScalarType> boost::function<void (ScalarType)> 
#       scalarPropertyDefinitionCallback (const std::string& element_name, const std::string& property_name);
# 
#       /** \brief function called when a list property is parsed
#         * \param[in] element_name element name to which the property belongs
#         * \param[in] property_name list property name
#         */
#       template <typename SizeType, typename ScalarType> 
#       boost::tuple<boost::function<void (SizeType)>, boost::function<void (ScalarType)>, boost::function<void ()> > 
#       listPropertyDefinitionCallback (const std::string& element_name, const std::string& property_name);
#       
#       /** Callback function for an anonymous vertex float property.
#         * Writes down a float value in cloud data.
#         * param[in] value float value parsed
#         */      
#       inline void
#       vertexFloatPropertyCallback (pcl::io::ply::float32 value);
# 
#       /** Callback function for vertex RGB color.
#         * This callback is in charge of packing red green and blue in a single int
#         * before writing it down in cloud data.
#         * param[in] color_name color name in {red, green, blue}
#         * param[in] color value of {red, green, blue} property
#         */      
#       inline void
#       vertexColorCallback (const std::string& color_name, pcl::io::ply::uint8 color);
# 
#       /** Callback function for vertex intensity.
#         * converts intensity from int to float before writing it down in cloud data.
#         * param[in] intensity
#         */
#       inline void vertexIntensityCallback (pcl::io::ply::uint8 intensity);
#       
#       /** Callback function for origin x component.
#         * param[in] value origin x value
#         */
#       inline void originXCallback (const float& value)
#       
#       /** Callback function for origin y component.
#         * param[in] value origin y value
#         */
#       inline void originYCallback (const float& value)
# 
#       /** Callback function for origin z component.
#         * param[in] value origin z value
#         */      
#       inline void originZCallback (const float& value)
#     
#       /** Callback function for orientation x axis x component.
#         * param[in] value orientation x axis x value
#         */
#       inline void orientationXaxisXCallback (const float& value)
#       
#       /** Callback function for orientation x axis y component.
#         * param[in] value orientation x axis y value
#         */
#       inline void orientationXaxisYCallback (const float& value)
#       
#       /** Callback function for orientation x axis z component.
#         * param[in] value orientation x axis z value
#         */
#       inline void orientationXaxisZCallback (const float& value)
#       
#       /** Callback function for orientation y axis x component.
#         * param[in] value orientation y axis x value
#         */
#       inline void orientationYaxisXCallback (const float& value)
#       
#       /** Callback function for orientation y axis y component.
#         * param[in] value orientation y axis y value
#         */
#       inline void orientationYaxisYCallback (const float& value)
# 
#       /** Callback function for orientation y axis z component.
#         * param[in] value orientation y axis z value
#         */
#       inline void orientationYaxisZCallback (const float& value)
#       
#       /** Callback function for orientation z axis x component.
#         * param[in] value orientation z axis x value
#         */
#       inline void orientationZaxisXCallback (const float& value)
#     
#       /** Callback function for orientation z axis y component.
#         * param[in] value orientation z axis y value
#         */
#       inline void orientationZaxisYCallback (const float& value)
#       
#       /** Callback function for orientation z axis z component.
#         * param[in] value orientation z axis z value
#         */
#       inline void orientationZaxisZCallback (const float& value)
#       
#       /** Callback function to set the cloud height
#         * param[in] height cloud height
#         */
#       inline void cloudHeightCallback (const int &height)
# 
#       /** Callback function to set the cloud width
#         * param[in] width cloud width
#         */
#       inline void cloudWidthCallback (const int &width)
#         
#       /** Append a float property to the cloud fields.
#         * param[in] name property name
#         * param[in] count property count: 1 for scalar properties and higher for a 
#         * list property.
#       void appendFloatProperty (const std::string& name, const size_t& count = 1);
# 
#       /** Callback function for the begin of vertex line */
#       void vertexBeginCallback ();
# 
#       /** Callback function for the end of vertex line */
#       void vertexEndCallback ();
# 
#       /** Callback function for the begin of range_grid line */
#       void rangeGridBeginCallback ();
# 
#       /** Callback function for the begin of range_grid vertex_indices property 
#         * param[in] size vertex_indices list size  
#         */
#       void rangeGridVertexIndicesBeginCallback (pcl::io::ply::uint8 size);
# 
#       /** Callback function for each range_grid vertex_indices element
#         * param[in] vertex_index index of the vertex in vertex_indices
#         */      
#       void rangeGridVertexIndicesElementCallback (pcl::io::ply::int32 vertex_index);
# 
#       /** Callback function for the end of a range_grid vertex_indices property */
#       void rangeGridVertexIndicesEndCallback ();
# 
#       /** Callback function for the end of a range_grid element end */
#       void rangeGridEndCallback ();
# 
#       /** Callback function for obj_info */
#       void objInfoCallback (const std::string& line);
# 
#       /// origin
#       Eigen::Vector4f origin_;
#       /// orientation
#       Eigen::Matrix3f orientation_;
#       //vertex element artifacts
#       sensor_msgs::PointCloud2 *cloud_;
#       size_t vertex_count_, vertex_properties_counter_;
#       int vertex_offset_before_;
#       //range element artifacts
#       std::vector<std::vector <int> > *range_grid_;
#       size_t range_count_, range_grid_vertex_indices_element_index_;
#       size_t rgb_offset_before_;
#     	public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
# 
#   /** \brief Point Cloud Data (PLY) file format writer.
#     * \author Nizar Sallem
#     * \ingroup io
#     */
#   class PCL_EXPORTS PLYWriter : public FileWriter
#   {
#     public:
#       ///Constructor
#       PLYWriter () : FileWriter () {};
# 
#       ///Destructor
#       ~PLYWriter () {};
# 
#       /** \brief Generate the header of a PLY v.7 file format
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor data acquisition origin (translation)
#         * \param[in] orientation the sensor data acquisition origin (rotation)
#         * \param[in] valid_points number of valid points (finite ones for range_grid and
#         * all of them for camer)
#         * \param[in] use_camera if set to true then PLY file will use element camera else
#         * element range_grid will be used.
#         */
#       inline std::string
#       generateHeaderBinary (const sensor_msgs::PointCloud2 &cloud, 
#                             const Eigen::Vector4f &origin, 
#                             const Eigen::Quaternionf &orientation,
#                             int valid_points,
#                             bool use_camera = true)
#       
#       /** \brief Generate the header of a PLY v.7 file format
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor data acquisition origin (translation)
#         * \param[in] orientation the sensor data acquisition origin (rotation)
#         * \param[in] valid_points number of valid points (finite ones for range_grid and
#         * all of them for camer)
#         * \param[in] use_camera if set to true then PLY file will use element camera else
#         * element range_grid will be used.
#         */
#       inline std::string
#       generateHeaderASCII (const sensor_msgs::PointCloud2 &cloud, 
#                            const Eigen::Vector4f &origin, 
#                            const Eigen::Quaternionf &orientation,
#                            int valid_points,
#                            bool use_camera = true)
# 
#       /** \brief Save point cloud data to a PLY file containing n-D points, in ASCII format
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor data acquisition origin (translation)
#         * \param[in] orientation the sensor data acquisition origin (rotation)
#         * \param[in] precision the specified output numeric stream precision (default: 8)
#         * \param[in] use_camera if set to true then PLY file will use element camera else
#         * element range_grid will be used.
#         */
#       int 
#       writeASCII (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud, 
#                   const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#                   const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#                   int precision = 8,
#                   bool use_camera = true);
# 
#       /** \brief Save point cloud data to a PLY file containing n-D points, in BINARY format
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor data acquisition origin (translation)
#         * \param[in] orientation the sensor data acquisition origin (rotation)
#         */
#       int 
#       writeBinary (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud,
#                    const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#                    const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity ());
# 
#       /** \brief Save point cloud data to a PLY file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         * \param[in] binary set to true if the file is to be written in a binary
#         * PLY format, false (default) for ASCII
#         */
#       inline int
#       write (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud, 
#              const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#              const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#              const bool binary = false)
# 
#       /** \brief Save point cloud data to a PLY file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         * \param[in] binary set to true if the file is to be written in a binary
#         * PLY format, false (default) for ASCII
#         * \param[in] use_camera set to true to used camera element and false to
#         * use range_grid element
#         */
#       inline int
#       write (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud, 
#              const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#              const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#              bool binary = false,
#              bool use_camera = true)
# 
#       /** \brief Save point cloud data to a PLY file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the point cloud data message (boost shared pointer)
#         * \param[in] origin the sensor acquisition origin
#         * \param[in] orientation the sensor acquisition orientation
#         * \param[in] binary set to true if the file is to be written in a binary
#         * PLY format, false (default) for ASCII
#         * \param[in] use_camera set to true to used camera element and false to
#         * use range_grid element
#         */
#       inline int
#       write (const std::string &file_name, const sensor_msgs::PointCloud2::ConstPtr &cloud, 
#              const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#              const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#              bool binary = false,
#              bool use_camera = true)
# 
#       /** \brief Save point cloud data to a PLY file containing n-D points
#         * \param[in] file_name the output file name
#         * \param[in] cloud the pcl::PointCloud data
#         * \param[in] binary set to true if the file is to be written in a binary
#         * PLY format, false (default) for ASCII
#         * \param[in] use_camera set to true to used camera element and false to
#         * use range_grid element
#         */
#       template<typename PointT> inline int
#       write (const std::string &file_name, 
#              const pcl::PointCloud<PointT> &cloud, 
#              bool binary = false,
#              bool use_camera = true)
#   };
# 
#   namespace io
#   {
#     /** \brief Load a PLY v.6 file into a templated PointCloud type.
#       *
#       * Any PLY files containg sensor data will generate a warning as a
#       * sensor_msgs/PointCloud2 message cannot hold the sensor origin.
#       *
#       * \param[in] file_name the name of the file to load
#       * \param[in] cloud the resultant templated point cloud
#       * \ingroup io
#       */
#     inline int
#     loadPLYFile (const std::string &file_name, sensor_msgs::PointCloud2 &cloud)
# 
#     /** \brief Load any PLY file into a templated PointCloud type.
#       * \param[in] file_name the name of the file to load
#       * \param[in] cloud the resultant templated point cloud
#       * \param[in] origin the sensor acquisition origin (only for > PLY_V7 - null if not present)
#       * \param[in] orientation the sensor acquisition orientation if availble, 
#       * identity if not present
#       * \ingroup io
#       */
#     inline int
#     loadPLYFile (const std::string &file_name, sensor_msgs::PointCloud2 &cloud,
#                  Eigen::Vector4f &origin, Eigen::Quaternionf &orientation)
# 
#     /** \brief Load any PLY file into a templated PointCloud type
#       * \param[in] file_name the name of the file to load
#       * \param[in] cloud the resultant templated point cloud
#       * \ingroup io
#       */
#     template<typename PointT> inline int
#     loadPLYFile (const std::string &file_name, pcl::PointCloud<PointT> &cloud)
# 
#     /** \brief Save point cloud data to a PLY file containing n-D points
#       * \param[in] file_name the output file name
#       * \param[in] cloud the point cloud data message
#       * \param[in] origin the sensor data acquisition origin (translation)
#       * \param[in] orientation the sensor data acquisition origin (rotation)
#       * \param[in] binary_mode true for binary mode, false (default) for ASCII
#       * \ingroup io
#       */
#     inline int 
#     savePLYFile (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud, 
#                  const Eigen::Vector4f &origin = Eigen::Vector4f::Zero (), 
#                  const Eigen::Quaternionf &orientation = Eigen::Quaternionf::Identity (),
#                  bool binary_mode = false, bool use_camera = true)
# 
#     /** \brief Templated version for saving point cloud data to a PLY file
#       * containing a specific given cloud format
#       * \param[in] file_name the output file name
#       * \param[in] cloud the point cloud data message
#       * \param[in] binary_mode true for binary mode, false (default) for ASCII
#       * \ingroup io
#       */
#     template<typename PointT> inline int
#     savePLYFile (const std::string &file_name, const pcl::PointCloud<PointT> &cloud, bool binary_mode = false)
# 
#     /** \brief Templated version for saving point cloud data to a PLY file
#       * containing a specific given cloud format.
#       * \param[in] file_name the output file name
#       * \param[in] cloud the point cloud data message
#       * \ingroup io
#       */
#     template<typename PointT> inline int
#     savePLYFileASCII (const std::string &file_name, const pcl::PointCloud<PointT> &cloud)
# 
#     /** \brief Templated version for saving point cloud data to a PLY file containing a specific given cloud format.
#       * \param[in] file_name the output file name
#       * \param[in] cloud the point cloud data message
#       * \ingroup io
#       */
#     template<typename PointT> inline int
#     savePLYFileBinary (const std::string &file_name, const pcl::PointCloud<PointT> &cloud)
# 
#     /** \brief Templated version for saving point cloud data to a PLY file containing a specific given cloud format
#       * \param[in] file_name the output file name
#       * \param[in] cloud the point cloud data message
#       * \param[in] indices the set of indices to save
#       * \param[in] binary_mode true for binary mode, false (default) for ASCII
#       * \ingroup io
#       */
#     template<typename PointT> int
#     savePLYFile (const std::string &file_name, const pcl::PointCloud<PointT> &cloud,
#                  const std::vector<int> &indices, bool binary_mode = false)
#
#     /** \brief Saves a PolygonMesh in ascii PLY format.
#       * \param[in] file_name the name of the file to write to disk
#       * \param[in] mesh the polygonal mesh to save
#       * \param[in] precision the output ASCII precision default 5
#       * \ingroup io
#       */
#     PCL_EXPORTS int
#     savePLYFile (const std::string &file_name, const pcl::PolygonMesh &mesh, unsigned precision = 5);
#   }
###

# tar.h
# namespace pcl
# {
#   namespace io
#   {
#     /** \brief A TAR file's header, as described on 
#       * http://en.wikipedia.org/wiki/Tar_%28file_format%29. 
#       */
#     struct TARHeader
#     {
#       char file_name[100];
#       char file_mode[8];
#       char uid[8];
#       char gid[8];
#       char file_size[12];
#       char mtime[12];
#       char chksum[8];
#       char file_type[1];
#       char link_file_name[100];
#       char ustar[6];
#       char ustar_version[2];
#       char uname[32];
#       char gname[32];
#       char dev_major[8];
#       char dev_minor[8];
#       char file_name_prefix[155];
#       char _padding[12];
# 
#       unsigned int 
#       getFileSize ()
#       {
#         unsigned int output = 0;
#         char *str = file_size;
#         for (int i = 0; i < 11; i++)
#         {
#           output = output * 8 + *str - '0';
#           str++;
#         }
#         return (output);
#       }
#     };
#   }
# }
###

# vtk_io.h
# namespace pcl
# namespace io
#     /** \brief Saves a PolygonMesh in ascii VTK format. 
#       * \param[in] file_name the name of the file to write to disk
#       * \param[in] triangles the polygonal mesh to save
#       * \param[in] precision the output ASCII precision
#       * \ingroup io
#       */
#     PCL_EXPORTS int 
#     saveVTKFile (const std::string &file_name, const pcl::PolygonMesh &triangles, unsigned precision = 5);
# 
#     /** \brief Saves a PointCloud in ascii VTK format. 
#       * \param[in] file_name the name of the file to write to disk
#       * \param[in] cloud the point cloud to save
#       * \param[in] precision the output ASCII precision
#       * \ingroup io
#       */
#     PCL_EXPORTS int 
#     saveVTKFile (const std::string &file_name, const sensor_msgs::PointCloud2 &cloud, unsigned precision = 5);    
# 
###

# vtk_lib_io.h
# namespace pcl
#   namespace io
#     /** \brief Convert vtkPolyData object to a PCL PolygonMesh
#       * \param[in] poly_data Pointer (vtkSmartPointer) to a vtkPolyData object
#       * \param[out] mesh PCL Polygon Mesh to fill
#       * \return Number of points in the point cloud of mesh.
#       */
#     PCL_EXPORTS int
#     vtk2mesh (const vtkSmartPointer<vtkPolyData>& poly_data, 
#               pcl::PolygonMesh& mesh);
# 
#     /** \brief Convert a PCL PolygonMesh to a vtkPolyData object
#       * \param[in] mesh Reference to PCL Polygon Mesh
#       * \param[out] poly_data Pointer (vtkSmartPointer) to a vtkPolyData object
#       * \return Number of points in the point cloud of mesh.
#       */
#     PCL_EXPORTS int
#     mesh2vtk (const pcl::PolygonMesh& mesh, 
#               vtkSmartPointer<vtkPolyData>& poly_data);
# 
#     /** \brief Load a \ref PolygonMesh object given an input file name, based on the file extension
#       * \param[in] file_name the name of the file containing the polygon data
#       * \param[out] mesh the object that we want to load the data in 
#       * \ingroup io
#       */ 
#     PCL_EXPORTS int
#     loadPolygonFile (const std::string &file_name, 
#                      pcl::PolygonMesh& mesh);
# 
#     /** \brief Save a \ref PolygonMesh object given an input file name, based on the file extension
#       * \param[in] file_name the name of the file to save the data to
#       * \param[in] mesh the object that contains the data
#       * \ingroup io
#       */
#     PCL_EXPORTS int
#     savePolygonFile (const std::string &file_name, 
#                      const pcl::PolygonMesh& mesh);
# 
#     /** \brief Load a VTK file into a \ref PolygonMesh object
#       * \param[in] file_name the name of the file that contains the data
#       * \param[out] mesh the object that we want to load the data in 
#       * \ingroup io
#       */
#     PCL_EXPORTS int
#     loadPolygonFileVTK (const std::string &file_name, 
#                         pcl::PolygonMesh& mesh);
# 
#     /** \brief Load a PLY file into a \ref PolygonMesh object
#       * \param[in] file_name the name of the file that contains the data
#       * \param[out] mesh the object that we want to load the data in 
#       * \ingroup io
#       */
#     PCL_EXPORTS int
#     loadPolygonFilePLY (const std::string &file_name, 
#                         pcl::PolygonMesh& mesh);
# 
#     /** \brief Load an OBJ file into a \ref PolygonMesh object
#       * \param[in] file_name the name of the file that contains the data
#       * \param[out] mesh the object that we want to load the data in 
#       * \ingroup io
#       */
#     PCL_EXPORTS int
#     loadPolygonFileOBJ (const std::string &file_name, 
#                         pcl::PolygonMesh& mesh);
# 
#     /** \brief Load an STL file into a \ref PolygonMesh object
#       * \param[in] file_name the name of the file that contains the data
#       * \param[out] mesh the object that we want to load the data in 
#       * \ingroup io
#       */
#     PCL_EXPORTS int
#     loadPolygonFileSTL (const std::string &file_name, 
#                         pcl::PolygonMesh& mesh);
# 
#     /** \brief Save a \ref PolygonMesh object into a VTK file
#       * \param[in] file_name the name of the file to save the data to
#       * \param[in] mesh the object that contains the data
#       * \ingroup io
#       */
#     PCL_EXPORTS int
#     savePolygonFileVTK (const std::string &file_name, 
#                         const pcl::PolygonMesh& mesh);
# 
#     /** \brief Save a \ref PolygonMesh object into a PLY file
#       * \param[in] file_name the name of the file to save the data to
#       * \param[in] mesh the object that contains the data
#       * \ingroup io
#       */
#     PCL_EXPORTS int
#     savePolygonFilePLY (const std::string &file_name, 
#                         const pcl::PolygonMesh& mesh);
# 
#     /** \brief Save a \ref PolygonMesh object into an STL file
#       * \param[in] file_name the name of the file to save the data to
#       * \param[in] mesh the object that contains the data
#       * \ingroup io
#       */
#     PCL_EXPORTS int
#     savePolygonFileSTL (const std::string &file_name, 
#                         const pcl::PolygonMesh& mesh);
# 
#     /** \brief Write a \ref RangeImagePlanar object to a PNG file
#       * \param[in] file_name the name of the file to save the data to
#       * \param[in] range_image the object that contains the data
#       * \ingroup io
#       */
#     PCL_EXPORTS void
#     saveRangeImagePlanarFilePNG (const std::string &file_name,
#                                  const pcl::RangeImagePlanar& range_image);
# 
#     /** \brief Convert a pcl::PointCloud object to a VTK PolyData one.
#       * \param[in] cloud the input pcl::PointCloud object
#       * \param[out] polydata the resultant VTK PolyData object
#       * \ingroup io
#       */
#     template <typename PointT> void
#     pointCloudTovtkPolyData (const pcl::PointCloud<PointT>& cloud, 
#                              vtkPolyData* const polydata);
# 
#     /** \brief Convert a pcl::PointCloud object to a VTK StructuredGrid one.
#       * \param[in] cloud the input pcl::PointCloud object
#       * \param[out] structured_grid the resultant VTK StructuredGrid object
#       * \ingroup io
#       */
#     template <typename PointT> void
#     pointCloudTovtkStructuredGrid (const pcl::PointCloud<PointT>& cloud, 
#                                    vtkStructuredGrid* const structured_grid);
# 
#     /** \brief Convert a VTK PolyData object to a pcl::PointCloud one.
#       * \param[in] polydata the input VTK PolyData object
#       * \param[out] cloud the resultant pcl::PointCloud object
#       * \ingroup io
#       */
#     template <typename PointT> void
#     vtkPolyDataToPointCloud (vtkPolyData* const polydata, 
#                              pcl::PointCloud<PointT>& cloud);
# 
#     /** \brief Convert a VTK StructuredGrid object to a pcl::PointCloud one.
#       * \param[in] structured_grid the input VTK StructuredGrid object
#       * \param[out] cloud the resultant pcl::PointCloud object
#       * \ingroup io
#       */
#     template <typename PointT> void
#     vtkStructuredGridToPointCloud (vtkStructuredGrid* const structured_grid, 
#                                    pcl::PointCloud<PointT>& cloud);
# 
# 
###

