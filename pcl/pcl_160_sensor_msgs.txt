from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

# boost
from boost_shared_ptr cimport shared_ptr

cimport pcl_160_std_msgs as std_msgs

cdef extern from "sensor_msgs/Image.h" namespace "sensor_msgs":
    cdef struct Image:
        Image ()
        std_msgs.Header  header
        pcl::uint32_t height;
        pcl::uint32_t width;
        string encoding;
        pcl::uint8_t is_bigendian;
        pcl::uint32_t step;
        vector[pcl::uint8_t] data;

ctypedef shared_ptr[Image] ImagePtr;
ctypedef shared_ptr[Image*] ImageConstPtr;

# inline std::ostream& operator<<(std::ostream& s, const  ::sensor_msgs::Image & v)

###

cdef extern from "sensor_msgs/PointCloud2.h" namespace "sensor_msgs":
    cdef struct PointCloud2:
        PointCloud2 ()
        
        std_msgs.Header  header
        pcl::uint32_t height;
        pcl::uint32_t width;
        std::vector< ::sensor_msgs::PointField>  fields;
        
        pcl::uint8_t is_bigendian;
        pcl::uint32_t point_step;
        pcl::uint32_t row_step;
        
        std::vector<pcl::uint8_t> data;
        
        pcl::uint8_t is_dense;

ctypedef sharedptr[PointCloud2] PointCloud2Ptr;
ctypedef sharedptr[PointCloud2*] PointCloud2ConstPtr;

#  inline std::ostream& operator<<(std::ostream& s, const  ::sensor_msgs::PointCloud2 &v)

###

cdef extern from "sensor_msgs/PointField.h" namespace "sensor_msgs":
    cdef struct PointField:
        PointField ()
        
        string name
        pcl::uint32_t offset;
        pcl::uint8_t datatype;
        pcl::uint32_t count;
        
        enum { INT8 = 1 };
        enum { UINT8 = 2 };
        enum { INT16 = 3 };
        enum { UINT16 = 4 };
        enum { INT32 = 5 };
        enum { UINT32 = 6 };
        enum { FLOAT32 = 7 };
        enum { FLOAT64 = 8 };

ctypedef sharedptr[PointField] PointFieldPtr;
ctypedef sharedptr[PointField*] PointFieldConstPtr;

# inline std::ostream& operator<<(std::ostream& s, const  ::sensor_msgs::PointField & v)

