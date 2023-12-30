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

cdef extern from "sensor_msgs/Image.h" namespace "std_msgs":
    cdef struct Header:
        Header ()
        # pcl::uint32_t seq
        # pcl::uint64_t stamp
        unsigned int seq
        unsigned long stamp
        string frame_id

# typedef boost::shared_ptr<Header> HeaderPtr;
# typedef boost::shared_ptr<Header const> HeaderConstPtr;
# inline std::ostream& operator << (std::ostream& out, const Header &h)

ctypedef shared_ptr[Header] HeaderPtr_t
ctypedef shared_ptr[const Header] HeaderConstPtr_t
###


cdef extern from "sensor_msgs/Image.h" namespace "sensor_msgs":
    cdef struct Image:
        Image ()
        Header header
        unsigned int height
        unsigned int width
        string encoding
        unsigned char is_bigendian
        unsigned int step;
        vector[unsigned int] data;

# inline std::ostream& operator<<(std::ostream& s, const  ::sensor_msgs::Image & v)

ctypedef shared_ptr[Image] ImagePtr_t
ctypedef shared_ptr[const Image] ImageConstPtr_t
###

cdef extern from "sensor_msgs/PointCloud2.h" namespace "sensor_msgs":
    cdef struct PointCloud2:
        PointCloud2 ()
        Header header
        unsigned int height
        unsigned int width
        vector[PointField] fields
        unsigned char is_bigendian
        unsigned int point_step
        unsigned int row_step
        vector[unsigned char] data
        unsigned char is_dense

#  inline std::ostream& operator<<(std::ostream& s, const  ::sensor_msgs::PointCloud2 &v)

# ctypedef sharedptr[PointCloud2] PointCloud2Ptr_t
# ctypedef sharedptr[const PointCloud2] PointCloud2ConstPtr_t
ctypedef cpp.PointCloud[PointCloud2] PointCloud_PointCloud2_t
ctypedef shared_ptr[cpp.PointCloud[PointCloud2]] PointCloud_PointCloud2Ptr_t
###

cdef extern from "sensor_msgs/PointField.h" namespace "sensor_msgs":
    cdef struct PointField:
        PointField ()
        string name
        unsigned int offset
        unsigned char datatype
        unsigned int count

# inline std::ostream& operator<<(std::ostream& s, const  ::sensor_msgs::PointField & v)

# ctypedef sharedptr[PointField] PointFieldPtr_t
# ctypedef sharedptr[const PointField] PointFieldConstPtr_t
###


###############################################################################
# Enum
###############################################################################

cdef extern from "sensor_msgs/PointField.h" namespace "sensor_msgs":
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

