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
        vector[PointField] fields
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

