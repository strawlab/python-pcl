from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

# main
cimport pcl_defs as cpp

# from boost_shared_ptr cimport shared_ptr

cdef extern from "pcl/io/pcd_io.h" namespace "pcl::io":
    # XYZ
    int load(string file_name, cpp.PointCloud[cpp.PointXYZ] &cloud) nogil except +
    int loadPCDFile(string file_name,
                    cpp.PointCloud[cpp.PointXYZ] &cloud) nogil except +
    int savePCDFile(string file_name, cpp.PointCloud[cpp.PointXYZ] &cloud,
                    bool binary_mode) nogil except +
    int savePCDFileASCII (string file_name, cpp.PointCloud[cpp.PointXYZ] &cloud) nogil except +
    int savePCDFileBinary (string &file_name, cpp.PointCloud[cpp.PointXYZ] &cloud) nogil except +
    int savePCDFile (string &file_name, 
                     cpp.PointCloud[cpp.PointXYZ] &cloud,
                     vector[int] &indices, 
                     bool binary_mode) nogil except +

    # XYZRGBA
    int load(string file_name, cpp.PointCloud[cpp.PointXYZRGBA] &cloud) nogil except +
    int loadPCDFile(string file_name,
                    cpp.PointCloud[cpp.PointXYZRGBA] &cloud) nogil except +
    int savePCDFile(string file_name, cpp.PointCloud[cpp.PointXYZRGBA] &cloud,
                    bool binary_mode) nogil except +

cdef extern from "pcl/io/ply_io.h" namespace "pcl::io":
    # XYZ
    int loadPLYFile(string file_name,
                    cpp.PointCloud[cpp.PointXYZ] &cloud) nogil except +
    int savePLYFile(string file_name, cpp.PointCloud[cpp.PointXYZ] &cloud,
                    bool binary_mode) nogil except +
    # XYZRGBA
    int loadPLYFile(string file_name,
                    cpp.PointCloud[cpp.PointXYZRGBA] &cloud) nogil except +
    int savePLYFile(string file_name, cpp.PointCloud[cpp.PointXYZRGBA] &cloud,
                    bool binary_mode) nogil except +

#http://dev.pointclouds.org/issues/624
#cdef extern from "pcl/io/ply_io.h" namespace "pcl::io":
#    int loadPLYFile (string file_name, PointCloud[cpp.PointXYZ] cloud)
#    int savePLYFile (string file_name, PointCloud[cpp.PointXYZ] cloud, bool binary_mode)


# cdef extern from "pcl/io/ply_io.h" namespace "pcl::io":
