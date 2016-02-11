from libcpp.vector cimport vector

# main
cimport pcl_defs as cpp

from boost_shared_ptr cimport shared_ptr

# cdef extern from "pcl/search/kdtree.h" namespace "pcl::search":
#     cdef cppclass KdTree[T]:
#         KdTree()
#         void setInputCloud (shared_ptr[cpp.PointCloud[T]])
# 
# ctypedef KdTree[cpp.PointXYZ] KdTree_t
# ctypedef KdTree[cpp.PointXYZRGBA] KdTree2_t

cdef extern from "pcl/kdtree/kdtree_flann.h" namespace "pcl":
    cdef cppclass KdTreeFLANN[T]:
        KdTreeFLANN()
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        int nearestKSearch (cpp.PointCloud[T], int, int, vector[int], vector[float])

ctypedef KdTreeFLANN[cpp.PointXYZ] KdTreeFLANN_t
ctypedef KdTreeFLANN[cpp.PointXYZRGBA] KdTreeFLANN_t2
