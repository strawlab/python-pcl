from libcpp cimport bool
from libcpp.vector cimport vector

# main
cimport pcl_defs as cpp
from boost_shared_ptr cimport shared_ptr

###

# octree.h
###

# octree2buf_base.h
###

# octree_base.h
###

# octree_container.h
###

# octree_impl.h
###

# octree_iterator.h
###

# octree_key.h
###

# octree_nodes.h
###

# octree_node_pool.h
###

# octree_pointcloud.h
cdef extern from "pcl/octree/octree_pointcloud.h" namespace "pcl::octree":
    cdef cppclass OctreePointCloud[T]:
        OctreePointCloud(double)
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        void defineBoundingBox()
        void defineBoundingBox(double, double, double, double, double, double)
        void addPointsFromInputCloud()
        void deleteTree()
        bool isVoxelOccupiedAtPoint(double, double, double)
        int getOccupiedVoxelCenters(cpp.AlignedPointTVector_t)  
        void deleteVoxelAtPoint(cpp.PointXYZ)

ctypedef OctreePointCloud[cpp.PointXYZ] OctreePointCloud_t
ctypedef OctreePointCloud[cpp.PointXYZRGBA] OctreePointCloud2_t

# octree_pointcloud_changedetector.h
###
# octree_pointcloud_density.h
###
# octree_pointcloud_occupancy.h
###
# octree_pointcloud_pointvector.h
###
# octree_pointcloud_singlepoint.h
###
# octree_pointcloud_voxelcentroid.h
###

# octree_search.h
cdef extern from "pcl/octree/octree_search.h" namespace "pcl::octree":
    cdef cppclass OctreePointCloudSearch[T]:
        OctreePointCloudSearch(double)
        int radiusSearch (cpp.PointXYZ, double, vector[int], vector[float], unsigned int)

ctypedef OctreePointCloudSearch[cpp.PointXYZ] OctreePointCloudSearch_t
ctypedef OctreePointCloudSearch[cpp.PointXYZRGBA] OctreePointCloudSearch2_t

###

