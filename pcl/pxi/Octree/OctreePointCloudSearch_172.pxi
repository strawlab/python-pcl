# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_octree_172 as pcloct

# include "PointXYZtoPointXYZ.pxi" --> multiple define ng
# include "OctreePointCloud.pxi"

cdef class OctreePointCloudSearch(OctreePointCloud):
    """
    Octree pointcloud search
    """

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = <pcloct.OctreePointCloud_t*> new pcloct.OctreePointCloudSearch_t(resolution)

    # nearestKSearch
    ###
    def nearest_k_search_for_cloud(self, PointCloud pc not None, int k=1):
        """
        Find the k nearest neighbours and squared distances for all points
        in the pointcloud. Results are in ndarrays, size (pc.size, k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.npy_intp n_points = pc.size
        cdef cnp.ndarray[float, ndim=2] sqdist = np.zeros((n_points, k),
                                                          dtype=np.float32)
        cdef cnp.ndarray[int, ndim=2] ind = np.zeros((n_points, k),
                                                     dtype=np.int32)

        for i in range(n_points):
            self._nearest_k(pc, i, k, ind[i], sqdist[i])
        return ind, sqdist

    def nearest_k_search_for_point(self, PointCloud pc not None, int index, int k=1):
        """
        Find the k nearest neighbours and squared distances for the point
        at pc[index]. Results are in ndarrays, size (k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.ndarray[float] sqdist = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] ind = np.zeros(k, dtype=np.int32)

        self._nearest_k(pc, index, k, ind, sqdist)
        return ind, sqdist

    @cython.boundscheck(False)
    cdef void _nearest_k(self, PointCloud pc, int index, int k,
                         cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
                         cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
                        ) except +:
        # k nearest neighbors query for a single point.
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        k_indices.resize(k)
        k_sqr_distances.resize(k)
        # self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices, k_sqr_distances)
        (<pcloct.OctreePointCloudSearch_t*>self.me).nearestKSearch(pc.thisptr()[0], index, k, k_indices, k_sqr_distances)

        for i in range(k):
            sqdist[i] = k_sqr_distances[i]
            ind[i] = k_indices[i]

    ###

    # radius Search
    ###
    def radius_search (self, point, double radius, unsigned int max_nn = 0):
        """
        Search for all neighbors of query point that are within a given radius.
        
        Returns: (k_indices, k_sqr_distances)
        """
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        if max_nn > 0:
            k_indices.resize(max_nn)
            k_sqr_distances.resize(max_nn)
        cdef int k = (<pcloct.OctreePointCloudSearch_t*>self.me).radiusSearch(to_point_t(point), radius, k_indices, k_sqr_distances, max_nn)
        cdef cnp.ndarray[float] np_k_sqr_distances = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] np_k_indices = np.zeros(k, dtype=np.int32)
        for i in range(k):
            np_k_sqr_distances[i] = k_sqr_distances[i]
            np_k_indices[i] = k_indices[i]
        return np_k_indices, np_k_sqr_distances

    ###

    # Voxel Search
    ### 
    def VoxelSearch (self, PointCloud pc):
        """
        Search for all neighbors of query point that are within a given voxel.
        
        Returns: (v_indices)
        """
        cdef vector[int] v_indices
        # cdef bool isVexelSearch = (<pcloct.OctreePointCloudSearch_t*>self.me).voxelSearch(pc.thisptr()[0], v_indices)
        # self._VoxelSearch(pc, v_indices)
        result = pc.to_array()
        cdef cpp.PointXYZ point
        point.x = result[0, 0]
        point.y = result[0, 1]
        point.z = result[0, 2]
        
        print ('VoxelSearch at (' + str(point.x) + ' ' + str(point.y) + ' ' + str(point.z) + ')')
        
        # print('before v_indices count = ' + str(v_indices.size()))
        self._VoxelSearch(point, v_indices)
        v = v_indices.size()
        # print('after v_indices count = ' + str(v))
        
        cdef cnp.ndarray[int] np_v_indices = np.zeros(v, dtype=np.int32)
        for i in range(v):
            np_v_indices[i] = v_indices[i]
        
        return np_v_indices

    @cython.boundscheck(False)
    cdef void _VoxelSearch(self, cpp.PointXYZ point, vector[int] &v_indices) except +:
        cdef vector[int] voxel_indices
        # k = 10
        # voxel_indices.resize(k)
        (<pcloct.OctreePointCloudSearch_t*>self.me).voxelSearch(point, voxel_indices)
        
        # print('_VoxelSearch k = ' + str(k))
        # print('_VoxelSearch voxel_indices = ' + str(voxel_indices.size()))
        k = voxel_indices.size()
        
        for i in range(k):
            v_indices.push_back(voxel_indices[i])

    ### 


#     def radius_search_for_cloud(self, PointCloud pc not None, double radius):
#         """
#         Find the radius and squared distances for all points
#         in the pointcloud. Results are in ndarrays, size (pc.size, k)
#         Returns: (radius_indices, radius_distances)
#         """
#         k = 10
#         cdef cnp.npy_intp n_points = pc.size
#         cdef cnp.ndarray[float, ndim=2] sqdist = np.zeros((n_points, k),
#                                                           dtype=np.float32)
#         cdef cnp.ndarray[int, ndim=2] ind = np.zeros((n_points, k),
#                                                      dtype=np.int32)
# 
#         for i in range(n_points):
#             self._search_radius(pc, i, k, radius, ind[i], sqdist[i])
#         return ind, sqdist
# 
#     @cython.boundscheck(False)
#     cdef void _search_radius(self, PointCloud pc, int index, int k, double radius,
#                          cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
#                          cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
#                         ) except +:
#         # radius query for a single point.
#         cdef vector[int] radius_indices
#         cdef vector[float] radius_distances
#         radius_indices.resize(k)
#         radius_distances.resize(k)
#         # self.me.radiusSearch(pc.thisptr()[0], index, radius, radius_indices, radius_distances)
#         k = (<pcloct.OctreePointCloudSearch_t*>self.me).radiusSearch(pc.thisptr()[0], index, radius, radius_indices, radius_distances, 10)
# 
#         for i in range(k):
#             sqdist[i] = radius_distances[i]
#             ind[i] = radius_indices[i]


cdef class OctreePointCloudSearch_PointXYZI(OctreePointCloud_PointXYZI):
    """
    Octree pointcloud search
    """

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = <pcloct.OctreePointCloud_PointXYZI_t*> new pcloct.OctreePointCloudSearch_PointXYZI_t(resolution)

    def radius_search (self, point, double radius, unsigned int max_nn = 0):
        """
        Search for all neighbors of query point that are within a given radius.

        Returns: (k_indices, k_sqr_distances)
        """
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        if max_nn > 0:
            k_indices.resize(max_nn)
            k_sqr_distances.resize(max_nn)
        cdef int k = (<pcloct.OctreePointCloudSearch_PointXYZI_t*>self.me).radiusSearch(to_point2_t(point), radius, k_indices, k_sqr_distances, max_nn)
        cdef cnp.ndarray[float] np_k_sqr_distances = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] np_k_indices = np.zeros(k, dtype=np.int32)
        for i in range(k):
            np_k_sqr_distances[i] = k_sqr_distances[i]
            np_k_indices[i] = k_indices[i]
        return np_k_indices, np_k_sqr_distances


cdef class OctreePointCloudSearch_PointXYZRGB(OctreePointCloud_PointXYZRGB):
    """
    Octree pointcloud search
    """

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = <pcloct.OctreePointCloud_PointXYZRGB_t*> new pcloct.OctreePointCloudSearch_PointXYZRGB_t(resolution)

    def radius_search (self, point, double radius, unsigned int max_nn = 0):
        """
        Search for all neighbors of query point that are within a given radius.

        Returns: (k_indices, k_sqr_distances)
        """
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        if max_nn > 0:
            k_indices.resize(max_nn)
            k_sqr_distances.resize(max_nn)
        cdef int k = (<pcloct.OctreePointCloudSearch_PointXYZRGB_t*>self.me).radiusSearch(to_point3_t(point), radius, k_indices, k_sqr_distances, max_nn)
        cdef cnp.ndarray[float] np_k_sqr_distances = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] np_k_indices = np.zeros(k, dtype=np.int32)
        for i in range(k):
            np_k_sqr_distances[i] = k_sqr_distances[i]
            np_k_indices[i] = k_indices[i]
        return np_k_indices, np_k_sqr_distances


cdef class OctreePointCloudSearch_PointXYZRGBA(OctreePointCloud_PointXYZRGBA):
    """
    Octree pointcloud search
    """

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = <pcloct.OctreePointCloud_PointXYZRGBA_t*> new pcloct.OctreePointCloudSearch_PointXYZRGBA_t(resolution)

    def radius_search (self, point, double radius, unsigned int max_nn = 0):
        """
        Search for all neighbors of query point that are within a given radius.

        Returns: (k_indices, k_sqr_distances)
        """
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        if max_nn > 0:
            k_indices.resize(max_nn)
            k_sqr_distances.resize(max_nn)
        cdef int k = (<pcloct.OctreePointCloudSearch_PointXYZRGBA_t*>self.me).radiusSearch(to_point4_t(point), radius, k_indices, k_sqr_distances, max_nn)
        cdef cnp.ndarray[float] np_k_sqr_distances = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] np_k_indices = np.zeros(k, dtype=np.int32)
        for i in range(k):
            np_k_sqr_distances[i] = k_sqr_distances[i]
            np_k_indices[i] = k_indices[i]
        return np_k_indices, np_k_sqr_distances

