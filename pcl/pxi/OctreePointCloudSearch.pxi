
cimport pcl_defs as cpp
cimport pcl_octree as pcloct

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

