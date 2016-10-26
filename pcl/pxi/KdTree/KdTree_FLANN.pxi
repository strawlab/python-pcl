# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_kdtree as pclkdt

cdef class KdTreeFLANN:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTreeFLANN(pc).
    """
    cdef pclkdt.KdTreeFLANN_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclkdt.KdTreeFLANN_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

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

    def nearest_k_search_for_point(self, PointCloud pc not None, int index,
                                   int k=1):
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
        self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices,
                               k_sqr_distances)

        for i in range(k):
            sqdist[i] = k_sqr_distances[i]
            ind[i] = k_indices[i]

    def radius_search_for_cloud(self, PointCloud pc not None, double radius, unsigned int max_nn = 0):
        """
        Find the radius and squared distances for all points
        in the pointcloud. Results are in ndarrays, size (pc.size, k)
        Returns: (radius_indices, radius_distances)
        """
        k = max_nn
        cdef cnp.npy_intp n_points = pc.size
        cdef cnp.ndarray[float, ndim=2] sqdist = np.zeros((n_points, k),
                                                          dtype=np.float32)
        cdef cnp.ndarray[int, ndim=2] ind = np.zeros((n_points, k),
                                                     dtype=np.int32)

        for i in range(n_points):
            self._search_radius(pc, i, k, radius, ind[i], sqdist[i])
        return ind, sqdist

    @cython.boundscheck(False)
    cdef void _search_radius(self, PointCloud pc, int index, int k, double radius,
                         cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
                         cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
                        ) except +:
        # radius query for a single point.
        cdef vector[int] radius_indices
        cdef vector[float] radius_distances
        radius_indices.resize(k)
        radius_distances.resize(k)
        self.me.radiusSearch(pc.thisptr()[0], index, radius, radius_indices, radius_distances)

        for i in range(k):
            sqdist[i] = radius_distances[i]
            ind[i] = radius_indices[i]

cdef class KdTreeFLANN_PointXYZI:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTreeFLANN(pc).
    """
    cdef pclkdt.KdTreeFLANN_PointXYZI_t *me

    def __cinit__(self, PointCloud_PointXYZI pc not None):
        self.me = new pclkdt.KdTreeFLANN_PointXYZI_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def nearest_k_search_for_cloud(self, PointCloud_PointXYZI pc not None, int k=1):
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

    def nearest_k_search_for_point(self, PointCloud_PointXYZI pc not None, int index,
                                   int k=1):
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
    cdef void _nearest_k(self, PointCloud_PointXYZI pc, int index, int k,
                         cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
                         cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
                        ) except +:
        # k nearest neighbors query for a single point.
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        k_indices.resize(k)
        k_sqr_distances.resize(k)
        self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices,
                               k_sqr_distances)

        for i in range(k):
            sqdist[i] = k_sqr_distances[i]
            ind[i] = k_indices[i]


cdef class KdTreeFLANN_PointXYZRGB:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTreeFLANN(pc).
    """
    cdef pclkdt.KdTreeFLANN_PointXYZRGB_t *me

    def __cinit__(self, PointCloud_PointXYZRGB pc not None):
        self.me = new pclkdt.KdTreeFLANN_PointXYZRGB_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def nearest_k_search_for_cloud(self, PointCloud_PointXYZRGB pc not None, int k=1):
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

    def nearest_k_search_for_point(self, PointCloud_PointXYZRGB pc not None, int index,
                                   int k=1):
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
    cdef void _nearest_k(self, PointCloud_PointXYZRGB pc, int index, int k,
                         cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
                         cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
                        ) except +:
        # k nearest neighbors query for a single point.
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        k_indices.resize(k)
        k_sqr_distances.resize(k)
        self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices,
                               k_sqr_distances)

        for i in range(k):
            sqdist[i] = k_sqr_distances[i]
            ind[i] = k_indices[i]


cdef class KdTreeFLANN_PointXYZRGBA:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTreeFLANN(pc).
    """
    cdef pclkdt.KdTreeFLANN_PointXYZRGBA_t *me

    def __cinit__(self, PointCloud_PointXYZRGBA pc not None):
        self.me = new pclkdt.KdTreeFLANN_PointXYZRGBA_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def nearest_k_search_for_cloud(self, PointCloud_PointXYZRGBA pc not None, int k=1):
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

    def nearest_k_search_for_point(self, PointCloud_PointXYZRGBA pc not None, int index,
                                   int k=1):
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
    cdef void _nearest_k(self, PointCloud_PointXYZRGBA pc, int index, int k,
                         cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
                         cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
                        ) except +:
        # k nearest neighbors query for a single point.
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        k_indices.resize(k)
        k_sqr_distances.resize(k)
        self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices,
                               k_sqr_distances)

        for i in range(k):
            sqdist[i] = k_sqr_distances[i]
            ind[i] = k_indices[i]

