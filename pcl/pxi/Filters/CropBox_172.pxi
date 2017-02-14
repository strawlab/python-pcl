# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_filters_172 as pclfil

cimport eigen as eigen3

from boost_shared_ptr cimport shared_ptr

cdef class CropBox:
    """
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in CropBox(pc).
    """
    cdef pclfil.CropBox_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclfil.CropBox_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    def set_Translation(self, tx, ty, tz):
        # Convert Eigen::Vector3f
        cdef eigen3.Vector3f origin
        cdef float *data = origin.data()
        data[0] = tx
        data[1] = ty
        data[2] = tz
        self.me.setTranslation(origin)

    # def set_Rotation(self, cnp.ndarray[ndim=3, dtype=int, mode='c'] ind):
    def set_Rotation(self, rx, ry, rz):
        # Convert Eigen::Vector3f
        cdef eigen3.Vector3f origin
        cdef float *data = origin.data()
        data[0] = rx
        data[1] = ry
        data[2] = rz
        self.me.setRotation(origin)

    def set_MinMax(self, minx, miny, minz, mins, maxx, maxy, maxz, maxs):
        # Convert Eigen::Vector4f
        cdef eigen3.Vector4f originMin
        cdef float *dataMin = originMin.data()
        dataMin[0] = minx
        dataMin[1] = miny
        dataMin[2] = minz
        dataMin[3] = mins
        self.me.setMin(originMin)
        
        cdef eigen3.Vector4f originMax;
        cdef float *dataMax = originMax.data()
        dataMax[0] = maxx
        dataMax[1] = maxy
        dataMax[2] = maxz
        dataMax[3] = maxs
        self.me.setMax(originMax)

    def Filtering(self, PointCloud outputCloud):
        # NG
        # self.me.filter(<cpp.PointCloud[cpp.PointXYZ]> outputCloud)
        # self.me.filter(outputCloud.thisptr())
        # Cython 0.25.2 NG(0.24.1 OK)
        # self.me.filter(deref(outputCloud.thisptr()))
        # Cython 0.24.1 NG(0.25.2 OK)
        # pcl 1.7.2 NG
        # self.me.filter(<vector[int]> outputCloud)
        self.me.filter(<vector[int]&> outputCloud)

    # @cython.boundscheck(False)
    # cdef void _nearest_k(self, PointCloud pc, int index, int k,
    #                      cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
    #                     ) except +:
    #     # k nearest neighbors query for a single point.
    #     cdef vector[int] k_indices
    #     cdef vector[float] k_sqr_distances
    #     k_indices.resize(k)
    #     k_sqr_distances.resize(k)
    #     self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices, k_sqr_distances)
    # 
    #     for i in range(k):
    #         sqdist[i] = k_sqr_distances[i]
    #         ind[i] = k_indices[i]
    # 
    # def nearest_k_search_for_cloud(self, PointCloud pc not None, int k=1):
    #     """
    #     Find the k nearest neighbours and squared distances for all points
    #     in the pointcloud. Results are in ndarrays, size (pc.size, k)
    #     Returns: (k_indices, k_sqr_distances)
    #     """
    #     cdef cnp.npy_intp n_points = pc.size
    #     cdef cnp.ndarray[float, ndim=2] sqdist = np.zeros((n_points, k),
    #                                                       dtype=np.float32)
    #     cdef cnp.ndarray[int, ndim=2] ind = np.zeros((n_points, k),
    #                                                  dtype=np.int32)
    # 
    #     for i in range(n_points):
    #         self._nearest_k(pc, i, k, ind[i], sqdist[i])
    #     return ind, sqdist

