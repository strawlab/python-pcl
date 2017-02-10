# -*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_filters_172 as pclfil

from boost_shared_ptr cimport shared_ptr

# include "Vertices.pxi"

cdef class CropHull:
    """
    
    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in CropHull(pc).
    """
    cdef pclfil.CropHull_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclfil.CropHull_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me

    # def SetParameter(self, shared_ptr[cpp.PointCloud[cpp.PointXYZ]] points, cpp.Vertices vt):
    def SetParameter(self, PointCloud points, Vertices vt):
        cdef const vector[cpp.Vertices] tmp_vertices
        # NG
        # tmp_vertices.push_back(deref(vt))
        # tmp_vertices.push_back<cpp.Vertices>(vt)
        # tmp_vertices.push_back[cpp.Vertices](vt)
        # tmp_vertices.push_back(vt)
        # tmp_vertices.push_back(deref(vt.thisptr()))
        self.me.setHullIndices(tmp_vertices)
        # self.me.setHullCloud(<shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> points)
        # convert <PointCloud> to <shared_ptr[cpp.PointCloud[cpp.PointXYZ]]>
        self.me.setHullCloud(points.thisptr_shared)
        self.me.setDim(<int> 2)
        self.me.setCropOutside(<bool> False)

    def Filtering(self, PointCloud outputCloud):
        # NG
        # self.me.filter(<cpp.PointCloud[cpp.PointXYZ]> outputCloud)
        # self.me.filter(outputCloud.thisptr())
        # Cython 0.25.2 NG
        # self.me.filter(deref(outputCloud.thisptr()))
        # Cython 0.24.1 NG
        # self.me.filter(<vector[int]> outputCloud)
        self.me.filter(<cpp.PointCloud[cpp.PointXYZ]> outputCloud.thisptr()[0])

    # @cython.boundscheck(False)
    # cdef void _nearest_k(self, PointCloud pc, int index, int k,
    #                      cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
    #                     ) except +:
    #     # k nearest neighbors query for a single point.
    #     cdef vector[int] k_indices
    #     cdef vector[float] k_sqr_distances
    #     k_indices.resize(k)
    #     k_sqr_distances.resize(k)
    #     self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices,
    #                            k_sqr_distances)
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

