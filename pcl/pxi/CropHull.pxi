
from libcpp.vector cimport vector
from libcpp cimport bool

cimport pcl_defs as cpp
cimport pcl_filters as pclfil

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

    # def SetParameter(self, shared_ptr[cpp.PointCloud[cpp.PointXYZ]] points, shared_ptr[cpp.Vertices] vt):
    def SetParameter(self, PointCloud points, Vertices vt):
        cdef vector[cpp.Vertices] tmp_vertices
        # NG
        # tmp_vertices.push_back(deref(vt))
        tmp_vertices.push_back(<cpp.Vertices> vt)
        self.me.setHullIndices(tmp_vertices)
        self.me.setHullCloud(<shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> points)
        self.me.setDim(<int> 2)
        self.me.setCropOutside(<bool> False)

    def Filtering(self, PointCloud outputCloud):
        self.me.filter(<cpp.PointCloud[cpp.PointXYZ]> outputCloud)

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

