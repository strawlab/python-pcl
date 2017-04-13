# -*- coding: utf-8 -*-

from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

# boost
from boost_shared_ptr cimport shared_ptr

# Base Interface
# Search.h
# namespace pcl
# namespace search
# template<typename PointT>
# class Search
cdef extern from "pcl/Search/Search.h" namespace "pcl::search":
    Search[T]:
        Search()
        # Search (const string& name = "", bool sorted = false)
        # public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<pcl::search::Search<PointT> > Ptr;
        # typedef boost::shared_ptr<const pcl::search::Search<PointT> > ConstPtr;
        # typedef boost::shared_ptr<std::vector<int> > IndicesPtr;
        # typedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;
        # /** \brief returns the search method name
        string getName ()
        
        # /** \brief sets whether the results should be sorted (ascending in the distance) or not
        #   * \param[in] sorted should be true if the results should be sorted by the distance in ascending order.
        #   * Otherwise the results may be returned in any order.
        void setSortedResults (bool sorted)
        
        # /** \brief Pass the input dataset that the search will be performed on.
        #   * \param[in] cloud a const pointer to the PointCloud data
        #   * \param[in] indices the point indices subset that is to be used from the cloud
        # virtual void setInputCloud (const PointCloudConstPtr& cloud, const IndicesConstPtr &indices = IndicesConstPtr ())
        
        # /** \brief Get a pointer to the input point cloud dataset. */
        # virtual PointCloudConstPtr getInputCloud () const
        
        # /** \brief Get a pointer to the vector of indices used. */
        # virtual IndicesConstPtr getIndices () const
        
        # /** \brief Search for the k-nearest neighbors for the given query point.
        #   * \param[in] point the given query point
        #   * \param[in] k the number of neighbors to search for
        #   * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k
        #   * a priori!)
        #   * \return number of neighbors found
        # virtual int nearestKSearch (const PointT &point, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const = 0;
        
        # /** \brief Search for k-nearest neighbors for the given query point.
        #   * This method accepts a different template parameter for the point type.
        #   * \param[in] point the given query point
        #   * \param[in] k the number of neighbors to search for
        #   * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k
        #   * a priori!)
        #   * \return number of neighbors found
        # template <typename PointTDiff>
        # inline int nearestKSearchT (const PointTDiff &point, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
        
        # /** \brief Search for k-nearest neighbors for the given query point.
        #   * \attention This method does not do any bounds checking for the input index
        #   * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        #   * \param[in] cloud the point cloud data
        #   * \param[in] index a \a valid index in \a cloud representing a \a valid (i.e., finite) query point
        #   * \param[in] k the number of neighbors to search for
        #   * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k
        #   * a priori!)
        #   * \return number of neighbors found
        #   * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        # virtual int nearestKSearch (const PointCloud &cloud, int index, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
        # 
        # /** \brief Search for k-nearest neighbors for the given query point (zero-copy).
        #   * \attention This method does not do any bounds checking for the input index
        #   * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        #   * \param[in] index a \a valid index representing a \a valid query point in the dataset given
        #   * by \a setInputCloud. If indices were given in setInputCloud, index will be the position in
        #   * the indices vector.
        #   * \param[in] k the number of neighbors to search for
        #   * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k
        #   * a priori!)
        #   * \return number of neighbors found
        #   * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        # virtual int nearestKSearch (int index, int k, vector[int] &k_indices, vector[float] &k_sqr_distances) const
        # 
        # /** \brief Search for the k-nearest neighbors for the given query point.
        #   * \param[in] cloud the point cloud data
        #   * \param[in] indices a vector of point cloud indices to query for nearest neighbors
        #   * \param[in] k the number of neighbors to search for
        #   * \param[out] k_indices the resultant indices of the neighboring points, k_indices[i] corresponds to the neighbors of the query point i
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points, k_sqr_distances[i] corresponds to the neighbors of the query point i
        # virtual void nearestKSearch (const PointCloud& cloud, vector[int]& indices, int k, vector[vector[int] ]& k_indices, std::vector< std::vector<float> >& k_sqr_distances) const
        # 
        # /** \brief Search for the k-nearest neighbors for the given query point. 
        #   *  Use this method if the query points are of a different type than the points in the data set (e.g. PointXYZRGBA instead of PointXYZ).
        #   * \param[in] cloud the point cloud data
        #   * \param[in] indices a vector of point cloud indices to query for nearest neighbors
        #   * \param[in] k the number of neighbors to search for
        #   * \param[out] k_indices the resultant indices of the neighboring points, k_indices[i] corresponds to the neighbors of the query point i
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points, k_sqr_distances[i] corresponds to the neighbors of the query point i
        #   * \note This method copies the input point cloud of type PointTDiff to a temporary cloud of type PointT and performs the batch search on the new cloud. You should prefer the single-point search if you don't use a search algorithm that accelerates batch NN search.
        # template <typename PointTDiff> 
        # void nearestKSearchT (const pcl::PointCloud<PointTDiff> &cloud, const std::vector<int>& indices, int k, std::vector< std::vector<int> > &k_indices, std::vector< std::vector<float> > &k_sqr_distances) const
        # 
        # /** \brief Search for all the nearest neighbors of the query point in a given radius.
        #   * \param[in] point the given query point
        #   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        #   * \param[out] k_indices the resultant indices of the neighboring points
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
        #   * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        #   * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        #   * returned.
        #   * \return number of neighbors found in radius
        # virtual int radiusSearch (const PointT& point, double radius, std::vector<int>& k_indices, std::vector<float>& k_sqr_distances, unsigned int max_nn = 0) const = 0;
        # 
        # /** \brief Search for all the nearest neighbors of the query point in a given radius.
        #   * \param[in] point the given query point
        #   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        #   * \param[out] k_indices the resultant indices of the neighboring points
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
        #   * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        #   * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        #   * returned.
        #   * \return number of neighbors found in radius
        # template <typename PointTDiff> 
        # inline int radiusSearchT (const PointTDiff &point, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
        # 
        # /** \brief Search for all the nearest neighbors of the query point in a given radius.
        #   * \attention This method does not do any bounds checking for the input index
        #   * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        #   * \param[in] cloud the point cloud data
        #   * \param[in] index a \a valid index in \a cloud representing a \a valid (i.e., finite) query point
        #   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        #   * \param[out] k_indices the resultant indices of the neighboring points
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
        #   * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        #   * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        #   * returned.
        #   * \return number of neighbors found in radius
        #   * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        # virtual int radiusSearch (const PointCloud &cloud, int index, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
        # 
        # /** \brief Search for all the nearest neighbors of the query point in a given radius (zero-copy).
        #   * \attention This method does not do any bounds checking for the input index
        #   * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        #   * \param[in] index a \a valid index representing a \a valid query point in the dataset given
        #   * by \a setInputCloud. If indices were given in setInputCloud, index will be the position in
        #   * the indices vector.
        #   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        #   * \param[out] k_indices the resultant indices of the neighboring points
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
        #   * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        #   * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        #   * returned.
        #   * \return number of neighbors found in radius
        #   * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        # virtual int radiusSearch (int index, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
        # 
        # /** \brief Search for all the nearest neighbors of the query point in a given radius.
        #   * \param[in] cloud the point cloud data
        #   * \param[in] indices the indices in \a cloud. If indices is empty, neighbors will be searched for all points.
        #   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        #   * \param[out] k_indices the resultant indices of the neighboring points, k_indices[i] corresponds to the neighbors of the query point i
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points, k_sqr_distances[i] corresponds to the neighbors of the query point i
        #   * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        #   * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        #   * returned.
        # virtual void radiusSearch (const PointCloud& cloud,
        #               const std::vector<int>& indices,
        #               double radius,
        #               std::vector< std::vector<int> >& k_indices,
        #               std::vector< std::vector<float> > &k_sqr_distances,
        #               unsigned int max_nn = 0) const
        # 
        # /** \brief Search for all the nearest neighbors of the query points in a given radius.
        #   * \param[in] cloud the point cloud data
        #   * \param[in] indices a vector of point cloud indices to query for nearest neighbors
        #   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        #   * \param[out] k_indices the resultant indices of the neighboring points, k_indices[i] corresponds to the neighbors of the query point i
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points, k_sqr_distances[i] corresponds to the neighbors of the query point i
        #   * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        #   * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        #   * returned.
        #   * \note This method copies the input point cloud of type PointTDiff to a temporary cloud of type PointT and performs the batch search on the new cloud. You should prefer the single-point search if you don't use a search algorithm that accelerates batch NN search.
        # template <typename PointTDiff> void
        # radiusSearchT (const pcl::PointCloud<PointTDiff> &cloud,
        #                const std::vector<int>& indices,
        #                double radius,
        #                std::vector< std::vector<int> > &k_indices,
        #                std::vector< std::vector<float> > &k_sqr_distances,
        #                unsigned int max_nn = 0) const


###

# template<typename PointT> void
# Search<PointT>::sortResults (std::vector<int>& indices, std::vector<float>& distances) const
# cdef extern from "pcl/Search/Search.h" namespace "pcl::search":
#   cdef Search[T]::sortResults (std::vector<int>& indices, std::vector<float>& distances) const
###

# pcl_search target out
cdef extern from "pcl/Search/brute_force.h" namespace "pcl::search":
    cdef cppclass BruteForce[PointT](Search[PointT]):
        BruteForce()
        # BruteForce (bool sorted_results = false)
        # ctypedef typename Search<PointT>::PointCloud PointCloud;
        # ctypedef typename Search<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # ctypedef shared_ptr[vector[int]] IndicesPtr;
        # ctypedef shared_ptr[vector[int]] IndicesConstPtr;
        # using Search<PointT>::input_;
        # using Search<PointT>::indices_;
        # using Search<PointT>::sorted_results_;
        # 
        # cdef struct Entry
        #     Entry(int , float)
        #     Entry ()
        #     unsigned index;
        #     float distance;
        
        # replace by some metric functor
        # float getDistSqr (const PointT& point1, const PointT& point2) const;
        float getDistSqr (const PointT& point1, const PointT& point2)
        
        # int nearestKSearch (const PointT &point, int k, std::vector<int> &k_indices, std::vector<float> &k_distances) const;
        int nearestKSearch (const PointT &point, int k, vector[int] &k_indices, vector[float] &k_distances)
        
        # int radiusSearch (const PointT& point, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const;
        int radiusSearch (const PointT& point, double radius, vector[int] &k_indices, vector[float] &k_sqr_distances, unsigned int max_nn)
        
        # int denseKSearch (const PointT &point, int k, std::vector<int> &k_indices, std::vector<float> &k_distances) const;
        int denseKSearch (const PointT &point, int k, vector[int] &k_indices, vector[float] &k_distances)
        
        # int sparseKSearch (const PointT &point, int k, std::vector<int> &k_indices, std::vector<float> &k_distances) const;
        int sparseKSearch (const PointT &point, int k, vector[int] &k_indices, vector[float] &k_distances)
        
        # int denseRadiusSearch (const PointT& point, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const;
        int denseRadiusSearch (const PointT& point, double radius, vector[int] &k_indices, vector[float] &k_sqr_distances, unsigned int max_nn)
        
        # int sparseRadiusSearch (const PointT& point, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const;
        int sparseRadiusSearch (const PointT& point, double radius, vector[int] &k_indices, vector[float] &k_sqr_distances, unsigned int max_nn)


# ctypedef BruteForce
###

# pcl_search target out
cdef extern from "pcl/Search/flann_search.h" namespace "pcl":
    cdef cppclass FlannSearch[T](Search[PointT]):
        VoxelGrid()
        
        void setLeafSize (float, float, float)
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        void filter(cpp.PointCloud[T] c)
        
        # # ctypedef typename Search<PointT>::PointCloud PointCloud;
        # # ctypedef typename Search<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # ctypedef sharedptr[vector[int]] IndicesPtr;
        # ctypedef sharedptr[vector[int]] IndicesConstPtr;
        # # ctypedef flann::NNIndex[FlannDistance] Index;
        # ctypedef sharedptr[flann::NNIndex[FlannDistance]] IndexPtr;
        # ctypedef sharedptr[flann::Matrix[float]] MatrixPtr;
        # ctypedef sharedptr[flann::Matrix[float]] MatrixConstPtr;
        # # ctypedef pcl::PointRepresentation<PointT> PointRepresentation;
        # //typedef boost::shared_ptr<PointRepresentation> PointRepresentationPtr;
        # ctypedef sharedptr[PointRepresentation] PointRepresentationConstPtr;
        # # using Search<PointT>::input_;
        # # using Search<PointT>::indices_;
        # # using Search<PointT>::sorted_results_;
        
        # public:
        # ctypedef sharedptr[FlannSearch[PointT]] Ptr;
        # ctypedef sharedptr[FlannSearch[PointT]] ConstPtr;
        # # cdef cppclass FlannIndexCreator
        # #    virtual IndexPtr createIndex (MatrixConstPtr data)=0;
        # # class KdTreeIndexCreator: public FlannIndexCreator
        # cdef cppclass KdTreeIndexCreator:
        # # KdTreeIndexCreator (unsigned int max_leaf_size=15)
        # KdTreeIndexCreator (unsigned int)
        # # virtual IndexPtr createIndex (MatrixConstPtr data);
        # cdef FlannSearch (bool sorted = true, FlannIndexCreator* creator = new KdTreeIndexCreator());
        # cdef void setEpsilon (double eps)
        # cdef double getEpsilon ()
        # cdef void setInputCloud (const PointCloudConstPtr& cloud, const IndicesConstPtr& indices = IndicesConstPtr ());
        # cdef int nearestKSearch (const PointT &point, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const;
        # cdef void nearestKSearch (const PointCloud& cloud, const std::vector<int>& indices, int k, 
        #                           std::vector< std::vector<int> >& k_indices, std::vector< std::vector<float> >& k_sqr_distances) const;
        # cdef int radiusSearch (const PointT& point, double radius, 
        #                       std::vector<int> &k_indices, std::vector<float> &k_sqr_distances,
        #                       unsigned int max_nn = 0) const;
        # cdef void radiusSearch (const PointCloud& cloud, const std::vector<int>& indices, double radius, std::vector< std::vector<int> >& k_indices,
        #                       vector[vector[float]] k_sqr_distances, unsigned int max_nn=0) const;
        # cdef void setPointRepresentation (const PointRepresentationConstPtr &point_representation)
        # cdef PointRepresentationConstPtr getPointRepresentation ()


###

# Conflict pcl_kdtree ?
# cdef extern from "pcl/Search/kdtree.h" namespace "pcl::search":
#     cdef cppclass KdTree[PointT](Search[PointT]):
#         # KdTree()
#         KdTree (bool)
#         # public:
#         # ctypedef typename Search<PointT>::PointCloud PointCloud;
#         # ctypedef typename Search<PointT>::PointCloudConstPtr PointCloudConstPtr;
#         
#         # ctypedef boost::shared_ptr<std::vector<int> > IndicesPtr;
#         # ctypedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;
#         # using pcl::search::Search<PointT>::indices_;
#         # using pcl::search::Search<PointT>::input_;
#         # using pcl::search::Search<PointT>::getIndices;
#         # using pcl::search::Search<PointT>::getInputCloud;
#         # using pcl::search::Search<PointT>::nearestKSearch;
#         # using pcl::search::Search<PointT>::radiusSearch;
#         # using pcl::search::Search<PointT>::sorted_results_;
#         # typedef boost::shared_ptr<KdTree<PointT> > Ptr;
#         # typedef boost::shared_ptr<const KdTree<PointT> > ConstPtr;
#         # typedef boost::shared_ptr<pcl::KdTreeFLANN<PointT> > KdTreeFLANNPtr;
#         # typedef boost::shared_ptr<const pcl::KdTreeFLANN<PointT> > KdTreeFLANNConstPtr;
#         
#         void setSortedResults (bool sorted_results)
#         
#         void setEpsilon (float eps)
#         
#         float getEpsilon ()
#         
#         # void setInputCloud (const PointCloudConstPtr& cloud, const IndicesConstPtr& indices = IndicesConstPtr ())
#         
#         # int nearestKSearch (const PointT &point, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
#         int nearestKSearch (const PointT &point, int k, vector[int] &k_indices, vector[float] &k_sqr_distances)
#         
#         int radiusSearch (const PointT& point, double radius, vector[int] &k_indices, vector[float] &k_sqr_distances, unsigned int max_nn)
# 
# 
###

# Conflict pcl_Octree ?
# cdef extern from "pcl/Search/Octree.h" namespace "pcl::search":
#     cdef cppclass Octree[PointT](Search[PointT]):
#         # Octree (const double resolution)
#         Octree (double)
#         
#         # public:
#         # ctypedef boost::shared_ptr<std::vector<int> > IndicesPtr;
#         # ctypedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;
#         # ctypedef pcl::PointCloud<PointT> PointCloud;
#         # ctypedef boost::shared_ptr<PointCloud> PointCloudPtr;
#         # ctypedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
#         # ctypedef boost::shared_ptr<pcl::octree::OctreePointCloudSearch<PointT, LeafTWrap, BranchTWrap> > Ptr;
#         # ctypedef boost::shared_ptr<const pcl::octree::OctreePointCloudSearch<PointT, LeafTWrap, BranchTWrap> > ConstPtr;
#         # Ptr tree_;
#         # using pcl::search::Search<PointT>::input_;
#         # using pcl::search::Search<PointT>::indices_;
#         # using pcl::search::Search<PointT>::sorted_results_;
#         
#         # void setInputCloud (const PointCloudConstPtr &cloud)
#         void setInputCloud (const shared_ptr[cpp.PointCloud[PointT]] &cloud)
#         
#         # void setInputCloud (const PointCloudConstPtr &cloud, const IndicesConstPtr& indices)
#         # void setInputCloud (const shared_ptr[cpp.PointCloud[PointT]] &cloud, const IndicesConstPtr& indices)
#         
#         int nearestKSearch (const cpp.PointCloud[PointT] &cloud, int index, int k, vector[int] &k_indices, vector[float] &k_sqr_distances)
#         
#         # int nearestKSearch (const PointT &point, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
#         int nearestKSearch (const PointT &point, int k, vector[int] &k_indices, vector[float] &k_sqr_distances)
#         
#         # int nearestKSearch (int index, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
#         int nearestKSearch (int index, int k, vector[int] &k_indices, vector[float] &k_sqr_distances)
#         
#         # int radiusSearch ( const PointCloud &cloud,  int index,  double radius, std::vector<int> &k_indices,  std::vector<float> &k_sqr_distances,  unsigned int max_nn = 0) const
#         int radiusSearch ( const cpp.PointCloud[PointT] &cloud,  int index,  double radius, vector[int] &k_indices, vector[float] &k_sqr_distances, unsigned int max_nn)
#         
#         # int radiusSearch (const PointT &p_q,  double radius,  std::vector<int> &k_indices, std::vector<float> &k_sqr_distances,  unsigned int max_nn = 0) const
#         int radiusSearch (const PointT &p_q,  double radius,  vector[int] &k_indices, vector[float] &k_sqr_distances,  unsigned int max_nn)
#         
#         # cdef int radiusSearch (int index, double radius, vector[int] &k_indices, vector[float] &k_sqr_distances, unsigned int max_nn = 0) const
#         int radiusSearch (int index, double radius, vector[int] &k_indices, vector[float] &k_sqr_distances, unsigned int max_nn)
#         
#         # cdef void approxNearestSearch ( const PointCloudConstPtr &cloud, int query_index, int &result_index, float &sqr_distance)
#         void approxNearestSearch ( const shared_ptr[cpp.PointCloud[PointT]] &cloud, int query_index, int &result_index, float &sqr_distance)
#         
#         # cdef void approxNearestSearch ( const PointT &p_q, int &result_index, float &sqr_distance)
#         
#         # cdef void approxNearestSearch (int query_index, int &result_index, float &sqr_distance)
# 
# 
####

cdef extern from "pcl/Search/organized.h" namespace "pcl::search":
    cdef cppclass OrganizedNeighbor[PointT](Search[PointT]):
        OrganizedNeighbor()
        # OrganizedNeighbor (bool sorted_results = false, float eps = 1e-4f, unsigned pyramid_level = 5)
        # public:
        # ctypedef pcl::PointCloud<PointT> PointCloud;
        # ctypedef boost::shared_ptr<PointCloud> PointCloudPtr;
        # ctypedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
        # ctypedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;
        # ctypedef boost::shared_ptr<pcl::search::OrganizedNeighbor<PointT> > Ptr;
        # ctypedef boost::shared_ptr<const pcl::search::OrganizedNeighbor<PointT> > ConstPtr;
        # using pcl::search::Search<PointT>::indices_;
        # using pcl::search::Search<PointT>::sorted_results_;
        # using pcl::search::Search<PointT>::input_;
        
        # bool isValid () const
        bool isValid ()
        
        # void computeCameraMatrix (Eigen::Matrix3f& camera_matrix) const;
        # void computeCameraMatrix (eigen3.Matrix3f& camera_matrix)
        
        # void setInputCloud (const PointCloudConstPtr& cloud, const IndicesConstPtr &indices = IndicesConstPtr ())
        
        # int radiusSearch (const PointT &p_q, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const;
        int radiusSearch (const PointT &p_q, double radius, vector[int] &k_indices, vector[float] &k_sqr_distances, unsigned int max_nn)
        
        void estimateProjectionMatrix ()
        
        int nearestKSearch ( const PointT &p_q, int k, vector[int] &k_indices, vector[float] &k_sqr_distances)
        
        # bool projectPoint (const PointT& p, pcl::PointXY& q) const;


###

# pcl_search.h
# include header
###


