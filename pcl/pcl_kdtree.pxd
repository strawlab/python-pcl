# -*- coding: utf-8 -*-
from libcpp.vector cimport vector

# main
cimport pcl_defs as cpp

from boost_shared_ptr cimport shared_ptr

# flann.h
###

# io.h
# namespace pcl
# {
#   /** \brief Get a set of approximate indices for a given point cloud into a reference point cloud. 
#     * The coordinates of the two point clouds can differ. The method uses an internal KdTree for 
#     * finding the closest neighbors from \a cloud_in in \a cloud_ref. 
#     *
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] cloud_ref the reference point cloud dataset
#     * \param[out] indices the resultant set of nearest neighbor indices of \a cloud_in in \a cloud_ref
#     * \ingroup kdtree
#     */
#   template <typename PointT> void 
#   getApproximateIndices (const typename pcl::PointCloud<PointT>::Ptr &cloud_in, 
#                          const typename pcl::PointCloud<PointT>::Ptr &cloud_ref,
#                          std::vector<int> &indices);
# 
#   /** \brief Get a set of approximate indices for a given point cloud into a reference point cloud. 
#     * The coordinates of the two point clouds can differ. The method uses an internal KdTree for 
#     * finding the closest neighbors from \a cloud_in in \a cloud_ref. 
#     *
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] cloud_ref the reference point cloud dataset
#     * \param[out] indices the resultant set of nearest neighbor indices of \a cloud_in in \a cloud_ref
#     * \ingroup kdtree
#     */
#   template <typename Point1T, typename Point2T> void 
#   getApproximateIndices (const typename pcl::PointCloud<Point1T>::Ptr &cloud_in, 
#                          const typename pcl::PointCloud<Point2T>::Ptr &cloud_ref,
#                          std::vector<int> &indices);
# }
###

# kdtree.h
# namespace pcl
# template <typename PointT>
# class KdTree
cdef extern from "pcl/kdtree/kdtree.h" namespace "pcl::search":
    cdef cppclass KdTree[T]:
        KdTree()
        # KdTree (bool sorted)
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        # public:
        # typedef boost::shared_ptr <std::vector<int> > IndicesPtr;
        # typedef boost::shared_ptr <const std::vector<int> > IndicesConstPtr;
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef boost::shared_ptr<PointCloud> PointCloudPtr;
        # typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
        # typedef pcl::PointRepresentation<PointT> PointRepresentation;
        # //typedef boost::shared_ptr<PointRepresentation> PointRepresentationPtr;
        # typedef boost::shared_ptr<const PointRepresentation> PointRepresentationConstPtr;
        # // Boost shared pointers
        # typedef boost::shared_ptr<KdTree<PointT> > Ptr;
        # typedef boost::shared_ptr<const KdTree<PointT> > ConstPtr;
        # /** \brief Empty constructor for KdTree. Sets some internal values to their defaults. 
        # * \param[in] sorted set to true if the application that the tree will be used for requires sorted nearest neighbor indices (default). False otherwise. 
        # */
        # KdTree (bool sorted = true)
        # /** \brief Provide a pointer to the input dataset.
        # * \param[in] cloud the const boost shared pointer to a PointCloud message
        # * \param[in] indices the point indices subset that is to be used from \a cloud - if NULL the whole cloud is used
        # virtual void setInputCloud (const PointCloudConstPtr &cloud, const IndicesConstPtr &indices = IndicesConstPtr ())
        # /** \brief Get a pointer to the vector of indices used. */
        # inline IndicesConstPtr getIndices () const
        # /** \brief Get a pointer to the input point cloud dataset. */
        # inline PointCloudConstPtr getInputCloud () const
        # /** \brief Provide a pointer to the point representation to use to convert points into k-D vectors. 
        # * \param[in] point_representation the const boost shared pointer to a PointRepresentation
        # inline void setPointRepresentation (const PointRepresentationConstPtr &point_representation)
        # /** \brief Get a pointer to the point representation used when converting points into k-D vectors. */
        # inline PointRepresentationConstPtr getPointRepresentation () const
        # /** \brief Search for k-nearest neighbors for the given query point.
        # * \param[in] p_q the given query point
        # * \param[in] k the number of neighbors to search for
        # * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
        # * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k 
        # * a priori!)
        # * \return number of neighbors found
        # virtual int nearestKSearch (const PointT &p_q, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const = 0;
        # * \brief Search for k-nearest neighbors for the given query point.
        # * \attention This method does not do any bounds checking for the input index
        # * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        # * \param[in] cloud the point cloud data
        # * \param[in] index a \a valid index in \a cloud representing a \a valid (i.e., finite) query point
        # * \param[in] k the number of neighbors to search for
        # * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
        # * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k 
        # * a priori!)
        # * \return number of neighbors found
        # * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        # virtual int nearestKSearch (const PointCloud &cloud, int index, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
        # * \brief Search for k-nearest neighbors for the given query point. 
        # * This method accepts a different template parameter for the point type.
        # * \param[in] point the given query point
        # * \param[in] k the number of neighbors to search for
        # * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
        # * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k 
        # * a priori!)
        # * \return number of neighbors found
        # template <typename PointTDiff> inline int nearestKSearchT (const PointTDiff &point, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
        # * \brief Search for k-nearest neighbors for the given query point (zero-copy).
        # * \attention This method does not do any bounds checking for the input index
        # * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        # * \param[in] index a \a valid index representing a \a valid query point in the dataset given 
        # * by \a setInputCloud. If indices were given in setInputCloud, index will be the position in 
        # * the indices vector.
        # * \param[in] k the number of neighbors to search for
        # * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
        # * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k 
        # * a priori!)
        # * \return number of neighbors found
        # * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        # virtual int nearestKSearch (int index, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
        # * \brief Search for all the nearest neighbors of the query point in a given radius.
        # * \param[in] p_q the given query point
        # * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        # * \param[out] k_indices the resultant indices of the neighboring points
        # * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
        # * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        # * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        # * returned.
        # * \return number of neighbors found in radius
        # virtual int radiusSearch (const PointT &p_q, double radius, std::vector<int> &k_indices,std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const = 0;
        # * \brief Search for all the nearest neighbors of the query point in a given radius.
        # * \attention This method does not do any bounds checking for the input index
        # * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        # * \param[in] cloud the point cloud data
        # * \param[in] index a \a valid index in \a cloud representing a \a valid (i.e., finite) query point
        # * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        # * \param[out] k_indices the resultant indices of the neighboring points
        # * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
        # * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        # * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        # * returned.
        # * \return number of neighbors found in radius
        # * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        # virtual int radiusSearch (const PointCloud &cloud, int index, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
        # * \brief Search for all the nearest neighbors of the query point in a given radius.
        # * \param[in] point the given query point
        # * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        # * \param[out] k_indices the resultant indices of the neighboring points
        # * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
        # * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        # * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        # * returned.
        # * \return number of neighbors found in radius
        # template <typename PointTDiff> inline int radiusSearchT (const PointTDiff &point, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
        # * \brief Search for all the nearest neighbors of the query point in a given radius (zero-copy).
        # * \attention This method does not do any bounds checking for the input index
        # * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        # * \param[in] index a \a valid index representing a \a valid query point in the dataset given 
        # * by \a setInputCloud. If indices were given in setInputCloud, index will be the position in 
        # * the indices vector.
        # * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        # * \param[out] k_indices the resultant indices of the neighboring points
        # * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
        # * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        # * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        # * returned.
        # * \return number of neighbors found in radius
        # * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        # virtual int radiusSearch (int index, double radius, std::vector<int> &k_indices,std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
        # * \brief Set the search epsilon precision (error bound) for nearest neighbors searches.
        # * \param[in] eps precision (error bound) for nearest neighbors searches
        # virtual inline void setEpsilon (float eps)
        # * \brief Get the search epsilon precision (error bound) for nearest neighbors searches. */
        # inline float getEpsilon () const
        # * \brief Minimum allowed number of k nearest neighbors points that a viable result must contain. 
        # * \param[in] min_pts the minimum number of neighbors in a viable neighborhood 
        # inline void setMinPts (int min_pts)
        # * \brief Get the minimum allowed number of k nearest neighbors points that a viable result must contain. */
        # inline int getMinPts () const

ctypedef KdTree[cpp.PointXYZ] KdTree_t
ctypedef KdTree[cpp.PointXYZI] KdTree_PointXYZI_t
ctypedef KdTree[cpp.PointXYZRGB] KdTree_PointXYZRGB_t
ctypedef KdTree[cpp.PointXYZRGBA] KdTree_PointXYZRGBA_t

ctypedef shared_ptr[KdTree[cpp.PointXYZ]] KdTreePtr_t
ctypedef shared_ptr[KdTree[cpp.PointXYZI]] KdTree_PointXYZI_Ptr_t
ctypedef shared_ptr[KdTree[cpp.PointXYZRGB]] KdTree_PointXYZRGB_Ptr_t
ctypedef shared_ptr[KdTree[cpp.PointXYZRGBA]] KdTree_PointXYZRGBA_Ptr_t
###

# kdtree_flann.h
# NG
# cdef cppclass KdTreeFLANN[T](KdTree[T]):
# namespace pcl
# template <typename PointT, typename Dist = flann::L2_Simple<float> >
# class KdTreeFLANN : public pcl::KdTree<PointT>
cdef extern from "pcl/kdtree/kdtree_flann.h" namespace "pcl":
    cdef cppclass KdTreeFLANN[T]:
        KdTreeFLANN()
        # KdTreeFLANN (bool sorted)
        # KdTreeFLANN (const KdTreeFLANN<PointT> &k) : 
        #       inline KdTreeFLANN<PointT>& operator = (const KdTreeFLANN<PointT>& k)
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        
        # /** \brief Search for k-nearest neighbors for the given query point.
        #   * \attention This method does not do any bounds checking for the input index
        #   * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        #   * \param[in] point a given \a valid (i.e., finite) query point
        #   * \param[in] k the number of neighbors to search for
        #   * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k 
        #   * a priori!)
        #   * \return number of neighbors found
        #   * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        #   */
        # int nearestKSearch (cpp.PointCloud[T], int, vector[int], vector[float])
        # inline define
        int nearestKSearch (cpp.PointCloud[T], int, int, vector[int], vector[float])
        # int nearestKSearch (const PointT &point, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const;
        
        # /** \brief Search for all the nearest neighbors of the query point in a given radius.
        #   * \attention This method does not do any bounds checking for the input index
        #   * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        #   * \param[in] point a given \a valid (i.e., finite) query point
        #   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        #   * \param[out] k_indices the resultant indices of the neighboring points
        #   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
        #   * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        #   * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        #   * returned.
        #   * \return number of neighbors found in radius
        #   * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        #   */
        # int radiusSearch (cpp.PointCloud[T], double, vector[int], vector[float])
        # int radiusSearch (cpp.PointCloud[T], double, vector[int], vector[float], unsigned int)
        # inline define
        int radiusSearch (cpp.PointCloud[T], int, double, vector[int], vector[float])
        # int radiusSearch (const PointT &point, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const;

        # using KdTree<PointT>::input_;
        # using KdTree<PointT>::indices_;
        # using KdTree<PointT>::epsilon_;
        # using KdTree<PointT>::sorted_;
        # using KdTree<PointT>::point_representation_;
        # using KdTree<PointT>::nearestKSearch;
        # using KdTree<PointT>::radiusSearch;
        # typedef typename KdTree<PointT>::PointCloud PointCloud;
        # typedef typename KdTree<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<std::vector<int> > IndicesPtr;
        # typedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;
        # typedef flann::Index<Dist> FLANNIndex;
        # // Boost shared pointers
        # typedef boost::shared_ptr<KdTreeFLANN<PointT> > Ptr;
        # typedef boost::shared_ptr<const KdTreeFLANN<PointT> > ConstPtr;
        # /** \brief Set the search epsilon precision (error bound) for nearest neighbors searches.
        #   * \param[in] eps precision (error bound) for nearest neighbors searches
        #   */
        # inline void setEpsilon (float eps)
        # inline void setSortedResults (bool sorted)
        # inline Ptr makeShared ()
###

# template <>
# class KdTreeFLANN <Eigen::MatrixXf>
#       public:
#       typedef pcl::PointCloud<Eigen::MatrixXf> PointCloud;
#       typedef PointCloud::ConstPtr PointCloudConstPtr;
#       typedef boost::shared_ptr<std::vector<int> > IndicesPtr;
#       typedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;
#       typedef flann::Index<flann::L2_Simple<float> > FLANNIndex;
#       typedef pcl::PointRepresentation<Eigen::MatrixXf> PointRepresentation;
#       typedef boost::shared_ptr<const PointRepresentation> PointRepresentationConstPtr;
#       // Boost shared pointers
#       typedef boost::shared_ptr<KdTreeFLANN<Eigen::MatrixXf> > Ptr;
#       typedef boost::shared_ptr<const KdTreeFLANN<Eigen::MatrixXf> > ConstPtr;
# 
#       KdTreeFLANN (bool sorted = true) : 
#       KdTreeFLANN (const KdTreeFLANN<Eigen::MatrixXf> &k) : 
#       inline KdTreeFLANN& operator = (const KdTreeFLANN<Eigen::MatrixXf>& k)
#       inline void setEpsilon (float eps)
#       inline Ptr makeShared ()
#       void setInputCloud (const PointCloudConstPtr &cloud, const IndicesConstPtr &indices = IndicesConstPtr ())
#       inline IndicesConstPtr getIndices () const
#       inline PointCloudConstPtr getInputCloud () const
#       template <typename T> int nearestKSearch (const T &point, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
#       inline int nearestKSearch (const PointCloud &cloud, int index, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
#       inline int nearestKSearch (int index, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const
#       template <typename T> int radiusSearch (const T &point, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_dists, unsigned int max_nn = 0) const
#       inline int radiusSearch (const PointCloud &cloud, int index, double radius, 
#                     std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, 
#                     unsigned int max_nn = 0) const
#       inline int radiusSearch (int index, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
#       /** \brief Get the search epsilon precision (error bound) for nearest neighbors searches. */
#       inline float getEpsilon () const
#       protected:
#       /** \brief The input point cloud dataset containing the points we need to use. */
#       PointCloudConstPtr input_;
#       /** \brief A pointer to the vector of point indices to use. */
#       IndicesConstPtr indices_;
#       /** \brief Epsilon precision (error bound) for nearest neighbors searches. */
#       float epsilon_;
#       /** \brief Return the radius search neighbours sorted **/
#       bool sorted_;
### 

ctypedef KdTreeFLANN[cpp.PointXYZ] KdTreeFLANN_t
ctypedef KdTreeFLANN[cpp.PointXYZI] KdTreeFLANN_PointXYZI_t
ctypedef KdTreeFLANN[cpp.PointXYZRGB] KdTreeFLANN_PointXYZRGB_t
ctypedef KdTreeFLANN[cpp.PointXYZRGBA] KdTreeFLANN_PointXYZRGBA_t

ctypedef shared_ptr[KdTreeFLANN[cpp.PointXYZ]] KdTreeFLANNPtr_t
ctypedef shared_ptr[KdTreeFLANN[cpp.PointXYZI]] KdTreeFLANN_PointXYZI_Ptr_t
ctypedef shared_ptr[KdTreeFLANN[cpp.PointXYZRGB]] KdTreeFLANN_PointXYZRGB_Ptr_t
ctypedef shared_ptr[KdTreeFLANN[cpp.PointXYZRGBA]] KdTreeFLANN_PointXYZRGBA_Ptr_t

ctypedef shared_ptr[const KdTreeFLANN[cpp.PointXYZ]] KdTreeFLANNConstPtr_t
ctypedef shared_ptr[const KdTreeFLANN[cpp.PointXYZI]] KdTreeFLANN_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const KdTreeFLANN[cpp.PointXYZRGB]] KdTreeFLANN_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const KdTreeFLANN[cpp.PointXYZRGBA]] KdTreeFLANN_PointXYZRGBA_ConstPtr_t

###

