from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from boost_shared_ptr cimport shared_ptr

# main
cimport pcl_defs as cpp

###############################################################################
# Types
###############################################################################

# # class ShapeContext3DEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
# cdef extern from "pcl/features/3dsc.h" namespace "pcl":
#     cdef cppclass ShapeContext3DEstimation[T, N]:
#         ShapeContext3DEstimation(bool)
#         # public:
#         # using Feature<PointInT, PointOutT>::feature_name_;
#         # using Feature<PointInT, PointOutT>::getClassName;
#         # using Feature<PointInT, PointOutT>::indices_;
#         # using Feature<PointInT, PointOutT>::search_parameter_;
#         # using Feature<PointInT, PointOutT>::search_radius_;
#         # using Feature<PointInT, PointOutT>::surface_;
#         # using Feature<PointInT, PointOutT>::input_;
#         # using Feature<PointInT, PointOutT>::searchForNeighbors;
#         # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
#        
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
# 
#         # brief Set the number of bins along the azimuth to \a bins.
#         # param[in] bins the number of bins along the azimuth
#         void setAzimuthBins (size_t)
# 
#         # return the number of bins along the azimuth
#         size_t getAzimuthBins () 
# 
#         # brief Set the number of bins along the elevation to \a bins.
#         # param[in] bins the number of bins along the elevation
#         void setElevationBins (size_t )
# 
#         # return The number of bins along the elevation
#         size_t getElevationBins ()
# 
#         # brief Set the number of bins along the radii to \a bins.
#         # param[in] bins the number of bins along the radii
#         void setRadiusBins (size_t )
# 
#         # return The number of bins along the radii direction
#         size_t getRadiusBins ()
# 
#         # brief The minimal radius value for the search sphere (rmin) in the original paper 
#         # param[in] radius the desired minimal radius
#         void setMinimalRadius (double )
# 
#         # return The minimal sphere radius
#         double getMinimalRadius ()
# 
#         # brief This radius is used to compute local point density 
#         # density = number of points within this radius
#         # param[in] radius value of the point density search radius
#         void setPointDensityRadius (double )
# 
#         # return The point density search radius
#         double getPointDensityRadius ()
#         
#         # protected:
#         # brief Initialize computation by allocating all the intervals and the volume lookup table. */
#         # bool initCompute ();
# 
#         # brief Estimate a descriptor for a given point.
#         # param[in] index the index of the point to estimate a descriptor for
#         # param[in] normals a pointer to the set of normals
#         # param[in] rf the reference frame
#         # param[out] desc the resultant estimated descriptor
#         # return true if the descriptor was computed successfully, false if there was an error 
#         # e.g. the nearest neighbor didn't return any neighbors)
#         # bool computePoint (size_t index, const pcl::PointCloud<PointNT> &normals, float rf[9], std::vector<float> &desc);
# 
#         # brief Estimate the actual feature. 
#         # param[out] output the resultant feature 
#         # void computeFeature (PointCloudOut &output);
# 
#         # brief Values of the radii interval
#         # vector<float> radii_interval_
# 
#         # brief Theta divisions interval
#         # std::vector<float> theta_divisions_;
# 
#         # brief Phi divisions interval
#         # std::vector<float> phi_divisions_;
# 
#         # brief Volumes look up table
#         # vector<float> volume_lut_;
# 
#         # brief Bins along the azimuth dimension
#         # size_t azimuth_bins_;
# 
#         # brief Bins along the elevation dimension
#         # size_t elevation_bins_;
# 
#         # brief Bins along the radius dimension
#         # size_t radius_bins_;
# 
#         # brief Minimal radius value
#         # double min_radius_;
# 
#         # brief Point density radius
#         # double point_density_radius_;
# 
#         # brief Descriptor length
#         # size_t descriptor_length_;
# 
#         # brief Boost-based random number generator algorithm.
#         # boost::mt19937 rng_alg_;
# 
#         # brief Boost-based random number generator distribution.
#         # boost::shared_ptr<boost::uniform_01<boost::mt19937> > rng_;
# 
#         # brief Shift computed descriptor "L" times along the azimuthal direction
#         # param[in] block_size the size of each azimuthal block
#         # param[in] desc at input desc == original descriptor and on output it contains 
#         # shifted descriptor resized descriptor_length_ * azimuth_bins_
#         # void shiftAlongAzimuth (size_t block_size, std::vector<float>& desc);
# 
#         # brief Boost-based random number generator.
#         # inline double rnd ()
#     
#         # private:
#         # void computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
# 
# # class ShapeContext3DEstimation<PointInT, PointNT, Eigen::MatrixXf> : public ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>
# # cdef extern from "pcl/features/3dsc.h" namespace "pcl":
# #     cdef cppclass ShapeContext3DEstimation[T, N, M]:
# #         ShapeContext3DEstimation(bool)
# #         # public:
# #         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::feature_name_;
# #         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::indices_;
# #         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::descriptor_length_;
# #         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::normals_;
# #         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::input_;
# #         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::compute;
# #         # private:
# #         # void computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# #         # void compute (pcl::PointCloud<pcl::SHOT> &) {}
# ###
# 
# # class BoundaryEstimation: public FeatureFromNormals<PointInT, PointNT, PointOutT>
# cdef extern from "pcl/features/boundary.h" namespace "pcl":
#     cdef cppclass BoundaryEstimation[I, N, O]:
#         BoundaryEstimation ()
#         # public:
#         # using Feature<PointInT, PointOutT>::feature_name_;
#         # using Feature<PointInT, PointOutT>::getClassName;
#         # using Feature<PointInT, PointOutT>::input_;
#         # using Feature<PointInT, PointOutT>::indices_;
#         # using Feature<PointInT, PointOutT>::k_;
#         # using Feature<PointInT, PointOutT>::tree_;
#         # using Feature<PointInT, PointOutT>::search_radius_;
#         # using Feature<PointInT, PointOutT>::search_parameter_;
#         # using Feature<PointInT, PointOutT>::surface_;
#         # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#         # brief Check whether a point is a boundary point in a planar patch of projected points given by indices.
#         # note A coordinate system u-v-n must be computed a-priori using \a getCoordinateSystemOnPlane
#         # param[in] cloud a pointer to the input point cloud
#         # param[in] q_idx the index of the query point in \a cloud
#         # param[in] indices the estimated point neighbors of the query point
#         # param[in] u the u direction
#         # param[in] v the v direction
#         # param[in] angle_threshold the threshold angle (default \f$\pi / 2.0\f$)
#         # bool isBoundaryPoint (const pcl::PointCloud<PointInT> &cloud, 
#         #                int q_idx, const std::vector<int> &indices, 
#         #                const Eigen::Vector4f &u, const Eigen::Vector4f &v, const float angle_threshold);
# 
#         # brief Check whether a point is a boundary point in a planar patch of projected points given by indices.
#         # note A coordinate system u-v-n must be computed a-priori using \a getCoordinateSystemOnPlane
#         # param[in] cloud a pointer to the input point cloud
#         # param[in] q_point a pointer to the querry point
#         # param[in] indices the estimated point neighbors of the query point
#         # param[in] u the u direction
#         # param[in] v the v direction
#         # param[in] angle_threshold the threshold angle (default \f$\pi / 2.0\f$)
#         # bool isBoundaryPoint (const pcl::PointCloud<PointInT> &cloud, 
#         #                const PointInT &q_point, 
#         #                const std::vector<int> &indices, 
#         #                const Eigen::Vector4f &u, const Eigen::Vector4f &v, const float angle_threshold);
# 
#         # brief Set the decision boundary (angle threshold) that marks points as boundary or regular. 
#         # (default \f$\pi / 2.0\f$) 
#         # param[in] angle the angle threshold
#         # inline void setAngleThreshold (float angle)
# 
#         # inline float getAngleThreshold ()
# 
#         # brief Get a u-v-n coordinate system that lies on a plane defined by its normal
#         # param[in] p_coeff the plane coefficients (containing the plane normal)
#         # param[out] u the resultant u direction
#         # param[out] v the resultant v direction
#         # inline void getCoordinateSystemOnPlane (const PointNT &p_coeff, 
#         #                           Eigen::Vector4f &u, Eigen::Vector4f &v)
# 
#         # protected:
#         # void computeFeature (PointCloudOut &output);
#         # float angle_threshold_;
# 
# ###
# 
# # class CVFHEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
# cdef extern from "pcl/features/cvfh.h" namespace "pcl":
#     cdef cppclass CVFHEstimation[I, N, O]:
#         CVFHEstimation()
#         # public:
#         # using Feature<PointInT, PointOutT>::feature_name_;
#         # using Feature<PointInT, PointOutT>::getClassName;
#         # using Feature<PointInT, PointOutT>::indices_;
#         # using Feature<PointInT, PointOutT>::k_;
#         # using Feature<PointInT, PointOutT>::search_radius_;
#         # using Feature<PointInT, PointOutT>::surface_;
#         # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#         # ctypedef typename pcl::search::Search<PointNormal>::Ptr KdTreePtr;
#         # ctypedef typename pcl::NormalEstimation<PointNormal, PointNormal> NormalEstimator;
#         # ctypedef typename pcl::VFHEstimation<PointInT, PointNT, pcl::VFHSignature308> VFHEstimator;
# 
#         # brief Removes normals with high curvature caused by real edges or noisy data
#         # param[in] cloud pointcloud to be filtered
#         # param[out] indices_out the indices of the points with higher curvature than threshold
#         # param[out] indices_in the indices of the remaining points after filtering
#         # param[in] threshold threshold value for curvature
#         #
#         # void filterNormalsWithHighCurvature (
#         #                                       const pcl::PointCloud<PointNT> & cloud, 
#         #                                       vector[int] &, vector[int] &,
#         #                                       vector[int] &, float);
# 
#         # brief Set the viewpoint.
#         # param[in] vpx the X coordinate of the viewpoint
#         # param[in] vpy the Y coordinate of the viewpoint
#         # param[in] vpz the Z coordinate of the viewpoint
#         inline void setViewPoint (float, float, float)
# 
#         # brief Set the radius used to compute normals
#         # param[in] radius_normals the radius
#         inline void setRadiusNormals (float)
# 
#         # brief Get the viewpoint. 
#         # param[out] vpx the X coordinate of the viewpoint
#         # param[out] vpy the Y coordinate of the viewpoint
#         # param[out] vpz the Z coordinate of the viewpoint
#         inline void getViewPoint (float &, float &, float &)
# 
#         # brief Get the centroids used to compute different CVFH descriptors
#         # param[out] centroids vector to hold the centroids
#         # inline void getCentroidClusters (vector[Eigen::Vector3f] &)
#       
#         # brief Get the normal centroids used to compute different CVFH descriptors
#         # param[out] centroids vector to hold the normal centroids
#         # inline void getCentroidNormalClusters (vector[Eigen::Vector3f] &)
# 
#         # brief Sets max. Euclidean distance between points to be added to the cluster 
#         # param[in] d the maximum Euclidean distance 
#         inline void setClusterTolerance (float)
# 
#         # brief Sets max. deviation of the normals between two points so they can be clustered together
#         # param[in] d the maximum deviation 
#         inline void setEPSAngleThreshold (float)
# 
#         # brief Sets curvature threshold for removing normals
#         # param[in] d the curvature threshold 
#         inline void setCurvatureThreshold (float)
# 
#         # brief Set minimum amount of points for a cluster to be considered
#         # param[in] min the minimum amount of points to be set 
#         inline void setMinPoints (size_t)
# 
#         # brief Sets wether if the CVFH signatures should be normalized or not
#         # param[in] normalize true if normalization is required, false otherwise 
#         inline void setNormalizeBins (bool)
# 
#         # brief Overloaded computed method from pcl::Feature.
#         # param[out] output the resultant point cloud model dataset containing the estimated features
#         # void compute (PointCloudOut &);
# 
#         # protected:
#         # /** \brief Centroids that were used to compute different CVFH descriptors */
#         # std::vector<Eigen::Vector3f> centroids_dominant_orientations_;
#         # /** \brief Normal centroids that were used to compute different CVFH descriptors */
#         # std::vector<Eigen::Vector3f> dominant_normals_;
# ###
# 
# 
# # class ESFEstimation: public Feature<PointInT, PointOutT>
# cdef extern from "pcl/features/esf.h" namespace "pcl":
#     cdef cppclass ESFEstimation[I, O]:
#         ESFEstimation ()
#         # public:
#         # using Feature<PointInT, PointOutT>::feature_name_;
#         # using Feature<PointInT, PointOutT>::getClassName;
#         # using Feature<PointInT, PointOutT>::indices_;
#         # using Feature<PointInT, PointOutT>::k_;
#         # using Feature<PointInT, PointOutT>::search_radius_;
#         # using Feature<PointInT, PointOutT>::input_;
#         # using Feature<PointInT, PointOutT>::surface_;
# 
#         # ctypedef typename pcl::PointCloud<PointInT> PointCloudIn;
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#         # void compute (PointCloudOut &output)
# 
#         # protected:
#         # void computeFeature (PointCloudOut &output);
# 
#         # int lci (const int x1, const int y1, const int z1, 
#         #    const int x2, const int y2, const int z2, 
#         #    float &ratio, int &incnt, int &pointcount);
# 
#         # void computeESF (PointCloudIn &pc, std::vector<float> &hist);
#         # void voxelize9 (PointCloudIn &cluster);
#         # void cleanup9 (PointCloudIn &cluster);
#         # void scale_points_unit_sphere (const pcl::PointCloud<PointInT> &pc, float scalefactor, Eigen::Vector4f& centroid);
# 
# ###
# 
# # 
# # cdef extern from "pcl/features/feature.h" namespace "pcl":
# #     cdef inline void solvePlaneParameters (const Eigen::Matrix3f &covariance_matrix,
# #                                             const Eigen::Vector4f &point,
# #                                             Eigen::Vector4f &plane_parameters, float &curvature);
# #     cdef inline void solvePlaneParameters (const Eigen::Matrix3f &covariance_matrix,
# #                         float &nx, float &ny, float &nz, float &curvature);
# 
# # class Feature : public PCLBase<PointInT>
# cdef extern from "pcl/features/feature.h" namespace "pcl":
#     cdef cppclass Feature[T]:
#         Feature ()
#         # public:
#         # using PCLBase<PointInT>::indices_;
#         # using PCLBase<PointInT>::input_;
# 
#         # ctypedef PCLBase<PointInT> BaseClass;
#         # ctypedef boost::shared_ptr< Feature<PointInT, PointOutT> > Ptr;
#         # ctypedef boost::shared_ptr< const Feature<PointInT, PointOutT> > ConstPtr;
#         # ctypedef typename pcl::search::Search<PointInT> KdTree;
#         # ctypedef typename pcl::search::Search<PointInT>::Ptr KdTreePtr;
#         # ctypedef pcl::PointCloud<PointInT> PointCloudIn;
#         # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
#         # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
#         # ctypedef pcl::PointCloud<PointOutT> PointCloudOut;
#         # ctypedef boost::function<int (size_t, double, std::vector<int> &, std::vector<float> &)> SearchMethod;
#         # ctypedef boost::function<int (const PointCloudIn &cloud, size_t index, double, std::vector<int> &, std::vector<float> &)> SearchMethodSurface;
# 
#         # public:
#         # inline void setSearchSurface (const PointCloudInConstPtr &)
#         # inline PointCloudInConstPtr getSearchSurface () const
#         # inline void setSearchMethod (const KdTreePtr &tree)
#         # inline KdTreePtr getSearchMethod () const
#         inline double getSearchParameter () const
#         inline void setKSearch (int)
#         inline int getKSearch () const
#         inline void setRadiusSearch (double radius)
#         inline double getRadiusSearch () const
#         # void compute (PointCloudOut &output);
#         # void computeEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#         # protected:
#         # /** \brief The feature name. */
#         # std::string feature_name_;
#         # /** \brief The search method template for points. */
#         # SearchMethodSurface search_method_surface_;
#         # PointCloudInConstPtr surface_;
#         # /** \brief A pointer to the spatial search object. */
#         # KdTreePtr tree_;
#         # /** \brief The actual search parameter (from either \a search_radius_ or \a k_). */
#         # double search_parameter_;
#         # /** \brief The nearest neighbors search radius for each point. */
#         # double search_radius_;
#         # /** \brief The number of K nearest neighbors to use for each point. */
#         # int k_;
# 
#         # /** \brief Get a string representation of the name of this class. */
#         # inline const std::string& getClassName () const { return (feature_name_); }
#         # virtual bool initCompute ();
# 
#         # /** \brief This method should get called after ending the actual computation. */
#         # virtual bool deinitCompute ();
# 
#         # /** \brief If no surface is given, we use the input PointCloud as the surface. */
#         # bool fake_surface_;
# 
#         # inline int
#         # searchForNeighbors (size_t index, double parameter,
#         #                   std::vector<int> &indices, std::vector<float> &distances) const
# 
#         # inline int
#         # searchForNeighbors (const PointCloudIn &cloud, size_t index, double parameter,
#         #                     std::vector<int> &indices, std::vector<float> &distances) const
# 
#         # public:
#         # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
# 
# 
# # template <typename PointInT, typename PointNT, typename PointOutT>
# # class FeatureFromNormals : public Feature<PointInT, PointOutT>
# cdef extern from "pcl/features/feature.h" namespace "pcl":
#     cdef cppclass FeatureFromNormals[T]:
#         FeatureFromNormals()
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
#         # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
#         # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#         # public:
#         # ctypedef typename pcl::PointCloud<PointNT> PointCloudN;
#         # ctypedef typename PointCloudN::Ptr PointCloudNPtr;
#         # ctypedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
#         # ctypedef boost::shared_ptr< FeatureFromNormals<PointInT, PointNT, PointOutT> > Ptr;
#         # ctypedef boost::shared_ptr< const FeatureFromNormals<PointInT, PointNT, PointOutT> > ConstPtr;
# 
#         # // Members derived from the base class
#         # using Feature<PointInT, PointOutT>::input_;
#         # using Feature<PointInT, PointOutT>::surface_;
#         # using Feature<PointInT, PointOutT>::getClassName;
# 
#         # /** \brief Empty constructor. */
#         # FeatureFromNormals () : normals_ () {}
# 
#         # /** \brief Provide a pointer to the input dataset that contains the point normals of
#         #         * the XYZ dataset.
#         # * In case of search surface is set to be different from the input cloud,
#         # * normals should correspond to the search surface, not the input cloud!
#         # * \param[in] normals the const boost shared pointer to a PointCloud of normals.
#         # * By convention, L2 norm of each normal should be 1.
#         # */
#         # inline void setInputNormals (const PointCloudNConstPtr &normals)
# 
#         # /** \brief Get a pointer to the normals of the input XYZ point cloud dataset. */
#         # inline PointCloudNConstPtr getInputNormals ()
# 
#         # protected:
#         # /** \brief A pointer to the input dataset that contains the point normals of the XYZ
#         # * dataset.
#         # */
#         # PointCloudNConstPtr normals_;
# 
#         # /** \brief This method should get called before starting the actual computation. */
#         # virtual bool
#         # initCompute ();
# 
#         # public:
#         # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
# 
# # template <typename PointInT, typename PointLT, typename PointOutT>
# # class FeatureFromLabels : public Feature<PointInT, PointOutT>
# cdef extern from "pcl/features/feature.h" namespace "pcl":
#     cdef cppclass FeatureFromLabels[T]:
#         FeatureFromLabels()
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
#         # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
#         # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
#         # ctypedef typename pcl::PointCloud<PointLT> PointCloudL;
#         # ctypedef typename PointCloudL::Ptr PointCloudNPtr;
#         # ctypedef typename PointCloudL::ConstPtr PointCloudLConstPtr;
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#         # public:
#         # ctypedef boost::shared_ptr< FeatureFromLabels<PointInT, PointLT, PointOutT> > Ptr;
#         # ctypedef boost::shared_ptr< const FeatureFromLabels<PointInT, PointLT, PointOutT> > ConstPtr;
# 
#         # // Members derived from the base class
#         # using Feature<PointInT, PointOutT>::input_;
#         # using Feature<PointInT, PointOutT>::surface_;
#         # using Feature<PointInT, PointOutT>::getClassName;
#         # using Feature<PointInT, PointOutT>::k_;
# 
#         # /** \brief Provide a pointer to the input dataset that contains the point labels of
#         #   * the XYZ dataset.
#         #   * In case of search surface is set to be different from the input cloud,
#         #   * labels should correspond to the search surface, not the input cloud!
#         #   * \param[in] labels the const boost shared pointer to a PointCloud of labels.
#         #   */
#         # inline void setInputLabels (const PointCloudLConstPtr &labels)
# 
#         # inline PointCloudLConstPtr getInputLabels () const
# 
#         # protected:
#         # /** \brief A pointer to the input dataset that contains the point labels of the XYZ
#         # * dataset.
#         # */
#         # PointCloudLConstPtr labels_;
# 
#         # /** \brief This method should get called before starting the actual computation. */
#         # virtual bool
#         # initCompute ();
# 
#         # public:
#         # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
# 
# # template <typename PointInT, typename PointRFT>
# # class FeatureWithLocalReferenceFrames
# cdef extern from "pcl/features/feature.h" namespace "pcl":
#     cdef cppclass FeatureWithLocalReferenceFrames[T, REF]:
#         FeatureWithLocalReferenceFrames ()
#         # public:
#         # ctypedef pcl::PointCloud<PointRFT> PointCloudLRF;
#         # ctypedef typename PointCloudLRF::Ptr PointCloudLRFPtr;
#         # ctypedef typename PointCloudLRF::ConstPtr PointCloudLRFConstPtr;
#         # inline void setInputReferenceFrames (const PointCloudLRFConstPtr &frames)
#         # cinline PointCloudLRFConstPtr getInputReferenceFrames () const
# 
#         # protected:
#         # /** \brief A boost shared pointer to the local reference frames. */
#         # PointCloudLRFConstPtr frames_;
#         # /** \brief The user has never set the frames. */
#         # bool frames_never_defined_;
# 
#         # /** \brief Check if frames_ has been correctly initialized and compute it if needed.
#         # * \param input the subclass' input cloud dataset.
#         # * \param lrf_estimation a pointer to a local reference frame estimation class to be used as default.
#         # * \return true if frames_ has been correctly initialized.
#         # */
#         # typedef typename Feature<PointInT, PointRFT>::Ptr LRFEstimationPtr;
#         # virtual bool
#         # initLocalReferenceFrames (const size_t& indices_size,
#                                     # const LRFEstimationPtr& lrf_estimation = LRFEstimationPtr());
# 
# ###
# 
# cdef extern from "pcl/features/feature.h" namespace "pcl":
#     cdef cppclass FPFHEstimation[T]:
#         FPFHEstimation()
#             # public:
#             # using Feature<PointInT, PointOutT>::feature_name_;
#             # using Feature<PointInT, PointOutT>::getClassName;
#             # using Feature<PointInT, PointOutT>::indices_;
#             # using Feature<PointInT, PointOutT>::k_;
#             # using Feature<PointInT, PointOutT>::search_parameter_;
#             # using Feature<PointInT, PointOutT>::input_;
#             # using Feature<PointInT, PointOutT>::surface_;
#             # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#             # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#         # * represented by Cartesian coordinates and normals.
#         # * \note For explanations about the features, please see the literature mentioned above (the order of the
#         # * features might be different).
#         # * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
#         # * \param[in] normals the dataset containing the surface normals (assuming normalized vectors) at each point in cloud
#         # * \param[in] p_idx the index of the first point (source)
#         # * \param[in] q_idx the index of the second point (target)
#         # * \param[out] f1 the first angular feature (angle between the projection of nq_idx and u)
#         # * \param[out] f2 the second angular feature (angle between nq_idx and v)
#         # * \param[out] f3 the third angular feature (angle between np_idx and |p_idx - q_idx|)
#         # * \param[out] f4 the distance feature (p_idx - q_idx)
#         # bool 
#         # computePairFeatures (const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals, 
#         #                    int p_idx, int q_idx, float &f1, float &f2, float &f3, float &f4);
# 
#         # * \brief Estimate the SPFH (Simple Point Feature Histograms) individual signatures of the three angular
#         # * (f1, f2, f3) features for a given point based on its spatial neighborhood of 3D points with normals
#         # * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
#         # * \param[in] normals the dataset containing the surface normals at each point in \a cloud
#         # * \param[in] p_idx the index of the query point (source)
#         # * \param[in] row the index row in feature histogramms
#         # * \param[in] indices the k-neighborhood point indices in the dataset
#         # * \param[out] hist_f1 the resultant SPFH histogram for feature f1
#         # * \param[out] hist_f2 the resultant SPFH histogram for feature f2
#         # * \param[out] hist_f3 the resultant SPFH histogram for feature f3
#         # void 
#         # computePointSPFHSignature (const pcl::PointCloud<PointInT> &cloud, 
#         #                          const pcl::PointCloud<PointNT> &normals, int p_idx, int row, 
#         #                          const std::vector<int> &indices, 
#         #                          Eigen::MatrixXf &hist_f1, Eigen::MatrixXf &hist_f2, Eigen::MatrixXf &hist_f3);
# 
#         # * \brief Weight the SPFH (Simple Point Feature Histograms) individual histograms to create the final FPFH
#         # * (Fast Point Feature Histogram) for a given point based on its 3D spatial neighborhood
#         # * \param[in] hist_f1 the histogram feature vector of \a f1 values over the given patch
#         # * \param[in] hist_f2 the histogram feature vector of \a f2 values over the given patch
#         # * \param[in] hist_f3 the histogram feature vector of \a f3 values over the given patch
#         # * \param[in] indices the point indices of p_idx's k-neighborhood in the point cloud
#         # * \param[in] dists the distances from p_idx to all its k-neighbors
#         # * \param[out] fpfh_histogram the resultant FPFH histogram representing the feature at the query point
#         # void 
#         # weightPointSPFHSignature (const Eigen::MatrixXf &hist_f1, 
#         #                         const Eigen::MatrixXf &hist_f2, 
#         #                         const Eigen::MatrixXf &hist_f3, 
#         #                         const std::vector<int> &indices, 
#         #                         const std::vector<float> &dists, 
#         #                         Eigen::VectorXf &fpfh_histogram);
# 
#         # * \brief Set the number of subdivisions for each angular feature interval.
#         # * \param[in] nr_bins_f1 number of subdivisions for the first angular feature
#         # * \param[in] nr_bins_f2 number of subdivisions for the second angular feature
#         # * \param[in] nr_bins_f3 number of subdivisions for the third angular feature
#         inline void setNrSubdivisions (int , int , int )
# 
#         # * \brief Get the number of subdivisions for each angular feature interval. 
#         # * \param[out] nr_bins_f1 number of subdivisions for the first angular feature
#         # * \param[out] nr_bins_f2 number of subdivisions for the second angular feature
#         # * \param[out] nr_bins_f3 number of subdivisions for the third angular feature
#         inline void getNrSubdivisions (int &, int &, int &)
# 
#         # protected:
#         # * \brief Estimate the set of all SPFH (Simple Point Feature Histograms) signatures for the input cloud
#         # * \param[out] spfh_hist_lookup a lookup table for all the SPF feature indices
#         # * \param[out] hist_f1 the resultant SPFH histogram for feature f1
#         # * \param[out] hist_f2 the resultant SPFH histogram for feature f2
#         # * \param[out] hist_f3 the resultant SPFH histogram for feature f3
#         # void computeSPFHSignatures (vector[int] &, Eigen::MatrixXf &hist_f1, Eigen::MatrixXf &hist_f2, Eigen::MatrixXf &hist_f3);
# 
#         # * \brief Estimate the Fast Point Feature Histograms (FPFH) descriptors at a set of points given by
#         # * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         # * setSearchMethod ()
#         # * \param[out] output the resultant point cloud model dataset that contains the FPFH feature estimates
#         # void 
#         # computeFeature (PointCloudOut &output);
# 
#         # * \brief The number of subdivisions for each angular feature interval. */
#         # int nr_bins_f1_, nr_bins_f2_, nr_bins_f3_;
#         # * \brief Placeholder for the f1 histogram. */
#         # Eigen::MatrixXf hist_f1_;
# 
#         # * \brief Placeholder for the f2 histogram. */
#         # Eigen::MatrixXf hist_f2_;
# 
#         # /** \brief Placeholder for the f3 histogram. */
#         # Eigen::MatrixXf hist_f3_;
# 
#         # /** \brief Placeholder for a point's FPFH signature. */
#         # Eigen::VectorXf fpfh_histogram_;
# 
#         # /** \brief Float constant = 1.0 / (2.0 * M_PI) */
#         # float d_pi_; 
# 
# #   template <typename PointInT, typename PointNT>
# #   class FPFHEstimation<PointInT, PointNT, Eigen::MatrixXf> : public FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>
# # cdef extern from "pcl/features/feature.h" namespace "pcl":
# #     cdef cppclass FPFHEstimation[T, NT]:
# #         FPFHEstimation()
# #         # public:
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::k_;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::nr_bins_f1_;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::nr_bins_f2_;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::nr_bins_f3_;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::hist_f1_;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::hist_f2_;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::hist_f3_;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::indices_;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::search_parameter_;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::input_;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::compute;
# #         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::fpfh_histogram_;
# 
# ###
#     # fpfh_omp
# #   template <typename PointInT, typename PointNT, typename PointOutT>
# #   class FPFHEstimationOMP : public FPFHEstimation<PointInT, PointNT, PointOutT>
# cdef extern from "pcl/features/fpfh_omp.h" namespace "pcl":
#     cdef cppclass FPFHEstimationOMP[I, NT, O]:
#         FPFHEstimationOMP ()
#         # FPFHEstimationOMP (unsigned int )
#         # public:
#         # using Feature<PointInT, PointOutT>::feature_name_;
#         # using Feature<PointInT, PointOutT>::getClassName;
#         # using Feature<PointInT, PointOutT>::indices_;
#         # using Feature<PointInT, PointOutT>::k_;
#         # using Feature<PointInT, PointOutT>::search_parameter_;
#         # using Feature<PointInT, PointOutT>::input_;
#         # using Feature<PointInT, PointOutT>::surface_;
#         # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
#         # using FPFHEstimation<PointInT, PointNT, PointOutT>::hist_f1_;
#         # using FPFHEstimation<PointInT, PointNT, PointOutT>::hist_f2_;
#         # using FPFHEstimation<PointInT, PointNT, PointOutT>::hist_f3_;
#         # using FPFHEstimation<PointInT, PointNT, PointOutT>::weightPointSPFHSignature;
# 
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#         # * \brief Initialize the scheduler and set the number of threads to use.
#         # * \param[in] nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         inline void setNumberOfThreads (unsigned) 
# 
#         # public:
#         # * \brief The number of subdivisions for each angular feature interval. */
#         # int nr_bins_f1_, nr_bins_f2_, nr_bins_f3_;
# 
# ###
# # integral_image_normal.h
# # template <typename PointInT, typename PointOutT>
# # class IntegralImageNormalEstimation: public Feature<PointInT, PointOutT>
# cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl":
# 	cdef cppclass IntegralImageNormalEstimation[I, O]:
#     	IntegralImageNormalEstimation ()
# 		# using Feature<PointInT, PointOutT>::input_;
#     	# using Feature<PointInT, PointOutT>::feature_name_;
#     	# using Feature<PointInT, PointOutT>::tree_;
#     	# using Feature<PointInT, PointOutT>::k_;
# 
#     	# public:
# 		# * \brief Different types of border handling. */
#         # enum BorderPolicy
#         # {
#         #   BORDER_POLICY_IGNORE,
#         #   BORDER_POLICY_MIRROR
#         # ;
# 
# 		# * \brief Different normal estimation methods.
#         # * <ul>
#         # *   <li><b>COVARIANCE_MATRIX</b> - creates 9 integral images to compute the normal for a specific point
#         # *   from the covariance matrix of its local neighborhood.</li>
#         # *   <li><b>AVERAGE_3D_GRADIENT</b> - creates 6 integral images to compute smoothed versions of
#         # *   horizontal and vertical 3D gradients and computes the normals using the cross-product between these
#         # *   two gradients.
#         # *   <li><b>AVERAGE_DEPTH_CHANGE</b> -  creates only a single integral image and computes the normals
#         # *   from the average depth changes.
#         # * </ul>
#       	# enum NormalEstimationMethod
#       	# {
#       	#   COVARIANCE_MATRIX,
#       	#   AVERAGE_3D_GRADIENT,
#       	#   AVERAGE_DEPTH_CHANGE,
#       	#   SIMPLE_3D_GRADIENT
#       	# };
# 
#       	# ctypedef typename Feature<PointInT, PointOutT>::PointCloudIn  PointCloudIn;
#       	# ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#       	# * \brief Set the regions size which is considered for normal estimation.
#       	#   * \param[in] width the width of the search rectangle
#       	#   * \param[in] height the height of the search rectangle
#       	void setRectSize (const int width, const int height);
# 
#       	# * \brief Sets the policy for handling borders.
#       	#   * \param[in] border_policy the border policy.
#       	#   */
#       void setBorderPolicy (const BorderPolicy border_policy)
# 
#       	# * \brief Computes the normal at the specified position.
#       	#   * \param[in] pos_x x position (pixel)
#       	#   * \param[in] pos_y y position (pixel)
#       	#   * \param[in] point_index the position index of the point
#       	#   * \param[out] normal the output estimated normal
#       	#   */
#       	void computePointNormal (const int pos_x, const int pos_y, const unsigned point_index, PointOutT &normal);
# 
#       	# * \brief Computes the normal at the specified position with mirroring for border handling.
#       	#   * \param[in] pos_x x position (pixel)
#       	#   * \param[in] pos_y y position (pixel)
#       	#   * \param[in] point_index the position index of the point
#       	#   * \param[out] normal the output estimated normal
#       void computePointNormalMirror (const int pos_x, const int pos_y, const unsigned point_index, PointOutT &normal);
# 
#       	# * \brief The depth change threshold for computing object borders
#       	#   * \param[in] max_depth_change_factor the depth change threshold for computing object borders based on
#       	#   * depth changes
#       	#   */
#       void setMaxDepthChangeFactor (float max_depth_change_factor)
# 
#       	# * \brief Set the normal smoothing size
#       	#   * \param[in] normal_smoothing_size factor which influences the size of the area used to smooth normals
#       	#   * (depth dependent if useDepthDependentSmoothing is true)
#       	#   */
#       void setNormalSmoothingSize (float normal_smoothing_size)
# 
#       	# * \brief Set the normal estimation method. The current implemented algorithms are:
#       	#   * <ul>
#       	#   *   <li><b>COVARIANCE_MATRIX</b> - creates 9 integral images to compute the normal for a specific point
#       	#   *   from the covariance matrix of its local neighborhood.</li>
#       	#   *   <li><b>AVERAGE_3D_GRADIENT</b> - creates 6 integral images to compute smoothed versions of
#       	#   *   horizontal and vertical 3D gradients and computes the normals using the cross-product between these
#       	#   *   two gradients.
#       	#   *   <li><b>AVERAGE_DEPTH_CHANGE</b> -  creates only a single integral image and computes the normals
#       	#   *   from the average depth changes.
#       	#   * </ul>
#       	#   * \param[in] normal_estimation_method the method used for normal estimation
#       	#   */
#       	void setNormalEstimationMethod (NormalEstimationMethod normal_estimation_method)
# 
#       	# /** \brief Set whether to use depth depending smoothing or not
#       	#   * \param[in] use_depth_dependent_smoothing decides whether the smoothing is depth dependent
#       	#   */
#       	void setDepthDependentSmoothing (bool use_depth_dependent_smoothing)
# 
#       	# * \brief Provide a pointer to the input dataset (overwrites the PCLBase::setInputCloud method)
#       	#    * \param[in] cloud the const boost shared pointer to a PointCloud message
#       	#    */
#       	virtual inline void setInputCloud (const typename PointCloudIn::ConstPtr &cloud)
# 
#       	# * \brief Returns a pointer to the distance map which was computed internally
#       	#   */
#       	inline float* getDistanceMap ()
# 
#       	# * \brief Set the viewpoint.
#       	#   * \param vpx the X coordinate of the viewpoint
#       	#   * \param vpy the Y coordinate of the viewpoint
#       	#   * \param vpz the Z coordinate of the viewpoint
#       	#   */
#       	inline void setViewPoint (float vpx, float vpy, float vpz)
# 
#       	# * \brief Get the viewpoint.
#       	#   * \param [out] vpx x-coordinate of the view point
#       	#   * \param [out] vpy y-coordinate of the view point
#       	#   * \param [out] vpz z-coordinate of the view point
#       	#   * \note this method returns the currently used viewpoint for normal flipping.
#       	#   * If the viewpoint is set manually using the setViewPoint method, this method will return the set view point coordinates.
#       	#   * If an input cloud is set, it will return the sensor origin otherwise it will return the origin (0, 0, 0)
#       	inline void getViewPoint (float &vpx, float &vpy, float &vpz)
# 
#       	# * \brief sets whether the sensor origin or a user given viewpoint should be used. After this method, the 
#       	#   * normal estimation method uses the sensor origin of the input cloud.
#       	#   * to use a user defined view point, use the method setViewPoint
#       	inline void useSensorOriginAsViewPoint ()
#       
#     	# protected:
# 
#       	# * \brief Computes the normal for the complete cloud.
#       	#   * \param[out] output the resultant normals
#       	#   */
#       	# void computeFeature (PointCloudOut &output);
# 
#       	# * \brief Initialize the data structures, based on the normal estimation method chosen. */
#       	# void initData ();
# 
#     	# private:
#       	# * \brief The normal estimation method to use. Currently, 3 implementations are provided:
#       	#   *
#       	#   * - COVARIANCE_MATRIX
#       	#   * - AVERAGE_3D_GRADIENT
#       	#   * - AVERAGE_DEPTH_CHANGE
#       	#   */
#       	# NormalEstimationMethod normal_estimation_method_;
# 
#       	# * \brief The policy for handling borders. */
#       	# BorderPolicy border_policy_;
# 
#       	# * The width of the neighborhood region used for computing the normal. */
#       	# int rect_width_;
#       	# int rect_width_2_;
#       	# int rect_width_4_;
#       	# /** The height of the neighborhood region used for computing the normal. */
#       	# int rect_height_;
#       	# int rect_height_2_;
#       	# int rect_height_4_;
# 
#       	# /** the threshold used to detect depth discontinuities */
#       	# float distance_threshold_;
# 
#       	# /** integral image in x-direction */
#       	# IntegralImage2D<float, 3> integral_image_DX_;
#       	# /** integral image in y-direction */
#       	# IntegralImage2D<float, 3> integral_image_DY_;
#       	# /** integral image */
#       	# IntegralImage2D<float, 1> integral_image_depth_;
#       	# /** integral image xyz */
#       	# IntegralImage2D<float, 3> integral_image_XYZ_;
# 
#       	# /** derivatives in x-direction */
#       	# float *diff_x_;
#       	# /** derivatives in y-direction */
#       	# float *diff_y_;
# 
#       	# /** depth data */
#       	# float *depth_data_;
# 
#       	# /** distance map */
#       	# float *distance_map_;
# 
#       	# /** \brief Smooth data based on depth (true/false). */
#       	# bool use_depth_dependent_smoothing_;
# 
#       	# /** \brief Threshold for detecting depth discontinuities */
#       	# float max_depth_change_factor_;
# 
#       	# /** \brief */
#       	# float normal_smoothing_size_;
# 
#       	# /** \brief True when a dataset has been received and the covariance_matrix data has been initialized. */
#       	# bool init_covariance_matrix_;
# 
#       	# /** \brief True when a dataset has been received and the average 3d gradient data has been initialized. */
#       	# bool init_average_3d_gradient_;
# 
#       	# /** \brief True when a dataset has been received and the simple 3d gradient data has been initialized. */
#       	# bool init_simple_3d_gradient_;
# 
#       	# /** \brief True when a dataset has been received and the depth change data has been initialized. */
#       	# bool init_depth_change_;
# 
#       	# /** \brief Values describing the viewpoint ("pinhole" camera model assumed). For per point viewpoints, inherit
#       	#   * from NormalEstimation and provide your own computeFeature (). By default, the viewpoint is set to 0,0,0. */
#       	# float vpx_, vpy_, vpz_;
# 
#       	# /** whether the sensor origin of the input cloud or a user given viewpoint should be used.*/
#       	# bool use_sensor_origin_;
#       
#       	# /** \brief This method should get called before starting the actual computation. */
#       	# bool initCompute ();
# 
#       	# /** \brief Internal initialization method for COVARIANCE_MATRIX estimation. */
#       	# void initCovarianceMatrixMethod ();
# 
#       	# /** \brief Internal initialization method for AVERAGE_3D_GRADIENT estimation. */
#       	# void initAverage3DGradientMethod ();
# 
#       	# /** \brief Internal initialization method for AVERAGE_DEPTH_CHANGE estimation. */
#       	# void initAverageDepthChangeMethod ();
# 
#       	# /** \brief Internal initialization method for SIMPLE_3D_GRADIENT estimation. */
#       	# void initSimple3DGradientMethod ();
# 
# ###
# 
#     # integral_image2D.h
# # template <class DataType, unsigned Dimension>
# # class IntegralImage2D
# cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl":
# 	cdef cppclass IntegralImage2D[Type, Dim]:
#     	# IntegralImage2D ()
#     	IntegralImage2D (bool )
#     	# public:
#       	# static const unsigned second_order_size = (Dimension * (Dimension + 1)) >> 1;
#       	# ctypedef Eigen::Matrix<typename IntegralImageTypeTraits<DataType>::IntegralType, Dimension, 1> ElementType;
#       	# ctypedef Eigen::Matrix<typename IntegralImageTypeTraits<DataType>::IntegralType, second_order_size, 1> SecondOrderType;
# 
#       	# void setSecondOrderComputation (bool compute_second_order_integral_images);
# 
#       	# * \brief Set the input data to compute the integral image for
#       	#   * \param[in] data the input data
#       	#   * \param[in] width the width of the data
#       	#   * \param[in] height the height of the data
#       	#   * \param[in] element_stride the element stride of the data
#       	#   * \param[in] row_stride the row stride of the data
#       	#   */
#       	# void
#       	# setInput (const DataType * data,
#       	#           unsigned width, unsigned height, unsigned element_stride, unsigned row_stride);
# 
#       	# * \brief Compute the first order sum within a given rectangle
#       	#   * \param[in] start_x x position of rectangle
#       	#   * \param[in] start_y y position of rectangle
#       	#   * \param[in] width width of rectangle
#       	#   * \param[in] height height of rectangle
#       	#   */
#       	# inline ElementType
#       	# getFirstOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;
# 
#       	# /** \brief Compute the first order sum within a given rectangle
#       	#   * \param[in] start_x x position of the start of the rectangle
#       	#   * \param[in] start_y x position of the start of the rectangle
#       	#   * \param[in] end_x x position of the end of the rectangle
#       	#   * \param[in] end_y x position of the end of the rectangle
#       	#   */
#       	# inline ElementType
#       	# getFirstOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;
# 
#       	# /** \brief Compute the second order sum within a given rectangle
#       	#   * \param[in] start_x x position of rectangle
#       	#   * \param[in] start_y y position of rectangle
#       	#   * \param[in] width width of rectangle
#       	#   * \param[in] height height of rectangle
#       	#   */
#       	# inline SecondOrderType
#       	# getSecondOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;
# 
#       	# /** \brief Compute the second order sum within a given rectangle
#       	#   * \param[in] start_x x position of the start of the rectangle
#       	#   * \param[in] start_y x position of the start of the rectangle
#       	#   * \param[in] end_x x position of the end of the rectangle
#       	#   * \param[in] end_y x position of the end of the rectangle
#       	#   */
#       	# inline SecondOrderType
#       	# getSecondOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;
# 
#       	# /** \brief Compute the number of finite elements within a given rectangle
#       	#   * \param[in] start_x x position of rectangle
#       	#   * \param[in] start_y y position of rectangle
#       	#   * \param[in] width width of rectangle
#       	#   * \param[in] height height of rectangle
#       	#   */
#       	# inline unsigned
#       	# getFiniteElementsCount (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;
# 
#       	# /** \brief Compute the number of finite elements within a given rectangle
#       	#   * \param[in] start_x x position of the start of the rectangle
#       	#   * \param[in] start_y x position of the start of the rectangle
#       	#   * \param[in] end_x x position of the end of the rectangle
#       	#   * \param[in] end_y x position of the end of the rectangle
#       	#   */
#       	# inline unsigned
#       	# getFiniteElementsCountSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;
# 
# # template <class DataType>
# # class IntegralImage2D <DataType, 1>
# cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl":
# 	cdef cppclass IntegralImage2D[Type]:
#     	# IntegralImage2D ()
#     	IntegralImage2D (bool )
# 
#     	# public:
#     	#   static const unsigned second_order_size = 1;
#     	#   typedef typename IntegralImageTypeTraits<DataType>::IntegralType ElementType;
#     	#   typedef typename IntegralImageTypeTraits<DataType>::IntegralType SecondOrderType;
# 
#       	# /** \brief Set the input data to compute the integral image for
#       	#   * \param[in] data the input data
#       	#   * \param[in] width the width of the data
#       	#   * \param[in] height the height of the data
#       	#   * \param[in] element_stride the element stride of the data
#       	#   * \param[in] row_stride the row stride of the data
#       	#   */
#       	# void
#       	# setInput (const DataType * data,
#       	#           unsigned width, unsigned height, unsigned element_stride, unsigned row_stride);
# 	
#       	# /** \brief Compute the first order sum within a given rectangle
#       	#   * \param[in] start_x x position of rectangle
#       	#   * \param[in] start_y y position of rectangle
#       	#   * \param[in] width width of rectangle
#       	#   * \param[in] height height of rectangle
#       	#   */
#       	# inline ElementType
#       	# getFirstOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;
# 
#       	# /** \brief Compute the first order sum within a given rectangle
#       	#   * \param[in] start_x x position of the start of the rectangle
#       	#   * \param[in] start_y x position of the start of the rectangle
#       	#   * \param[in] end_x x position of the end of the rectangle
#       	#   * \param[in] end_y x position of the end of the rectangle
#       	#   */
#       	# inline ElementType
#       	# getFirstOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;
# 
#       	# /** \brief Compute the second order sum within a given rectangle
#       	#   * \param[in] start_x x position of rectangle
#       	#   * \param[in] start_y y position of rectangle
#       	#   * \param[in] width width of rectangle
#       	#   * \param[in] height height of rectangle
#       	#   */
#       	# inline SecondOrderType
#       	# getSecondOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;
# 
#       	# /** \brief Compute the second order sum within a given rectangle
#       	#   * \param[in] start_x x position of the start of the rectangle
#       	#   * \param[in] start_y x position of the start of the rectangle
#       	#   * \param[in] end_x x position of the end of the rectangle
#       	#   * \param[in] end_y x position of the end of the rectangle
#       	#   */
#       	# inline SecondOrderType
#       	# getSecondOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;
# 
#       	# /** \brief Compute the number of finite elements within a given rectangle
#       	#   * \param[in] start_x x position of rectangle
#       	#   * \param[in] start_y y position of rectangle
#       	#   * \param[in] width width of rectangle
#       	#   * \param[in] height height of rectangle
#       	#   */
#       	# inline unsigned
#       	# getFiniteElementsCount (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;
# 
#       	# /** \brief Compute the number of finite elements within a given rectangle
#       	#   * \param[in] start_x x position of the start of the rectangle
#       	#   * \param[in] start_y x position of the start of the rectangle
#       	#   * \param[in] end_x x position of the end of the rectangle
#       	#   * \param[in] end_y x position of the end of the rectangle
#       	#   */
#       	# inline unsigned
#       	# getFiniteElementsCountSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;
# 
# ###
#     # intensity_gradient.h
# # template <typename PointInT, typename PointNT, typename PointOutT, typename IntensitySelectorT = pcl::common::IntensityFieldAccessor<PointInT> >
# # class IntensityGradientEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
# cdef extern from "pcl/features/intensity_gradient.h" namespace "pcl":
# 	cdef cppclass IntensityGradientEstimation[In, NT, Out, Intensity]:
#     	IntensityGradientEstimation ()
#     	# public:
#       	# using Feature<PointInT, PointOutT>::feature_name_;
#       	# using Feature<PointInT, PointOutT>::getClassName;
#       	# using Feature<PointInT, PointOutT>::indices_;
#       	# using Feature<PointInT, PointOutT>::surface_;
#       	# using Feature<PointInT, PointOutT>::k_;
#       	# using Feature<PointInT, PointOutT>::search_parameter_;
#       	# using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#       	# typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#       	# /** \brief Initialize the scheduler and set the number of threads to use.
#       	#   * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#       	#   */
#       	# inline void
#       	# setNumberOfThreads (int nr_threads)
# 
#     	# protected:
#     	#   /** \brief Estimate the intensity gradients for a set of points given in <setInputCloud (), setIndices ()> using
#     	#     *  the surface in setSearchSurface () and the spatial locator in setSearchMethod ().
#     	#     *  \param output the resultant point cloud that contains the intensity gradient vectors
#     	#     */
#       	# void computeFeature (PointCloudOut &output);
# 
#       	# /** \brief Estimate the intensity gradient around a given point based on its spatial neighborhood of points
#       	#   * \param cloud a point cloud dataset containing XYZI coordinates (Cartesian coordinates + intensity)
#       	#   * \param indices the indices of the neighoring points in the dataset
#       	#   * \param point the 3D Cartesian coordinates of the point at which to estimate the gradient
#       	#   * \param normal the 3D surface normal of the given point
#       	#   * \param gradient the resultant 3D gradient vector
#       	#   */
#       	# void
#       	# computePointIntensityGradient (const pcl::PointCloud<PointInT> &cloud,
#       	#                                const std::vector<int> &indices,
#       	#                                const Eigen::Vector3f &point, 
#       	#                                float mean_intensity, 
#       	#                                const Eigen::Vector3f &normal,
#       	#                                Eigen::Vector3f &gradient);
# 
#     	# protected:
#     	#   ///intensity field accessor structure
#     	#   IntensitySelectorT intensity_;
#     	#   ///number of threads to be used, default 1
#     	#   int threads_;
# 
# # template <typename PointInT, typename PointNT>
# # class IntensityGradientEstimation<PointInT, PointNT, Eigen::MatrixXf>: public IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>
# cdef extern from "pcl/features/intensity_gradient.h" namespace "pcl":
# 	cdef cppclass IntensityGradientEstimation[In, NT]:
#     	IntensityGradientEstimation ()
#     	# public:
#     	#   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::indices_;
#     	#   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::normals_;
#     	#   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::input_;
#     	#   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::surface_;
#     	#   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::k_;
#     	#   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::search_parameter_;
#     	#   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::compute;
# 
#     	# protected:
#     	#   /** \brief Estimate the intensity gradients for a set of points given in <setInputCloud (), setIndices ()> using
#     	#     *  the surface in setSearchSurface () and the spatial locator in setSearchMethod ().
#     	#     *  \param output the resultant point cloud that contains the intensity gradient vectors
#     	#     */
#     	#   void
#     	#   computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
#     	#   /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#     	#     * \param[out] output the output point cloud
#     	#     */
#     	#   void
#     	#   compute (pcl::PointCloud<pcl::Normal> &) {}
# 
# ###
#     # intensity_spin.h
# # template <typename PointInT, typename PointOutT>
# # class IntensitySpinEstimation: public Feature<PointInT, PointOutT>
# cdef extern from "pcl/features/intensity_spin.h" namespace "pcl":
# 	cdef cppclass IntensitySpinEstimation[In, Out]:
#     	IntensitySpinEstimation ()
# 
#     	# public:
#     	#   using Feature<PointInT, PointOutT>::feature_name_;
#     	#   using Feature<PointInT, PointOutT>::getClassName;
#       	# using Feature<PointInT, PointOutT>::input_;
#       	# using Feature<PointInT, PointOutT>::indices_;
#       	# using Feature<PointInT, PointOutT>::surface_;
#       	# using Feature<PointInT, PointOutT>::tree_;
#       	# using Feature<PointInT, PointOutT>::search_radius_;
#       
#       	# ctypedef typename pcl::PointCloud<PointInT> PointCloudIn;
#       	# ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#       	# /** \brief Estimate the intensity-domain spin image descriptor for a given point based on its spatial
#       	#   * neighborhood of 3D points and their intensities. 
#       	#   * \param[in] cloud the dataset containing the Cartesian coordinates and intensity values of the points
#       	#   * \param[in] radius the radius of the feature
#       	#   * \param[in] sigma the standard deviation of the Gaussian smoothing kernel to use during the soft histogram update
#       	#   * \param[in] k the number of neighbors to use from \a indices and \a squared_distances
#       	#   * \param[in] indices the indices of the points that comprise the query point's neighborhood
#       	#   * \param[in] squared_distances the squared distances from the query point to each point in the neighborhood
#       	#   * \param[out] intensity_spin_image the resultant intensity-domain spin image descriptor
#       	#   */
#       	# void 
#       	# computeIntensitySpinImage (const PointCloudIn &cloud, 
#       	#                            float radius, float sigma, int k, 
#       	#                            const std::vector<int> &indices, 
#       	#                            const std::vector<float> &squared_distances, 
#       	#                            Eigen::MatrixXf &intensity_spin_image);
# 
#       	# /** \brief Set the number of bins to use in the distance dimension of the spin image
#       	#   * \param[in] nr_distance_bins the number of bins to use in the distance dimension of the spin image
#       	#   */
#       	# inline void 
#       	# setNrDistanceBins (size_t nr_distance_bins) { nr_distance_bins_ = static_cast<int> (nr_distance_bins); };
# 
#       	# /** \brief Returns the number of bins in the distance dimension of the spin image. */
#       	# inline int 
#       	# getNrDistanceBins () { return (nr_distance_bins_); };
# 
#       	# /** \brief Set the number of bins to use in the intensity dimension of the spin image.
#       	#   * \param[in] nr_intensity_bins the number of bins to use in the intensity dimension of the spin image
#       	#   */
#       	# inline void 
#       	# setNrIntensityBins (size_t nr_intensity_bins) { nr_intensity_bins_ = static_cast<int> (nr_intensity_bins); };
# 
#       	# /** \brief Returns the number of bins in the intensity dimension of the spin image. */
#       	# inline int 
#       	# getNrIntensityBins () { return (nr_intensity_bins_); };
# 
#       	# /** \brief Set the standard deviation of the Gaussian smoothing kernel to use when constructing the spin images.  
#       	#   * \param[in] sigma the standard deviation of the Gaussian smoothing kernel to use when constructing the spin images
#       	#   */
#       	# inline void 
#       	# setSmoothingBandwith (float sigma) { sigma_ = sigma; };
# 
#       	# /** \brief Returns the standard deviation of the Gaussian smoothing kernel used to construct the spin images.  */
#       	# inline float 
#       	# getSmoothingBandwith () { return (sigma_); };
# 
#       	# /** \brief Estimate the intensity-domain descriptors at a set of points given by <setInputCloud (), setIndices ()>
#       	#   *  using the surface in setSearchSurface (), and the spatial locator in setSearchMethod ().
#       	#   * \param[out] output the resultant point cloud model dataset that contains the intensity-domain spin image features
#       	#   */
#       	# void 
#       	# computeFeature (PointCloudOut &output);
# 
#       	# /** \brief The number of distance bins in the descriptor. */
#       	# int nr_distance_bins_;
# 
#       	# /** \brief The number of intensity bins in the descriptor. */
#       	# int nr_intensity_bins_;
# 
#       	# /** \brief The standard deviation of the Gaussian smoothing kernel used to construct the spin images. */
#       	# float sigma_;
# 
# # template <typename PointInT>
# # class IntensitySpinEstimation<PointInT, Eigen::MatrixXf>: public IntensitySpinEstimation<PointInT, pcl::Histogram<20> >
# cdef extern from "pcl/features/intensity_spin.h" namespace "pcl":
# 	cdef cppclass IntensitySpinEstimation[In]:
#     	IntensitySpinEstimation ()
#     	# public:
#     	#   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::getClassName;
#     	#   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::input_;
#     	#   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::indices_;
#     	#   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::surface_;
#     	#   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::search_radius_;
#     	#   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::nr_intensity_bins_;
#     	#   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::nr_distance_bins_;
#     	#   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::tree_;
#     	#   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::sigma_;
#     	#   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::compute;
# 
# ###
#     # moment_invariants.h
#   template <typename PointInT, typename PointOutT>
#   class MomentInvariantsEstimation: public Feature<PointInT, PointOutT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::input_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#       /** \brief Empty constructor. */
#       MomentInvariantsEstimation () : xyz_centroid_ (), temp_pt_ ()
#       {
#         feature_name_ = "MomentInvariantsEstimation";
#       };
# 
#       /** \brief Compute the 3 moment invariants (j1, j2, j3) for a given set of points, using their indices.
#         * \param[in] cloud the input point cloud
#         * \param[in] indices the point cloud indices that need to be used
#         * \param[out] j1 the resultant first moment invariant
#         * \param[out] j2 the resultant second moment invariant
#         * \param[out] j3 the resultant third moment invariant
#         */
#       void 
#       computePointMomentInvariants (const pcl::PointCloud<PointInT> &cloud, 
#                                     const std::vector<int> &indices, 
#                                     float &j1, float &j2, float &j3);
# 
#       /** \brief Compute the 3 moment invariants (j1, j2, j3) for a given set of points, using their indices.
#         * \param[in] cloud the input point cloud
#         * \param[out] j1 the resultant first moment invariant
#         * \param[out] j2 the resultant second moment invariant
#         * \param[out] j3 the resultant third moment invariant
#         */
#       void 
#       computePointMomentInvariants (const pcl::PointCloud<PointInT> &cloud, 
#                                     float &j1, float &j2, float &j3);
# 
#     protected:
# 
#       /** \brief Estimate moment invariants for all points given in <setInputCloud (), setIndices ()> using the surface
#         * in setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the moment invariants
#         */
#       void 
#       computeFeature (PointCloudOut &output);
# 
#     private:
#       /** \brief 16-bytes aligned placeholder for the XYZ centroid of a surface patch. */
#       Eigen::Vector4f xyz_centroid_;
# 
#       /** \brief Internal data vector. */
#       Eigen::Vector4f temp_pt_;
# 
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   /** \brief MomentInvariantsEstimation estimates the 3 moment invariants (j1, j2, j3) at each 3D point.
#     *
#     * \note The code is stateful as we do not expect this class to be multicore parallelized. Please look at
#     * \ref NormalEstimationOMP for an example on how to extend this to parallel implementations.
#     * \author Radu B. Rusu
#     * \ingroup features
#     */
#   template <typename PointInT>
#   class MomentInvariantsEstimation<PointInT, Eigen::MatrixXf>: public MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>
#   {
#     public:
#       using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::k_;
#       using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::indices_;
#       using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::search_parameter_;
#       using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::surface_;
#       using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::input_;
#       using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::compute;
# 
#    private:
#       /** \brief Estimate moment invariants for all points given in <setInputCloud (), setIndices ()> using the surface
#         * in setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the moment invariants
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       compute (pcl::PointCloud<pcl::Normal> &) {}
#   };
# 
# 
# ###
#     # multiscale_feature_persistence.h
#   template <typename PointSource, typename PointFeature>
#   class MultiscaleFeaturePersistence : public PCLBase<PointSource>
#   {
#     public:
#       typedef pcl::PointCloud<PointFeature> FeatureCloud;
#       typedef typename pcl::PointCloud<PointFeature>::Ptr FeatureCloudPtr;
#       typedef typename pcl::Feature<PointSource, PointFeature>::Ptr FeatureEstimatorPtr;
#       typedef boost::shared_ptr<const pcl::PointRepresentation <PointFeature> > FeatureRepresentationConstPtr;
# 
#       using pcl::PCLBase<PointSource>::input_;
# 
#       /** \brief Empty constructor */
#       MultiscaleFeaturePersistence ();
# 
#       /** \brief Method that calls computeFeatureAtScale () for each scale parameter */
#       void
#       computeFeaturesAtAllScales ();
# 
#       /** \brief Central function that computes the persistent features
#        * \param output_features a cloud containing the persistent features
#        * \param output_indices vector containing the indices of the points in the input cloud
#        * that have persistent features, under a one-to-one correspondence with the output_features cloud
#        */
#       void
#       determinePersistentFeatures (FeatureCloud &output_features,
#                                    boost::shared_ptr<std::vector<int> > &output_indices);
# 
#       /** \brief Method for setting the scale parameters for the algorithm
#        * \param scale_values vector of scales to determine the characteristic of each scaling step
#        */
#       inline void
#       setScalesVector (std::vector<float> &scale_values) { scale_values_ = scale_values; }
# 
#       /** \brief Method for getting the scale parameters vector */
#       inline std::vector<float>
#       getScalesVector () { return scale_values_; }
# 
#       /** \brief Setter method for the feature estimator
#        * \param feature_estimator pointer to the feature estimator instance that will be used
#        * \note the feature estimator instance should already have the input data given beforehand
#        * and everything set, ready to be given the compute () command
#        */
#       inline void
#       setFeatureEstimator (FeatureEstimatorPtr feature_estimator) { feature_estimator_ = feature_estimator; };
# 
#       /** \brief Getter method for the feature estimator */
#       inline FeatureEstimatorPtr
#       getFeatureEstimator () { return feature_estimator_; }
# 
#       /** \brief Provide a pointer to the feature representation to use to convert features to k-D vectors.
#        * \param feature_representation the const boost shared pointer to a PointRepresentation
#        */
#       inline void
#       setPointRepresentation (const FeatureRepresentationConstPtr& feature_representation) { feature_representation_ = feature_representation; }
# 
#       /** \brief Get a pointer to the feature representation used when converting features into k-D vectors. */
#       inline FeatureRepresentationConstPtr const
#       getPointRepresentation () { return feature_representation_; }
# 
#       /** \brief Sets the alpha parameter
#        * \param alpha value to replace the current alpha with
#        */
#       inline void
#       setAlpha (float alpha) { alpha_ = alpha; }
# 
#       /** \brief Get the value of the alpha parameter */
#       inline float
#       getAlpha () { return alpha_; }
# 
#       /** \brief Method for setting the distance metric that will be used for computing the difference between feature vectors
#        * \param distance_metric the new distance metric chosen from the NormType enum
#        */
#       inline void
#       setDistanceMetric (NormType distance_metric) { distance_metric_ = distance_metric; }
# 
#       /** \brief Returns the distance metric that is currently used to calculate the difference between feature vectors */
#       inline NormType
#       getDistanceMetric () { return distance_metric_; }
# 
# 
#     private:
#       /** \brief Checks if all the necessary input was given and the computations can successfully start */
#       bool
#       initCompute ();
# 
# 
#       /** \brief Method to compute the features for the point cloud at the given scale */
#       virtual void
#       computeFeatureAtScale (float &scale,
#                              FeatureCloudPtr &features);
# 
# 
#       /** \brief Function that calculates the scalar difference between two features
#        * \return the difference as a floating point type
#        */
#       float
#       distanceBetweenFeatures (const std::vector<float> &a,
#                                const std::vector<float> &b);
# 
#       /** \brief Method that averages all the features at all scales in order to obtain the global mean feature;
#        * this value is stored in the mean_feature field
#        */
#       void
#       calculateMeanFeature ();
# 
#       /** \brief Selects the so-called 'unique' features from the cloud of features at each level.
#        * These features are the ones that fall outside the standard deviation * alpha_
#        */
#       void
#       extractUniqueFeatures ();
# 
# 
#       /** \brief The general parameter for determining each scale level */
#       std::vector<float> scale_values_;
# 
#       /** \brief Parameter that determines if a feature is to be considered unique or not */
#       float alpha_;
# 
#       /** \brief Parameter that determines which distance metric is to be usedto calculate the difference between feature vectors */
#       NormType distance_metric_;
# 
#       /** \brief the feature estimator that will be used to determine the feature set at each scale level */
#       FeatureEstimatorPtr feature_estimator_;
# 
#       std::vector<FeatureCloudPtr> features_at_scale_;
#       std::vector<std::vector<std::vector<float> > > features_at_scale_vectorized_;
#       std::vector<float> mean_feature_;
#       FeatureRepresentationConstPtr feature_representation_;
# 
#       /** \brief Two structures in which to hold the results of the unique feature extraction process.
#        * They are superfluous with respect to each other, but improve the time performance of the algorithm
#        */
#       std::vector<std::list<size_t> > unique_features_indices_;
#       std::vector<std::vector<bool> > unique_features_table_;
#   };
# 
# ###
#     # narf.h
#     # narf_descriptor.h
# ###
# 
# cdef extern from "pcl/features/normal_3d.h" namespace "pcl":
#     cdef cppclass NormalEstimation[I, N, O]:
#         NormalEstimation()
# 
#   template <typename PointT> inline void
#   computePointNormal (const pcl::PointCloud<PointT> &cloud,
#                       Eigen::Vector4f &plane_parameters, float &curvature)
#   {
#     // Placeholder for the 3x3 covariance matrix at each surface patch
#     EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
#     // 16-bytes aligned placeholder for the XYZ centroid of a surface patch
#     Eigen::Vector4f xyz_centroid;
# 
#     if (computeMeanAndCovarianceMatrix (cloud, covariance_matrix, xyz_centroid) == 0)
#     {
#       plane_parameters.setConstant (std::numeric_limits<float>::quiet_NaN ());
#       curvature = std::numeric_limits<float>::quiet_NaN ();
#       return;
#     }
# 
#     // Get the plane normal and surface curvature
#     solvePlaneParameters (covariance_matrix, xyz_centroid, plane_parameters, curvature);
#   }
# 
#   /** \brief Compute the Least-Squares plane fit for a given set of points, using their indices,
#     * and return the estimated plane parameters together with the surface curvature.
#     * \param cloud the input point cloud
#     * \param indices the point cloud indices that need to be used
#     * \param plane_parameters the plane parameters as: a, b, c, d (ax + by + cz + d = 0)
#     * \param curvature the estimated surface curvature as a measure of
#     * \f[
#     * \lambda_0 / (\lambda_0 + \lambda_1 + \lambda_2)
#     * \f]
#     * \ingroup features
#     */
#   template <typename PointT> inline void
#   computePointNormal (const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices,
#                       Eigen::Vector4f &plane_parameters, float &curvature)
#   {
#     // Placeholder for the 3x3 covariance matrix at each surface patch
#     EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
#     // 16-bytes aligned placeholder for the XYZ centroid of a surface patch
#     Eigen::Vector4f xyz_centroid;
#     if (computeMeanAndCovarianceMatrix (cloud, indices, covariance_matrix, xyz_centroid) == 0)
#     {
#       plane_parameters.setConstant (std::numeric_limits<float>::quiet_NaN ());
#       curvature = std::numeric_limits<float>::quiet_NaN ();
#       return;
#     }
#     // Get the plane normal and surface curvature
#     solvePlaneParameters (covariance_matrix, xyz_centroid, plane_parameters, curvature);
#   }
# 
#   /** \brief Flip (in place) the estimated normal of a point towards a given viewpoint
#     * \param point a given point
#     * \param vp_x the X coordinate of the viewpoint
#     * \param vp_y the X coordinate of the viewpoint
#     * \param vp_z the X coordinate of the viewpoint
#     * \param normal the plane normal to be flipped
#     * \ingroup features
#     */
#   template <typename PointT, typename Scalar> inline void
#   flipNormalTowardsViewpoint (const PointT &point, float vp_x, float vp_y, float vp_z,
#                               Eigen::Matrix<Scalar, 4, 1>& normal)
#   {
#     Eigen::Matrix <Scalar, 4, 1> vp (vp_x - point.x, vp_y - point.x, vp_z - point.z, 0);
# 
#     // Dot product between the (viewpoint - point) and the plane normal
#     float cos_theta = vp.dot (normal);
# 
#     // Flip the plane normal
#     if (cos_theta < 0)
#     {
#       normal *= -1;
#       normal[3] = 0.0f;
#       // Hessian form (D = nc . p_plane (centroid here) + p)
#       normal[3] = -1 * normal.dot (point.getVector4fMap ());
#     }
#   }
# 
#   /** \brief Flip (in place) the estimated normal of a point towards a given viewpoint
#     * \param point a given point
#     * \param vp_x the X coordinate of the viewpoint
#     * \param vp_y the X coordinate of the viewpoint
#     * \param vp_z the X coordinate of the viewpoint
#     * \param normal the plane normal to be flipped
#     * \ingroup features
#     */
#   template <typename PointT, typename Scalar> inline void
#   flipNormalTowardsViewpoint (const PointT &point, float vp_x, float vp_y, float vp_z,
#                               Eigen::Matrix<Scalar, 3, 1>& normal)
#   {
#     Eigen::Matrix <Scalar, 3, 1> vp (vp_x - point.x, vp_y - point.x, vp_z - point.z);
# 
#     // Flip the plane normal
#     if (vp.dot (normal) < 0)
#       normal *= -1;
#   }
#   
#   /** \brief Flip (in place) the estimated normal of a point towards a given viewpoint
#     * \param point a given point
#     * \param vp_x the X coordinate of the viewpoint
#     * \param vp_y the X coordinate of the viewpoint
#     * \param vp_z the X coordinate of the viewpoint
#     * \param nx the resultant X component of the plane normal
#     * \param ny the resultant Y component of the plane normal
#     * \param nz the resultant Z component of the plane normal
#     * \ingroup features
#     */
#   template <typename PointT> inline void
#   flipNormalTowardsViewpoint (const PointT &point, float vp_x, float vp_y, float vp_z,
#                               float &nx, float &ny, float &nz)
#   {
#     // See if we need to flip any plane normals
#     vp_x -= point.x;
#     vp_y -= point.y;
#     vp_z -= point.z;
# 
#     // Dot product between the (viewpoint - point) and the plane normal
#     float cos_theta = (vp_x * nx + vp_y * ny + vp_z * nz);
# 
#     // Flip the plane normal
#     if (cos_theta < 0)
#     {
#       nx *= -1;
#       ny *= -1;
#       nz *= -1;
#     }
#   }
# 
#   /** \brief NormalEstimation estimates local surface properties (surface normals and curvatures)at each
#     * 3D point. If PointOutT is specified as pcl::Normal, the normal is stored in the first 3 components (0-2),
#     * and the curvature is stored in component 3.
#     *
#     * \note The code is stateful as we do not expect this class to be multicore parallelized. Please look at
#     * \ref NormalEstimationOMP for a parallel implementation.
#     * \author Radu B. Rusu
#     * \ingroup features
#     */
#   template <typename PointInT, typename PointOutT>
#   class NormalEstimation: public Feature<PointInT, PointOutT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::input_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_radius_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#       typedef typename Feature<PointInT, PointOutT>::PointCloudConstPtr PointCloudConstPtr;
#       
#       /** \brief Empty constructor. */
#       NormalEstimation () 
#       : vpx_ (0)
#       , vpy_ (0)
#       , vpz_ (0)
#       , covariance_matrix_ ()
#       , xyz_centroid_ ()
#       , use_sensor_origin_ (true)
#       {
#         feature_name_ = "NormalEstimation";
#       };
# 
#       /** \brief Compute the Least-Squares plane fit for a given set of points, using their indices,
#         * and return the estimated plane parameters together with the surface curvature.
#         * \param cloud the input point cloud
#         * \param indices the point cloud indices that need to be used
#         * \param plane_parameters the plane parameters as: a, b, c, d (ax + by + cz + d = 0)
#         * \param curvature the estimated surface curvature as a measure of
#         * \f[
#         * \lambda_0 / (\lambda_0 + \lambda_1 + \lambda_2)
#         * \f]
#         */
#       inline void
#       computePointNormal (const pcl::PointCloud<PointInT> &cloud, const std::vector<int> &indices, Eigen::Vector4f &plane_parameters, float &curvature)
#       {
#         if (computeMeanAndCovarianceMatrix (cloud, indices, covariance_matrix_, xyz_centroid_) == 0)
#         {
#           plane_parameters.setConstant (std::numeric_limits<float>::quiet_NaN ());
#           curvature = std::numeric_limits<float>::quiet_NaN ();
#           return;
#         }
# 
#         // Get the plane normal and surface curvature
#         solvePlaneParameters (covariance_matrix_, xyz_centroid_, plane_parameters, curvature);
#       }
# 
#       /** \brief Compute the Least-Squares plane fit for a given set of points, using their indices,
#         * and return the estimated plane parameters together with the surface curvature.
#         * \param cloud the input point cloud
#         * \param indices the point cloud indices that need to be used
#         * \param nx the resultant X component of the plane normal
#         * \param ny the resultant Y component of the plane normal
#         * \param nz the resultant Z component of the plane normal
#         * \param curvature the estimated surface curvature as a measure of
#         * \f[
#         * \lambda_0 / (\lambda_0 + \lambda_1 + \lambda_2)
#         * \f]
#         */
#       inline void
#       computePointNormal (const pcl::PointCloud<PointInT> &cloud, const std::vector<int> &indices, float &nx, float &ny, float &nz, float &curvature)
#       {
#         if (computeMeanAndCovarianceMatrix (cloud, indices, covariance_matrix_, xyz_centroid_) == 0)
#         {
#           nx = ny = nz = curvature = std::numeric_limits<float>::quiet_NaN ();
#           return;
#         }
# 
#         // Get the plane normal and surface curvature
#         solvePlaneParameters (covariance_matrix_, nx, ny, nz, curvature);
#       }
# 
#       /** \brief Provide a pointer to the input dataset
#         * \param cloud the const boost shared pointer to a PointCloud message
#         */
#       virtual inline void 
#       setInputCloud (const PointCloudConstPtr &cloud)
#       {
#         input_ = cloud;
#         if (use_sensor_origin_)
#         {
#           vpx_ = input_->sensor_origin_.coeff (0);
#           vpy_ = input_->sensor_origin_.coeff (1);
#           vpz_ = input_->sensor_origin_.coeff (2);
#         }
#       }
#       
#       /** \brief Set the viewpoint.
#         * \param vpx the X coordinate of the viewpoint
#         * \param vpy the Y coordinate of the viewpoint
#         * \param vpz the Z coordinate of the viewpoint
#         */
#       inline void
#       setViewPoint (float vpx, float vpy, float vpz)
#       {
#         vpx_ = vpx;
#         vpy_ = vpy;
#         vpz_ = vpz;
#         use_sensor_origin_ = false;
#       }
# 
#       /** \brief Get the viewpoint.
#         * \param [out] vpx x-coordinate of the view point
#         * \param [out] vpy y-coordinate of the view point
#         * \param [out] vpz z-coordinate of the view point
#         * \note this method returns the currently used viewpoint for normal flipping.
#         * If the viewpoint is set manually using the setViewPoint method, this method will return the set view point coordinates.
#         * If an input cloud is set, it will return the sensor origin otherwise it will return the origin (0, 0, 0)
#         */
#       inline void
#       getViewPoint (float &vpx, float &vpy, float &vpz)
#       {
#         vpx = vpx_;
#         vpy = vpy_;
#         vpz = vpz_;
#       }
# 
#       /** \brief sets whether the sensor origin or a user given viewpoint should be used. After this method, the 
#         * normal estimation method uses the sensor origin of the input cloud.
#         * to use a user defined view point, use the method setViewPoint
#         */
#       inline void
#       useSensorOriginAsViewPoint ()
#       {
#         use_sensor_origin_ = true;
#         if (input_)
#         {
#           vpx_ = input_->sensor_origin_.coeff (0);
#           vpy_ = input_->sensor_origin_.coeff (1);
#           vpz_ = input_->sensor_origin_.coeff (2);
#         }
#         else
#         {
#           vpx_ = 0;
#           vpy_ = 0;
#           vpz_ = 0;
#         }
#       }
#       
#     protected:
#       /** \brief Estimate normals for all points given in <setInputCloud (), setIndices ()> using the surface in
#         * setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \note In situations where not enough neighbors are found, the normal and curvature values are set to -1.
#         * \param output the resultant point cloud model dataset that contains surface normals and curvatures
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief Values describing the viewpoint ("pinhole" camera model assumed). For per point viewpoints, inherit
#         * from NormalEstimation and provide your own computeFeature (). By default, the viewpoint is set to 0,0,0. */
#       float vpx_, vpy_, vpz_;
# 
#       /** \brief Placeholder for the 3x3 covariance matrix at each surface patch. */
#       EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix_;
# 
#       /** \brief 16-bytes aligned placeholder for the XYZ centroid of a surface patch. */
#       Eigen::Vector4f xyz_centroid_;
#       
#       /** whether the sensor origin of the input cloud or a user given viewpoint should be used.*/
#       bool use_sensor_origin_;
# 
#     private:
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#    };
# 
#   /** \brief NormalEstimation estimates local surface properties at each 3D point, such as surface normals and
#     * curvatures.
#     *
#     * \note The code is stateful as we do not expect this class to be multicore parallelized. Please look at
#     * \ref NormalEstimationOMP for a parallel implementation.
#     * \author Radu B. Rusu
#     * \ingroup features
#     */
#   template <typename PointInT>
#   class NormalEstimation<PointInT, Eigen::MatrixXf>: public NormalEstimation<PointInT, pcl::Normal>
#   {
#     public:
#       using NormalEstimation<PointInT, pcl::Normal>::indices_;
#       using NormalEstimation<PointInT, pcl::Normal>::input_;
#       using NormalEstimation<PointInT, pcl::Normal>::surface_;
#       using NormalEstimation<PointInT, pcl::Normal>::k_;
#       using NormalEstimation<PointInT, pcl::Normal>::search_parameter_;
#       using NormalEstimation<PointInT, pcl::Normal>::vpx_;
#       using NormalEstimation<PointInT, pcl::Normal>::vpy_;
#       using NormalEstimation<PointInT, pcl::Normal>::vpz_;
#       using NormalEstimation<PointInT, pcl::Normal>::computePointNormal;
#       using NormalEstimation<PointInT, pcl::Normal>::compute;
# 
#     private:
#       /** \brief Estimate normals for all points given in <setInputCloud (), setIndices ()> using the surface in
#         * setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \note In situations where not enough neighbors are found, the normal and curvature values are set to NaN
#         * \param[out] output the resultant point cloud model dataset that contains surface normals and curvatures
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       compute (pcl::PointCloud<pcl::Normal> &) {}
#   };
# 
# ###
#     # normal_3d_omp.h
#   template <typename PointInT, typename PointOutT>
#   class NormalEstimationOMP: public NormalEstimation<PointInT, PointOutT>
#   {
#     public:
#       using NormalEstimation<PointInT, PointOutT>::feature_name_;
#       using NormalEstimation<PointInT, PointOutT>::getClassName;
#       using NormalEstimation<PointInT, PointOutT>::indices_;
#       using NormalEstimation<PointInT, PointOutT>::input_;
#       using NormalEstimation<PointInT, PointOutT>::k_;
#       using NormalEstimation<PointInT, PointOutT>::search_parameter_;
#       using NormalEstimation<PointInT, PointOutT>::surface_;
#       using NormalEstimation<PointInT, PointOutT>::getViewPoint;
# 
#       typedef typename NormalEstimation<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#     public:
#       /** \brief Empty constructor. */
#       NormalEstimationOMP () : threads_ (1) 
#       {
#         feature_name_ = "NormalEstimationOMP";
#       };
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       NormalEstimationOMP (unsigned int nr_threads) : threads_ (1)
#       {
#         setNumberOfThreads (nr_threads);
#         feature_name_ = "NormalEstimationOMP";
#       }
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       inline void 
#       setNumberOfThreads (unsigned int nr_threads)
#       { 
#         if (nr_threads == 0)
#           nr_threads = 1;
#         threads_ = nr_threads; 
#       }
# 
# 
#     protected:
#       /** \brief The number of threads the scheduler should use. */
#       unsigned int threads_;
# 
#     private:
#       /** \brief Estimate normals for all points given in <setInputCloud (), setIndices ()> using the surface in
#         * setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains surface normals and curvatures
#         */
#       void 
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   /** \brief NormalEstimationOMP estimates local surface properties at each 3D point, such as surface normals and
#     * curvatures, in parallel, using the OpenMP standard.
#     * \author Radu Bogdan Rusu
#     * \ingroup features
#     */
#   template <typename PointInT>
#   class NormalEstimationOMP<PointInT, Eigen::MatrixXf>: public NormalEstimationOMP<PointInT, pcl::Normal>
#   {
#     public:
#       using NormalEstimationOMP<PointInT, pcl::Normal>::indices_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::search_parameter_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::k_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::input_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::surface_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::getViewPoint;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::threads_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::compute;
# 
#       /** \brief Default constructor.
#         */
#       NormalEstimationOMP () : NormalEstimationOMP<PointInT, pcl::Normal> () {}
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       NormalEstimationOMP (unsigned int nr_threads) : NormalEstimationOMP<PointInT, pcl::Normal> (nr_threads) {}
# 
#     private:
#       /** \brief Estimate normals for all points given in <setInputCloud (), setIndices ()> using the surface in
#         * setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains surface normals and curvatures
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       compute (pcl::PointCloud<pcl::Normal> &) {}
#     };
# 
# ###
#     # normal_based_signature.h
#   template <typename PointInT, typename PointOutT>
#   class NormalEstimationOMP: public NormalEstimation<PointInT, PointOutT>
#   {
#     public:
#       using NormalEstimation<PointInT, PointOutT>::feature_name_;
#       using NormalEstimation<PointInT, PointOutT>::getClassName;
#       using NormalEstimation<PointInT, PointOutT>::indices_;
#       using NormalEstimation<PointInT, PointOutT>::input_;
#       using NormalEstimation<PointInT, PointOutT>::k_;
#       using NormalEstimation<PointInT, PointOutT>::search_parameter_;
#       using NormalEstimation<PointInT, PointOutT>::surface_;
#       using NormalEstimation<PointInT, PointOutT>::getViewPoint;
# 
#       typedef typename NormalEstimation<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#     public:
#       /** \brief Empty constructor. */
#       NormalEstimationOMP () : threads_ (1) 
#       {
#         feature_name_ = "NormalEstimationOMP";
#       };
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       NormalEstimationOMP (unsigned int nr_threads) : threads_ (1)
#       {
#         setNumberOfThreads (nr_threads);
#         feature_name_ = "NormalEstimationOMP";
#       }
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       inline void 
#       setNumberOfThreads (unsigned int nr_threads)
#       { 
#         if (nr_threads == 0)
#           nr_threads = 1;
#         threads_ = nr_threads; 
#       }
# 
# 
#     protected:
#       /** \brief The number of threads the scheduler should use. */
#       unsigned int threads_;
# 
#     private:
#       /** \brief Estimate normals for all points given in <setInputCloud (), setIndices ()> using the surface in
#         * setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains surface normals and curvatures
#         */
#       void 
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   /** \brief NormalEstimationOMP estimates local surface properties at each 3D point, such as surface normals and
#     * curvatures, in parallel, using the OpenMP standard.
#     * \author Radu Bogdan Rusu
#     * \ingroup features
#     */
#   template <typename PointInT>
#   class NormalEstimationOMP<PointInT, Eigen::MatrixXf>: public NormalEstimationOMP<PointInT, pcl::Normal>
#   {
#     public:
#       using NormalEstimationOMP<PointInT, pcl::Normal>::indices_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::search_parameter_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::k_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::input_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::surface_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::getViewPoint;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::threads_;
#       using NormalEstimationOMP<PointInT, pcl::Normal>::compute;
# 
#       /** \brief Default constructor.
#         */
#       NormalEstimationOMP () : NormalEstimationOMP<PointInT, pcl::Normal> () {}
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       NormalEstimationOMP (unsigned int nr_threads) : NormalEstimationOMP<PointInT, pcl::Normal> (nr_threads) {}
# 
#     private:
#       /** \brief Estimate normals for all points given in <setInputCloud (), setIndices ()> using the surface in
#         * setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains surface normals and curvatures
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       compute (pcl::PointCloud<pcl::Normal> &) {}
#     };
# 
# ###
#     # pfh.h
#   template <typename PointInT, typename PointNT, typename PointOutT = pcl::PFHSignature125>
#   class PFHEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::input_;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#       typedef typename Feature<PointInT, PointOutT>::PointCloudIn  PointCloudIn;
# 
#       /** \brief Empty constructor. 
#         * Sets \a use_cache_ to false, \a nr_subdiv_ to 5, and the internal maximum cache size to 1GB.
#         */
#       PFHEstimation () : 
#         nr_subdiv_ (5), 
#         pfh_histogram_ (),
#         pfh_tuple_ (),
#         d_pi_ (1.0f / (2.0f * static_cast<float> (M_PI))), 
#         feature_map_ (),
#         key_list_ (),
#         // Default 1GB memory size. Need to set it to something more conservative.
#         max_cache_size_ ((1ul*1024ul*1024ul*1024ul) / sizeof (std::pair<std::pair<int, int>, Eigen::Vector4f>)),
#         use_cache_ (false)
#       {
#         feature_name_ = "PFHEstimation";
#       };
# 
#       /** \brief Set the maximum internal cache size. Defaults to 2GB worth of entries.
#         * \param[in] cache_size maximum cache size 
#         */
#       inline void
#       setMaximumCacheSize (unsigned int cache_size)
#       {
#         max_cache_size_ = cache_size;
#       }
# 
#       /** \brief Get the maximum internal cache size. */
#       inline unsigned int 
#       getMaximumCacheSize ()
#       {
#         return (max_cache_size_);
#       }
# 
#       /** \brief Set whether to use an internal cache mechanism for removing redundant calculations or not. 
#         *
#         * \note Depending on how the point cloud is ordered and how the nearest
#         * neighbors are estimated, using a cache could have a positive or a
#         * negative influence. Please test with and without a cache on your
#         * data, and choose whatever works best!
#         *
#         * See \ref setMaximumCacheSize for setting the maximum cache size
#         *
#         * \param[in] use_cache set to true to use the internal cache, false otherwise
#         */
#       inline void
#       setUseInternalCache (bool use_cache)
#       {
#         use_cache_ = use_cache;
#       }
# 
#       /** \brief Get whether the internal cache is used or not for computing the PFH features. */
#       inline bool
#       getUseInternalCache ()
#       {
#         return (use_cache_);
#       }
# 
#       /** \brief Compute the 4-tuple representation containing the three angles and one distance between two points
#         * represented by Cartesian coordinates and normals.
#         * \note For explanations about the features, please see the literature mentioned above (the order of the
#         * features might be different).
#         * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
#         * \param[in] normals the dataset containing the surface normals (assuming normalized vectors) at each point in cloud
#         * \param[in] p_idx the index of the first point (source)
#         * \param[in] q_idx the index of the second point (target)
#         * \param[out] f1 the first angular feature (angle between the projection of nq_idx and u)
#         * \param[out] f2 the second angular feature (angle between nq_idx and v)
#         * \param[out] f3 the third angular feature (angle between np_idx and |p_idx - q_idx|)
#         * \param[out] f4 the distance feature (p_idx - q_idx)
#         * \note For efficiency reasons, we assume that the point data passed to the method is finite.
#         */
#       bool 
#       computePairFeatures (const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals, 
#                            int p_idx, int q_idx, float &f1, float &f2, float &f3, float &f4);
# 
#       /** \brief Estimate the PFH (Point Feature Histograms) individual signatures of the three angular (f1, f2, f3)
#         * features for a given point based on its spatial neighborhood of 3D points with normals
#         * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
#         * \param[in] normals the dataset containing the surface normals at each point in \a cloud
#         * \param[in] indices the k-neighborhood point indices in the dataset
#         * \param[in] nr_split the number of subdivisions for each angular feature interval
#         * \param[out] pfh_histogram the resultant (combinatorial) PFH histogram representing the feature at the query point
#         */
#       void 
#       computePointPFHSignature (const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals, 
#                                 const std::vector<int> &indices, int nr_split, Eigen::VectorXf &pfh_histogram);
# 
#     protected:
#       /** \brief Estimate the Point Feature Histograms (PFH) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the PFH feature estimates
#         */
#       void 
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief The number of subdivisions for each angular feature interval. */
#       int nr_subdiv_;
# 
#       /** \brief Placeholder for a point's PFH signature. */
#       Eigen::VectorXf pfh_histogram_;
# 
#       /** \brief Placeholder for a PFH 4-tuple. */
#       Eigen::Vector4f pfh_tuple_;
# 
#       /** \brief Placeholder for a histogram index. */
#       int f_index_[3];
# 
#       /** \brief Float constant = 1.0 / (2.0 * M_PI) */
#       float d_pi_; 
# 
#       /** \brief Internal hashmap, used to optimize efficiency of redundant computations. */
#       std::map<std::pair<int, int>, Eigen::Vector4f, std::less<std::pair<int, int> >, Eigen::aligned_allocator<Eigen::Vector4f> > feature_map_;
# 
#       /** \brief Queue of pairs saved, used to constrain memory usage. */
#       std::queue<std::pair<int, int> > key_list_;
# 
#       /** \brief Maximum size of internal cache memory. */
#       unsigned int max_cache_size_;
# 
#       /** \brief Set to true to use the internal cache for removing redundant computations. */
#       bool use_cache_;
#     private:
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   /** \brief PFHEstimation estimates the Point Feature Histogram (PFH) descriptor for a given point cloud dataset
#     * containing points and normals.
#     *
#     * \note If you use this code in any academic work, please cite:
#     *
#     *   - R.B. Rusu, N. Blodow, Z.C. Marton, M. Beetz.
#     *     Aligning Point Cloud Views using Persistent Feature Histograms.
#     *     In Proceedings of the 21st IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),
#     *     Nice, France, September 22-26 2008.
#     *   - R.B. Rusu, Z.C. Marton, N. Blodow, M. Beetz.
#     *     Learning Informative Point Classes for the Acquisition of Object Model Maps.
#     *     In Proceedings of the 10th International Conference on Control, Automation, Robotics and Vision (ICARCV),
#     *     Hanoi, Vietnam, December 17-20 2008.
#     *
#     * \attention 
#     * The convention for PFH features is:
#     *   - if a query point's nearest neighbors cannot be estimated, the PFH feature will be set to NaN 
#     *     (not a number)
#     *   - it is impossible to estimate a PFH descriptor for a point that
#     *     doesn't have finite 3D coordinates. Therefore, any point that contains
#     *     NaN data on x, y, or z, will have its PFH feature property set to NaN.
#     *
#     * \note The code is stateful as we do not expect this class to be multicore parallelized. Please look at
#     * \ref FPFHEstimationOMP for examples on parallel implementations of the FPFH (Fast Point Feature Histogram).
#     *
#     * \author Radu B. Rusu
#     * \ingroup features
#     */
#   template <typename PointInT, typename PointNT>
#   class PFHEstimation<PointInT, PointNT, Eigen::MatrixXf> : public PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>
#   {
#     public:
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::pfh_histogram_;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::nr_subdiv_;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::k_;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::indices_;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::search_parameter_;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::surface_;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::input_;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::normals_;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::computePointPFHSignature;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::compute;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::feature_map_;
#       using PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>::key_list_;
# 
# ###
#     # pfhrgb.h
#   template <typename PointInT, typename PointNT, typename PointOutT = pcl::PFHRGBSignature250>
#   class PFHRGBEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
#   {
#     public:
#       using PCLBase<PointInT>::indices_;
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
# 
#       PFHRGBEstimation ()
#         : nr_subdiv_ (5), pfhrgb_histogram_ (), pfhrgb_tuple_ (), d_pi_ (1.0f / (2.0f * static_cast<float> (M_PI)))
#       {
#         feature_name_ = "PFHRGBEstimation";
#       }
# 
#       bool
#       computeRGBPairFeatures (const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals,
#                               int p_idx, int q_idx,
#                               float &f1, float &f2, float &f3, float &f4, float &f5, float &f6, float &f7);
# 
#       void
#       computePointPFHRGBSignature (const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals,
#                                    const std::vector<int> &indices, int nr_split, Eigen::VectorXf &pfhrgb_histogram);
# 
#     protected:
#       void
#       computeFeature (PointCloudOut &output);
# 
#     private:
#       /** \brief The number of subdivisions for each angular feature interval. */
#       int nr_subdiv_;
# 
#       /** \brief Placeholder for a point's PFHRGB signature. */
#       Eigen::VectorXf pfhrgb_histogram_;
# 
#       /** \brief Placeholder for a PFHRGB 7-tuple. */
#       Eigen::VectorXf pfhrgb_tuple_;
# 
#       /** \brief Placeholder for a histogram index. */
#       int f_index_[7];
# 
#       /** \brief Float constant = 1.0 / (2.0 * M_PI) */
#       float d_pi_;
# 
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
# ###
# 	# ppf.h
#   template <typename PointInT, typename PointNT, typename PointOutT>
#   class PPFEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
#   {
#     public:
#       using PCLBase<PointInT>::indices_;
#       using Feature<PointInT, PointOutT>::input_;
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#       typedef pcl::PointCloud<PointOutT> PointCloudOut;
# 
#       /** \brief Empty Constructor. */
#       PPFEstimation ();
# 
# 
#     private:
#       /** \brief The method called for actually doing the computations
#         * \param[out] output the resulting point cloud (which should be of type pcl::PPFSignature);
#         * its size is the size of the input cloud, squared (i.e., one point for each pair in
#         * the input cloud);
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   /** \brief Class that calculates the "surflet" features for each pair in the given
#     * pointcloud. Please refer to the following publication for more details:
#     *    B. Drost, M. Ulrich, N. Navab, S. Ilic
#     *    Model Globally, Match Locally: Efficient and Robust 3D Object Recognition
#     *    2010 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
#     *    13-18 June 2010, San Francisco, CA
#     *
#     * PointOutT is meant to be pcl::PPFSignature - contains the 4 values of the Surflet
#     * feature and in addition, alpha_m for the respective pair - optimization proposed by
#     * the authors (see above)
#     *
#     * \author Alexandru-Eugen Ichim
#     */
#   template <typename PointInT, typename PointNT>
#   class PPFEstimation<PointInT, PointNT, Eigen::MatrixXf> : public PPFEstimation<PointInT, PointNT, pcl::PPFSignature>
#   {
#     public:
#       using PPFEstimation<PointInT, PointNT, pcl::PPFSignature>::getClassName;
#       using PPFEstimation<PointInT, PointNT, pcl::PPFSignature>::input_;
#       using PPFEstimation<PointInT, PointNT, pcl::PPFSignature>::normals_;
#       using PPFEstimation<PointInT, PointNT, pcl::PPFSignature>::indices_;
# 
#     private:
#       /** \brief The method called for actually doing the computations
#         * \param[out] output the resulting point cloud
#         * its size is the size of the input cloud, squared (i.e., one point for each pair in
#         * the input cloud);
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       compute (pcl::PointCloud<pcl::Normal> &) {}
#   };
# 
# ###
#     # ppfrgb.h
#   template <typename PointInT, typename PointNT, typename PointOutT>
#   class PPFRGBEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
#   {
#     public:
#       using PCLBase<PointInT>::indices_;
#       using Feature<PointInT, PointOutT>::input_;
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#       typedef pcl::PointCloud<PointOutT> PointCloudOut;
# 
#       /**
#         * \brief Empty Constructor
#         */
#       PPFRGBEstimation ();
# 
# 
#     private:
#       /** \brief The method called for actually doing the computations
#         * \param output the resulting point cloud (which should be of type pcl::PPFRGBSignature);
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   template <typename PointInT, typename PointNT, typename PointOutT>
#   class PPFRGBRegionEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
#   {
#     public:
#       using PCLBase<PointInT>::indices_;
#       using Feature<PointInT, PointOutT>::input_;
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::search_radius_;
#       using Feature<PointInT, PointOutT>::tree_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#       typedef pcl::PointCloud<PointOutT> PointCloudOut;
# 
#       PPFRGBRegionEstimation ();
# 
#     private:
#       void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief Make the computeFeature (pcl::PointCloud<Eigen::MatrixXf> &output); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
# ###
#     # principal_curvatures.h
#   template <typename PointInT, typename PointNT, typename PointOutT = pcl::PrincipalCurvatures>
#   class PrincipalCurvaturesEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::input_;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#       typedef pcl::PointCloud<PointInT> PointCloudIn;
# 
#       /** \brief Empty constructor. */
#       PrincipalCurvaturesEstimation () : 
#         projected_normals_ (), 
#         xyz_centroid_ (Eigen::Vector3f::Zero ()), 
#         demean_ (Eigen::Vector3f::Zero ()),
#         covariance_matrix_ (Eigen::Matrix3f::Zero ()),
#         eigenvector_ (Eigen::Vector3f::Zero ()),
#         eigenvalues_ (Eigen::Vector3f::Zero ())
#       {
#         feature_name_ = "PrincipalCurvaturesEstimation";
#       };
# 
#       /** \brief Perform Principal Components Analysis (PCA) on the point normals of a surface patch in the tangent
#        *  plane of the given point normal, and return the principal curvature (eigenvector of the max eigenvalue),
#        *  along with both the max (pc1) and min (pc2) eigenvalues
#        * \param[in] normals the point cloud normals
#        * \param[in] p_idx the query point at which the least-squares plane was estimated
#        * \param[in] indices the point cloud indices that need to be used
#        * \param[out] pcx the principal curvature X direction
#        * \param[out] pcy the principal curvature Y direction
#        * \param[out] pcz the principal curvature Z direction
#        * \param[out] pc1 the max eigenvalue of curvature
#        * \param[out] pc2 the min eigenvalue of curvature
#        */
#       void
#       computePointPrincipalCurvatures (const pcl::PointCloud<PointNT> &normals,
#                                        int p_idx, const std::vector<int> &indices,
#                                        float &pcx, float &pcy, float &pcz, float &pc1, float &pc2);
# 
#     protected:
# 
#       /** \brief Estimate the principal curvature (eigenvector of the max eigenvalue), along with both the max (pc1)
#         * and min (pc2) eigenvalues for all points given in <setInputCloud (), setIndices ()> using the surface in
#         * setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the principal curvature estimates
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#     private:
#       /** \brief A pointer to the input dataset that contains the point normals of the XYZ dataset. */
#       std::vector<Eigen::Vector3f> projected_normals_;
# 
#       /** \brief SSE aligned placeholder for the XYZ centroid of a surface patch. */
#       Eigen::Vector3f xyz_centroid_;
# 
#       /** \brief Temporary point placeholder. */
#       Eigen::Vector3f demean_;
# 
#       /** \brief Placeholder for the 3x3 covariance matrix at each surface patch. */
#       EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix_;
# 
#       /** \brief SSE aligned eigenvectors placeholder for a covariance matrix. */
#       Eigen::Vector3f eigenvector_;
#       /** \brief eigenvalues placeholder for a covariance matrix. */
#       Eigen::Vector3f eigenvalues_;
# 
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   /** \brief PrincipalCurvaturesEstimation estimates the directions (eigenvectors) and magnitudes (eigenvalues) of
#     * principal surface curvatures for a given point cloud dataset containing points and normals.
#     *
#     * \note The code is stateful as we do not expect this class to be multicore parallelized. Please look at
#     * \ref NormalEstimationOMP for an example on how to extend this to parallel implementations.
#     *
#     * \author Radu B. Rusu, Jared Glover
#     * \ingroup features
#     */
#   template <typename PointInT, typename PointNT>
#   class PrincipalCurvaturesEstimation<PointInT, PointNT, Eigen::MatrixXf> : public PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>
#   {
#     public:
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::indices_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::k_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::search_parameter_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::surface_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::compute;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::input_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::normals_;
# 
#     private:
#       /** \brief Estimate the principal curvature (eigenvector of the max eigenvalue), along with both the max (pc1)
#         * and min (pc2) eigenvalues for all points given in <setInputCloud (), setIndices ()> using the surface in
#         * setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the principal curvature estimates
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       compute (pcl::PointCloud<pcl::Normal> &) {}
#   };
# 
# ###
#     # range_image_border_extractor.h
# 
# ###
#     # rift.h
#   template <typename PointInT, typename GradientT, typename PointOutT>
#   class RIFTEstimation: public Feature<PointInT, PointOutT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
# 
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::indices_;
# 
#       using Feature<PointInT, PointOutT>::tree_;
#       using Feature<PointInT, PointOutT>::search_radius_;
#       
#       typedef typename pcl::PointCloud<PointInT> PointCloudIn;
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#       typedef typename pcl::PointCloud<GradientT> PointCloudGradient;
#       typedef typename PointCloudGradient::Ptr PointCloudGradientPtr;
#       typedef typename PointCloudGradient::ConstPtr PointCloudGradientConstPtr;
# 
#       typedef typename boost::shared_ptr<RIFTEstimation<PointInT, GradientT, PointOutT> > Ptr;
#       typedef typename boost::shared_ptr<const RIFTEstimation<PointInT, GradientT, PointOutT> > ConstPtr;
# 
# 
#       /** \brief Empty constructor. */
#       RIFTEstimation () : gradient_ (), nr_distance_bins_ (4), nr_gradient_bins_ (8)
#       {
#         feature_name_ = "RIFTEstimation";
#       };
# 
#       /** \brief Provide a pointer to the input gradient data
#         * \param[in] gradient a pointer to the input gradient data
#         */
#       inline void 
#       setInputGradient (const PointCloudGradientConstPtr &gradient) { gradient_ = gradient; };
# 
#       /** \brief Returns a shared pointer to the input gradient data */
#       inline PointCloudGradientConstPtr 
#       getInputGradient () const { return (gradient_); };
# 
#       /** \brief Set the number of bins to use in the distance dimension of the RIFT descriptor
#         * \param[in] nr_distance_bins the number of bins to use in the distance dimension of the RIFT descriptor
#         */
#       inline void 
#       setNrDistanceBins (int nr_distance_bins) { nr_distance_bins_ = nr_distance_bins; };
# 
#       /** \brief Returns the number of bins in the distance dimension of the RIFT descriptor. */
#       inline int 
#       getNrDistanceBins () const { return (nr_distance_bins_); };
# 
#       /** \brief Set the number of bins to use in the gradient orientation dimension of the RIFT descriptor
#         * \param[in] nr_gradient_bins the number of bins to use in the gradient orientation dimension of the RIFT descriptor
#         */
#       inline void 
#       setNrGradientBins (int nr_gradient_bins) { nr_gradient_bins_ = nr_gradient_bins; };
# 
#       /** \brief Returns the number of bins in the gradient orientation dimension of the RIFT descriptor. */
#       inline int 
#       getNrGradientBins () const { return (nr_gradient_bins_); };
# 
#       /** \brief Estimate the Rotation Invariant Feature Transform (RIFT) descriptor for a given point based on its 
#         * spatial neighborhood of 3D points and the corresponding intensity gradient vector field
#         * \param[in] cloud the dataset containing the Cartesian coordinates of the points
#         * \param[in] gradient the dataset containing the intensity gradient at each point in \a cloud
#         * \param[in] p_idx the index of the query point in \a cloud (i.e. the center of the neighborhood)
#         * \param[in] radius the radius of the RIFT feature
#         * \param[in] indices the indices of the points that comprise \a p_idx's neighborhood in \a cloud
#         * \param[in] squared_distances the squared distances from the query point to each point in the neighborhood
#         * \param[out] rift_descriptor the resultant RIFT descriptor
#         */
#       void 
#       computeRIFT (const PointCloudIn &cloud, const PointCloudGradient &gradient, int p_idx, float radius,
#                    const std::vector<int> &indices, const std::vector<float> &squared_distances, 
#                    Eigen::MatrixXf &rift_descriptor);
# 
#     protected:
# 
#       /** \brief Estimate the Rotation Invariant Feature Transform (RIFT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface (), the gradient in 
#         * setInputGradient (), and the spatial locator in setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the RIFT feature estimates
#         */
#       void 
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief The intensity gradient of the input point cloud data*/
#       PointCloudGradientConstPtr gradient_;
# 
#       /** \brief The number of distance bins in the descriptor. */
#       int nr_distance_bins_;
# 
#       /** \brief The number of gradient orientation bins in the descriptor. */
#       int nr_gradient_bins_;
# 
#     private:
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf>&) {}
#   };
# 
#   /** \brief RIFTEstimation estimates the Rotation Invariant Feature Transform descriptors for a given point cloud 
#     * dataset containing points and intensity.  For more information about the RIFT descriptor, see:
#     *
#     *  Svetlana Lazebnik, Cordelia Schmid, and Jean Ponce. 
#     *  A sparse texture representation using local affine regions. 
#     *  In IEEE Transactions on Pattern Analysis and Machine Intelligence, volume 27, pages 1265-1278, August 2005.
#     *
#     * \author Michael Dixon
#     * \ingroup features
#     */
# 
#   template <typename PointInT, typename GradientT>
#   class RIFTEstimation<PointInT, GradientT, Eigen::MatrixXf>: public RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >
#   {
#     public:
#       using RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >::getClassName;
#       using RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >::surface_;
#       using RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >::indices_;
#       using RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >::tree_;
#       using RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >::search_radius_;
#       using RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >::gradient_;
#       using RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >::nr_gradient_bins_;
#       using RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >::nr_distance_bins_;
#       using RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >::compute;
#       
#     private:
#       /** \brief Estimate the Rotation Invariant Feature Transform (RIFT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface (), the gradient in 
#         * setInputGradient (), and the spatial locator in setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the RIFT feature estimates
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       compute (pcl::PointCloud<pcl::Normal>&) {}
#   };
# 
# ###
#     # shot.h
#   template <typename PointInT, typename PointNT, typename PointOutT, typename PointRFT = pcl::ReferenceFrame>
#   class SHOTEstimationBase : public FeatureFromNormals<PointInT, PointNT, PointOutT>,
#                              public FeatureWithLocalReferenceFrames<PointInT, PointRFT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::input_;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       using Feature<PointInT, PointOutT>::search_radius_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::fake_surface_;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
# 
#     protected:
#       /** \brief Empty constructor.
#         * \param[in] nr_shape_bins the number of bins in the shape histogram
#         */
#       SHOTEstimationBase (int nr_shape_bins = 10) :
#         nr_shape_bins_ (nr_shape_bins),
#         shot_ (),
#         sqradius_ (0), radius3_4_ (0), radius1_4_ (0), radius1_2_ (0),
#         nr_grid_sector_ (32),
#         maxAngularSectors_ (28),
#         descLength_ (0)
#       {
#         feature_name_ = "SHOTEstimation";
#       };
# 
#     public:
#        /** \brief Estimate the SHOT descriptor for a given point based on its spatial neighborhood of 3D points with normals
#          * \param[in] index the index of the point in indices_
#          * \param[in] indices the k-neighborhood point indices in surface_
#          * \param[in] sqr_dists the k-neighborhood point distances in surface_
#          * \param[out] shot the resultant SHOT descriptor representing the feature at the query point
#          */
#       virtual void
#       computePointSHOT (const int index,
#                         const std::vector<int> &indices,
#                         const std::vector<float> &sqr_dists,
#                         Eigen::VectorXf &shot) = 0;
# 
#     protected:
# 
#       /** \brief This method should get called before starting the actual computation. */
#       virtual bool
#       initCompute ();
# 
#       /** \brief Quadrilinear interpolation used when color and shape descriptions are NOT activated simultaneously
#         *
#         * \param[in] indices the neighborhood point indices
#         * \param[in] sqr_dists the neighborhood point distances
#         * \param[in] index the index of the point in indices_
#         * \param[out] binDistance the resultant distance shape histogram
#         * \param[in] nr_bins the number of bins in the shape histogram
#         * \param[out] shot the resultant SHOT histogram
#         */
#       void
#       interpolateSingleChannel (const std::vector<int> &indices,
#                                 const std::vector<float> &sqr_dists,
#                                 const int index,
#                                 std::vector<double> &binDistance,
#                                 const int nr_bins,
#                                 Eigen::VectorXf &shot);
# 
#       /** \brief Normalize the SHOT histogram.
#         * \param[in,out] shot the SHOT histogram
#         * \param[in] desc_length the length of the histogram
#         */
#       void
#       normalizeHistogram (Eigen::VectorXf &shot, int desc_length);
# 
# 
#       /** \brief Create a binned distance shape histogram
#         * \param[in] index the index of the point in indices_
#         * \param[in] indices the k-neighborhood point indices in surface_
#         * \param[in] sqr_dists the k-neighborhood point distances in surface_
#         * \param[out] bin_distance_shape the resultant histogram
#         */
#       void
#       createBinDistanceShape (int index, const std::vector<int> &indices,
#                               std::vector<double> &bin_distance_shape);
# 
#       /** \brief The number of bins in each shape histogram. */
#       int nr_shape_bins_;
# 
#       /** \brief Placeholder for a point's SHOT. */
#       Eigen::VectorXf shot_;
# 
#       /** \brief The squared search radius. */
#       double sqradius_;
# 
#       /** \brief 3/4 of the search radius. */
#       double radius3_4_;
# 
#       /** \brief 1/4 of the search radius. */
#       double radius1_4_;
# 
#       /** \brief 1/2 of the search radius. */
#       double radius1_2_;
# 
#       /** \brief Number of azimuthal sectors. */
#       const int nr_grid_sector_;
# 
#       /** \brief ... */
#       const int maxAngularSectors_;
# 
#       /** \brief One SHOT length. */
#       int descLength_;
# 
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   /** \brief SHOTEstimation estimates the Signature of Histograms of OrienTations (SHOT) descriptor for
#     * a given point cloud dataset containing points and normals.
#     *
#     * \note If you use this code in any academic work, please cite:
#     *
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     Unique Signatures of Histograms for Local Surface Description.
#     *     In Proceedings of the 11th European Conference on Computer Vision (ECCV),
#     *     Heraklion, Greece, September 5-11 2010.
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     A Combined Texture-Shape Descriptor For Enhanced 3D Feature Matching.
#     *     In Proceedings of the 18th International Conference on Image Processing (ICIP),
#     *     Brussels, Belgium, September 11-14 2011.
#     *
#     * \author Samuele Salti, Federico Tombari
#     * \ingroup features
#     */
#   //template <typename PointInT, typename PointNT, typename PointRFT>
#   //PCL_DEPRECATED (class, "SHOTEstimationBase<..., Eigen::MatrixXf, ...> IS DEPRECATED")
#     //SHOTEstimationBase<PointInT, PointNT, Eigen::MatrixXf, PointRFT> : public SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>
#   //{
#     //public:
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::getClassName;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::input_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::indices_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::k_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::search_parameter_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::search_radius_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::surface_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::descLength_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::nr_grid_sector_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::nr_shape_bins_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::sqradius_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::radius3_4_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::radius1_4_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::radius1_2_;
#       //using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::shot_;
#       //using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
# //
#       ///** \brief Empty constructor.
#         //* \param[in] nr_shape_bins the number of bins in the shape histogram
#         //*/
#       //SHOTEstimationBase (int nr_shape_bins = 10) : SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT> (nr_shape_bins) {};
# //
#       ///** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         //* <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         //* setSearchMethod ()
#         //* \param output the resultant point cloud model dataset that contains the SHOT feature estimates
#         //*/
#       //void
#       //computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# //
#       ///** \brief Base method for feature estimation for all points given in
#         //* <setInputCloud (), setIndices ()> using the surface in setSearchSurface ()
#         //* and the spatial locator in setSearchMethod ()
#         //* \param[out] output the resultant point cloud model dataset containing the estimated features
#         //*/
#       //void
#       //computeEigen (pcl::PointCloud<Eigen::MatrixXf> &output)
#       //{
#         //pcl::SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::computeEigen (output);
#       //}
# //
#       ///** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         //* \param[out] output the output point cloud
#         //*/
#       //void
#       //compute (pcl::PointCloud<pcl::SHOT> &) {}
#   //};
# 
#   /** \brief SHOTEstimation estimates the Signature of Histograms of OrienTations (SHOT) descriptor for
#     * a given point cloud dataset containing points and normals.
#     *
#     * \note If you use this code in any academic work, please cite:
#     *
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     Unique Signatures of Histograms for Local Surface Description.
#     *     In Proceedings of the 11th European Conference on Computer Vision (ECCV),
#     *     Heraklion, Greece, September 5-11 2010.
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     A Combined Texture-Shape Descriptor For Enhanced 3D Feature Matching.
#     *     In Proceedings of the 18th International Conference on Image Processing (ICIP),
#     *     Brussels, Belgium, September 11-14 2011.
#     *
#     * \author Samuele Salti, Federico Tombari
#     * \ingroup features
#     */
#   template <typename PointInT, typename PointNT, typename PointOutT = pcl::SHOT352, typename PointRFT = pcl::ReferenceFrame>
#   class SHOTEstimation : public SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>
#   {
#     public:
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::feature_name_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::getClassName;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::indices_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::k_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::search_parameter_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::search_radius_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::surface_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::input_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::normals_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::descLength_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::nr_grid_sector_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::nr_shape_bins_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::sqradius_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::radius3_4_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::radius1_4_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::radius1_2_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::maxAngularSectors_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::interpolateSingleChannel;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::shot_;
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
# 
#       /** \brief Empty constructor. */
#       SHOTEstimation () : SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT> (10)
#       {
#         feature_name_ = "SHOTEstimation";
#       };
# 
#       /** \brief Estimate the SHOT descriptor for a given point based on its spatial neighborhood of 3D points with normals
#         * \param[in] index the index of the point in indices_
#         * \param[in] indices the k-neighborhood point indices in surface_
#         * \param[in] sqr_dists the k-neighborhood point distances in surface_
#         * \param[out] shot the resultant SHOT descriptor representing the feature at the query point
#         */
#       virtual void
#       computePointSHOT (const int index,
#                         const std::vector<int> &indices,
#                         const std::vector<float> &sqr_dists,
#                         Eigen::VectorXf &shot);
#     protected:
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeature (pcl::PointCloud<PointOutT> &output);
#   };
# 
#   /** \brief SHOTEstimation estimates the Signature of Histograms of OrienTations (SHOT) descriptor for
#     * a given point cloud dataset containing points and normals.
#     *
#     * \note If you use this code in any academic work, please cite:
#     *
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     Unique Signatures of Histograms for Local Surface Description.
#     *     In Proceedings of the 11th European Conference on Computer Vision (ECCV),
#     *     Heraklion, Greece, September 5-11 2010.
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     A Combined Texture-Shape Descriptor For Enhanced 3D Feature Matching.
#     *     In Proceedings of the 18th International Conference on Image Processing (ICIP),
#     *     Brussels, Belgium, September 11-14 2011.
#     *
#     * \author Samuele Salti, Federico Tombari
#     * \ingroup features
#     */
#   template <typename PointInT, typename PointNT, typename PointRFT>
#   class PCL_DEPRECATED_CLASS (SHOTEstimation, "SHOTEstimation<..., pcl::SHOT, ...> IS DEPRECATED, USE SHOTEstimation<..., pcl::SHOT352, ...> INSTEAD")
#     <PointInT, PointNT, pcl::SHOT, PointRFT>
#     : public SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>
#   {
#     public:
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::feature_name_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::getClassName;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::indices_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::k_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::search_parameter_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::search_radius_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::surface_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::input_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::normals_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::descLength_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::nr_grid_sector_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::nr_shape_bins_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::sqradius_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::radius3_4_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::radius1_4_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::radius1_2_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::maxAngularSectors_;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::interpolateSingleChannel;
#       using SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>::shot_;
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
# 
#       typedef typename Feature<PointInT, pcl::SHOT>::PointCloudIn PointCloudIn;
# 
#       /** \brief Empty constructor.
#         * \param[in] nr_shape_bins the number of bins in the shape histogram
#         */
#       SHOTEstimation (int nr_shape_bins = 10) : SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT> (nr_shape_bins)
#       {
#         feature_name_ = "SHOTEstimation";
#       };
# 
#       /** \brief Estimate the SHOT descriptor for a given point based on its spatial neighborhood of 3D points with normals
#         * \param[in] index the index of the point in indices_
#         * \param[in] indices the k-neighborhood point indices in surface_
#         * \param[in] sqr_dists the k-neighborhood point distances in surface_
#         * \param[out] shot the resultant SHOT descriptor representing the feature at the query point
#         */
#       virtual void
#       computePointSHOT (const int index,
#                         const std::vector<int> &indices,
#                         const std::vector<float> &sqr_dists,
#                         Eigen::VectorXf &shot);
# 
#     protected:
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeature (pcl::PointCloud<pcl::SHOT> &output);
#   };
# 
#   /** \brief SHOTEstimation estimates the Signature of Histograms of OrienTations (SHOT) descriptor for
#     * a given point cloud dataset containing points and normals.
#     *
#     * \note If you use this code in any academic work, please cite:
#     *
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     Unique Signatures of Histograms for Local Surface Description.
#     *     In Proceedings of the 11th European Conference on Computer Vision (ECCV),
#     *     Heraklion, Greece, September 5-11 2010.
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     A Combined Texture-Shape Descriptor For Enhanced 3D Feature Matching.
#     *     In Proceedings of the 18th International Conference on Image Processing (ICIP),
#     *     Brussels, Belgium, September 11-14 2011.
#     *
#     * \author Samuele Salti, Federico Tombari
#     * \ingroup features
#     */
#   template <typename PointInT, typename PointNT, typename PointRFT>
#   class SHOTEstimation<PointInT, PointNT, Eigen::MatrixXf, PointRFT> : public SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>
#   {
#     public:
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::feature_name_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::getClassName;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::indices_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::k_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::search_parameter_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::search_radius_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::surface_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::input_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::normals_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::descLength_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::nr_grid_sector_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::nr_shape_bins_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::sqradius_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::radius3_4_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::radius1_4_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::radius1_2_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::maxAngularSectors_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::interpolateSingleChannel;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::shot_;
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
# 
#       /** \brief Empty constructor. */
#       SHOTEstimation (int nr_shape_bins = 10) : SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT> ()
#       {
#         feature_name_ = "SHOTEstimation";
#         nr_shape_bins_ = nr_shape_bins;
#       };
# 
#       /** \brief Base method for feature estimation for all points given in
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface ()
#         * and the spatial locator in setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset containing the estimated features
#         */
#       void
#       computeEigen (pcl::PointCloud<Eigen::MatrixXf> &output)
#       {
#         pcl::SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>::computeEigen (output);
#       }
# 
#       /** \brief Estimate the SHOT descriptor for a given point based on its spatial neighborhood of 3D points with normals
#         * \param[in] index the index of the point in indices_
#         * \param[in] indices the k-neighborhood point indices in surface_
#         * \param[in] sqr_dists the k-neighborhood point distances in surface_
#         * \param[out] shot the resultant SHOT descriptor representing the feature at the query point
#         */
#       //virtual void
#       //computePointSHOT (const int index,
#                         //const std::vector<int> &indices,
#                         //const std::vector<float> &sqr_dists,
#                         //Eigen::VectorXf &shot);
# 
#     protected:
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
# 	  
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       compute (pcl::PointCloud<pcl::SHOT352> &) { assert(0); }
#   };
# 
#   /** \brief SHOTColorEstimation estimates the Signature of Histograms of OrienTations (SHOT) descriptor for a given point cloud dataset
#     * containing points, normals and colors.
#     *
#     * \note If you use this code in any academic work, please cite:
#     *
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     Unique Signatures of Histograms for Local Surface Description.
#     *     In Proceedings of the 11th European Conference on Computer Vision (ECCV),
#     *     Heraklion, Greece, September 5-11 2010.
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     A Combined Texture-Shape Descriptor For Enhanced 3D Feature Matching.
#     *     In Proceedings of the 18th International Conference on Image Processing (ICIP),
#     *     Brussels, Belgium, September 11-14 2011.
#     *
#     * \author Samuele Salti, Federico Tombari
#     * \ingroup features
#     */
#   template <typename PointInT, typename PointNT, typename PointOutT = pcl::SHOT1344, typename PointRFT = pcl::ReferenceFrame>
#   class SHOTColorEstimation : public SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>
#   {
#     public:
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::feature_name_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::getClassName;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::indices_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::k_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::search_parameter_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::search_radius_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::surface_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::input_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::normals_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::descLength_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::nr_grid_sector_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::nr_shape_bins_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::sqradius_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::radius3_4_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::radius1_4_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::radius1_2_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::maxAngularSectors_;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::interpolateSingleChannel;
#       using SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::shot_;
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
# 
#       /** \brief Empty constructor.
#         * \param[in] describe_shape
#         * \param[in] describe_color
#         */
#       SHOTColorEstimation (bool describe_shape = true,
#                            bool describe_color = true)
#         : SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT> (10),
#           b_describe_shape_ (describe_shape),
#           b_describe_color_ (describe_color),
#           nr_color_bins_ (30)
#       {
#         feature_name_ = "SHOTColorEstimation";
#       };
# 
#       /** \brief Estimate the SHOT descriptor for a given point based on its spatial neighborhood of 3D points with normals
#         * \param[in] index the index of the point in indices_
#         * \param[in] indices the k-neighborhood point indices in surface_
#         * \param[in] sqr_dists the k-neighborhood point distances in surface_
#         * \param[out] shot the resultant SHOT descriptor representing the feature at the query point
#         */
#       virtual void
#       computePointSHOT (const int index,
#                         const std::vector<int> &indices,
#                         const std::vector<float> &sqr_dists,
#                         Eigen::VectorXf &shot);
#     protected:
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeature (pcl::PointCloud<PointOutT> &output);
# 
#       /** \brief Quadrilinear interpolation; used when color and shape descriptions are both activated
#         * \param[in] indices the neighborhood point indices
#         * \param[in] sqr_dists the neighborhood point distances
#         * \param[in] index the index of the point in indices_
#         * \param[out] binDistanceShape the resultant distance shape histogram
#         * \param[out] binDistanceColor the resultant color shape histogram
#         * \param[in] nr_bins_shape the number of bins in the shape histogram
#         * \param[in] nr_bins_color the number of bins in the color histogram
#         * \param[out] shot the resultant SHOT histogram
#         */
#       void
#       interpolateDoubleChannel (const std::vector<int> &indices,
#                                 const std::vector<float> &sqr_dists,
#                                 const int index,
#                                 std::vector<double> &binDistanceShape,
#                                 std::vector<double> &binDistanceColor,
#                                 const int nr_bins_shape,
#                                 const int nr_bins_color,
#                                 Eigen::VectorXf &shot);
# 
#       /** \brief Compute shape descriptor. */
#       bool b_describe_shape_;
# 
#       /** \brief Compute color descriptor. */
#       bool b_describe_color_;
# 
#       /** \brief The number of bins in each color histogram. */
#       int nr_color_bins_;
# 
#     public:
#       /** \brief Converts RGB triplets to CIELab space.
#         * \param[in] R the red channel
#         * \param[in] G the green channel
#         * \param[in] B the blue channel
#         * \param[out] L the lightness
#         * \param[out] A the first color-opponent dimension
#         * \param[out] B2 the second color-opponent dimension
#         */
#       static void
#       RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A, float &B2);
# 
#       static float sRGB_LUT[256];
#       static float sXYZ_LUT[4000];
#   };
# 
#   template <typename PointInT, typename PointNT, typename PointRFT>
#   class SHOTColorEstimation<PointInT, PointNT, Eigen::MatrixXf, PointRFT> : public SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>
#   {
#     public:
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::feature_name_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::getClassName;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::indices_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::k_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::search_parameter_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::search_radius_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::surface_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::input_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::normals_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::descLength_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::nr_grid_sector_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::nr_shape_bins_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::sqradius_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::radius3_4_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::radius1_4_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::radius1_2_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::maxAngularSectors_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::interpolateSingleChannel;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::shot_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::b_describe_shape_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::b_describe_color_;
#       using SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::nr_color_bins_;
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
# 
#       /** \brief Empty constructor.
#         * \param[in] describe_shape
#         * \param[in] describe_color
#         */
#       SHOTColorEstimation (bool describe_shape = true,
#                            bool describe_color = true,
#                            int nr_shape_bins = 10,
#                            int nr_color_bins = 30)
#         : SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT> (describe_shape, describe_color)
#       {
#         feature_name_ = "SHOTColorEstimation";
#         nr_shape_bins_ = nr_shape_bins;
#         nr_color_bins_ = nr_color_bins;
#       };
# 
#       /** \brief Base method for feature estimation for all points given in
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface ()
#         * and the spatial locator in setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset containing the estimated features
#         */
#       void
#       computeEigen (pcl::PointCloud<Eigen::MatrixXf> &output)
#       {
#         pcl::SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>::computeEigen (output);
#       }
# 
#     protected:
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
# 	  
# 	  /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       compute (pcl::PointCloud<pcl::SHOT1344> &) { assert(0); }
#   };
# 
#   /** \brief SHOTEstimation estimates the Signature of Histograms of OrienTations (SHOT) descriptor for a given point cloud dataset
#     * containing points and normals.
#     *
#     * \note If you use this code in any academic work, please cite:
#     *
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     Unique Signatures of Histograms for Local Surface Description.
#     *     In Proceedings of the 11th European Conference on Computer Vision (ECCV),
#     *     Heraklion, Greece, September 5-11 2010.
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     A Combined Texture-Shape Descriptor For Enhanced 3D Feature Matching.
#     *     In Proceedings of the 18th International Conference on Image Processing (ICIP),
#     *     Brussels, Belgium, September 11-14 2011.
#     *
#     * \author Samuele Salti, Federico Tombari
#     * \ingroup features
#     */
#   template <typename PointNT, typename PointRFT>
#   class PCL_DEPRECATED_CLASS (SHOTEstimation, "SHOTEstimation<pcl::PointXYZRGBA,...,pcl::SHOT,...> IS DEPRECATED, USE SHOTEstimation<pcl::PointXYZRGBA,...,pcl::SHOT352,...> FOR SHAPE AND SHOTColorEstimation<pcl::PointXYZRGBA,...,pcl::SHOT1344,...> FOR SHAPE+COLOR INSTEAD")
#     <pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>
#     : public SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT> 
#   {
#     public:
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::feature_name_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::indices_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::k_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::search_parameter_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::search_radius_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::surface_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::input_;
#       using FeatureFromNormals<pcl::PointXYZRGBA, PointNT, pcl::SHOT>::normals_;
#       using FeatureWithLocalReferenceFrames<pcl::PointXYZRGBA, PointRFT>::frames_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::getClassName;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::descLength_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::nr_grid_sector_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::nr_shape_bins_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::sqradius_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::radius3_4_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::radius1_4_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::radius1_2_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::maxAngularSectors_;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::interpolateSingleChannel;
#       using SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::shot_;
# 
#       typedef typename Feature<pcl::PointXYZRGBA, pcl::SHOT>::PointCloudOut PointCloudOut;
#       typedef typename Feature<pcl::PointXYZRGBA, pcl::SHOT>::PointCloudIn PointCloudIn;
# 
#       /** \brief Empty constructor.
#         * \param[in] describe_shape
#         * \param[in] describe_color
#         * \param[in] nr_shape_bins
#         * \param[in] nr_color_bins
#         */
#       SHOTEstimation (bool describe_shape = true,
#                       bool describe_color = false,
#                       const int nr_shape_bins = 10,
#                       const int nr_color_bins = 30)
#         : SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT> (nr_shape_bins),
#           b_describe_shape_ (describe_shape),
#           b_describe_color_ (describe_color),
#           nr_color_bins_ (nr_color_bins)
#       {
#         feature_name_ = "SHOTEstimation";
#       };
# 
#       /** \brief Estimate the SHOT descriptor for a given point based on its spatial neighborhood of 3D points with normals
#         * \param[in] index the index of the point in indices_
#         * \param[in] indices the k-neighborhood point indices in surface_
#         * \param[in] sqr_dists the k-neighborhood point distances in surface_
#         * \param[out] shot the resultant SHOT descriptor representing the feature at the query point
#         */
#       virtual void
#       computePointSHOT (const int index,
#                         const std::vector<int> &indices,
#                         const std::vector<float> &sqr_dists,
#                         Eigen::VectorXf &shot);
# 
#     protected:
# 
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief Quadrilinear interpolation; used when color and shape descriptions are both activated
#         * \param[in] indices the neighborhood point indices
#         * \param[in] sqr_dists the neighborhood point distances
#         * \param[in] index the index of the point in indices_
#         * \param[out] binDistanceShape the resultant distance shape histogram
#         * \param[out] binDistanceColor the resultant color shape histogram
#         * \param[in] nr_bins_shape the number of bins in the shape histogram
#         * \param[in] nr_bins_color the number of bins in the color histogram
#         * \param[out] shot the resultant SHOT histogram
#         */
#       void
#       interpolateDoubleChannel (const std::vector<int> &indices,
#                                 const std::vector<float> &sqr_dists,
#                                 const int index,
#                                 std::vector<double> &binDistanceShape,
#                                 std::vector<double> &binDistanceColor,
#                                 const int nr_bins_shape,
#                                 const int nr_bins_color,
#                                 Eigen::VectorXf &shot);
# 
#       /** \brief Converts RGB triplets to CIELab space.
#         * \param[in] R the red channel
#         * \param[in] G the green channel
#         * \param[in] B the blue channel
#         * \param[out] L the lightness
#         * \param[out] A the first color-opponent dimension
#         * \param[out] B2 the second color-opponent dimension
#         */
#       static void
#       RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A, float &B2);
# 
#       /** \brief Compute shape descriptor. */
#       bool b_describe_shape_;
# 
#       /** \brief Compute color descriptor. */
#       bool b_describe_color_;
# 
#       /** \brief The number of bins in each color histogram. */
#       int nr_color_bins_;
# 
#     public:
#       static float sRGB_LUT[256];
#       static float sXYZ_LUT[4000];
#   };
# 
#   /** \brief SHOTEstimation estimates the Signature of Histograms of OrienTations (SHOT) descriptor for a given point cloud dataset
#     * containing points and normals.
#     *
#     * \note If you use this code in any academic work, please cite:
#     *
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     Unique Signatures of Histograms for Local Surface Description.
#     *     In Proceedings of the 11th European Conference on Computer Vision (ECCV),
#     *     Heraklion, Greece, September 5-11 2010.
#     *   - F. Tombari, S. Salti, L. Di Stefano
#     *     A Combined Texture-Shape Descriptor For Enhanced 3D Feature Matching.
#     *     In Proceedings of the 18th International Conference on Image Processing (ICIP),
#     *     Brussels, Belgium, September 11-14 2011.
#     *
#     * \author Samuele Salti, Federico Tombari
#     * \ingroup features
#     */
#   template <typename PointNT, typename PointRFT>
#   class PCL_DEPRECATED_CLASS (SHOTEstimation, "SHOTEstimation<pcl::PointXYZRGBA,...,Eigen::MatrixXf,...> IS DEPRECATED, USE SHOTColorEstimation<pcl::PointXYZRGBA,...,Eigen::MatrixXf,...> FOR SHAPE AND SHAPE+COLOR INSTEAD")
#     <pcl::PointXYZRGBA, PointNT, Eigen::MatrixXf, PointRFT>
#     : public SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>
#   {
#     public:
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::feature_name_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::getClassName;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::indices_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::k_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::search_parameter_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::search_radius_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::surface_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::input_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::descLength_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::nr_grid_sector_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::nr_shape_bins_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::sqradius_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::radius3_4_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::radius1_4_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::radius1_2_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::maxAngularSectors_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::interpolateSingleChannel;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::shot_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::b_describe_shape_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::b_describe_color_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::nr_color_bins_;
#       using FeatureWithLocalReferenceFrames<pcl::PointXYZRGBA, PointRFT>::frames_;
# 
#       /** \brief Empty constructor.
#         * \param[in] describe_shape
#         * \param[in] describe_color
#         * \param[in] nr_shape_bins
#         * \param[in] nr_color_bins
#         */
#       SHOTEstimation (bool describe_shape = true,
#                       bool describe_color = false,
#                       const int nr_shape_bins = 10,
#                       const int nr_color_bins = 30)
#         : SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT> (describe_shape, describe_color, nr_shape_bins, nr_color_bins) {};
# 
#    protected:
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
# 	  
# 	  /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       compute (pcl::PointCloud<pcl::SHOT> &) { assert(0); }
#   };
# 
# ###
#     # shot_lrf.h
#   template <typename PointInT, typename PointNT, typename PointOutT = pcl::PrincipalCurvatures>
#   class PrincipalCurvaturesEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::input_;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#       typedef pcl::PointCloud<PointInT> PointCloudIn;
# 
#       /** \brief Empty constructor. */
#       PrincipalCurvaturesEstimation () : 
#         projected_normals_ (), 
#         xyz_centroid_ (Eigen::Vector3f::Zero ()), 
#         demean_ (Eigen::Vector3f::Zero ()),
#         covariance_matrix_ (Eigen::Matrix3f::Zero ()),
#         eigenvector_ (Eigen::Vector3f::Zero ()),
#         eigenvalues_ (Eigen::Vector3f::Zero ())
#       {
#         feature_name_ = "PrincipalCurvaturesEstimation";
#       };
# 
#       /** \brief Perform Principal Components Analysis (PCA) on the point normals of a surface patch in the tangent
#        *  plane of the given point normal, and return the principal curvature (eigenvector of the max eigenvalue),
#        *  along with both the max (pc1) and min (pc2) eigenvalues
#        * \param[in] normals the point cloud normals
#        * \param[in] p_idx the query point at which the least-squares plane was estimated
#        * \param[in] indices the point cloud indices that need to be used
#        * \param[out] pcx the principal curvature X direction
#        * \param[out] pcy the principal curvature Y direction
#        * \param[out] pcz the principal curvature Z direction
#        * \param[out] pc1 the max eigenvalue of curvature
#        * \param[out] pc2 the min eigenvalue of curvature
#        */
#       void
#       computePointPrincipalCurvatures (const pcl::PointCloud<PointNT> &normals,
#                                        int p_idx, const std::vector<int> &indices,
#                                        float &pcx, float &pcy, float &pcz, float &pc1, float &pc2);
# 
#     protected:
# 
#       /** \brief Estimate the principal curvature (eigenvector of the max eigenvalue), along with both the max (pc1)
#         * and min (pc2) eigenvalues for all points given in <setInputCloud (), setIndices ()> using the surface in
#         * setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the principal curvature estimates
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#     private:
#       /** \brief A pointer to the input dataset that contains the point normals of the XYZ dataset. */
#       std::vector<Eigen::Vector3f> projected_normals_;
# 
#       /** \brief SSE aligned placeholder for the XYZ centroid of a surface patch. */
#       Eigen::Vector3f xyz_centroid_;
# 
#       /** \brief Temporary point placeholder. */
#       Eigen::Vector3f demean_;
# 
#       /** \brief Placeholder for the 3x3 covariance matrix at each surface patch. */
#       EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix_;
# 
#       /** \brief SSE aligned eigenvectors placeholder for a covariance matrix. */
#       Eigen::Vector3f eigenvector_;
#       /** \brief eigenvalues placeholder for a covariance matrix. */
#       Eigen::Vector3f eigenvalues_;
# 
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   /** \brief PrincipalCurvaturesEstimation estimates the directions (eigenvectors) and magnitudes (eigenvalues) of
#     * principal surface curvatures for a given point cloud dataset containing points and normals.
#     *
#     * \note The code is stateful as we do not expect this class to be multicore parallelized. Please look at
#     * \ref NormalEstimationOMP for an example on how to extend this to parallel implementations.
#     *
#     * \author Radu B. Rusu, Jared Glover
#     * \ingroup features
#     */
#   template <typename PointInT, typename PointNT>
#   class PrincipalCurvaturesEstimation<PointInT, PointNT, Eigen::MatrixXf> : public PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>
#   {
#     public:
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::indices_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::k_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::search_parameter_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::surface_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::compute;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::input_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::normals_;
# 
#     private:
#       /** \brief Estimate the principal curvature (eigenvector of the max eigenvalue), along with both the max (pc1)
#         * and min (pc2) eigenvalues for all points given in <setInputCloud (), setIndices ()> using the surface in
#         * setSearchSurface () and the spatial locator in setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the principal curvature estimates
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       compute (pcl::PointCloud<pcl::Normal> &) {}
#   };
# 
# ###
#     # shot_lrf_omp.h
#   template<typename PointInT, typename PointOutT = ReferenceFrame>
#   class SHOTLocalReferenceFrameEstimationOMP : public SHOTLocalReferenceFrameEstimation<PointInT, PointOutT>
#   {
#     public:
#       /** \brief Constructor */
#     SHOTLocalReferenceFrameEstimationOMP ()
#       {
#         feature_name_ = "SHOTLocalReferenceFrameEstimationOMP";
#         threads_ = 1;
#       }
# 
#     /** \brief Initialize the scheduler and set the number of threads to use.
#      * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#      */
#      inline void
#      setNumberOfThreads (unsigned int nr_threads)
#      {
#        if (nr_threads == 0)
#          nr_threads = 1;
#        threads_ = nr_threads;
#      }
# 
#     protected:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       //using Feature<PointInT, PointOutT>::searchForNeighbors;
#       using Feature<PointInT, PointOutT>::input_;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::tree_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       using SHOTLocalReferenceFrameEstimation<PointInT, PointOutT>::getLocalRF;
#       typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#       /** \brief Feature estimation method.
#         * \param[out] output the resultant features
#         */
#       virtual void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief Feature estimation method.
#         * \param[out] output the resultant features
#         */
#       virtual void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#       /** \brief The number of threads the scheduler should use. */
#       int threads_;
# 
# ###
#     # shot_omp.h
#   template <typename PointInT, typename PointNT, typename PointOutT = pcl::SHOT352, typename PointRFT = pcl::ReferenceFrame>
#   class SHOTEstimationOMP : public SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::input_;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       using Feature<PointInT, PointOutT>::search_radius_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::fake_surface_;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
#       using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::descLength_;
#       using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::nr_grid_sector_;
#       using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::nr_shape_bins_;
#       using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::sqradius_;
#       using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::radius3_4_;
#       using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::radius1_4_;
#       using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::radius1_2_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#       typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
# 
#       /** \brief Empty constructor. */
#       SHOTEstimationOMP (unsigned int nr_threads = - 1) : SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT> (), threads_ ()
#       {
#         setNumberOfThreads (nr_threads);
#       }
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       inline void
#       setNumberOfThreads (unsigned int nr_threads)
#       {
#         if (nr_threads == 0)
#           nr_threads = 1;
#         threads_ = nr_threads;
#       }
# 
#     protected:
# 
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief This method should get called before starting the actual computation. */
#       bool
#       initCompute ();
# 
#       /** \brief The number of threads the scheduler should use. */
#       int threads_;
#   };
# 
#   template <typename PointInT, typename PointNT, typename PointOutT = pcl::SHOT1344, typename PointRFT = pcl::ReferenceFrame>
#   class SHOTColorEstimationOMP : public SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::input_;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       using Feature<PointInT, PointOutT>::search_radius_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::fake_surface_;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
#       using SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>::descLength_;
#       using SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>::nr_grid_sector_;
#       using SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>::nr_shape_bins_;
#       using SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>::sqradius_;
#       using SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>::radius3_4_;
#       using SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>::radius1_4_;
#       using SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>::radius1_2_;
#       using SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>::b_describe_shape_;
#       using SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>::b_describe_color_;
#       using SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>::nr_color_bins_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#       typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
# 
#       /** \brief Empty constructor. */
#       SHOTColorEstimationOMP (bool describe_shape = true,
#                               bool describe_color = true,
#                               unsigned int nr_threads = - 1)
#         : SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT> (describe_shape, describe_color), threads_ ()
#       {
#         setNumberOfThreads (nr_threads);
#       }
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       inline void
#       setNumberOfThreads (unsigned int nr_threads)
#       {
#         if (nr_threads == 0)
#           nr_threads = 1;
#         threads_ = nr_threads;
#       }
# 
#     protected:
# 
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief This method should get called before starting the actual computation. */
#       bool
#       initCompute ();
# 
#       /** \brief The number of threads the scheduler should use. */
#       int threads_;
#   };
# 
#   template <typename PointInT, typename PointNT, typename PointRFT>
#   class PCL_DEPRECATED_CLASS (SHOTEstimationOMP, "SHOTEstimationOMP<..., pcl::SHOT, ...> IS DEPRECATED, USE SHOTEstimationOMP<..., pcl::SHOT352, ...> INSTEAD")
#     <PointInT, PointNT, pcl::SHOT, PointRFT>
#     : public SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT>
#   {
#     public:
#       using Feature<PointInT, pcl::SHOT>::feature_name_;
#       using Feature<PointInT, pcl::SHOT>::getClassName;
#       using Feature<PointInT, pcl::SHOT>::input_;
#       using Feature<PointInT, pcl::SHOT>::indices_;
#       using Feature<PointInT, pcl::SHOT>::k_;
#       using Feature<PointInT, pcl::SHOT>::search_parameter_;
#       using Feature<PointInT, pcl::SHOT>::search_radius_;
#       using Feature<PointInT, pcl::SHOT>::surface_;
#       using Feature<PointInT, pcl::SHOT>::fake_surface_;
#       using FeatureFromNormals<PointInT, PointNT, pcl::SHOT>::normals_;
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT>::descLength_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT>::nr_grid_sector_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT>::nr_shape_bins_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT>::sqradius_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT>::radius3_4_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT>::radius1_4_;
#       using SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT>::radius1_2_;
# 
#       typedef typename Feature<PointInT, pcl::SHOT>::PointCloudOut PointCloudOut;
#       typedef typename Feature<PointInT, pcl::SHOT>::PointCloudIn PointCloudIn;
# 
#       /** \brief Empty constructor. */
#       SHOTEstimationOMP (unsigned int nr_threads = - 1, int nr_shape_bins = 10)
#         : SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT> (nr_shape_bins), threads_ ()
#       {
#         setNumberOfThreads (nr_threads);
#       }
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       inline void
#       setNumberOfThreads (unsigned int nr_threads)
#       {
#         if (nr_threads == 0)
#           nr_threads = 1;
#         threads_ = nr_threads;
#       }
# 
#     protected:
# 
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief This method should get called before starting the actual computation. */
#       bool
#       initCompute ();
# 
#       /** \brief The number of threads the scheduler should use. */
#       int threads_;
#   };
# 
#   template <typename PointNT, typename PointRFT>
#   class PCL_DEPRECATED_CLASS (SHOTEstimationOMP, "SHOTEstimationOMP<pcl::PointXYZRGBA,...,pcl::SHOT,...> IS DEPRECATED, USE SHOTEstimationOMP<pcl::PointXYZRGBA,...,pcl::SHOT352,...> FOR SHAPE AND SHOTColorEstimationOMP<pcl::PointXYZRGBA,...,pcl::SHOT1344,...> FOR SHAPE+COLOR INSTEAD")
#     <pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>
#     : public SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>
#   {
#     public:
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::feature_name_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::getClassName;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::input_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::indices_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::k_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::search_parameter_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::search_radius_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::surface_;
#       using FeatureFromNormals<pcl::PointXYZRGBA, PointNT, pcl::SHOT>::normals_;
#       using FeatureWithLocalReferenceFrames<pcl::PointXYZRGBA, PointRFT>::frames_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::descLength_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::nr_grid_sector_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::nr_shape_bins_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::sqradius_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::radius3_4_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::radius1_4_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::radius1_2_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::b_describe_shape_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::b_describe_color_;
#       using SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>::nr_color_bins_;
# 
#       typedef typename Feature<pcl::PointXYZRGBA, pcl::SHOT>::PointCloudOut PointCloudOut;
#       typedef typename Feature<pcl::PointXYZRGBA, pcl::SHOT>::PointCloudIn PointCloudIn;
# 
#       /** \brief Empty constructor. */
#       SHOTEstimationOMP (bool describeShape = true,
#                          bool describeColor = false,
#                          unsigned int nr_threads = - 1,
#                          const int nr_shape_bins = 10,
#                          const int nr_color_bins = 30)
#         : SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT> (describeShape, describeColor, nr_shape_bins, nr_color_bins),
#           threads_ ()
#       {
#         setNumberOfThreads (nr_threads);
#       }
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       inline void
#       setNumberOfThreads (unsigned int nr_threads)
#       {
#         if (nr_threads == 0)
#           nr_threads = 1;
#         threads_ = nr_threads;
#       }
# 
#     private:
# 
#       /** \brief Estimate the Signatures of Histograms of OrienTations (SHOT) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param output the resultant point cloud model dataset that contains the SHOT feature estimates
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief The number of threads the scheduler should use. */
#       int threads_;
# 
# ###
#     # spin_image.h
#   template <typename PointInT, typename PointNT, typename PointOutT>
#   class SpinImageEstimation : public Feature<PointInT, PointOutT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::search_radius_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::fake_surface_;
#       using PCLBase<PointInT>::input_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
# 
#       typedef typename pcl::PointCloud<PointNT> PointCloudN;
#       typedef typename PointCloudN::Ptr PointCloudNPtr;
#       typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
# 
#       typedef typename pcl::PointCloud<PointInT> PointCloudIn;
#       typedef typename PointCloudIn::Ptr PointCloudInPtr;
#       typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
#       
#       typedef typename boost::shared_ptr<SpinImageEstimation<PointInT, PointNT, PointOutT> > Ptr;
#       typedef typename boost::shared_ptr<const SpinImageEstimation<PointInT, PointNT, PointOutT> > ConstPtr;
# 
#       /** \brief Constructs empty spin image estimator.
#         * 
#         * \param[in] image_width spin-image resolution, number of bins along one dimension
#         * \param[in] support_angle_cos minimal allowed cosine of the angle between 
#         *   the normals of input point and search surface point for the point 
#         *   to be retained in the support
#         * \param[in] min_pts_neighb min number of points in the support to correctly estimate 
#         *   spin-image. If at some point the support contains less points, exception is thrown
#         */
#       SpinImageEstimation (unsigned int image_width = 8,
#                            double support_angle_cos = 0.0,   // when 0, this is bogus, so not applied
#                            unsigned int min_pts_neighb = 0);
# 
#       /** \brief Sets spin-image resolution.
#         * 
#         * \param[in] bin_count spin-image resolution, number of bins along one dimension
#         */
#       void 
#       setImageWidth (unsigned int bin_count)
#       {
#         image_width_ = bin_count;
#       }
# 
#       /** \brief Sets the maximum angle for the point normal to get to support region.
#         * 
#         * \param[in] support_angle_cos minimal allowed cosine of the angle between 
#         *   the normals of input point and search surface point for the point 
#         *   to be retained in the support
#         */
#       void 
#       setSupportAngle (double support_angle_cos)
#       {
#         if (0.0 > support_angle_cos || support_angle_cos > 1.0)  // may be permit negative cosine?
#         {
#           throw PCLException ("Cosine of support angle should be between 0 and 1",
#             "spin_image.h", "setSupportAngle");
#         }
# 
#         support_angle_cos_ = support_angle_cos;
#       }
# 
#       /** \brief Sets minimal points count for spin image computation.
#         *
#         * \param[in] min_pts_neighb min number of points in the support to correctly estimate 
#         *   spin-image. If at some point the support contains less points, exception is thrown
#         */
#       void 
#       setMinPointCountInNeighbourhood (unsigned int min_pts_neighb)
#       {
#         min_pts_neighb_ = min_pts_neighb;
#       }
# 
#       /** \brief Provide a pointer to the input dataset that contains the point normals of 
#         * the input XYZ dataset given by \ref setInputCloud
#         * 
#         * \attention The input normals given by \ref setInputNormals have to match
#         * the input point cloud given by \ref setInputCloud. This behavior is
#         * different than feature estimation methods that extend \ref
#         * FeatureFromNormals, which match the normals with the search surface.
#         * \param[in] normals the const boost shared pointer to a PointCloud of normals. 
#         * By convention, L2 norm of each normal should be 1. 
#         */
#       inline void 
#       setInputNormals (const PointCloudNConstPtr &normals)
#       { 
#         input_normals_ = normals; 
#       }
# 
#       /** \brief Sets single vector a rotation axis for all input points.
#         * 
#         * It could be useful e.g. when the vertical axis is known.
#         * \param[in] axis unit-length vector that serves as rotation axis for reference frame
#         */
#       void 
#       setRotationAxis (const PointNT& axis)
#       {
#         rotation_axis_ = axis;
#         use_custom_axis_ = true;
#         use_custom_axes_cloud_ = false;
#       }
# 
#       /** \brief Sets array of vectors as rotation axes for input points.
#         * 
#         * Useful e.g. when one wants to use tangents instead of normals as rotation axes
#         * \param[in] axes unit-length vectors that serves as rotation axes for 
#         *   the corresponding input points' reference frames
#         */
#       void 
#       setInputRotationAxes (const PointCloudNConstPtr& axes)
#       {
#         rotation_axes_cloud_ = axes;
# 
#         use_custom_axes_cloud_ = true;
#         use_custom_axis_ = false;
#       }
# 
#       /** \brief Sets input normals as rotation axes (default setting). */
#       void 
#       useNormalsAsRotationAxis () 
#       { 
#         use_custom_axis_ = false; 
#         use_custom_axes_cloud_ = false;
#       }
# 
#       /** \brief Sets/unsets flag for angular spin-image domain.
#         * 
#         * Angular spin-image differs from the vanilla one in the way that not 
#         * the points are collected in the bins but the angles between their
#         * normals and the normal to the reference point. For further
#         * information please see 
#         * Endres, F., Plagemann, C., Stachniss, C., & Burgard, W. (2009). 
#         * Unsupervised Discovery of Object Classes from Range Data using Latent Dirichlet Allocation. 
#         * In Robotics: Science and Systems. Seattle, USA.
#         * \param[in] is_angular true for angular domain, false for point domain
#         */
#       void 
#       setAngularDomain (bool is_angular = true) { is_angular_ = is_angular; }
# 
#       /** \brief Sets/unsets flag for radial spin-image structure.
#         * 
#         * Instead of rectangular coordinate system for reference frame 
#         * polar coordinates are used. Binning is done depending on the distance and 
#         * inclination angle from the reference point
#         * \param[in] is_radial true for radial spin-image structure, false for rectangular
#         */
#       void 
#       setRadialStructure (bool is_radial = true) { is_radial_ = is_radial; }
# 
#     protected:
#       /** \brief Estimate the Spin Image descriptors at a set of points given by
#         * setInputWithNormals() using the surface in setSearchSurfaceWithNormals() and the spatial locator 
#         * \param[out] output the resultant point cloud that contains the Spin Image feature estimates
#         */
#       virtual void 
#       computeFeature (PointCloudOut &output); 
# 
#       /** \brief initializes computations specific to spin-image.
#         * 
#         * \return true iff input data and initialization are correct
#         */
#       virtual bool
#       initCompute ();
# 
#       /** \brief Computes a spin-image for the point of the scan. 
#         * \param[in] index the index of the reference point in the input cloud
#         * \return estimated spin-image (or its variant) as a matrix
#         */
#       Eigen::ArrayXXd 
#       computeSiForPoint (int index) const;
# 
#     private:
#       PointCloudNConstPtr input_normals_;
#       PointCloudNConstPtr rotation_axes_cloud_;
#       
#       bool is_angular_;
# 
#       PointNT rotation_axis_;
#       bool use_custom_axis_;
#       bool use_custom_axes_cloud_;
# 
#       bool is_radial_;
# 
#       unsigned int image_width_;
#       double support_angle_cos_;
#       unsigned int min_pts_neighb_;
# 
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   /** \brief Estimates spin-image descriptors in the  given input points. 
#     *  
#     *  This class represents spin image descriptor. Spin image is
#     *  a histogram of point locations summed along the bins of the image.
#     *  A 2D accumulator indexed by <VAR>a</VAR> and <VAR>b</VAR> is created. Next, 
#     *  the coordinates (<VAR>a</VAR>, <VAR>b</VAR>) are computed for a vertex in 
#     *  the surface mesh that is within the support of the spin image 
#     *  (explained below). The bin indexed by (<VAR>a</VAR>, <VAR>b</VAR>) in 
#     *  the accumulator is then incremented; bilinear interpolation is used 
#     *  to smooth the contribution of the vertex. This procedure is repeated 
#     *  for all vertices within the support of the spin image. 
#     *  The resulting accumulator can be thought of as an image; 
#     *  dark areas in the image correspond to bins that contain many projected points. 
#     *  As long as the size of the bins in the accumulator is greater 
#     *  than the median distance between vertices in the mesh 
#     *  (the definition of mesh resolution), the position of individual 
#     *  vertices will be averaged out during spin image generation.
#     *  
#     * For further information please see:
#     *
#     *  - Johnson, A. E., & Hebert, M. (1998). Surface Matching for Object
#     *    Recognition in Complex 3D Scenes. Image and Vision Computing, 16,
#     *    635-651.
#     *  
#     *  The class also implements radial spin images and spin-images in angular domain 
#     *  (or both).
#     *  
#     *  \author Roman Shapovalov, Alexander Velizhev
#     *  \ingroup features
#     */
#   template <typename PointInT, typename PointNT>
#   class SpinImageEstimation<PointInT, PointNT, Eigen::MatrixXf> : public SpinImageEstimation<PointInT, PointNT, pcl::Histogram<153> >
#   {
#     public:
#       using SpinImageEstimation<PointInT, PointNT, pcl::Histogram<153> >::indices_;
#       using SpinImageEstimation<PointInT, PointNT, pcl::Histogram<153> >::search_radius_;
#       using SpinImageEstimation<PointInT, PointNT, pcl::Histogram<153> >::k_;
#       using SpinImageEstimation<PointInT, PointNT, pcl::Histogram<153> >::surface_;
#       using SpinImageEstimation<PointInT, PointNT, pcl::Histogram<153> >::fake_surface_;
#       using SpinImageEstimation<PointInT, PointNT, pcl::Histogram<153> >::compute;
# 
#       /** \brief Constructs empty spin image estimator.
#         * 
#         * \param[in] image_width spin-image resolution, number of bins along one dimension
#         * \param[in] support_angle_cos minimal allowed cosine of the angle between 
#         *   the normals of input point and search surface point for the point 
#         *   to be retained in the support
#         * \param[in] min_pts_neighb min number of points in the support to correctly estimate 
#         *   spin-image. If at some point the support contains less points, exception is thrown
#         */
#       SpinImageEstimation (unsigned int image_width = 8,
#                            double support_angle_cos = 0.0,   // when 0, this is bogus, so not applied
#                            unsigned int min_pts_neighb = 0) : 
#         SpinImageEstimation<PointInT, PointNT, pcl::Histogram<153> > (image_width, support_angle_cos, min_pts_neighb) {}
# 
#     private:
#       /** \brief Estimate the Spin Image descriptors at a set of points given by
#         * setInputWithNormals() using the surface in setSearchSurfaceWithNormals() and the spatial locator 
#         * \param[out] output the resultant point cloud that contains the Spin Image feature estimates
#         */
#       virtual void 
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output); 
# 
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud 
#         */
#       void 
#       compute (pcl::PointCloud<pcl::Normal> &) {}
#   };
# 
# ###
#     # statistical_multiscale_interest_region_extraction.h
#   template <typename PointT>
#   class StatisticalMultiscaleInterestRegionExtraction : public PCLBase<PointT>
#   {
#     public:
#       typedef boost::shared_ptr <std::vector<int> > IndicesPtr;
#       typedef typename boost::shared_ptr<StatisticalMultiscaleInterestRegionExtraction<PointT> > Ptr;
#       typedef typename boost::shared_ptr<const StatisticalMultiscaleInterestRegionExtraction<PointT> > ConstPtr;
# 
# 
#       /** \brief Empty constructor */
#       StatisticalMultiscaleInterestRegionExtraction () :
#         scale_values_ (), geodesic_distances_ (), F_scales_ ()
#       {};
# 
#       /** \brief Method that generates the underlying nearest neighbor graph based on the
#        * input point cloud
#        */
#       void
#       generateCloudGraph ();
# 
#       /** \brief The method to be called in order to run the algorithm and produce the resulting
#        * set of regions of interest
#        */
#       void
#       computeRegionsOfInterest (std::list<IndicesPtr>& rois);
# 
#       /** \brief Method for setting the scale parameters for the algorithm
#        * \param scale_values vector of scales to determine the size of each scaling step
#        */
#       inline void
#       setScalesVector (std::vector<float> &scale_values) { scale_values_ = scale_values; }
# 
#       /** \brief Method for getting the scale parameters vector */
#       inline std::vector<float>
#       getScalesVector () { return scale_values_; }
# 
# 
#     private:
#       /** \brief Checks if all the necessary input was given and the computations can successfully start */
#       bool
#       initCompute ();
# 
#       void
#       geodesicFixedRadiusSearch (size_t &query_index,
#                                  float &radius,
#                                  std::vector<int> &result_indices);
# 
#       void
#       computeF ();
# 
#       void
#       extractExtrema (std::list<IndicesPtr>& rois);
# 
#       using PCLBase<PointT>::initCompute;
#       using PCLBase<PointT>::input_;
#       std::vector<float> scale_values_;
#       std::vector<std::vector<float> > geodesic_distances_;
#       std::vector<std::vector<float> > F_scales_;
# 
# ###
#     # usc.h
#   template <typename PointInT, typename PointOutT, typename PointRFT = pcl::ReferenceFrame>
#   class UniqueShapeContext : public Feature<PointInT, PointOutT>,
#                              public FeatureWithLocalReferenceFrames<PointInT, PointRFT>
#   {
#     public:
#        using Feature<PointInT, PointOutT>::feature_name_;
#        using Feature<PointInT, PointOutT>::getClassName;
#        using Feature<PointInT, PointOutT>::indices_;
#        using Feature<PointInT, PointOutT>::search_parameter_;
#        using Feature<PointInT, PointOutT>::search_radius_;
#        using Feature<PointInT, PointOutT>::surface_;
#        using Feature<PointInT, PointOutT>::fake_surface_;
#        using Feature<PointInT, PointOutT>::input_;
#        using Feature<PointInT, PointOutT>::searchForNeighbors;
#        using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
# 
#        typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#        typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
#        typedef typename boost::shared_ptr<UniqueShapeContext<PointInT, PointOutT, PointRFT> > Ptr;
#        typedef typename boost::shared_ptr<const UniqueShapeContext<PointInT, PointOutT, PointRFT> > ConstPtr;
# 
# 
#        /** \brief Constructor. */
#        UniqueShapeContext () :
#          radii_interval_(0), theta_divisions_(0), phi_divisions_(0), volume_lut_(0),
#          azimuth_bins_(12), elevation_bins_(11), radius_bins_(15),
#          min_radius_(0.1), point_density_radius_(0.2), descriptor_length_ (), local_radius_ (2.5)
#        {
#          feature_name_ = "UniqueShapeContext";
#          search_radius_ = 2.5;
#        }
# 
#       virtual ~UniqueShapeContext() { }
# 
#       /** \brief Set the number of bins along the azimuth
#         * \param[in] bins the number of bins along the azimuth
#         */
#       inline void
#       setAzimuthBins (size_t bins) { azimuth_bins_ = bins; }
# 
#       /** \return The number of bins along the azimuth. */
#       inline size_t
#       getAzimuthBins () const { return (azimuth_bins_); }
# 
#       /** \brief Set the number of bins along the elevation
#         * \param[in] bins the number of bins along the elevation
#         */
#       inline void
#       setElevationBins (size_t bins) { elevation_bins_ = bins; }
# 
#       /** \return The number of bins along the elevation */
#       inline size_t
#       getElevationBins () const { return (elevation_bins_); }
# 
#       /** \brief Set the number of bins along the radii
#         * \param[in] bins the number of bins along the radii
#         */
#       inline void
#       setRadiusBins (size_t bins) { radius_bins_ = bins; }
# 
#       /** \return The number of bins along the radii direction. */
#       inline size_t
#       getRadiusBins () const { return (radius_bins_); }
# 
#       /** The minimal radius value for the search sphere (rmin) in the original paper
#         * \param[in] radius the desired minimal radius
#         */
#       inline void
#       setMinimalRadius (double radius) { min_radius_ = radius; }
# 
#       /** \return The minimal sphere radius. */
#       inline double
#       getMinimalRadius () const { return (min_radius_); }
# 
#       /** This radius is used to compute local point density
#         * density = number of points within this radius
#         * \param[in] radius Value of the point density search radius
#         */
#       inline void
#       setPointDensityRadius (double radius) { point_density_radius_ = radius; }
# 
#       /** \return The point density search radius. */
#       inline double
#       getPointDensityRadius () const { return (point_density_radius_); }
# 
#       /** Set the local RF radius value
#         * \param[in] radius the desired local RF radius
#         */
#       inline void
#       setLocalRadius (double radius) { local_radius_ = radius; }
# 
#       /** \return The local RF radius. */
#       inline double
#       getLocalRadius () const { return (local_radius_); }
# 
#     protected:
#       /** Compute 3D shape context feature descriptor
#         * \param[in] index point index in input_
#         * \param[out] desc descriptor to compute
#         */
#       void
#       computePointDescriptor (size_t index, std::vector<float> &desc);
# 
#       /** \brief Initialize computation by allocating all the intervals and the volume lookup table. */
#       virtual bool
#       initCompute ();
# 
#       /** \brief The actual feature computation.
#         * \param[out] output the resultant features
#         */
#       virtual void
#       computeFeature (PointCloudOut &output);
# 
#       /** \brief values of the radii interval. */
#       std::vector<float> radii_interval_;
# 
#       /** \brief Theta divisions interval. */
#       std::vector<float> theta_divisions_;
# 
#       /** \brief Phi divisions interval. */
#       std::vector<float> phi_divisions_;
# 
#       /** \brief Volumes look up table. */
#       std::vector<float> volume_lut_;
# 
#       /** \brief Bins along the azimuth dimension. */
#       size_t azimuth_bins_;
# 
#       /** \brief Bins along the elevation dimension. */
#       size_t elevation_bins_;
# 
#       /** \brief Bins along the radius dimension. */
#       size_t radius_bins_;
# 
#       /** \brief Minimal radius value. */
#       double min_radius_;
# 
#       /** \brief Point density radius. */
#       double point_density_radius_;
# 
#       /** \brief Descriptor length. */
#       size_t descriptor_length_;
# 
#       /** \brief Radius to compute local RF. */
#       double local_radius_;
#    private:
#       /** \brief Make the computeFeature (&Eigen::MatrixXf); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}
#   };
# 
#   /** \brief UniqueShapeContext implements the Unique Shape Descriptor
#     * described here:
#     *
#     *   - F. Tombari, S. Salti, L. Di Stefano,
#     *     "Unique Shape Context for 3D data description",
#     *     International Workshop on 3D Object Retrieval (3DOR 10) -
#     *     in conjuction with ACM Multimedia 2010
#     *
#     * The USC computed feature has the following structure:
#     *   - rf float[9] = x_axis | y_axis | normal and represents the local frame
#     *     desc std::vector<float> which size is determined by the number of bins
#     *     radius_bins_, elevation_bins_ and azimuth_bins_.
#     *
#     * \author Alessandro Franchi, Federico Tombari, Samuele Salti (original code)
#     * \author Nizar Sallem (port to PCL)
#     * \ingroup features
#     */
#   template <typename PointInT, typename PointRFT>
#   class UniqueShapeContext<PointInT, Eigen::MatrixXf, PointRFT> : public UniqueShapeContext<PointInT, pcl::SHOT, PointRFT>
#   {
#     public:
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
#       using UniqueShapeContext<PointInT, pcl::SHOT, PointRFT>::indices_;
#       using UniqueShapeContext<PointInT, pcl::SHOT, PointRFT>::descriptor_length_;
#       using UniqueShapeContext<PointInT, pcl::SHOT, PointRFT>::compute;
#       using UniqueShapeContext<PointInT, pcl::SHOT, PointRFT>::computePointDescriptor;
# 
#     private:
#       /** \brief The actual feature computation.
#         * \param[out] output the resultant features
#         */
#       virtual void
#       computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void
#       compute (pcl::PointCloud<pcl::SHOT> &) {}
# 
# 
# ###
#     # vfh.h
#   template<typename PointInT, typename PointNT, typename PointOutT = pcl::VFHSignature308>
#   class VFHEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
#   {
#     public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_radius_;
#       using Feature<PointInT, PointOutT>::input_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
# 
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#       typedef typename boost::shared_ptr<VFHEstimation<PointInT, PointNT, PointOutT> > Ptr;
#       typedef typename boost::shared_ptr<const VFHEstimation<PointInT, PointNT, PointOutT> > ConstPtr;
# 
# 
#       /** \brief Empty constructor. */
#       VFHEstimation () :
#         nr_bins_f1_ (45), nr_bins_f2_ (45), nr_bins_f3_ (45), nr_bins_f4_ (45), nr_bins_vp_ (128),
#         vpx_ (0), vpy_ (0), vpz_ (0),
#         hist_f1_ (), hist_f2_ (), hist_f3_ (), hist_f4_ (), hist_vp_ (),
#         normal_to_use_ (), centroid_to_use_ (), use_given_normal_ (false), use_given_centroid_ (false),
#         normalize_bins_ (true), normalize_distances_ (false), size_component_ (false),
#         d_pi_ (1.0f / (2.0f * static_cast<float> (M_PI)))
#       {
#         hist_f1_.setZero (nr_bins_f1_);
#         hist_f2_.setZero (nr_bins_f2_);
#         hist_f3_.setZero (nr_bins_f3_);
#         hist_f4_.setZero (nr_bins_f4_);
#         search_radius_ = 0;
#         k_ = 0;
#         feature_name_ = "VFHEstimation";
#       }
# 
#       /** \brief Estimate the SPFH (Simple Point Feature Histograms) signatures of the angular
#         * (f1, f2, f3) and distance (f4) features for a given point from its neighborhood
#         * \param[in] centroid_p the centroid point
#         * \param[in] centroid_n the centroid normal
#         * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
#         * \param[in] normals the dataset containing the surface normals at each point in \a cloud
#         * \param[in] indices the k-neighborhood point indices in the dataset
#         */
#       void
#       computePointSPFHSignature (const Eigen::Vector4f &centroid_p, const Eigen::Vector4f &centroid_n,
#                                  const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals,
#                                  const std::vector<int> &indices);
# 
#       /** \brief Set the viewpoint.
#         * \param[in] vpx the X coordinate of the viewpoint
#         * \param[in] vpy the Y coordinate of the viewpoint
#         * \param[in] vpz the Z coordinate of the viewpoint
#         */
#       inline void
#       setViewPoint (float vpx, float vpy, float vpz)
#       {
#         vpx_ = vpx;
#         vpy_ = vpy;
#         vpz_ = vpz;
#       }
# 
#       /** \brief Get the viewpoint. */
#       inline void
#       getViewPoint (float &vpx, float &vpy, float &vpz)
#       {
#         vpx = vpx_;
#         vpy = vpy_;
#         vpz = vpz_;
#       }
# 
#       /** \brief Set use_given_normal_
#         * \param[in] use Set to true if you want to use the normal passed to setNormalUse(normal)
#         */
#       inline void
#       setUseGivenNormal (bool use)
#       {
#         use_given_normal_ = use;
#       }
# 
#       /** \brief Set the normal to use
#         * \param[in] normal Sets the normal to be used in the VFH computation. It is is used
#         * to build the Darboux Coordinate system.
#         */
#       inline void
#       setNormalToUse (const Eigen::Vector3f &normal)
#       {
#         normal_to_use_ = Eigen::Vector4f (normal[0], normal[1], normal[2], 0);
#       }
# 
#       /** \brief Set use_given_centroid_
#         * \param[in] use Set to true if you want to use the centroid passed through setCentroidToUse(centroid)
#         */
#       inline void
#       setUseGivenCentroid (bool use)
#       {
#         use_given_centroid_ = use;
#       }
# 
#       /** \brief Set centroid_to_use_
#         * \param[in] centroid Centroid to be used in the VFH computation. It is used to compute the distances
#         * from all points to this centroid.
#         */
#       inline void
#       setCentroidToUse (const Eigen::Vector3f &centroid)
#       {
#         centroid_to_use_ = Eigen::Vector4f (centroid[0], centroid[1], centroid[2], 0);
#       }
# 
#       /** \brief set normalize_bins_
#         * \param[in] normalize If true, the VFH bins are normalized using the total number of points
#         */
#       inline void
#       setNormalizeBins (bool normalize)
#       {
#         normalize_bins_ = normalize;
#       }
# 
#       /** \brief set normalize_distances_
#         * \param[in] normalize If true, the 4th component of VFH (shape distribution component) get normalized
#         * by the maximum size between the centroid and the point cloud
#         */
#       inline void
#       setNormalizeDistance (bool normalize)
#       {
#         normalize_distances_ = normalize;
#       }
# 
#       /** \brief set size_component_
#         * \param[in] fill_size True if the 4th component of VFH (shape distribution component) needs to be filled.
#         * Otherwise, it is set to zero.
#         */
#       inline void
#       setFillSizeComponent (bool fill_size)
#       {
#         size_component_ = fill_size;
#       }
# 
#       /** \brief Overloaded computed method from pcl::Feature.
#         * \param[out] output the resultant point cloud model dataset containing the estimated features
#         */
#       void
#       compute (PointCloudOut &output);
# 
#     private:
# 
#       /** \brief The number of subdivisions for each feature interval. */
#       int nr_bins_f1_, nr_bins_f2_, nr_bins_f3_, nr_bins_f4_, nr_bins_vp_;
# 
#       /** \brief Values describing the viewpoint ("pinhole" camera model assumed). For per point viewpoints, inherit
#         * from VFHEstimation and provide your own computeFeature (). By default, the viewpoint is set to 0,0,0.
#         */
#       float vpx_, vpy_, vpz_;
# 
#       /** \brief Estimate the Viewpoint Feature Histograms (VFH) descriptors at a set of points given by
#         * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
#         * setSearchMethod ()
#         * \param[out] output the resultant point cloud model dataset that contains the VFH feature estimates
#         */
#       void
#       computeFeature (PointCloudOut &output);
# 
#     protected:
#       /** \brief This method should get called before starting the actual computation. */
#       bool
#       initCompute ();
# 
#       /** \brief Placeholder for the f1 histogram. */
#       Eigen::VectorXf hist_f1_;
#       /** \brief Placeholder for the f2 histogram. */
#       Eigen::VectorXf hist_f2_;
#       /** \brief Placeholder for the f3 histogram. */
#       Eigen::VectorXf hist_f3_;
#       /** \brief Placeholder for the f4 histogram. */
#       Eigen::VectorXf hist_f4_;
#       /** \brief Placeholder for the vp histogram. */
#       Eigen::VectorXf hist_vp_;
# 
#       /** \brief Normal to be used to computed VFH. Default, the average normal of the whole point cloud */
#       Eigen::Vector4f normal_to_use_;
#       /** \brief Centroid to be used to computed VFH. Default, the centroid of the whole point cloud */
#       Eigen::Vector4f centroid_to_use_;
# 
#       // VFH configuration parameters because CVFH instantiates it. See constructor for default values.
# 
#       /** \brief Use the normal_to_use_ */
#       bool use_given_normal_;
#       /** \brief Use the centroid_to_use_ */
#       bool use_given_centroid_;
#       /** \brief Normalize bins by the number the total number of points. */
#       bool normalize_bins_;
#       /** \brief Normalize the shape distribution component of VFH */
#       bool normalize_distances_;
#       /** \brief Activate or deactivate the size component of VFH */
#       bool size_component_;
# 
# ###

###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################
