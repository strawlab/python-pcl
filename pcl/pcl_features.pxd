# -*- coding: utf-8 -*-

from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from boost_shared_ptr cimport shared_ptr

cimport eigen as eigen3

# main
cimport pcl_defs as cpp
cimport pcl_kdtree as pclkdt
cimport pcl_range_image as pcl_r_img

###############################################################################
# Types
###############################################################################

### base class ###

# feature.h
# class Feature : public PCLBase<PointInT>
cdef extern from "pcl/features/feature.h" namespace "pcl":
    cdef cppclass Feature[In, Out](cpp.PCLBase[In]):
        Feature ()
        # public:
        # using PCLBase<PointInT>::indices_;
        # using PCLBase<PointInT>::input_;
        # ctypedef PCLBase<PointInT> BaseClass;
        # ctypedef boost::shared_ptr< Feature<PointInT, PointOutT> > Ptr;
        # ctypedef boost::shared_ptr< const Feature<PointInT, PointOutT> > ConstPtr;
        # ctypedef typename pcl::search::Search<PointInT> KdTree;
        # ctypedef typename pcl::search::Search<PointInT>::Ptr KdTreePtr;
        # ctypedef pcl::PointCloud<PointInT> PointCloudIn;
        # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
        # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # ctypedef pcl::PointCloud<PointOutT> PointCloudOut;
        # ctypedef boost::function<int (size_t, double, std::vector<int> &, std::vector<float> &)> SearchMethod;
        # ctypedef boost::function<int (const PointCloudIn &cloud, size_t index, double, std::vector<int> &, std::vector<float> &)> SearchMethodSurface;
        # public:
        # inline void setSearchSurface (const cpp.PointCloudPtr_t &)
        # inline cpp.PointCloudPtr_t getSearchSurface () const
        void setSearchSurface (const In &)
        In getSearchSurface () const
        
        # inline void setSearchMethod (const KdTreePtr &tree)
        # void setSearchMethod (pclkdt.KdTreePtr_t tree)
        # void setSearchMethod (pclkdt.KdTreeFLANNPtr_t tree)
        # void setSearchMethod (pclkdt.KdTreeFLANNConstPtr_t &tree)
        void setSearchMethod (const pclkdt.KdTreePtr_t &tree)
        
        # inline KdTreePtr getSearchMethod () const
        # pclkdt.KdTreePtr_t getSearchMethod ()
        # pclkdt.KdTreeFLANNPtr_t getSearchMethod ()
        # pclkdt.KdTreeFLANNConstPtr_t getSearchMethod ()
        
        double getSearchParameter ()
        void setKSearch (int search)
        int getKSearch () const
        void setRadiusSearch (double radius)
        double getRadiusSearch ()
        
        # void compute (PointCloudOut &output);
        # void compute (cpp.PointCloudPtr_t output)
        # void compute (cpp.PointCloud_PointXYZI_Ptr_t output)
        # void compute (cpp.PointCloud_PointXYZRGB_Ptr_t output)
        # void compute (cpp.PointCloud_PointXYZRGBA_Ptr_t output)
        void compute (cpp.PointCloud[Out] &output)
        
        # void computeEigen (cpp.PointCloud[Eigen::MatrixXf] &output);


ctypedef Feature[cpp.PointXYZ, cpp.Normal] Feature_t
ctypedef Feature[cpp.PointXYZI, cpp.Normal] Feature_PointXYZI_t
ctypedef Feature[cpp.PointXYZRGB, cpp.Normal] Feature_PointXYZRGB_t
ctypedef Feature[cpp.PointXYZRGBA, cpp.Normal] Feature_PointXYZRGBA_t
###

# template <typename PointInT, typename PointNT, typename PointOutT>
# class FeatureFromNormals : public Feature<PointInT, PointOutT>
# cdef cppclass FeatureFromNormals(Feature[In, Out]):
cdef extern from "pcl/features/feature.h" namespace "pcl":
    cdef cppclass FeatureFromNormals[In, NT, Out](Feature[In, Out]):
        FeatureFromNormals()
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
        # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # public:
        # ctypedef typename pcl::PointCloud<PointNT> PointCloudN;
        # ctypedef typename PointCloudN::Ptr PointCloudNPtr;
        # ctypedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
        # ctypedef boost::shared_ptr< FeatureFromNormals<PointInT, PointNT, PointOutT> > Ptr;
        # ctypedef boost::shared_ptr< const FeatureFromNormals<PointInT, PointNT, PointOutT> > ConstPtr;
        # // Members derived from the base class
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::getClassName;
        
        # /** \brief Provide a pointer to the input dataset that contains the point normals of
        #         * the XYZ dataset.
        # * In case of search surface is set to be different from the input cloud,
        # * normals should correspond to the search surface, not the input cloud!
        # * \param[in] normals the const boost shared pointer to a PointCloud of normals.
        # * By convention, L2 norm of each normal should be 1.
        # inline void setInputNormals (const PointCloudNConstPtr &normals)
        void setInputNormals (cpp.PointCloud_Normal_Ptr_t normals)
        
        # /** \brief Get a pointer to the normals of the input XYZ point cloud dataset. */
        # inline PointCloudNConstPtr getInputNormals ()


###

# 3dsc.h
# class ShapeContext3DEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/3dsc.h" namespace "pcl":
    cdef cppclass ShapeContext3DEstimation[In, NT, Out](FeatureFromNormals[In, NT, Out]):
        ShapeContext3DEstimation(bool)
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::searchForNeighbors;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        ##
        # brief Set the number of bins along the azimuth to \a bins.
        # param[in] bins the number of bins along the azimuth
        void setAzimuthBins (size_t bins)
        # return the number of bins along the azimuth
        size_t getAzimuthBins () 
        # brief Set the number of bins along the elevation to \a bins.
        # param[in] bins the number of bins along the elevation
        void setElevationBins (size_t )
        # return The number of bins along the elevation
        size_t getElevationBins ()
        # brief Set the number of bins along the radii to \a bins.
        # param[in] bins the number of bins along the radii
        void setRadiusBins (size_t )
        # return The number of bins along the radii direction
        size_t getRadiusBins ()
        # brief The minimal radius value for the search sphere (rmin) in the original paper 
        # param[in] radius the desired minimal radius
        void setMinimalRadius (double radius)
        # return The minimal sphere radius
        double getMinimalRadius ()
        # brief This radius is used to compute local point density 
        # density = number of points within this radius
        # param[in] radius value of the point density search radius
        void setPointDensityRadius (double radius)
        # return The point density search radius
        double getPointDensityRadius ()
        
###

# feature.h
# cdef extern from "pcl/features/feature.h" namespace "pcl":
#     cdef inline void solvePlaneParameters (const Eigen::Matrix3f &covariance_matrix,
#                                             const Eigen::Vector4f &point,
#                                             Eigen::Vector4f &plane_parameters, float &curvature);
#     cdef inline void solvePlaneParameters (const Eigen::Matrix3f &covariance_matrix,
#                         float &nx, float &ny, float &nz, float &curvature);
# template <typename PointInT, typename PointLT, typename PointOutT>
# class FeatureFromLabels : public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/feature.h" namespace "pcl":
    cdef cppclass FeatureFromLabels[In, LT, Out](Feature[In, Out]):
        FeatureFromLabels()
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
        # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # ctypedef typename pcl::PointCloud<PointLT> PointCloudL;
        # ctypedef typename PointCloudL::Ptr PointCloudNPtr;
        # ctypedef typename PointCloudL::ConstPtr PointCloudLConstPtr;
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # public:
        # ctypedef boost::shared_ptr< FeatureFromLabels<PointInT, PointLT, PointOutT> > Ptr;
        # ctypedef boost::shared_ptr< const FeatureFromLabels<PointInT, PointLT, PointOutT> > ConstPtr;
        # // Members derived from the base class
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::k_;
        # /** \brief Provide a pointer to the input dataset that contains the point labels of
        #   * the XYZ dataset.
        #   * In case of search surface is set to be different from the input cloud,
        #   * labels should correspond to the search surface, not the input cloud!
        #   * \param[in] labels the const boost shared pointer to a PointCloud of labels.
        #   */
        # inline void setInputLabels (const PointCloudLConstPtr &labels)
        # inline PointCloudLConstPtr getInputLabels () const
###

### Inheritance class ###

# 3dsc.h
# class ShapeContext3DEstimation<PointInT, PointNT, Eigen::MatrixXf> : public ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>
# cdef extern from "pcl/features/3dsc.h" namespace "pcl":
#     cdef cppclass ShapeContext3DEstimation[T, N, M]:
#         ShapeContext3DEstimation(bool)
#         # public:
#         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::feature_name_;
#         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::indices_;
#         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::descriptor_length_;
#         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::normals_;
#         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::input_;
#         # using ShapeContext3DEstimation<PointInT, PointNT, pcl::SHOT>::compute;
###

# class BoundaryEstimation: public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/boundary.h" namespace "pcl":
    cdef cppclass BoundaryEstimation[In, NT, Out](FeatureFromNormals[In, NT, Out]):
        BoundaryEstimation ()
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::tree_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        ##
        # brief Check whether a point is a boundary point in a planar patch of projected points given by indices.
        # note A coordinate system u-v-n must be computed a-priori using \a getCoordinateSystemOnPlane
        # param[in] cloud a pointer to the input point cloud
        # param[in] q_idx the index of the query point in \a cloud
        # param[in] indices the estimated point neighbors of the query point
        # param[in] u the u direction
        # param[in] v the v direction
        # param[in] angle_threshold the threshold angle (default \f$\pi / 2.0\f$)
        # bool isBoundaryPoint (const cpp.PointCloud[In] &cloud, 
        #                int q_idx, const vector[int] &indices, 
        #                const Eigen::Vector4f &u, const Eigen::Vector4f &v, const float angle_threshold);
        # brief Check whether a point is a boundary point in a planar patch of projected points given by indices.
        # note A coordinate system u-v-n must be computed a-priori using \a getCoordinateSystemOnPlane
        # param[in] cloud a pointer to the input point cloud
        # param[in] q_point a pointer to the querry point
        # param[in] indices the estimated point neighbors of the query point
        # param[in] u the u direction
        # param[in] v the v direction
        # param[in] angle_threshold the threshold angle (default \f$\pi / 2.0\f$)
        # bool isBoundaryPoint (const cpp.PointCloud[In] &cloud, 
        #                const [In] &q_point, 
        #                const vector[int] &indices, 
        #                const Eigen::Vector4f &u, const Eigen::Vector4f &v, const float angle_threshold);
        # brief Set the decision boundary (angle threshold) that marks points as boundary or regular. 
        # (default \f$\pi / 2.0\f$) 
        # param[in] angle the angle threshold
        inline void setAngleThreshold (float angle)
        inline float getAngleThreshold ()
        # brief Get a u-v-n coordinate system that lies on a plane defined by its normal
        # param[in] p_coeff the plane coefficients (containing the plane normal)
        # param[out] u the resultant u direction
        # param[out] v the resultant v direction
        # inline void getCoordinateSystemOnPlane (const PointNT &p_coeff, 
        #                           Eigen::Vector4f &u, Eigen::Vector4f &v)

###

# class CVFHEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
# cdef extern from "pcl/features/cvfh.h" namespace "pcl":
#     cdef cppclass CVFHEstimation[In, NT, Out](FeatureFromNormals[In, NT, Out]):
#         CVFHEstimation()
#         # public:
#         # using Feature<PointInT, PointOutT>::feature_name_;
#         # using Feature<PointInT, PointOutT>::getClassName;
#         # using Feature<PointInT, PointOutT>::indices_;
#         # using Feature<PointInT, PointOutT>::k_;
#         # using Feature<PointInT, PointOutT>::search_radius_;
#         # using Feature<PointInT, PointOutT>::surface_;
#         # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
#         # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#         # ctypedef typename pcl::search::Search<PointNormal>::Ptr KdTreePtr;
#         # ctypedef typename pcl::NormalEstimation<PointNormal, PointNormal> NormalEstimator;
#         # ctypedef typename pcl::VFHEstimation<PointInT, PointNT, pcl::VFHSignature308> VFHEstimator;
#         ##
#         # brief Removes normals with high curvature caused by real edges or noisy data
#         # param[in] cloud pointcloud to be filtered
#         # param[out] indices_out the indices of the points with higher curvature than threshold
#         # param[out] indices_in the indices of the remaining points after filtering
#         # param[in] threshold threshold value for curvature
#         void filterNormalsWithHighCurvature (
#                                               const cpp.PointCloud[NT] & cloud, 
#                                               vector[int] &indices, vector[int] &indices2,
#                                               vector[int] &, float);
#         # brief Set the viewpoint.
#         # param[in] vpx the X coordinate of the viewpoint
#         # param[in] vpy the Y coordinate of the viewpoint
#         # param[in] vpz the Z coordinate of the viewpoint
#         inline void setViewPoint (float x, float y, float z)
#         # brief Set the radius used to compute normals
#         # param[in] radius_normals the radius
#         inline void setRadiusNormals (float radius)
#         # brief Get the viewpoint. 
#         # param[out] vpx the X coordinate of the viewpoint
#         # param[out] vpy the Y coordinate of the viewpoint
#         # param[out] vpz the Z coordinate of the viewpoint
#         inline void getViewPoint (float &x, float &y, float &z)
#         # brief Get the centroids used to compute different CVFH descriptors
#         # param[out] centroids vector to hold the centroids
#         # inline void getCentroidClusters (vector[Eigen::Vector3f] &)
#         # brief Get the normal centroids used to compute different CVFH descriptors
#         # param[out] centroids vector to hold the normal centroids
#         # inline void getCentroidNormalClusters (vector[Eigen::Vector3f] &)
#         # brief Sets max. Euclidean distance between points to be added to the cluster 
#         # param[in] d the maximum Euclidean distance 
#         inline void setClusterTolerance (float tolerance)
#         # brief Sets max. deviation of the normals between two points so they can be clustered together
#         # param[in] d the maximum deviation 
#         inline void setEPSAngleThreshold (float angle)
#         # brief Sets curvature threshold for removing normals
#         # param[in] d the curvature threshold 
#         inline void setCurvatureThreshold (float curve)
#         # brief Set minimum amount of points for a cluster to be considered
#         # param[in] min the minimum amount of points to be set 
#         inline void setMinPoints (size_t points)
#         # brief Sets wether if the CVFH signatures should be normalized or not
#         # param[in] normalize true if normalization is required, false otherwise 
#         inline void setNormalizeBins (bool bins)
#         # brief Overloaded computed method from pcl::Feature.
#         # param[out] output the resultant point cloud model dataset containing the estimated features
#         # void compute (cpp.PointCloud[Out] &);


###

# esf.h
# class ESFEstimation: public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/esf.h" namespace "pcl":
    cdef cppclass ESFEstimation[In, Out](Feature[In, Out]):
        ESFEstimation ()
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::surface_;
        # ctypedef typename pcl::PointCloud<PointInT> PointCloudIn;
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # void compute (cpp.PointCloud[Out] &output)
###

# template <typename PointInT, typename PointRFT>
# class FeatureWithLocalReferenceFrames
cdef extern from "pcl/features/feature.h" namespace "pcl":
    cdef cppclass FeatureWithLocalReferenceFrames[T, REF]:
        FeatureWithLocalReferenceFrames ()
        # public:
        # ctypedef cpp.PointCloud[RFT] PointCloudLRF;
        # ctypedef typename PointCloudLRF::Ptr PointCloudLRFPtr;
        # ctypedef typename PointCloudLRF::ConstPtr PointCloudLRFConstPtr;
        # inline void setInputReferenceFrames (const PointCloudLRFConstPtr &frames)
        # inline PointCloudLRFConstPtr getInputReferenceFrames () const
        # protected:
        # /** \brief A boost shared pointer to the local reference frames. */
        # PointCloudLRFConstPtr frames_;
        # /** \brief The user has never set the frames. */
        # bool frames_never_defined_;
        # /** \brief Check if frames_ has been correctly initialized and compute it if needed.
        # * \param input the subclass' input cloud dataset.
        # * \param lrf_estimation a pointer to a local reference frame estimation class to be used as default.
        # * \return true if frames_ has been correctly initialized.
        # */
        # typedef typename Feature<PointInT, PointRFT>::Ptr LRFEstimationPtr;
        # virtual bool
        # initLocalReferenceFrames (const size_t& indices_size,
        #                           const LRFEstimationPtr& lrf_estimation = LRFEstimationPtr());
###

# fpfh
# template <typename PointInT, typename PointNT, typename PointOutT = pcl::FPFHSignature33>
# class FPFHEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/fpfh.h" namespace "pcl":
    cdef cppclass FPFHEstimation[In, NT, Out](FeatureFromNormals[In, NT, Out]):
        FPFHEstimation()
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # * represented by Cartesian coordinates and normals.
        # * \note For explanations about the features, please see the literature mentioned above (the order of the
        # * features might be different).
        # * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
        # * \param[in] normals the dataset containing the surface normals (assuming normalized vectors) at each point in cloud
        # * \param[in] p_idx the index of the first point (source)
        # * \param[in] q_idx the index of the second point (target)
        # * \param[out] f1 the first angular feature (angle between the projection of nq_idx and u)
        # * \param[out] f2 the second angular feature (angle between nq_idx and v)
        # * \param[out] f3 the third angular feature (angle between np_idx and |p_idx - q_idx|)
        # * \param[out] f4 the distance feature (p_idx - q_idx)
        # bool computePairFeatures (const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals, int p_idx, int q_idx, float &f1, float &f2, float &f3, float &f4);
        
        # \brief Estimate the SPFH (Simple Point Feature Histograms) individual signatures of the three angular
        # (f1, f2, f3) features for a given point based on its spatial neighborhood of 3D points with normals
        # \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
        # \param[in] normals the dataset containing the surface normals at each point in \a cloud
        # \param[in] p_idx the index of the query point (source)
        # \param[in] row the index row in feature histogramms
        # \param[in] indices the k-neighborhood point indices in the dataset
        # \param[out] hist_f1 the resultant SPFH histogram for feature f1
        # \param[out] hist_f2 the resultant SPFH histogram for feature f2
        # \param[out] hist_f3 the resultant SPFH histogram for feature f3
        # void computePointSPFHSignature (
        #                               const pcl::PointCloud<PointInT> &cloud, 
        #                               const pcl::PointCloud<PointNT> &normals, int p_idx, int row, 
        #                               const std::vector<int> &indices, 
        #                               Eigen::MatrixXf &hist_f1, Eigen::MatrixXf &hist_f2, Eigen::MatrixXf &hist_f3);
        
        # \brief Weight the SPFH (Simple Point Feature Histograms) individual histograms to create the final FPFH
        # (Fast Point Feature Histogram) for a given point based on its 3D spatial neighborhood
        # \param[in] hist_f1 the histogram feature vector of \a f1 values over the given patch
        # \param[in] hist_f2 the histogram feature vector of \a f2 values over the given patch
        # \param[in] hist_f3 the histogram feature vector of \a f3 values over the given patch
        # \param[in] indices the point indices of p_idx's k-neighborhood in the point cloud
        # \param[in] dists the distances from p_idx to all its k-neighbors
        # \param[out] fpfh_histogram the resultant FPFH histogram representing the feature at the query point
        # void weightPointSPFHSignature (
        #                           const Eigen::MatrixXf &hist_f1, 
        #                           const Eigen::MatrixXf &hist_f2, 
        #                           const Eigen::MatrixXf &hist_f3, 
        #                           const std::vector<int> &indices, 
        #                           const std::vector<float> &dists, 
        #                           Eigen::VectorXf &fpfh_histogram);
        
        # \brief Set the number of subdivisions for each angular feature interval.
        # \param[in] nr_bins_f1 number of subdivisions for the first angular feature
        # \param[in] nr_bins_f2 number of subdivisions for the second angular feature
        # \param[in] nr_bins_f3 number of subdivisions for the third angular feature
        inline void setNrSubdivisions (int , int , int )
        
        # \brief Get the number of subdivisions for each angular feature interval. 
        # \param[out] nr_bins_f1 number of subdivisions for the first angular feature
        # \param[out] nr_bins_f2 number of subdivisions for the second angular feature
        # \param[out] nr_bins_f3 number of subdivisions for the third angular feature
        inline void getNrSubdivisions (int &, int &, int &)


ctypedef FPFHEstimation[cpp.PointXYZ, cpp.Normal, cpp.PFHSignature125] FPFHEstimation_t
ctypedef shared_ptr[FPFHEstimation[cpp.PointXYZ, cpp.Normal, cpp.PFHSignature125]] FPFHEstimationPtr_t
# template <typename PointInT, typename PointNT>
# class FPFHEstimation<PointInT, PointNT, Eigen::MatrixXf> : public FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>
# cdef extern from "pcl/features/feature.h" namespace "pcl":
#     cdef cppclass FPFHEstimation[T, NT]:
#         FPFHEstimation()
# ctypedef FPFHEstimation[cpp.PointXYZ, cpp.Normal, eigen3.MatrixXf] FPFHEstimation2_t
# ctypedef shared_ptr[FPFHEstimation[cpp.PointXYZ, cpp.Normal, eigen3.MatrixXf]] FPFHEstimation2Ptr_t
###

# fpfh_omp
# template <typename PointInT, typename PointNT, typename PointOutT>
# class FPFHEstimationOMP : public FPFHEstimation<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/fpfh_omp.h" namespace "pcl":
    cdef cppclass FPFHEstimationOMP[In, NT, Out](FPFHEstimation[In, NT, Out]):
        FPFHEstimationOMP ()
        # FPFHEstimationOMP (unsigned int )
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # using FPFHEstimation<PointInT, PointNT, PointOutT>::hist_f1_;
        # using FPFHEstimation<PointInT, PointNT, PointOutT>::hist_f2_;
        # using FPFHEstimation<PointInT, PointNT, PointOutT>::hist_f3_;
        # using FPFHEstimation<PointInT, PointNT, PointOutT>::weightPointSPFHSignature;
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # * \brief Initialize the scheduler and set the number of threads to use.
        # * \param[in] nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
        inline void setNumberOfThreads (unsigned threads)
        # public:
        # * \brief The number of subdivisions for each angular feature interval. */
        # int nr_bins_f1_, nr_bins_f2_, nr_bins_f3_;

###

# integral_image_normal.h
# template <typename PointInT, typename PointOutT>
# class IntegralImageNormalEstimation : public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl":
    cdef cppclass IntegralImageNormalEstimation[In, Out](Feature[In, Out]):
        IntegralImageNormalEstimation ()
        # public:
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudIn  PointCloudIn;
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # 
        # * \brief Set the regions size which is considered for normal estimation.
        # * \param[in] width the width of the search rectangle
        # * \param[in] height the height of the search rectangle
        void setRectSize (const int width, const int height)
        
        # * \brief Sets the policy for handling borders.
        # * \param[in] border_policy the border policy.
        # minipcl
        # void setBorderPolicy (BorderPolicy border_policy)
        # * \brief Computes the normal at the specified position.
        # * \param[in] pos_x x position (pixel)
        # * \param[in] pos_y y position (pixel)
        # * \param[in] point_index the position index of the point
        # * \param[out] normal the output estimated normal
        void computePointNormal (const int pos_x, const int pos_y, const unsigned point_index, Out &normal)
        
        # * \brief Computes the normal at the specified position with mirroring for border handling.
        # * \param[in] pos_x x position (pixel)
        # * \param[in] pos_y y position (pixel)
        # * \param[in] point_index the position index of the point
        # * \param[out] normal the output estimated normal
        void computePointNormalMirror (const int pos_x, const int pos_y, const unsigned point_index, Out &normal)
        
        # * \brief The depth change threshold for computing object borders
        # * \param[in] max_depth_change_factor the depth change threshold for computing object borders based on
        # * depth changes
        void setMaxDepthChangeFactor (float max_depth_change_factor)
        
        # * \brief Set the normal smoothing size
        # * \param[in] normal_smoothing_size factor which influences the size of the area used to smooth normals
        # * (depth dependent if useDepthDependentSmoothing is true)
        void setNormalSmoothingSize (float normal_smoothing_size)
        
        # TODO : use minipcl.cpp/h
        # * \brief Set the normal estimation method. The current implemented algorithms are:
        # * <ul>
        # *   <li><b>COVARIANCE_MATRIX</b> - creates 9 integral images to compute the normal for a specific point
        # *   from the covariance matrix of its local neighborhood.</li>
        # *   <li><b>AVERAGE_3D_GRADIENT</b> - creates 6 integral images to compute smoothed versions of
        # *   horizontal and vertical 3D gradients and computes the normals using the cross-product between these
        # *   two gradients.
        # *   <li><b>AVERAGE_DEPTH_CHANGE</b> -  creates only a single integral image and computes the normals
        # *   from the average depth changes.
        # * </ul>
        # * \param[in] normal_estimation_method the method used for normal estimation
        # void setNormalEstimationMethod (NormalEstimationMethod2 normal_estimation_method)
       
        # brief Set whether to use depth depending smoothing or not
        # param[in] use_depth_dependent_smoothing decides whether the smoothing is depth dependent
        void setDepthDependentSmoothing (bool use_depth_dependent_smoothing)
        
        # brief Provide a pointer to the input dataset (overwrites the PCLBase::setInputCloud method)
        # \param[in] cloud the const boost shared pointer to a PointCloud message
        # void setInputCloud (const typename PointCloudIn::ConstPtr &cloud)
        void setInputCloud (In cloud)
        
        # brief Returns a pointer to the distance map which was computed internally
        inline float* getDistanceMap ()
        
        # * \brief Set the viewpoint.
        # * \param vpx the X coordinate of the viewpoint
        # * \param vpy the Y coordinate of the viewpoint
        # * \param vpz the Z coordinate of the viewpoint
        inline void setViewPoint (float vpx, float vpy, float vpz)
        
        # * \brief Get the viewpoint.
        # * \param [out] vpx x-coordinate of the view point
        # * \param [out] vpy y-coordinate of the view point
        # * \param [out] vpz z-coordinate of the view point
        # * \note this method returns the currently used viewpoint for normal flipping.
        # * If the viewpoint is set manually using the setViewPoint method, this method will return the set view point coordinates.
        # * If an input cloud is set, it will return the sensor origin otherwise it will return the origin (0, 0, 0)
        inline void getViewPoint (float &vpx, float &vpy, float &vpz)
        
        # * \brief sets whether the sensor origin or a user given viewpoint should be used. After this method, the 
        # * normal estimation method uses the sensor origin of the input cloud.
        # * to use a user defined view point, use the method setViewPoint
        inline void useSensorOriginAsViewPoint ()


ctypedef IntegralImageNormalEstimation[cpp.PointXYZ, cpp.Normal] IntegralImageNormalEstimation_t
ctypedef IntegralImageNormalEstimation[cpp.PointXYZI, cpp.Normal] IntegralImageNormalEstimation_PointXYZI_t
ctypedef IntegralImageNormalEstimation[cpp.PointXYZRGB, cpp.Normal] IntegralImageNormalEstimation_PointXYZRGB_t
ctypedef IntegralImageNormalEstimation[cpp.PointXYZRGBA, cpp.Normal] IntegralImageNormalEstimation_PointXYZRGBA_t
ctypedef shared_ptr[IntegralImageNormalEstimation[cpp.PointXYZ, cpp.Normal]] IntegralImageNormalEstimationPtr_t
ctypedef shared_ptr[IntegralImageNormalEstimation[cpp.PointXYZI, cpp.Normal]] IntegralImageNormalEstimation_PointXYZI_Ptr_t
ctypedef shared_ptr[IntegralImageNormalEstimation[cpp.PointXYZRGB, cpp.Normal]] IntegralImageNormalEstimation_PointXYZRGB_Ptr_t
ctypedef shared_ptr[IntegralImageNormalEstimation[cpp.PointXYZRGBA, cpp.Normal]] IntegralImageNormalEstimation_PointXYZRGBA_Ptr_t
###

# integral_image2D.h
# template <class DataType, unsigned Dimension>
# class IntegralImage2D
cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl":
    cdef cppclass IntegralImage2D[Type, Dim]:
        # IntegralImage2D ()
        IntegralImage2D (bool flag)
        # public:
        # static const unsigned second_order_size = (Dimension * (Dimension + 1)) >> 1;
        # ctypedef Eigen::Matrix<typename IntegralImageTypeTraits<DataType>::IntegralType, Dimension, 1> ElementType;
        # ctypedef Eigen::Matrix<typename IntegralImageTypeTraits<DataType>::IntegralType, second_order_size, 1> SecondOrderType;
        # void setSecondOrderComputation (bool compute_second_order_integral_images);
        # * \brief Set the input data to compute the integral image for
        #   * \param[in] data the input data
        #   * \param[in] width the width of the data
        #   * \param[in] height the height of the data
        #   * \param[in] element_stride the element stride of the data
        #   * \param[in] row_stride the row stride of the data
        # void setInput (const DataType * data, unsigned width, unsigned height, unsigned element_stride, unsigned row_stride)
        # * \brief Compute the first order sum within a given rectangle
        #   * \param[in] start_x x position of rectangle
        #   * \param[in] start_y y position of rectangle
        #   * \param[in] width width of rectangle
        #   * \param[in] height height of rectangle
        # inline ElementType getFirstOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const
        # /** \brief Compute the first order sum within a given rectangle
        #   * \param[in] start_x x position of the start of the rectangle
        #   * \param[in] start_y x position of the start of the rectangle
        #   * \param[in] end_x x position of the end of the rectangle
        #   * \param[in] end_y x position of the end of the rectangle
        # inline ElementType getFirstOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const
        # /** \brief Compute the second order sum within a given rectangle
        #   * \param[in] start_x x position of rectangle
        #   * \param[in] start_y y position of rectangle
        #   * \param[in] width width of rectangle
        #   * \param[in] height height of rectangle
        # inline SecondOrderType getSecondOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const
        # /** \brief Compute the second order sum within a given rectangle
        #   * \param[in] start_x x position of the start of the rectangle
        #   * \param[in] start_y x position of the start of the rectangle
        #   * \param[in] end_x x position of the end of the rectangle
        #   * \param[in] end_y x position of the end of the rectangle
        # inline SecondOrderType getSecondOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const
        # /** \brief Compute the number of finite elements within a given rectangle
        #   * \param[in] start_x x position of rectangle
        #   * \param[in] start_y y position of rectangle
        #   * \param[in] width width of rectangle
        #   * \param[in] height height of rectangle
        inline unsigned getFiniteElementsCount (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const
        # /** \brief Compute the number of finite elements within a given rectangle
        #   * \param[in] start_x x position of the start of the rectangle
        #   * \param[in] start_y x position of the start of the rectangle
        #   * \param[in] end_x x position of the end of the rectangle
        #   * \param[in] end_y x position of the end of the rectangle
        inline unsigned getFiniteElementsCountSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const
###

# template <class DataType>
# class IntegralImage2D <DataType, 1>
# cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl":
#     cdef cppclass IntegralImage2D[Type]:
#         # IntegralImage2D ()
#         IntegralImage2D (bool flag)
#         # public:
#         # static const unsigned second_order_size = 1;
#         # ctypedef typename IntegralImageTypeTraits<DataType>::IntegralType ElementType;
#         # ctypedef typename IntegralImageTypeTraits<DataType>::IntegralType SecondOrderType;
#         # /** \brief Set the input data to compute the integral image for
#         #   * \param[in] data the input data
#         #   * \param[in] width the width of the data
#         #   * \param[in] height the height of the data
#         #   * \param[in] element_stride the element stride of the data
#         #   * \param[in] row_stride the row stride of the data
#         #   */
#         # void setInput (const DataType * data, unsigned width, unsigned height, unsigned element_stride, unsigned row_stride);
#         # /** \brief Compute the first order sum within a given rectangle
#         #   * \param[in] start_x x position of rectangle
#         #   * \param[in] start_y y position of rectangle
#         #   * \param[in] width width of rectangle
#         #   * \param[in] height height of rectangle
#         #   */
#         # inline ElementType getFirstOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;
#         # /** \brief Compute the first order sum within a given rectangle
#         #   * \param[in] start_x x position of the start of the rectangle
#         #   * \param[in] start_y x position of the start of the rectangle
#         #   * \param[in] end_x x position of the end of the rectangle
#         #   * \param[in] end_y x position of the end of the rectangle
#         #   */
#         # inline ElementType getFirstOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;
#         # /** \brief Compute the second order sum within a given rectangle
#         #   * \param[in] start_x x position of rectangle
#         #   * \param[in] start_y y position of rectangle
#         #   * \param[in] width width of rectangle
#         #   * \param[in] height height of rectangle
#         #   */
#         # inline SecondOrderType getSecondOrderSum (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;
#         # /** \brief Compute the second order sum within a given rectangle
#         #   * \param[in] start_x x position of the start of the rectangle
#         #   * \param[in] start_y x position of the start of the rectangle
#         #   * \param[in] end_x x position of the end of the rectangle
#         #   * \param[in] end_y x position of the end of the rectangle
#         # inline SecondOrderType getSecondOrderSumSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;
#         # /** \brief Compute the number of finite elements within a given rectangle
#         #   * \param[in] start_x x position of rectangle
#         #   * \param[in] start_y y position of rectangle
#         #   * \param[in] width width of rectangle
#         #   * \param[in] height height of rectangle
#         #   */
#         inline unsigned getFiniteElementsCount (unsigned start_x, unsigned start_y, unsigned width, unsigned height) const;
#         # /** \brief Compute the number of finite elements within a given rectangle
#         #   * \param[in] start_x x position of the start of the rectangle
#         #   * \param[in] start_y x position of the start of the rectangle
#         #   * \param[in] end_x x position of the end of the rectangle
#         #   * \param[in] end_y x position of the end of the rectangle
#         #   */
#         inline unsigned getFiniteElementsCountSE (unsigned start_x, unsigned start_y, unsigned end_x, unsigned end_y) const;

###

# intensity_gradient.h
# template <typename PointInT, typename PointNT, typename PointOutT, typename IntensitySelectorT = pcl::common::IntensityFieldAccessor<PointInT> >
# class IntensityGradientEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/intensity_gradient.h" namespace "pcl":
    cdef cppclass IntensityGradientEstimation[In, NT, Out, Intensity](FeatureFromNormals[In, NT, Out]):
        IntensityGradientEstimation ()
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # brief Initialize the scheduler and set the number of threads to use.
        # param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
        # inline void setNumberOfThreads (int nr_threads)
###

# template <typename PointInT, typename PointNT>
# class IntensityGradientEstimation<PointInT, PointNT, Eigen::MatrixXf>: public IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>
# cdef extern from "pcl/features/intensity_gradient.h" namespace "pcl":
#     cdef cppclass IntensityGradientEstimation[In, NT]:
#         IntensityGradientEstimation ()
#         # public:
#         #   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::indices_;
#         #   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::normals_;
#         #   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::input_;
#         #   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::surface_;
#         #   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::k_;
#         #   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::search_parameter_;
#         #   using IntensityGradientEstimation<PointInT, PointNT, pcl::IntensityGradient>::compute;

###

# intensity_spin.h
# template <typename PointInT, typename PointOutT>
# class IntensitySpinEstimation: public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/intensity_spin.h" namespace "pcl":
    cdef cppclass IntensitySpinEstimation[In, Out](Feature[In, Out]):
        IntensitySpinEstimation ()
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::tree_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # ctypedef typename pcl::PointCloud<PointInT> PointCloudIn;
        # ctypedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        ##
        # /** \brief Estimate the intensity-domain spin image descriptor for a given point based on its spatial
        #   * neighborhood of 3D points and their intensities. 
        #   * \param[in] cloud the dataset containing the Cartesian coordinates and intensity values of the points
        #   * \param[in] radius the radius of the feature
        #   * \param[in] sigma the standard deviation of the Gaussian smoothing kernel to use during the soft histogram update
        #   * \param[in] k the number of neighbors to use from \a indices and \a squared_distances
        #   * \param[in] indices the indices of the points that comprise the query point's neighborhood
        #   * \param[in] squared_distances the squared distances from the query point to each point in the neighborhood
        #   * \param[out] intensity_spin_image the resultant intensity-domain spin image descriptor
        #   */
        # void computeIntensitySpinImage (const PointCloudIn &cloud, 
        #                            float radius, float sigma, int k, 
        #                            const std::vector<int> &indices, 
        #                            const std::vector<float> &squared_distances, 
        #                            Eigen::MatrixXf &intensity_spin_image);

        # /** \brief Set the number of bins to use in the distance dimension of the spin image
        #   * \param[in] nr_distance_bins the number of bins to use in the distance dimension of the spin image
        #   */
        # inline void setNrDistanceBins (size_t nr_distance_bins) { nr_distance_bins_ = static_cast<int> (nr_distance_bins); };
        # /** \brief Returns the number of bins in the distance dimension of the spin image. */
        # inline int getNrDistanceBins ()
        # /** \brief Set the number of bins to use in the intensity dimension of the spin image.
        #   * \param[in] nr_intensity_bins the number of bins to use in the intensity dimension of the spin image
        #   */
        # inline void setNrIntensityBins (size_t nr_intensity_bins)
        # /** \brief Returns the number of bins in the intensity dimension of the spin image. */
        # inline int getNrIntensityBins ()
        # /** \brief Set the standard deviation of the Gaussian smoothing kernel to use when constructing the spin images.  
        #   * \param[in] sigma the standard deviation of the Gaussian smoothing kernel to use when constructing the spin images
        # inline void setSmoothingBandwith (float sigma)
        # /** \brief Returns the standard deviation of the Gaussian smoothing kernel used to construct the spin images.  */
        # inline float getSmoothingBandwith ()
        # /** \brief Estimate the intensity-domain descriptors at a set of points given by <setInputCloud (), setIndices ()>
        #   *  using the surface in setSearchSurface (), and the spatial locator in setSearchMethod ().
        #   * \param[out] output the resultant point cloud model dataset that contains the intensity-domain spin image features
        # void computeFeature (PointCloudOut &output);
        # /** \brief The number of distance bins in the descriptor. */
        # int nr_distance_bins_;
        # /** \brief The number of intensity bins in the descriptor. */
        # int nr_intensity_bins_;
        # /** \brief The standard deviation of the Gaussian smoothing kernel used to construct the spin images. */
        # float sigma_;

###

# template <typename PointInT>
# class IntensitySpinEstimation<PointInT, Eigen::MatrixXf>: public IntensitySpinEstimation<PointInT, pcl::Histogram<20> >
# cdef extern from "pcl/features/intensity_spin.h" namespace "pcl":
#     cdef cppclass IntensitySpinEstimation[In](IntensitySpinEstimation[In]):
#         IntensitySpinEstimation ()
#         # public:
#         #   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::getClassName;
#         #   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::input_;
#         #   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::indices_;
#         #   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::surface_;
#         #   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::search_radius_;
#         #   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::nr_intensity_bins_;
#         #   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::nr_distance_bins_;
#         #   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::tree_;
#         #   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::sigma_;
#         #   using IntensitySpinEstimation<PointInT, pcl::Histogram<20> >::compute;
###

# moment_invariants.h
# template <typename PointInT, typename PointOutT>
# class MomentInvariantsEstimation: public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/moment_invariants.h" namespace "pcl":
    cdef cppclass MomentInvariantsEstimation[In, Out](Feature[In, Out]):
        MomentInvariantsEstimation ()
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::input_;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # /** \brief Compute the 3 moment invariants (j1, j2, j3) for a given set of points, using their indices.
        # * \param[in] cloud the input point cloud
        # * \param[in] indices the point cloud indices that need to be used
        # * \param[out] j1 the resultant first moment invariant
        # * \param[out] j2 the resultant second moment invariant
        # * \param[out] j3 the resultant third moment invariant
        # */
        # void computePointMomentInvariants (const pcl::PointCloud<PointInT> &cloud, 
        #                             const std::vector<int> &indices, 
        #                             float &j1, float &j2, float &j3);
        # * \brief Compute the 3 moment invariants (j1, j2, j3) for a given set of points, using their indices.
        # * \param[in] cloud the input point cloud
        # * \param[out] j1 the resultant first moment invariant
        # * \param[out] j2 the resultant second moment invariant
        # * \param[out] j3 the resultant third moment invariant
        # void computePointMomentInvariants (const pcl::PointCloud<PointInT> &cloud, 
        #                             float &j1, float &j2, float &j3);
###

# template <typename PointInT>
# class MomentInvariantsEstimation<PointInT, Eigen::MatrixXf>: public MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>
# cdef extern from "pcl/features/moment_invariants.h" namespace "pcl":
#     cdef cppclass MomentInvariantsEstimation[In, Out](MomentInvariantsEstimation[In]):
#         MomentInvariantsEstimation ()
#         public:
#         using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::k_;
#         using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::indices_;
#         using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::search_parameter_;
#         using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::surface_;
#         using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::input_;
#         using MomentInvariantsEstimation<PointInT, pcl::MomentInvariants>::compute;
###

# multiscale_feature_persistence.h
# template <typename PointSource, typename PointFeature>
# class MultiscaleFeaturePersistence : public PCLBase<PointSource>
cdef extern from "pcl/features/multiscale_feature_persistence.h" namespace "pcl":
    cdef cppclass MultiscaleFeaturePersistence[Source, Feature](cpp.PCLBase[Source]):
        MultiscaleFeaturePersistence ()
        # public:
        # typedef pcl::PointCloud<PointFeature> FeatureCloud;
        # typedef typename pcl::PointCloud<PointFeature>::Ptr FeatureCloudPtr;
        # typedef typename pcl::Feature<PointSource, PointFeature>::Ptr FeatureEstimatorPtr;
        # typedef boost::shared_ptr<const pcl::PointRepresentation <PointFeature> > FeatureRepresentationConstPtr;
        # using pcl::PCLBase<PointSource>::input_;
        # 
        # /** \brief Method that calls computeFeatureAtScale () for each scale parameter */
        # void computeFeaturesAtAllScales ();
        
        # /** \brief Central function that computes the persistent features
        #  * \param output_features a cloud containing the persistent features
        #  * \param output_indices vector containing the indices of the points in the input cloud
        #  * that have persistent features, under a one-to-one correspondence with the output_features cloud
        #  */
        # void determinePersistentFeatures (FeatureCloud &output_features, boost::shared_ptr<std::vector<int> > &output_indices);
        
        # /** \brief Method for setting the scale parameters for the algorithm
        #  * \param scale_values vector of scales to determine the characteristic of each scaling step
        #  */
        inline void setScalesVector (vector[float] &scale_values)
        
        # /** \brief Method for getting the scale parameters vector */
        inline vector[float] getScalesVector ()
        
        # /** \brief Setter method for the feature estimator
        #  * \param feature_estimator pointer to the feature estimator instance that will be used
        #  * \note the feature estimator instance should already have the input data given beforehand
        #  * and everything set, ready to be given the compute () command
        #  */
        # inline void setFeatureEstimator (FeatureEstimatorPtr feature_estimator)
        
        # /** \brief Getter method for the feature estimator */
        # inline FeatureEstimatorPtr getFeatureEstimator ()
        
        # \brief Provide a pointer to the feature representation to use to convert features to k-D vectors.
        # \param feature_representation the const boost shared pointer to a PointRepresentation
        # inline void setPointRepresentation (const FeatureRepresentationConstPtr& feature_representation)
        
        # \brief Get a pointer to the feature representation used when converting features into k-D vectors. */
        # inline FeatureRepresentationConstPtr const getPointRepresentation ()
        
        # \brief Sets the alpha parameter
        # \param alpha value to replace the current alpha with
        inline void setAlpha (float alpha)
        
        # /** \brief Get the value of the alpha parameter */
        inline float getAlpha ()
        
        # /** \brief Method for setting the distance metric that will be used for computing the difference between feature vectors
        # * \param distance_metric the new distance metric chosen from the NormType enum
        # inline void setDistanceMetric (NormType distance_metric)
        
        # /** \brief Returns the distance metric that is currently used to calculate the difference between feature vectors */
        # inline NormType getDistanceMetric ()
###

# narf.h
# namespace pcl 
# {
#   // Forward declarations
#   class RangeImage;
#   struct InterestPoint;
# 
# #define NARF_DEFAULT_SURFACE_PATCH_PIXEL_SIZE 10
# narf.h
# namespace pcl 
# /**
# * \brief NARF (Normal Aligned Radial Features) is a point feature descriptor type for 3D data.
# * Please refer to pcl/features/narf_descriptor.h if you want the class derived from pcl Feature.
# * See B. Steder, R. B. Rusu, K. Konolige, and W. Burgard
# *     Point Feature Extraction on 3D Range Scans Taking into Account Object Boundaries
# *     In Proc. of the IEEE Int. Conf. on Robotics &Automation (ICRA). 2011. 
# * \author Bastian Steder
# * \ingroup features
# */
# class PCL_EXPORTS Narf
        # public:
        # // =====CONSTRUCTOR & DESTRUCTOR=====
        # //! Constructor
        # Narf();
        # //! Copy Constructor
        # Narf(const Narf& other);
        # //! Destructor
        # ~Narf();
        # 
        # // =====Operators=====
        # //! Assignment operator
        # const Narf& operator=(const Narf& other);
        # 
        # // =====STATIC=====
        # /** The maximum number of openmp threads that can be used in this class */
        # static int max_no_of_threads;
        # 
        # /** Add features extracted at the given interest point and add them to the list */
        # static void extractFromRangeImageAndAddToList (const RangeImage& range_image, const Eigen::Vector3f& interest_point, int descriptor_size, float support_size, bool rotation_invariant, std::vector<Narf*>& feature_list);
        # 
        # /** Same as above */
        # static void extractFromRangeImageAndAddToList (const RangeImage& range_image, float image_x, float image_y, int descriptor_size,float support_size, bool rotation_invariant, std::vector<Narf*>& feature_list);
        # 
        # /** Get a list of features from the given interest points. */
        # static void extractForInterestPoints (const RangeImage& range_image, const PointCloud<InterestPoint>& interest_points, int descriptor_size, float support_size, bool rotation_invariant, std::vector<Narf*>& feature_list);
        # 
        # /** Extract an NARF for every point in the range image. */
        # static void extractForEveryRangeImagePointAndAddToList (const RangeImage& range_image, int descriptor_size, float support_size, bool rotation_invariant, std::vector<Narf*>& feature_list);
        # 
        # // =====PUBLIC METHODS=====
        # /** Method to extract a NARF feature from a certain 3D point using a range image.
        # *  pose determines the coordinate system of the feature, whereas it transforms a point from the world into the feature system.
        # *  This means the interest point at which the feature is extracted will be the inverse application of pose onto (0,0,0).
        # *  descriptor_size_ determines the size of the descriptor,
        # *  support_size determines the support size of the feature, meaning the size in the world it covers */
        # bool extractFromRangeImage (const RangeImage& range_image, const Eigen::Affine3f& pose, int descriptor_size, float support_size,int surface_patch_world_size=NARF_DEFAULT_SURFACE_PATCH_PIXEL_SIZE);
        # 
        # //! Same as above, but determines the transformation from the surface in the range image
        # bool extractFromRangeImage (const RangeImage& range_image, float x, float y, int descriptor_size, float support_size);
        # 
        # //! Same as above
        # bool extractFromRangeImage (const RangeImage& range_image, const InterestPoint& interest_point, int descriptor_size, float support_size);
        # 
        # //! Same as above
        # bool extractFromRangeImage (const RangeImage& range_image, const Eigen::Vector3f& interest_point, int descriptor_size, float support_size);
        # 
        # /** Same as above, but using the rotational invariant version by choosing the best extracted rotation around the normal.
        # *  Use extractFromRangeImageAndAddToList if you want to enable the system to return multiple features with different rotations. */
        # bool extractFromRangeImageWithBestRotation (const RangeImage& range_image, const Eigen::Vector3f& interest_point, int descriptor_size, float support_size);
        # 
        # /* Get the dominant rotations of the current descriptor
        # * \param rotations the returned rotations
        # * \param strength values describing how pronounced the corresponding rotations are
        # */
        # void getRotations (std::vector<float>& rotations, std::vector<float>& strengths) const;
        # 
        # /* Get the feature with a different rotation around the normal
        # * You are responsible for deleting the new features!
        # * \param range_image the source from which the feature is extracted
        # * \param rotations list of angles (in radians)
        # * \param rvps returned features
        # */
        # void getRotatedVersions (const RangeImage& range_image, const std::vector<float>& rotations, std::vector<Narf*>& features) const;
        # 
        # //! Calculate descriptor distance, value in [0,1] with 0 meaning identical and 1 every cell above maximum distance
        # inline float getDescriptorDistance (const Narf& other) const;
        # 
        # //! How many points on each beam of the gradient star are used to calculate the descriptor?
        # inline int getNoOfBeamPoints () const { return (static_cast<int> (pcl_lrint (ceil (0.5f * float (surface_patch_pixel_size_))))); }
        # 
        # //! Copy the descriptor and pose to the point struct Narf36
        # inline void copyToNarf36 (Narf36& narf36) const;
        # 
        # /** Write to file */
        # void saveBinary (const std::string& filename) const;
        # 
        # /** Write to output stream */
        # void saveBinary (std::ostream& file) const;
        # 
        # /** Read from file */
        # void loadBinary (const std::string& filename);
        # /** Read from input stream */
        # void loadBinary (std::istream& file);
        # 
        # //! Create the descriptor from the already set other members
        # bool extractDescriptor (int descriptor_size);
        # 
        # // =====GETTERS=====
        # //! Getter (const) for the descriptor
        # inline const float* getDescriptor () const { return descriptor_;}
        # //! Getter for the descriptor
        # inline float* getDescriptor () { return descriptor_;}
        # //! Getter (const) for the descriptor length
        # inline const int& getDescriptorSize () const { return descriptor_size_;}
        # //! Getter for the descriptor length
        # inline int& getDescriptorSize () { return descriptor_size_;}
        # //! Getter (const) for the position
        # inline const Eigen::Vector3f& getPosition () const { return position_;}
        # //! Getter for the position
        # inline Eigen::Vector3f& getPosition () { return position_;}
        # //! Getter (const) for the 6DoF pose
        # inline const Eigen::Affine3f& getTransformation () const { return transformation_;}
        # //! Getter for the 6DoF pose
        # inline Eigen::Affine3f& getTransformation () { return transformation_;}
        # //! Getter (const) for the pixel size of the surface patch (only one dimension)
        # inline const int& getSurfacePatchPixelSize () const { return surface_patch_pixel_size_;}
        # //! Getter for the pixel size of the surface patch (only one dimension)
        # inline int& getSurfacePatchPixelSize () { return surface_patch_pixel_size_;}
        # //! Getter (const) for the world size of the surface patch
        # inline const float& getSurfacePatchWorldSize () const { return surface_patch_world_size_;}
        # //! Getter for the world size of the surface patch
        # inline float& getSurfacePatchWorldSize () { return surface_patch_world_size_;}
        # //! Getter (const) for the rotation of the surface patch
        # inline const float& getSurfacePatchRotation () const { return surface_patch_rotation_;}
        # //! Getter for the rotation of the surface patch
        # inline float& getSurfacePatchRotation () { return surface_patch_rotation_;}
        # //! Getter (const) for the surface patch
        # inline const float* getSurfacePatch () const { return surface_patch_;}
        # //! Getter for the surface patch
        # inline float* getSurfacePatch () { return surface_patch_;}
        # //! Method to erase the surface patch and free the memory
        # inline void freeSurfacePatch () { delete[] surface_patch_; surface_patch_=NULL; surface_patch_pixel_size_=0; }
        # 
        # // =====SETTERS=====
        # //! Setter for the descriptor
        # inline void setDescriptor (float* descriptor) { descriptor_ = descriptor;}
        # //! Setter for the surface patch
        # inline void setSurfacePatch (float* surface_patch) { surface_patch_ = surface_patch;}
        # 
        # // =====PUBLIC MEMBER VARIABLES=====
        # 
        # // =====PUBLIC STRUCTS=====
        # struct FeaturePointRepresentation : public PointRepresentation<Narf*>
        # {
        #     typedef Narf* PointT;
        #     FeaturePointRepresentation(int nr_dimensions) { this->nr_dimensions_ = nr_dimensions; }
        #     /** \brief Empty destructor */
        #     virtual ~FeaturePointRepresentation () {}
        #     virtual void copyToFloatArray (const PointT& p, float* out) const { memcpy(out, p->getDescriptor(), sizeof(*p->getDescriptor())*this->nr_dimensions_); }
        # };


###

# narf_descriptor.h
# namespace pcl
#     // Forward declarations
#     class RangeImage;
##
# narf_descriptor.h
# namespace pcl
# /** @b Computes NARF feature descriptors for points in a range image
# * See B. Steder, R. B. Rusu, K. Konolige, and W. Burgard
# *     Point Feature Extraction on 3D Range Scans Taking into Account Object Boundaries
# *     In Proc. of the IEEE Int. Conf. on Robotics &Automation (ICRA). 2011. 
# * \author Bastian Steder
# * \ingroup features
# */
# class PCL_EXPORTS NarfDescriptor : public Feature<PointWithRange,Narf36>
        # public:
        # typedef boost::shared_ptr<NarfDescriptor> Ptr;
        # typedef boost::shared_ptr<const NarfDescriptor> ConstPtr;
        # // =====TYPEDEFS=====
        # typedef Feature<PointWithRange,Narf36> BaseClass;
        # 
        # // =====STRUCTS/CLASSES=====
        # struct Parameters
        # {
        #   Parameters() : support_size(-1.0f), rotation_invariant(true) {}
        #   float support_size;
        #   bool rotation_invariant;
        # };
        # 
        # // =====CONSTRUCTOR & DESTRUCTOR=====
        # /** Constructor */
        # NarfDescriptor (const RangeImage* range_image=NULL, const std::vector<int>* indices=NULL);
        # /** Destructor */
        # virtual ~NarfDescriptor();
        # 
        # // =====METHODS=====
        # //! Set input data
        # void setRangeImage (const RangeImage* range_image, const std::vector<int>* indices=NULL);
        # 
        # //! Overwrite the compute function of the base class
        # void compute (cpp.PointCloud[Out]& output);
        # 
        # // =====GETTER=====
        # //! Get a reference to the parameters struct
        # Parameters& getParameters () { return parameters_;}


###

# normal_3d.h
# cdef extern from "pcl/features/normal_3d.h" namespace "pcl":
#     cdef cppclass NormalEstimation[I, N, O]:
#         NormalEstimation()
# 
#   template <typename PointT> inline void
#   computePointNormal (const pcl::PointCloud<PointT> &cloud,
#                       Eigen::Vector4f &plane_parameters, float &curvature)
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
#

# template <typename PointInT, typename PointOutT>
# class NormalEstimation : public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/normal_3d.h" namespace "pcl":
    cdef cppclass NormalEstimation[In, Out](Feature[In, Out]):
        NormalEstimation ()
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudConstPtr PointCloudConstPtr;
        
        # * \brief Compute the Least-Squares plane fit for a given set of points, using their indices,
        # * and return the estimated plane parameters together with the surface curvature.
        # * \param cloud the input point cloud
        # * \param indices the point cloud indices that need to be used
        # * \param plane_parameters the plane parameters as: a, b, c, d (ax + by + cz + d = 0)
        # * \param curvature the estimated surface curvature as a measure of
        # * \f[
        # * \lambda_0 / (\lambda_0 + \lambda_1 + \lambda_2)
        # * \f]
        # inline void computePointNormal (const cpp.PointCloud[In] &cloud, const vector[int] &indices, Eigen::Vector4f &plane_parameters, float &curvature)
        # void computePointNormal (const cpp.PointCloud[In] &cloud, const vector[int] &indices, eigen3.Vector4f &plane_parameters, float &curvature)
        
        # * \brief Compute the Least-Squares plane fit for a given set of points, using their indices,
        # * and return the estimated plane parameters together with the surface curvature.
        # * \param cloud the input point cloud
        # * \param indices the point cloud indices that need to be used
        # * \param nx the resultant X component of the plane normal
        # * \param ny the resultant Y component of the plane normal
        # * \param nz the resultant Z component of the plane normal
        # * \param curvature the estimated surface curvature as a measure of
        # * \f[
        # * \lambda_0 / (\lambda_0 + \lambda_1 + \lambda_2)
        # * \f]
        # inline void computePointNormal (const cpp.PointCloud[In] &cloud, const vector[int] &indices, float &nx, float &ny, float &nz, float &curvature)
        void computePointNormal (const cpp.PointCloud[In] &cloud, const vector[int] &indices, float &nx, float &ny, float &nz, float &curvature)
        
        # * \brief Provide a pointer to the input dataset
        # * \param cloud the const boost shared pointer to a PointCloud message
        # virtual inline void setInputCloud (const PointCloudConstPtr &cloud)
        # * \brief Set the viewpoint.
        # * \param vpx the X coordinate of the viewpoint
        # * \param vpy the Y coordinate of the viewpoint
        # * \param vpz the Z coordinate of the viewpoint
        inline void setViewPoint (float vpx, float vpy, float vpz)
        
        # * \brief Get the viewpoint.
        # * \param [out] vpx x-coordinate of the view point
        # * \param [out] vpy y-coordinate of the view point
        # * \param [out] vpz z-coordinate of the view point
        # * \note this method returns the currently used viewpoint for normal flipping.
        # * If the viewpoint is set manually using the setViewPoint method, this method will return the set view point coordinates.
        # * If an input cloud is set, it will return the sensor origin otherwise it will return the origin (0, 0, 0)
        inline void getViewPoint (float &vpx, float &vpy, float &vpz)
        
        # * \brief sets whether the sensor origin or a user given viewpoint should be used. After this method, the 
        # * normal estimation method uses the sensor origin of the input cloud.
        # * to use a user defined view point, use the method setViewPoint
        inline void useSensorOriginAsViewPoint ()
        

ctypedef NormalEstimation[cpp.PointXYZ, cpp.Normal] NormalEstimation_t
ctypedef NormalEstimation[cpp.PointXYZI, cpp.Normal] NormalEstimation_PointXYZI_t
ctypedef NormalEstimation[cpp.PointXYZRGB, cpp.Normal] NormalEstimation_PointXYZRGB_t
ctypedef NormalEstimation[cpp.PointXYZRGBA, cpp.Normal] NormalEstimation_PointXYZRGBA_t
ctypedef shared_ptr[NormalEstimation[cpp.PointXYZ, cpp.Normal]] NormalEstimationPtr_t
ctypedef shared_ptr[NormalEstimation[cpp.PointXYZI, cpp.Normal]] NormalEstimation_PointXYZI_Ptr_t
ctypedef shared_ptr[NormalEstimation[cpp.PointXYZRGB, cpp.Normal]] NormalEstimation_PointXYZRGB_Ptr_t
ctypedef shared_ptr[NormalEstimation[cpp.PointXYZRGBA, cpp.Normal]] NormalEstimation_PointXYZRGBA_Ptr_t
###

# template <typename PointInT>
# class NormalEstimation<PointInT, Eigen::MatrixXf>: public NormalEstimation<PointInT, pcl::Normal>
# cdef extern from "pcl/features/normal_3d.h" namespace "pcl":
#     cdef cppclass NormalEstimation[In, Eigen::MatrixXf](NormalEstimation[In, pcl::Normal]):
#         NormalEstimation ()
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
###

# normal_3d_omp.h
# template <typename PointInT, typename PointOutT>
# class NormalEstimationOMP: public NormalEstimation<PointInT, PointOutT>
cdef extern from "pcl/features/normal_3d_omp.h" namespace "pcl":
    cdef cppclass NormalEstimationOMP[In, Out](NormalEstimation[In, Out]):
        NormalEstimationOMP ()
        NormalEstimationOMP (unsigned int nr_threads)
        # public:
        # using NormalEstimation<PointInT, PointOutT>::feature_name_;
        # using NormalEstimation<PointInT, PointOutT>::getClassName;
        # using NormalEstimation<PointInT, PointOutT>::indices_;
        # using NormalEstimation<PointInT, PointOutT>::input_;
        # using NormalEstimation<PointInT, PointOutT>::k_;
        # using NormalEstimation<PointInT, PointOutT>::search_parameter_;
        # using NormalEstimation<PointInT, PointOutT>::surface_;
        # using NormalEstimation<PointInT, PointOutT>::getViewPoint;
        # typedef typename NormalEstimation<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # public:
        # /** \brief Initialize the scheduler and set the number of threads to use.
        #     * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
        # */
        inline void setNumberOfThreads (unsigned int nr_threads)
###

# template <typename PointInT>
# class NormalEstimationOMP<PointInT, Eigen::MatrixXf>: public NormalEstimationOMP<PointInT, pcl::Normal>
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


###

# normal_based_signature.h
# template <typename PointT, typename PointNT, typename PointFeature>
# class NormalBasedSignatureEstimation : public FeatureFromNormals<PointT, PointNT, PointFeature>
cdef extern from "pcl/features/normal_based_signature.h" namespace "pcl":
    cdef cppclass NormalBasedSignatureEstimation[In, NT, Feature](FeatureFromNormals[In,  NT, Feature]):
        NormalBasedSignatureEstimation ()
        # public:
        # using Feature<PointT, PointFeature>::input_;
        # using Feature<PointT, PointFeature>::tree_;
        # using Feature<PointT, PointFeature>::search_radius_;
        # using PCLBase<PointT>::indices_;
        # using FeatureFromNormals<PointT, PointNT, PointFeature>::normals_;
        # typedef pcl::PointCloud<PointFeature> FeatureCloud;
        # typedef typename boost::shared_ptr<NormalBasedSignatureEstimation<PointT, PointNT, PointFeature> > Ptr;
        # typedef typename boost::shared_ptr<const NormalBasedSignatureEstimation<PointT, PointNT, PointFeature> > ConstPtr;
        # /** \brief Setter method for the N parameter - the length of the columns used for the Discrete Fourier Transform. 
        # * \param[in] n the length of the columns used for the Discrete Fourier Transform. 
        inline void setN (size_t n)
        # /** \brief Returns the N parameter - the length of the columns used for the Discrete Fourier Transform. */
        # inline size_t getN ()
        # /** \brief Setter method for the M parameter - the length of the rows used for the Discrete Cosine Transform.
        # * \param[in] m the length of the rows used for the Discrete Cosine Transform.
        inline void setM (size_t m)
        # /** \brief Returns the M parameter - the length of the rows used for the Discrete Cosine Transform */
        inline size_t getM ()
        # /** \brief Setter method for the N' parameter - the number of columns to be taken from the matrix of DFT and DCT
        # * values that will be contained in the output feature vector
        # * \note This value directly influences the dimensions of the type of output points (PointFeature)
        # * \param[in] n_prime the number of columns from the matrix of DFT and DCT that will be contained in the output
        inline void setNPrime (size_t n_prime)
        # /** \brief Returns the N' parameter - the number of rows to be taken from the matrix of DFT and DCT
        # * values that will be contained in the output feature vector
        # * \note This value directly influences the dimensions of the type of output points (PointFeature)
        inline size_t getNPrime ()
        # * \brief Setter method for the M' parameter - the number of rows to be taken from the matrix of DFT and DCT
        # * values that will be contained in the output feature vector
        # * \note This value directly influences the dimensions of the type of output points (PointFeature)
        # * \param[in] m_prime the number of rows from the matrix of DFT and DCT that will be contained in the output
        inline void setMPrime (size_t m_prime)
        # * \brief Returns the M' parameter - the number of rows to be taken from the matrix of DFT and DCT
        # * values that will be contained in the output feature vector
        # * \note This value directly influences the dimensions of the type of output points (PointFeature)
        inline size_t getMPrime ()
        # * \brief Setter method for the scale parameter - used to determine the radius of the sampling disc around the
        # * point of interest - linked to the smoothing scale of the input cloud
        inline void setScale (float scale)
        # * \brief Returns the scale parameter - used to determine the radius of the sampling disc around the
        # * point of interest - linked to the smoothing scale of the input cloud
        inline float getScale ()
###

# pfh.h
# template <typename PointInT, typename PointNT, typename PointOutT = pcl::PFHSignature125>
# class PFHEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/pfh.h" namespace "pcl":
    cdef cppclass PFHEstimation[In, NT, Out](FeatureFromNormals[In,  NT, Out]):
        PFHEstimation ()
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::input_;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudIn  PointCloudIn;
        # * \brief Set the maximum internal cache size. Defaults to 2GB worth of entries.
        # * \param[in] cache_size maximum cache size 
        inline void setMaximumCacheSize (unsigned int cache_size)
        # /** \brief Get the maximum internal cache size. */
        inline unsigned int getMaximumCacheSize ()
        # * \brief Set whether to use an internal cache mechanism for removing redundant calculations or not. 
        # * \note Depending on how the point cloud is ordered and how the nearest
        # * neighbors are estimated, using a cache could have a positive or a
        # * negative influence. Please test with and without a cache on your
        # * data, and choose whatever works best!
        # * See \ref setMaximumCacheSize for setting the maximum cache size
        # * \param[in] use_cache set to true to use the internal cache, false otherwise
        inline void setUseInternalCache (bool use_cache)
        # /** \brief Get whether the internal cache is used or not for computing the PFH features. */
        inline bool getUseInternalCache ()
        # * \brief Compute the 4-tuple representation containing the three angles and one distance between two points
        # * represented by Cartesian coordinates and normals.
        # * \note For explanations about the features, please see the literature mentioned above (the order of the
        # * features might be different).
        # * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
        # * \param[in] normals the dataset containing the surface normals (assuming normalized vectors) at each point in cloud
        # * \param[in] p_idx the index of the first point (source)
        # * \param[in] q_idx the index of the second point (target)
        # * \param[out] f1 the first angular feature (angle between the projection of nq_idx and u)
        # * \param[out] f2 the second angular feature (angle between nq_idx and v)
        # * \param[out] f3 the third angular feature (angle between np_idx and |p_idx - q_idx|)
        # * \param[out] f4 the distance feature (p_idx - q_idx)
        # * \note For efficiency reasons, we assume that the point data passed to the method is finite.
        bool computePairFeatures (const cpp.PointCloud[In] &cloud, const cpp.PointCloud[NT] &normals, 
                                    int p_idx, int q_idx, float &f1, float &f2, float &f3, float &f4);
        # * \brief Estimate the PFH (Point Feature Histograms) individual signatures of the three angular (f1, f2, f3)
        # * features for a given point based on its spatial neighborhood of 3D points with normals
        # * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
        # * \param[in] normals the dataset containing the surface normals at each point in \a cloud
        # * \param[in] indices the k-neighborhood point indices in the dataset
        # * \param[in] nr_split the number of subdivisions for each angular feature interval
        # * \param[out] pfh_histogram the resultant (combinatorial) PFH histogram representing the feature at the query point
        # void computePointPFHSignature (const cpp.PointCloud[In] &cloud, const cpp.PointCloud[NT] &normals, 
        #                         const vector[int] &indices, int nr_split, Eigen::VectorXf &pfh_histogram);


###

# template <typename PointInT, typename PointNT>
# class PFHEstimation<PointInT, PointNT, Eigen::MatrixXf> : public PFHEstimation<PointInT, PointNT, pcl::PFHSignature125>
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

###

# pfhrgb.h
# template <typename PointInT, typename PointNT, typename PointOutT = pcl::PFHRGBSignature250>
# class PFHRGBEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/pfhrgb.h" namespace "pcl":
    cdef cppclass PFHRGBEstimation[In, NT, Out](FeatureFromNormals[In,  NT, Out]):
        PFHRGBEstimation ()
        # public:
        # using PCLBase<PointInT>::indices_;
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        bool computeRGBPairFeatures (const cpp.PointCloud[In] &cloud, const cpp.PointCloud[NT] &normals,
                              int p_idx, int q_idx,
                              float &f1, float &f2, float &f3, float &f4, float &f5, float &f6, float &f7)
        # void computePointPFHRGBSignature (const cpp.PointCloud[In] &cloud, const cpp.PointCloud[NT] &normals,
        #                            const vector[int] &indices, int nr_split, Eigen::VectorXf &pfhrgb_histogram)


###

# ppf.h
# template <typename PointInT, typename PointNT, typename PointOutT>
# class PPFEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/ppf.h" namespace "pcl":
    cdef cppclass PPFEstimation[In, NT, Out](FeatureFromNormals[In,  NT, Out]):
        PPFEstimation ()
        # public:
        # using PCLBase<PointInT>::indices_;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # typedef pcl::PointCloud<PointOutT> PointCloudOut;

# template <typename PointInT, typename PointNT>
# class PPFEstimation<PointInT, PointNT, Eigen::MatrixXf> : public PPFEstimation<PointInT, PointNT, pcl::PPFSignature>
#     public:
#       using PPFEstimation<PointInT, PointNT, pcl::PPFSignature>::getClassName;
#       using PPFEstimation<PointInT, PointNT, pcl::PPFSignature>::input_;
#       using PPFEstimation<PointInT, PointNT, pcl::PPFSignature>::normals_;
#       using PPFEstimation<PointInT, PointNT, pcl::PPFSignature>::indices_;
# 
###

# ppfrgb.h
# template <typename PointInT, typename PointNT, typename PointOutT>
# class PPFRGBEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/ppfrgb.h" namespace "pcl":
    cdef cppclass PPFRGBEstimation[In, NT, Out](FeatureFromNormals[In,  NT, Out]):
        PPFRGBEstimation ()
        # public:
        # using PCLBase<PointInT>::indices_;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # typedef pcl::PointCloud<PointOutT> PointCloudOut;

# template <typename PointInT, typename PointNT, typename PointOutT>
# class PPFRGBRegionEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
#       PPFRGBRegionEstimation ();
#     public:
#       using PCLBase<PointInT>::indices_;
#       using Feature<PointInT, PointOutT>::input_;
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::search_radius_;
#       using Feature<PointInT, PointOutT>::tree_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
#       typedef pcl::PointCloud<PointOutT> PointCloudOut;
###

# principal_curvatures.h
# template <typename PointInT, typename PointNT, typename PointOutT = pcl::PrincipalCurvatures>
# class PrincipalCurvaturesEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/principal_curvatures.h" namespace "pcl":
    cdef cppclass PrincipalCurvaturesEstimation[In, NT, Out](FeatureFromNormals[In,  NT, Out]):
        PrincipalCurvaturesEstimation ()
#       public:
#       using Feature<PointInT, PointOutT>::feature_name_;
#       using Feature<PointInT, PointOutT>::getClassName;
#       using Feature<PointInT, PointOutT>::indices_;
#       using Feature<PointInT, PointOutT>::k_;
#       using Feature<PointInT, PointOutT>::search_parameter_;
#       using Feature<PointInT, PointOutT>::surface_;
#       using Feature<PointInT, PointOutT>::input_;
#       using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#       typedef pcl::PointCloud<PointInT> PointCloudIn;
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
#       void computePointPrincipalCurvatures (const pcl::PointCloud<PointNT> &normals,
#                                        int p_idx, const std::vector<int> &indices,
#                                        float &pcx, float &pcy, float &pcz, float &pc1, float &pc2);

# template <typename PointInT, typename PointNT>
# class PrincipalCurvaturesEstimation<PointInT, PointNT, Eigen::MatrixXf> : public PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>
#     public:
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::indices_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::k_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::search_parameter_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::surface_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::compute;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::input_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::normals_;
###

# range_image_border_extractor.h
# namespace pcl
# class RangeImage;
# template <typename PointType>
# class PointCloud;

# class PCL_EXPORTS RangeImageBorderExtractor : public Feature<PointWithRange, BorderDescription>
cdef extern from "pcl/features/range_image_border_extractor.h" namespace "pcl":
    cdef cppclass RangeImageBorderExtractor(Feature[cpp.PointWithRange, cpp.BorderDescription]):
        RangeImageBorderExtractor ()
        RangeImageBorderExtractor (const pcl_r_img.RangeImage range_image)
        # =====CONSTRUCTOR & DESTRUCTOR=====
        # Constructor
        # RangeImageBorderExtractor (const RangeImage* range_image = NULL)
        # /** Destructor */
        # ~RangeImageBorderExtractor ();
        # 
        
        # public:
        # // =====PUBLIC STRUCTS=====
        # Stores some information extracted from the neighborhood of a point
        # struct LocalSurface
        # {
        #   LocalSurface () : 
        #     normal (), neighborhood_mean (), eigen_values (), normal_no_jumps (), 
        #     neighborhood_mean_no_jumps (), eigen_values_no_jumps (), max_neighbor_distance_squared () {}
        # 
        #   Eigen::Vector3f normal;
        #   Eigen::Vector3f neighborhood_mean;
        #   Eigen::Vector3f eigen_values;
        #   Eigen::Vector3f normal_no_jumps;
        #   Eigen::Vector3f neighborhood_mean_no_jumps;
        #   Eigen::Vector3f eigen_values_no_jumps;
        #   float max_neighbor_distance_squared;
        # };
        
        # Stores the indices of the shadow border corresponding to obstacle borders
        # struct ShadowBorderIndices 
        # {
        #   ShadowBorderIndices () : left (-1), right (-1), top (-1), bottom (-1) {}
        #   int left, right, top, bottom;
        # };
        
        # Parameters used in this class
        # struct Parameters
        # {
        #   Parameters () : max_no_of_threads(1), pixel_radius_borders (3), pixel_radius_plane_extraction (2), pixel_radius_border_direction (2), 
        #                  minimum_border_probability (0.8f), pixel_radius_principal_curvature (2) {}
        #   int max_no_of_threads;
        #   int pixel_radius_borders;
        #   int pixel_radius_plane_extraction;
        #   int pixel_radius_border_direction;
        #   float minimum_border_probability;
        #   int pixel_radius_principal_curvature;
        # };
        
        # =====STATIC METHODS=====
        # brief Take the information from BorderTraits to calculate the local direction of the border
        # param border_traits contains the information needed to calculate the border angle
        # 
        # static inline float getObstacleBorderAngle (const BorderTraits& border_traits);
        
        # // =====METHODS=====
        # /** \brief Provide a pointer to the range image
        #   * \param range_image a pointer to the range_image
        # void setRangeImage (const RangeImage* range_image);
        void setRangeImage (const pcl_r_img.RangeImage range_image)
        
        # brief Erase all data calculated for the current range image
        void clearData ()
        
        # brief Get the 2D directions in the range image from the border directions - probably mainly useful for 
        # visualization 
        # float* getAnglesImageForBorderDirections ();
        # float[] getAnglesImageForBorderDirections ()
        
        # brief Get the 2D directions in the range image from the surface change directions - probably mainly useful for visualization 
        # float* getAnglesImageForSurfaceChangeDirections ();
        # float[] getAnglesImageForSurfaceChangeDirections ()
        
        # /** Overwrite the compute function of the base class */
        # void compute (PointCloudOut& output);
        # void compute (cpp.PointCloud[Out]& output)
        
        # =====GETTER=====
        # Parameters& getParameters () { return (parameters_); }
        # Parameters& getParameters ()
        # 
        # bool hasRangeImage () const { return range_image_ != NULL; }
        bool hasRangeImage ()
        
        # const RangeImage& getRangeImage () const { return *range_image_; }
        const pcl_r_img.RangeImage getRangeImage ()
        
        # float* getBorderScoresLeft ()   { extractBorderScoreImages (); return border_scores_left_; }
        # float* getBorderScoresRight ()  { extractBorderScoreImages (); return border_scores_right_; }
        # float* getBorderScoresTop ()    { extractBorderScoreImages (); return border_scores_top_; }
        # float* getBorderScoresBottom () { extractBorderScoreImages (); return border_scores_bottom_; }
        # 
        # LocalSurface** getSurfaceStructure () { extractLocalSurfaceStructure (); return surface_structure_; }
        # PointCloudOut& getBorderDescriptions () { classifyBorders (); return *border_descriptions_; }
        # ShadowBorderIndices** getShadowBorderInformations () { findAndEvaluateShadowBorders (); return shadow_border_informations_; }
        # Eigen::Vector3f** getBorderDirections () { calculateBorderDirections (); return border_directions_; }
        # float* getSurfaceChangeScores () { calculateSurfaceChanges (); return surface_change_scores_; }
        # Eigen::Vector3f* getSurfaceChangeDirections () { calculateSurfaceChanges (); return surface_change_directions_; }


###

# rift.h
# template <typename PointInT, typename GradientT, typename PointOutT>
# class RIFTEstimation: public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/rift.h" namespace "pcl":
    cdef cppclass RIFTEstimation[In, GradientT, Out](Feature[In, Out]):
        RIFTEstimation ()
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::tree_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # typedef typename pcl::PointCloud<PointInT> PointCloudIn;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename pcl::PointCloud<GradientT> PointCloudGradient;
        # typedef typename PointCloudGradient::Ptr PointCloudGradientPtr;
        # typedef typename PointCloudGradient::ConstPtr PointCloudGradientConstPtr;
        # typedef typename boost::shared_ptr<RIFTEstimation<PointInT, GradientT, PointOutT> > Ptr;
        # typedef typename boost::shared_ptr<const RIFTEstimation<PointInT, GradientT, PointOutT> > ConstPtr;
        
        # brief Provide a pointer to the input gradient data
        # param[in] gradient a pointer to the input gradient data
        # inline void setInputGradient (const PointCloudGradientConstPtr &gradient)
        
        # /** \brief Returns a shared pointer to the input gradient data */
        # inline PointCloudGradientConstPtr getInputGradient () const 
        
        # brief Set the number of bins to use in the distance dimension of the RIFT descriptor
        # param[in] nr_distance_bins the number of bins to use in the distance dimension of the RIFT descriptor
        # inline void setNrDistanceBins (int nr_distance_bins)
        
        # /** \brief Returns the number of bins in the distance dimension of the RIFT descriptor. */
        # inline int getNrDistanceBins () const
        
        # /** \brief Set the number of bins to use in the gradient orientation dimension of the RIFT descriptor
        # * \param[in] nr_gradient_bins the number of bins to use in the gradient orientation dimension of the RIFT descriptor
        # inline void setNrGradientBins (int nr_gradient_bins)
        
        # /** \brief Returns the number of bins in the gradient orientation dimension of the RIFT descriptor. */
        # inline int getNrGradientBins () const
        
        # /** \brief Estimate the Rotation Invariant Feature Transform (RIFT) descriptor for a given point based on its 
        # * spatial neighborhood of 3D points and the corresponding intensity gradient vector field
        # * \param[in] cloud the dataset containing the Cartesian coordinates of the points
        # * \param[in] gradient the dataset containing the intensity gradient at each point in \a cloud
        # * \param[in] p_idx the index of the query point in \a cloud (i.e. the center of the neighborhood)
        # * \param[in] radius the radius of the RIFT feature
        # * \param[in] indices the indices of the points that comprise \a p_idx's neighborhood in \a cloud
        # * \param[in] squared_distances the squared distances from the query point to each point in the neighborhood
        # * \param[out] rift_descriptor the resultant RIFT descriptor
        # void computeRIFT (const PointCloudIn &cloud, const PointCloudGradient &gradient, int p_idx, float radius,
        #            const std::vector<int> &indices, const std::vector<float> &squared_distances, 
        #            Eigen::MatrixXf &rift_descriptor);


# ctypedef
# 
###

# template <typename PointInT, typename GradientT>
# class RIFTEstimation<PointInT, GradientT, Eigen::MatrixXf>: public RIFTEstimation<PointInT, GradientT, pcl::Histogram<32> >
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
###

# shot.h
# template <typename PointInT, typename PointNT, typename PointOutT, typename PointRFT = pcl::ReferenceFrame>
# class SHOTEstimationBase : public FeatureFromNormals<PointInT, PointNT, PointOutT>,
#                            public FeatureWithLocalReferenceFrames<PointInT, PointRFT>
cdef extern from "pcl/features/shot.h" namespace "pcl":
    cdef cppclass SHOTEstimationBase[In, NT, Out, RET](Feature[In, Out]):
        SHOTEstimationBase ()
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
#       typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
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
###

# template <typename PointInT, typename PointNT, typename PointOutT = pcl::SHOT352, typename PointRFT = pcl::ReferenceFrame>
# class SHOTEstimation : public SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>
cdef extern from "pcl/features/shot.h" namespace "pcl":
    cdef cppclass SHOTEstimation[In, NT, Out, RFT](SHOTEstimationBase[In, NT, Out, RFT]):
        SHOTEstimation ()
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
#       typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
# 
#       /** \brief Estimate the SHOT descriptor for a given point based on its spatial neighborhood of 3D points with normals
#         * \param[in] index the index of the point in indices_
#         * \param[in] indices the k-neighborhood point indices in surface_
#         * \param[in] sqr_dists the k-neighborhood point distances in surface_
#         * \param[out] shot the resultant SHOT descriptor representing the feature at the query point
#         */
#       virtual void computePointSHOT (const int index,
#                         const std::vector<int> &indices,
#                         const std::vector<float> &sqr_dists,
#                         Eigen::VectorXf &shot);


###

# template <typename PointInT, typename PointNT, typename PointRFT>
# class PCL_DEPRECATED_CLASS (SHOTEstimation, "SHOTEstimation<..., pcl::SHOT, ...> IS DEPRECATED, USE SHOTEstimation<..., pcl::SHOT352, ...> INSTEAD")
# <PointInT, PointNT, pcl::SHOT, PointRFT>
# : public SHOTEstimationBase<PointInT, PointNT, pcl::SHOT, PointRFT>
# cdef extern from "pcl/features/shot.h" namespace "pcl":
#    cdef cppclass PCL_DEPRECATED_CLASS[In, NT, RFT](SHOTEstimation[In, NT, pcl::SHOT, RFT]):
#        SHOTEstimation ()
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


###

# template <typename PointInT, typename PointNT, typename PointRFT>
# class SHOTEstimation<PointInT, PointNT, Eigen::MatrixXf, PointRFT> : public SHOTEstimation<PointInT, PointNT, pcl::SHOT352, PointRFT>
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
#       void computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
# 
#     
#       /** \brief Make the compute (&PointCloudOut); inaccessible from outside the class
#         * \param[out] output the output point cloud
#         */
#       void compute (pcl::PointCloud<pcl::SHOT352> &) { assert(0); }
#   };


###

# template <typename PointInT, typename PointNT, typename PointOutT = pcl::SHOT1344, typename PointRFT = pcl::ReferenceFrame>
# class SHOTColorEstimation : public SHOTEstimationBase<PointInT, PointNT, PointOutT, PointRFT>
cdef extern from "pcl/features/shot.h" namespace "pcl":
    cdef cppclass SHOTColorEstimation[In, NT, Out, RFT](SHOTEstimationBase[In, NT, Out, RFT]):
        SHOTColorEstimation ()
        #       SHOTColorEstimation (bool describe_shape = true,
        #                            bool describe_color = true)
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
        #       typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
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
###

# template <typename PointInT, typename PointNT, typename PointRFT>
# class SHOTColorEstimation<PointInT, PointNT, Eigen::MatrixXf, PointRFT> : public SHOTColorEstimation<PointInT, PointNT, pcl::SHOT1344, PointRFT>
# cdef extern from "pcl/features/shot.h" namespace "pcl":
#     cdef cppclass SHOTColorEstimation[In, NT, Out, RFT](SHOTColorEstimation[In, NT, Out, RFT]):
#         SHOTColorEstimation ()
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
###

# template <typename PointNT, typename PointRFT>
# class PCL_DEPRECATED_CLASS (SHOTEstimation, "SHOTEstimation<pcl::PointXYZRGBA,...,pcl::SHOT,...> IS DEPRECATED, USE SHOTEstimation<pcl::PointXYZRGBA,...,pcl::SHOT352,...> FOR SHAPE AND SHOTColorEstimation<pcl::PointXYZRGBA,...,pcl::SHOT1344,...> FOR SHAPE+COLOR INSTEAD")
#   <pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>
#   : public SHOTEstimationBase<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT> 
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

###

# template <typename PointNT, typename PointRFT>
# class PCL_DEPRECATED_CLASS (SHOTEstimation, "SHOTEstimation<pcl::PointXYZRGBA,...,Eigen::MatrixXf,...> IS DEPRECATED, USE SHOTColorEstimation<pcl::PointXYZRGBA,...,Eigen::MatrixXf,...> FOR SHAPE AND SHAPE+COLOR INSTEAD")
# <pcl::PointXYZRGBA, PointNT, Eigen::MatrixXf, PointRFT>
# : public SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>
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
###

# shot_lrf.h
#  template<typename PointInT, typename PointOutT = ReferenceFrame>
#  class SHOTLocalReferenceFrameEstimation : public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/shot_lrf.h" namespace "pcl":
    cdef cppclass SHOTLocalReferenceFrameEstimation[In, Out](Feature[In, Out]):
        PrincipalCurvaturesEstimation ()
        # protected:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # //using Feature<PointInT, PointOutT>::searchForNeighbors;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::tree_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # * \brief Computes disambiguated local RF for a point index
        # * \param[in] cloud input point cloud
        # * \param[in] search_radius the neighborhood radius
        # * \param[in] central_point the point from the input_ cloud at which the local RF is computed
        # * \param[in] indices the neighbours indices
        # * \param[in] dists the squared distances to the neighbours
        # * \param[out] rf reference frame to compute
        # float getLocalRF (const int &index, Eigen::Matrix3f &rf)
        # * \brief Feature estimation method.
        # \param[out] output the resultant features
        # virtual void computeFeature (PointCloudOut &output)
        # * \brief Feature estimation method.
        # * \param[out] output the resultant features
        # virtual void computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output)
###

# template <typename PointInT, typename PointNT>
# class PrincipalCurvaturesEstimation<PointInT, PointNT, Eigen::MatrixXf> : public PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>
#     public:
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::indices_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::k_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::search_parameter_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::surface_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::compute;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::input_;
#       using PrincipalCurvaturesEstimation<PointInT, PointNT, pcl::PrincipalCurvatures>::normals_;
###

# shot_lrf_omp.h
# template<typename PointInT, typename PointOutT = ReferenceFrame>
# class SHOTLocalReferenceFrameEstimationOMP : public SHOTLocalReferenceFrameEstimation<PointInT, PointOutT>
cdef extern from "pcl/features/shot_lrf_omp.h" namespace "pcl":
    cdef cppclass SHOTLocalReferenceFrameEstimationOMP[In, Out](SHOTLocalReferenceFrameEstimation[In, Out]):
        SHOTLocalReferenceFrameEstimationOMP ()
        # public:
        # brief Initialize the scheduler and set the number of threads to use.
        # param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
        # inline void setNumberOfThreads (unsigned int nr_threads)

###

# shot_omp.h
# template <typename PointInT, typename PointNT, typename PointOutT = pcl::SHOT352, typename PointRFT = pcl::ReferenceFrame>
# class SHOTEstimationOMP : public SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>
cdef extern from "pcl/features/shot_omp.h" namespace "pcl":
    cdef cppclass SHOTEstimationOMP[In, NT, Out, RFT](SHOTEstimation[In, NT, Out, RFT]):
        SHOTEstimationOMP ()
        # SHOTEstimationOMP (unsigned int nr_threads = - 1)
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::fake_surface_;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
        # using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::descLength_;
        # using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::nr_grid_sector_;
        # using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::nr_shape_bins_;
        # using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::sqradius_;
        # using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::radius3_4_;
        # using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::radius1_4_;
        # using SHOTEstimation<PointInT, PointNT, PointOutT, PointRFT>::radius1_2_;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # 
        # /** \brief Initialize the scheduler and set the number of threads to use.
        #  * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
        inline void setNumberOfThreads (unsigned int nr_threads)
        
###

# template <typename PointInT, typename PointNT, typename PointOutT = pcl::SHOT1344, typename PointRFT = pcl::ReferenceFrame>
# class SHOTColorEstimationOMP : public SHOTColorEstimation<PointInT, PointNT, PointOutT, PointRFT>
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
#       typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#       typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
# 
#       /** \brief Empty constructor. */
#       SHOTColorEstimationOMP (bool describe_shape = true,
#                               bool describe_color = true,
#                               unsigned int nr_threads = - 1)
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       inline void setNumberOfThreads (unsigned int nr_threads)
###

# template <typename PointInT, typename PointNT, typename PointRFT>
# class PCL_DEPRECATED_CLASS (SHOTEstimationOMP, "SHOTEstimationOMP<..., pcl::SHOT, ...> IS DEPRECATED, USE SHOTEstimationOMP<..., pcl::SHOT352, ...> INSTEAD")
# <PointInT, PointNT, pcl::SHOT, PointRFT>
# : public SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT>
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
#       typedef typename Feature<PointInT, pcl::SHOT>::PointCloudOut PointCloudOut;
#       typedef typename Feature<PointInT, pcl::SHOT>::PointCloudIn PointCloudIn;
#       /** \brief Empty constructor. */
#       SHOTEstimationOMP (unsigned int nr_threads = - 1, int nr_shape_bins = 10)
#         : SHOTEstimation<PointInT, PointNT, pcl::SHOT, PointRFT> (nr_shape_bins), threads_ ()
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       inline void setNumberOfThreads (unsigned int nr_threads)
# 
###

# template <typename PointNT, typename PointRFT>
# class PCL_DEPRECATED_CLASS (SHOTEstimationOMP, "SHOTEstimationOMP<pcl::PointXYZRGBA,...,pcl::SHOT,...> IS DEPRECATED, USE SHOTEstimationOMP<pcl::PointXYZRGBA,...,pcl::SHOT352,...> FOR SHAPE AND SHOTColorEstimationOMP<pcl::PointXYZRGBA,...,pcl::SHOT1344,...> FOR SHAPE+COLOR INSTEAD")
# <pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>
# : public SHOTEstimation<pcl::PointXYZRGBA, PointNT, pcl::SHOT, PointRFT>
#       public:
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
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       inline void
#       setNumberOfThreads (unsigned int nr_threads)
###

# spin_image.h
# template <typename PointInT, typename PointNT, typename PointOutT>
# class SpinImageEstimation : public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/spin_image.h" namespace "pcl":
    cdef cppclass SpinImageEstimation[In, NT, Out](Feature[In, Out]):
        SpinImageEstimation ()
        # SpinImageEstimation (unsigned int image_width = 8,
        #                    double support_angle_cos = 0.0,   // when 0, this is bogus, so not applied
        #                    unsigned int min_pts_neighb = 0);
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using Feature<PointInT, PointOutT>::fake_surface_;
        # using PCLBase<PointInT>::input_;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename pcl::PointCloud<PointNT> PointCloudN;
        # typedef typename PointCloudN::Ptr PointCloudNPtr;
        # typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
        # typedef typename pcl::PointCloud<PointInT> PointCloudIn;
        # typedef typename PointCloudIn::Ptr PointCloudInPtr;
        # typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # typedef typename boost::shared_ptr<SpinImageEstimation<PointInT, PointNT, PointOutT> > Ptr;
        # typedef typename boost::shared_ptr<const SpinImageEstimation<PointInT, PointNT, PointOutT> > ConstPtr;
        # /** \brief Sets spin-image resolution.
        #  * \param[in] bin_count spin-image resolution, number of bins along one dimension
        void setImageWidth (unsigned int bin_count)
        # /** \brief Sets the maximum angle for the point normal to get to support region.
        #   * \param[in] support_angle_cos minimal allowed cosine of the angle between 
        #   *   the normals of input point and search surface point for the point 
        #   *   to be retained in the support
        void setSupportAngle (double support_angle_cos)
        # /** \brief Sets minimal points count for spin image computation.
        #   * \param[in] min_pts_neighb min number of points in the support to correctly estimate 
        #   *   spin-image. If at some point the support contains less points, exception is thrown
        void setMinPointCountInNeighbourhood (unsigned int min_pts_neighb)
        # /** \brief Provide a pointer to the input dataset that contains the point normals of 
        #  * the input XYZ dataset given by \ref setInputCloud
        #  * \attention The input normals given by \ref setInputNormals have to match
        #  * the input point cloud given by \ref setInputCloud. This behavior is
        #  * different than feature estimation methods that extend \ref
        #  * FeatureFromNormals, which match the normals with the search surface.
        #  * \param[in] normals the const boost shared pointer to a PointCloud of normals. 
        #  * By convention, L2 norm of each normal should be 1. 
        # inline void setInputNormals (const PointCloudNConstPtr &normals)
        # /** \brief Sets single vector a rotation axis for all input points.
        #   * It could be useful e.g. when the vertical axis is known.
        #   * \param[in] axis unit-length vector that serves as rotation axis for reference frame
        # void setRotationAxis (const PointNT& axis)
        # /** \brief Sets array of vectors as rotation axes for input points.
        #  * Useful e.g. when one wants to use tangents instead of normals as rotation axes
        #  * \param[in] axes unit-length vectors that serves as rotation axes for 
        #  *   the corresponding input points' reference frames
        # void setInputRotationAxes (const PointCloudNConstPtr& axes)
        # /** \brief Sets input normals as rotation axes (default setting). */
        void useNormalsAsRotationAxis () 
        # /** \brief Sets/unsets flag for angular spin-image domain.
        #   * Angular spin-image differs from the vanilla one in the way that not 
        #   * the points are collected in the bins but the angles between their
        #   * normals and the normal to the reference point. For further
        #   * information please see 
        #   * Endres, F., Plagemann, C., Stachniss, C., & Burgard, W. (2009). 
        #   * Unsupervised Discovery of Object Classes from Range Data using Latent Dirichlet Allocation. 
        #   * In Robotics: Science and Systems. Seattle, USA.
        #   * \param[in] is_angular true for angular domain, false for point domain
        void setAngularDomain (bool is_angular = true)
        # /** \brief Sets/unsets flag for radial spin-image structure.
        #   * 
        #   * Instead of rectangular coordinate system for reference frame 
        #   * polar coordinates are used. Binning is done depending on the distance and 
        #   * inclination angle from the reference point
        #   * \param[in] is_radial true for radial spin-image structure, false for rectangular
        # */
        void setRadialStructure (bool is_radial = true)


####

# template <typename PointInT, typename PointNT>
# class SpinImageEstimation<PointInT, PointNT, Eigen::MatrixXf> : public SpinImageEstimation<PointInT, PointNT, pcl::Histogram<153> >
# cdef extern from "pcl/features/spin_image.h" namespace "pcl":
#    cdef cppclass SpinImageEstimation[In, NT, Eigen::MatrixXf](SpinImageEstimation[In, NT, pcl::Histogram<153>]):
#       SpinImageEstimation ()
#       public:
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
#       SpinImageEstimation<PointInT, PointNT, pcl::Histogram<153> > (image_width, support_angle_cos, min_pts_neighb) {}
###

# statistical_multiscale_interest_region_extraction.h
# template <typename PointT>
# class StatisticalMultiscaleInterestRegionExtraction : public PCLBase<PointT>
cdef extern from "pcl/features/statistical_multiscale_interest_region_extraction.h" namespace "pcl":
    cdef cppclass StatisticalMultiscaleInterestRegionExtraction[T](cpp.PCLBase[T]):
        StatisticalMultiscaleInterestRegionExtraction ()
        # public:
        # typedef boost::shared_ptr <std::vector<int> > IndicesPtr;
        # typedef typename boost::shared_ptr<StatisticalMultiscaleInterestRegionExtraction<PointT> > Ptr;
        # typedef typename boost::shared_ptr<const StatisticalMultiscaleInterestRegionExtraction<PointT> > ConstPtr;
        
        # brief Method that generates the underlying nearest neighbor graph based on the input point cloud
        void generateCloudGraph ()
        
        # brief The method to be called in order to run the algorithm and produce the resulting
        # set of regions of interest
        # void computeRegionsOfInterest (list[IndicesPtr_t]& rois)
        
        # brief Method for setting the scale parameters for the algorithm
        # param scale_values vector of scales to determine the size of each scaling step
        inline void setScalesVector (vector[float] &scale_values)
        
        # brief Method for getting the scale parameters vector */
        inline vector[float] getScalesVector ()
###

# usc.h
# template <typename PointInT, typename PointOutT, typename PointRFT = pcl::ReferenceFrame>
# class UniqueShapeContext : public Feature<PointInT, PointOutT>,
#                            public FeatureWithLocalReferenceFrames<PointInT, PointRFT>
# cdef extern from "pcl/features/usc.h" namespace "pcl":
#     cdef cppclass UniqueShapeContext[In, Out, RFT](Feature[In, Out], FeatureWithLocalReferenceFrames[In, RFT]):
#        VFHEstimation ()
#        public:
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
#        typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#        typedef typename Feature<PointInT, PointOutT>::PointCloudIn PointCloudIn;
#        typedef typename boost::shared_ptr<UniqueShapeContext<PointInT, PointOutT, PointRFT> > Ptr;
#        typedef typename boost::shared_ptr<const UniqueShapeContext<PointInT, PointOutT, PointRFT> > ConstPtr;
#        /** \brief Constructor. */
#        UniqueShapeContext () :
#       /** \brief Set the number of bins along the azimuth
#         * \param[in] bins the number of bins along the azimuth
#       inline void setAzimuthBins (size_t bins)
#       /** \return The number of bins along the azimuth. */
#       inline size_t getAzimuthBins () const
#       /** \brief Set the number of bins along the elevation
#         * \param[in] bins the number of bins along the elevation
#         */
#       inline void setElevationBins (size_t bins)
#       /** \return The number of bins along the elevation */
#       inline size_t getElevationBins () const
#       /** \brief Set the number of bins along the radii
#         * \param[in] bins the number of bins along the radii
#       inline void setRadiusBins (size_t bins)
#       /** \return The number of bins along the radii direction. */
#       inline size_t getRadiusBins () const
#       /** The minimal radius value for the search sphere (rmin) in the original paper
#         * \param[in] radius the desired minimal radius
#       inline void setMinimalRadius (double radius)
#       /** \return The minimal sphere radius. */
#       inline double
#       getMinimalRadius () const
#       /** This radius is used to compute local point density
#         * density = number of points within this radius
#         * \param[in] radius Value of the point density search radius
#       inline void setPointDensityRadius (double radius)
#       /** \return The point density search radius. */
#       inline double getPointDensityRadius () const
#       /** Set the local RF radius value
#         * \param[in] radius the desired local RF radius
#       inline void setLocalRadius (double radius)
#       /** \return The local RF radius. */
#       inline double getLocalRadius () const
#
###

# usc.h
# template <typename PointInT, typename PointRFT>
# class UniqueShapeContext<PointInT, Eigen::MatrixXf, PointRFT> : public UniqueShapeContext<PointInT, pcl::SHOT, PointRFT>
# cdef extern from "pcl/features/usc.h" namespace "pcl":
#     cdef cppclass UniqueShapeContext[In, Eigen::MatrixXf, RET](UniqueShapeContext[In, pcl::SHOT, RET]):
#       UniqueShapeContext ()
#       public:
#       using FeatureWithLocalReferenceFrames<PointInT, PointRFT>::frames_;
#       using UniqueShapeContext<PointInT, pcl::SHOT, PointRFT>::indices_;
#       using UniqueShapeContext<PointInT, pcl::SHOT, PointRFT>::descriptor_length_;
#       using UniqueShapeContext<PointInT, pcl::SHOT, PointRFT>::compute;
#       using UniqueShapeContext<PointInT, pcl::SHOT, PointRFT>::computePointDescriptor;
###

# vfh.h
# template<typename PointInT, typename PointNT, typename PointOutT = pcl::VFHSignature308>
# class VFHEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/vfh.h" namespace "pcl":
    cdef cppclass VFHEstimation[In, NT, Out](FeatureFromNormals[In, NT, Out]):
        VFHEstimation ()
        # public:
        # /** \brief Estimate the SPFH (Simple Point Feature Histograms) signatures of the angular
        #   * (f1, f2, f3) and distance (f4) features for a given point from its neighborhood
        #   * \param[in] centroid_p the centroid point
        #   * \param[in] centroid_n the centroid normal
        #   * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
        #   * \param[in] normals the dataset containing the surface normals at each point in \a cloud
        #   * \param[in] indices the k-neighborhood point indices in the dataset
        # void computePointSPFHSignature (const Eigen::Vector4f &centroid_p, const Eigen::Vector4f &centroid_n,
        #                            const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals,
        #                            const std::vector<int> &indices);
        # 
        # /** \brief Set the viewpoint.
        #   * \param[in] vpx the X coordinate of the viewpoint
        #   * \param[in] vpy the Y coordinate of the viewpoint
        #   * \param[in] vpz the Z coordinate of the viewpoint
        # inline void setViewPoint (float vpx, float vpy, float vpz)
        # 
        # /** \brief Get the viewpoint. */
        # inline void getViewPoint (float &vpx, float &vpy, float &vpz)
        # 
        # /** \brief Set use_given_normal_
        #   * \param[in] use Set to true if you want to use the normal passed to setNormalUse(normal)
        #   */
        # inline void setUseGivenNormal (bool use)
        # 
        # /** \brief Set the normal to use
        #   * \param[in] normal Sets the normal to be used in the VFH computation. It is is used
        #   * to build the Darboux Coordinate system.
        #   */
        # inline void setNormalToUse (const Eigen::Vector3f &normal)
        # 
        # /** \brief Set use_given_centroid_
        #   * \param[in] use Set to true if you want to use the centroid passed through setCentroidToUse(centroid)
        #   */
        # inline void setUseGivenCentroid (bool use)
        # 
        # /** \brief Set centroid_to_use_
        #   * \param[in] centroid Centroid to be used in the VFH computation. It is used to compute the distances
        #   * from all points to this centroid.
        #   */
        # inline void setCentroidToUse (const Eigen::Vector3f &centroid)
        # 
        # /** \brief set normalize_bins_
        #   * \param[in] normalize If true, the VFH bins are normalized using the total number of points
        #   */
        # inline void setNormalizeBins (bool normalize)
        # 
        # /** \brief set normalize_distances_
        #   * \param[in] normalize If true, the 4th component of VFH (shape distribution component) get normalized
        #   * by the maximum size between the centroid and the point cloud
        #   */
        # inline void setNormalizeDistance (bool normalize)
        # 
        # /** \brief set size_component_
        #   * \param[in] fill_size True if the 4th component of VFH (shape distribution component) needs to be filled.
        #   * Otherwise, it is set to zero.
        #   */
        # inline void setFillSizeComponent (bool fill_size)
        # 
        # /** \brief Overloaded computed method from pcl::Feature.
        #   * \param[out] output the resultant point cloud model dataset containing the estimated features
        #   */
        # void compute (cpp.PointCloud[Out] &output);


ctypedef VFHEstimation[cpp.PointXYZ, cpp.Normal, cpp.VFHSignature308] VFHEstimation_t
ctypedef VFHEstimation[cpp.PointXYZI, cpp.Normal, cpp.VFHSignature308] VFHEstimation_PointXYZI_t
ctypedef VFHEstimation[cpp.PointXYZRGB, cpp.Normal, cpp.VFHSignature308] VFHEstimation_PointXYZRGB_t
ctypedef VFHEstimation[cpp.PointXYZRGBA, cpp.Normal, cpp.VFHSignature308] VFHEstimation_PointXYZRGBA_t
###


###############################################################################
# Enum
###############################################################################

# Template
# # enum CoordinateFrame
# # CAMERA_FRAME = 0,
# # LASER_FRAME = 1
# Start
# cdef extern from "pcl/range_image/range_image.h" namespace "pcl":
#     ctypedef enum CoordinateFrame2 "pcl::RangeImage::CoordinateFrame":
#         COORDINATEFRAME_CAMERA "pcl::RangeImage::CAMERA_FRAME"
#         COORDINATEFRAME_LASER "pcl::RangeImage::LASER_FRAME"
###

# integral_image_normal.h
# cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl::IntegralImageNormalEstimation":
#         cdef enum BorderPolicy:
#             BORDER_POLICY_IGNORE
#             BORDER_POLICY_MIRROR
# NG : IntegralImageNormalEstimation use Template
# cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl::IntegralImageNormalEstimation":
#     ctypedef enum BorderPolicy2 "pcl::IntegralImageNormalEstimation::BorderPolicy":
#         BORDERPOLICY_IGNORE "pcl::IntegralImageNormalEstimation::BORDER_POLICY_IGNORE"
#         BORDERPOLICY_MIRROR "pcl::IntegralImageNormalEstimation::BORDER_POLICY_MIRROR"
###

# cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl::IntegralImageNormalEstimation":
#         cdef enum NormalEstimationMethod:
#             COVARIANCE_MATRIX
#             AVERAGE_3D_GRADIENT
#             AVERAGE_DEPTH_CHANGE
#             SIMPLE_3D_GRADIENT
# 
# NG : IntegralImageNormalEstimation use Template
# cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl":
#     ctypedef enum NormalEstimationMethod2 "pcl::IntegralImageNormalEstimation::NormalEstimationMethod":
#         ESTIMATIONMETHOD_COVARIANCE_MATRIX "pcl::IntegralImageNormalEstimation::COVARIANCE_MATRIX"
#         ESTIMATIONMETHOD_AVERAGE_3D_GRADIENT "pcl::IntegralImageNormalEstimation::AVERAGE_3D_GRADIENT"
#         ESTIMATIONMETHOD_AVERAGE_DEPTH_CHANGE "pcl::IntegralImageNormalEstimation::AVERAGE_DEPTH_CHANGE"
#         ESTIMATIONMETHOD_SIMPLE_3D_GRADIENT "pcl::IntegralImageNormalEstimation::SIMPLE_3D_GRADIENT"
# NG : (Test Cython 0.24.1)
# define __PYX_VERIFY_RETURN_INT/__PYX_VERIFY_RETURN_INT_EXC etc... , Convert Error "pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal>::NormalEstimationMethod"
# cdef extern from "pcl/features/integral_image_normal.h" namespace "pcl::IntegralImageNormalEstimation":
#     ctypedef enum NormalEstimationMethod2 "pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal>::NormalEstimationMethod":
#         ESTIMATIONMETHOD_COVARIANCE_MATRIX "pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal>::COVARIANCE_MATRIX"
#         ESTIMATIONMETHOD_AVERAGE_3D_GRADIENT "pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal>::AVERAGE_3D_GRADIENT"
#         ESTIMATIONMETHOD_AVERAGE_DEPTH_CHANGE "pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal>::AVERAGE_DEPTH_CHANGE"
#         ESTIMATIONMETHOD_SIMPLE_3D_GRADIENT "pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal>::SIMPLE_3D_GRADIENT"
###


###############################################################################
# Activation
###############################################################################

