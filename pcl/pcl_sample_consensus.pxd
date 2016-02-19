from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from boost_shared_ptr cimport shared_ptr

###############################################################################
# Types
###############################################################################

### base class ###

# # sac_model.h
# # namespace pcl
# # template<class T> class ProgressiveSampleConsensus;
# # 
# # namespace pcl
# # template <typename PointT>
# # class SampleConsensusModel
# cdef extern from "pcl/sample_consensus/sac_model.h" namespace "pcl":
#     cdef cppclass SampleConsensusModel[T]:
#         SampleConsensusModel()
#         # SampleConsensusModel (bool random = false) 
#         # SampleConsensusModel (const PointCloudConstPtr &cloud, bool random = false)
#         # SampleConsensusModel (const PointCloudConstPtr &cloud, const std::vector<int> &indices, bool random = false)
#         # public:
#         # typedef typename pcl::PointCloud<PointT> PointCloud;
#         # typedef typename pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
#         # typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
#         # typedef typename pcl::search::Search<PointT>::Ptr SearchPtr;
#         # typedef boost::shared_ptr<SampleConsensusModel> Ptr;
#         # typedef boost::shared_ptr<const SampleConsensusModel> ConstPtr;
#         # protected:
#         # public:
#         # /** \brief Get a set of random data samples and return them as point
#         # * indices. Pure virtual.  
#         # * \param[out] iterations the internal number of iterations used by SAC methods
#         # * \param[out] samples the resultant model samples
#         # */
#         # void getSamples (int &iterations, std::vector<int> &samples)
#         # /** \brief Check whether the given index samples can form a valid model,
#         # * compute the model coefficients from these samples and store them
#         # * in model_coefficients. Pure virtual.
#         # * \param[in] samples the point indices found as possible good candidates
#         # * for creating a valid model 
#         # * \param[out] model_coefficients the computed model coefficients
#         # */
#         # virtual bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients) = 0;
#         # /** \brief Recompute the model coefficients using the given inlier set
#         # * and return them to the user. Pure virtual.
#         # * @note: these are the coefficients of the model after refinement
#         # * (e.g., after a least-squares optimization)
#         # * \param[in] inliers the data inliers supporting the model
#         # * \param[in] model_coefficients the initial guess for the model coefficients
#         # * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
#         # */
#         # virtual void optimizeModelCoefficients (const std::vector<int> &inliers, 
#         #                          const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients) = 0;
#         # /** \brief Compute all distances from the cloud data to a given model. Pure virtual.
#         # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to 
#         # * \param[out] distances the resultant estimated distances
#         # virtual void  getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances) = 0;
#         # /** \brief Select all the points which respect the given model
#         # * coefficients as inliers. Pure virtual.
#         # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from 
#         # * the outliers
#         # * \param[out] inliers the resultant model inliers
#         # virtual void selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
#         #                     const double threshold, std::vector<int> &inliers) = 0;
#         # /** \brief Count all the points which respect the given model
#         # * coefficients as inliers. Pure virtual.
#         # * \param[in] model_coefficients the coefficients of a model that we need to
#         # * compute distances to
#         # * \param[in] threshold a maximum admissible distance threshold for
#         # * determining the inliers from the outliers
#         # * \return the resultant number of inliers
#         # */
#         # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold) = 0;
#         # /** \brief Create a new point cloud with inliers projected onto the model. Pure virtual.
#         # * \param[in] inliers the data inliers that we want to project on the model
#         # * \param[in] model_coefficients the coefficients of a model
#         # * \param[out] projected_points the resultant projected points
#         # * \param[in] copy_data_fields set to true (default) if we want the \a
#         # * projected_points cloud to be an exact copy of the input dataset minus
#         # * the point projections on the plane model
#         # virtual void projectPoints (const std::vector<int> &inliers, 
#         #              const Eigen::VectorXf &model_coefficients,
#         #              PointCloud &projected_points, 
#         #              bool copy_data_fields = true) = 0;
#         # /** \brief Verify whether a subset of indices verifies a given set of
#         # * model coefficients. Pure virtual.
#         # * \param[in] indices the data indices that need to be tested against the model
#         # * \param[in] model_coefficients the set of model coefficients
#         # * \param[in] threshold a maximum admissible distance threshold for
#         # * determining the inliers from the outliers
#         # virtual bool doSamplesVerifyModel (const std::set<int> &indices, 
#         #                     const Eigen::VectorXf &model_coefficients, 
#         #                     const double threshold) = 0;
#         # /** \brief Provide a pointer to the input dataset
#         # * \param[in] cloud the const boost shared pointer to a PointCloud message
#         # inline virtual void setInputCloud (const PointCloudConstPtr &cloud)
#         # /** \brief Get a pointer to the input point cloud dataset. */
#         # inline PointCloudConstPtr getInputCloud () const
#         # /** \brief Provide a pointer to the vector of indices that represents the input data.
#         # * \param[in] indices a pointer to the vector of indices that represents the input data.
#         # inline void setIndices (const boost::shared_ptr <std::vector<int> > &indices) 
#         # /** \brief Provide the vector of indices that represents the input data.
#         # * \param[out] indices the vector of indices that represents the input data.
#         # inline void setIndices (const std::vector<int> &indices) 
#         # /** \brief Get a pointer to the vector of indices used. */
#         # inline boost::shared_ptr <std::vector<int> > getIndices () const
#         # /** \brief Return an unique id for each type of model employed. */
#         # virtual SacModel getModelType () const = 0;
#         # /** \brief Return the size of a sample from which a model is computed */
#         # inline unsigned int getSampleSize () const 
#         # /** \brief Set the minimum and maximum allowable radius limits for the
#         # * model (applicable to models that estimate a radius)
#         # * \param[in] min_radius the minimum radius model
#         # * \param[in] max_radius the maximum radius model
#         # * \todo change this to set limits on the entire model
#         # inline void setRadiusLimits (const double &min_radius, const double &max_radius)
#         # /** \brief Get the minimum and maximum allowable radius limits for the
#         # * model as set by the user.
#         # * \param[out] min_radius the resultant minimum radius model
#         # * \param[out] max_radius the resultant maximum radius model
#         # inline void getRadiusLimits (double &min_radius, double &max_radius)
#         # /** \brief Set the maximum distance allowed when drawing random samples
#         # * \param[in] radius the maximum distance (L2 norm)
#         # inline void setSamplesMaxDist (const double &radius, SearchPtr search)
#         # /** \brief Get maximum distance allowed when drawing random samples
#         # * \param[out] radius the maximum distance (L2 norm)
#         # inline void getSamplesMaxDist (double &radius)
#         # friend class ProgressiveSampleConsensus<PointT>;
#         # protected:
#         # /** \brief Fills a sample array with random samples from the indices_ vector
#         # * \param[out] sample the set of indices of target_ to analyze
#         # inline void drawIndexSample (std::vector<int> &sample)
#         # /** \brief Fills a sample array with one random sample from the indices_ vector
#         # *        and other random samples that are closer than samples_radius_
#         # * \param[out] sample the set of indices of target_ to analyze
#         # inline void drawIndexSampleRadius (std::vector<int> &sample)
#         # /** \brief Check whether a model is valid given the user constraints.
#         # * \param[in] model_coefficients the set of model coefficients
#         # virtual inline bool isModelValid (const Eigen::VectorXf &model_coefficients) = 0;
#         # /** \brief Check if a sample of indices results in a good sample of points
#         # * indices. Pure virtual.
#         # * \param[in] samples the resultant index samples
#         # virtual bool isSampleGood (const std::vector<int> &samples) const = 0;
#         # /** \brief A boost shared pointer to the point cloud data array. */
#         # PointCloudConstPtr input_;
#         # /** \brief A pointer to the vector of point indices to use. */
#         # boost::shared_ptr <std::vector<int> > indices_;
#         # /** The maximum number of samples to try until we get a good one */
#         # static const unsigned int max_sample_checks_ = 1000;
#         # /** \brief The minimum and maximum radius limits for the model.
#         # * Applicable to all models that estimate a radius. 
#         # double radius_min_, radius_max_;
#         # /** \brief The maximum distance of subsequent samples from the first (radius search) */
#         # double samples_radius_;
#         # /** \brief The search object for picking subsequent samples using radius search */
#         # SearchPtr samples_radius_search_;
#         # /** Data containing a shuffled version of the indices. This is used and modified when drawing samples. */
#         # std::vector<int> shuffled_indices_;
#         # /** \brief Boost-based random number generator algorithm. */
#         # boost::mt19937 rng_alg_;
#         # /** \brief Boost-based random number generator distribution. */
#         # boost::shared_ptr<boost::uniform_int<> > rng_dist_;
#         # /** \brief Boost-based random number generator. */
#         # boost::shared_ptr<boost::variate_generator< boost::mt19937&, boost::uniform_int<> > > rng_gen_;
#         # /** \brief Boost-based random number generator. */
#         # inline int rnd ()
#         # public:
#         # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
# ###
# 
# # template <typename PointT, typename PointNT>
# # class SampleConsensusModelFromNormals
# cdef extern from "pcl/sample_consensus/sac_model.h" namespace "pcl":
#     cdef cppclass SampleConsensusModelFromNormals[T, NT]:
#         SampleConsensusModelFromNormals ()
#         # public:
#         # typedef typename pcl::PointCloud<PointNT>::ConstPtr PointCloudNConstPtr;
#         # typedef typename pcl::PointCloud<PointNT>::Ptr PointCloudNPtr;
#         # typedef boost::shared_ptr<SampleConsensusModelFromNormals> Ptr;
#         # typedef boost::shared_ptr<const SampleConsensusModelFromNormals> ConstPtr;
#         # /** \brief Empty constructor for base SampleConsensusModelFromNormals. */
#         # /** \brief Set the normal angular distance weight.
#         # * \param[in] w the relative weight (between 0 and 1) to give to the angular
#         # * distance (0 to pi/2) between point normals and the plane normal.
#         # * (The Euclidean distance will have weight 1-w.)
#         # */
#         # inline void setNormalDistanceWeight (const double w) 
#         # /** \brief Get the normal angular distance weight. */
#         # inline double getNormalDistanceWeight ()
#         # /** \brief Provide a pointer to the input dataset that contains the point
#         # * normals of the XYZ dataset.
#         # * \param[in] normals the const boost shared pointer to a PointCloud message
#         # inline void setInputNormals (const PointCloudNConstPtr &normals) 
#         # /** \brief Get a pointer to the normals of the input XYZ point cloud dataset. */
#         # inline PointCloudNConstPtr getInputNormals ()
#         # protected:
#         # /** \brief The relative weight (between 0 and 1) to give to the angular
#         # * distance (0 to pi/2) between point normals and the plane normal. 
#         # double normal_distance_weight_;
#         # /** \brief A pointer to the input dataset that contains the point normals
#         # * of the XYZ dataset. 
#         # PointCloudNConstPtr normals_;
###

# sac.h
# namespace pcl
# template <typename T>
# class SampleConsensus
# cdef extern from "pcl/sample_consensus/sac.h" namespace "pcl":
#     cdef cppclass SampleConsensus[T]:
#         # private:
#         # SampleConsensus ()
#         # typedef typename SampleConsensusModel<T>::Ptr SampleConsensusModelPtr;
#         # public:
#         # SampleConsensus (const SampleConsensusModelPtr &model, bool random = false)
#         # SampleConsensus (const SampleConsensusModelPtr &model, double threshold, bool random = false) : 
#         # typedef boost::shared_ptr<SampleConsensus> Ptr;
#         # typedef boost::shared_ptr<const SampleConsensus> ConstPtr;
#         # /** \brief Constructor for base SAC.
#         # * \param[in] model a Sample Consensus model
#         # * \param[in] random if true set the random seed to the current time, else set to 12345 (default: false)
#         # /** \brief Set the distance to model threshold.
#         # * \param[in] threshold distance to model threshold
#         # inline void setDistanceThreshold (double threshold)
#         # /** \brief Get the distance to model threshold, as set by the user. */
#         # inline double getDistanceThreshold ()
#         # /** \brief Set the maximum number of iterations.
#         # * \param[in] max_iterations maximum number of iterations
#         # inline void setMaxIterations (int max_iterations)
#         # /** \brief Get the maximum number of iterations, as set by the user. */
#         # inline int getMaxIterations ()
#         # /** \brief Set the desired probability of choosing at least one sample free from outliers.
#         # * \param[in] probability the desired probability of choosing at least one sample free from outliers
#         # * \note internally, the probability is set to 99% (0.99) by default.
#         # inline void setProbability (double probability)
#         # /** \brief Obtain the probability of choosing at least one sample free from outliers, as set by the user. */
#         # inline double getProbability ()
#         # /** \brief Compute the actual model. Pure virtual. */
#         # virtual bool computeModel (int debug_verbosity_level = 0) = 0;
#         # /** \brief Get a set of randomly selected indices.
#         # * \param[in] indices the input indices vector
#         # * \param[in] nr_samples the desired number of point indices to randomly select
#         # * \param[out] indices_subset the resultant output set of randomly selected indices
#         # inline void getRandomSamples (const boost::shared_ptr <std::vector<int> > &indices, 
#         #                 size_t nr_samples, std::set<int> &indices_subset)
#         # /** \brief Return the best model found so far. 
#         # * \param[out] model the resultant model
#         # inline void getModel (std::vector<int> &model)
#         # /** \brief Return the best set of inliers found so far for this model. 
#         # * \param[out] inliers the resultant set of inliers
#         # inline void getInliers (std::vector<int> &inliers)
#         # /** \brief Return the model coefficients of the best model found so far. 
#         # * \param[out] model_coefficients the resultant model coefficients
#         # inline void  getModelCoefficients (Eigen::VectorXf &model_coefficients)
#         # protected:
#         # /** \brief The underlying data model used (i.e. what is it that we attempt to search for). */
#         # SampleConsensusModelPtr sac_model_;
#         # /** \brief The model found after the last computeModel () as point cloud indices. */
#         # std::vector<int> model_;
#         # /** \brief The indices of the points that were chosen as inliers after the last computeModel () call. */
#         # std::vector<int> inliers_;
#         # /** \brief The coefficients of our model computed directly from the model found. */
#         # Eigen::VectorXf model_coefficients_;
#         # /** \brief Desired probability of choosing at least one sample free from outliers. */
#         # double probability_;
#         # /** \brief Total number of internal loop iterations that we've done so far. */
#         # int iterations_;
#         # /** \brief Distance to model threshold. */
#         # double threshold_;
#         # /** \brief Maximum number of iterations before giving up. */
#         # int max_iterations_;
#         # /** \brief Boost-based random number generator algorithm. */
#         # boost::mt19937 rng_alg_;
#         # /** \brief Boost-based random number generator distribution. */
#         # boost::shared_ptr<boost::uniform_01<boost::mt19937> > rng_;
#         # /** \brief Boost-based random number generator. */
#         # inline double rnd ()
# ###

# template<typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
# struct Functor
# cdef extern from "pcl/sample_consensus/rransac.h" namespace "pcl":
#     cdef cppclass Functor[_Scalar, int]:
#         Functor ()
#         # Functor (int m_data_points)
#         # typedef _Scalar Scalar;
#         # enum 
#         # {
#         #   InputsAtCompileTime = NX,
#         #   ValuesAtCompileTime = NY
#         # };
#         # typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
#         # typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
#         # typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
#         # /** \brief Get the number of values. */ 
#         # int values () const
###

### Inheritance class ###

# # lmeds.h
# # namespace pcl
# # template <typename PointT>
# # class LeastMedianSquares : public SampleConsensus<PointT>
# cdef extern from "pcl/sample_consensus/lmeds.h" namespace "pcl":
#     cdef cppclass LeastMedianSquares[T](SampleConsensus[T]):
#         # LeastMedianSquares ()
#         # LeastMedianSquares (const SampleConsensusModelPtr &model)
#         # LeastMedianSquares (const SampleConsensusModelPtr &model, double threshold)
#         # using SampleConsensus<PointT>::max_iterations_;
#         # using SampleConsensus<PointT>::threshold_;
#         # using SampleConsensus<PointT>::iterations_;
#         # using SampleConsensus<PointT>::sac_model_;
#         # using SampleConsensus<PointT>::model_;
#         # using SampleConsensus<PointT>::model_coefficients_;
#         # using SampleConsensus<PointT>::inliers_;
#         # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
#         # public:
#         # /** \brief Compute the actual model and find the inliers
#         #   * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
#         #   */
#         # bool computeModel (int debug_verbosity_level = 0)
# ###
# 
# # mlesac.h
# # namespace pcl
# # template <typename PointT>
# # class MaximumLikelihoodSampleConsensus : public SampleConsensus<PointT>
# cdef extern from "pcl/sample_consensus/mlesac.h" namespace "pcl":
#     cdef cppclass MaximumLikelihoodSampleConsensus[T](SampleConsensus[T]):
#         MaximumLikelihoodSampleConsensus ()
#         # MaximumLikelihoodSampleConsensus (const SampleConsensusModelPtr &model)
#         # MaximumLikelihoodSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
#         # using SampleConsensus<PointT>::max_iterations_;
#         # using SampleConsensus<PointT>::threshold_;
#         # using SampleConsensus<PointT>::iterations_;
#         # using SampleConsensus<PointT>::sac_model_;
#         # using SampleConsensus<PointT>::model_;
#         # using SampleConsensus<PointT>::model_coefficients_;
#         # using SampleConsensus<PointT>::inliers_;
#         # using SampleConsensus<PointT>::probability_;
#         # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
#         # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr; 
#         # public:
#         # * \brief MLESAC (Maximum Likelihood Estimator SAmple Consensus) main constructor
#         # * \param[in] model a Sample Consensus model
#         # /** \brief Compute the actual model and find the inliers
#         # * \param[in] debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
#         # bool computeModel (int debug_verbosity_level = 0);
#         # /** \brief Set the number of EM iterations.
#         # * \param[in] iterations the number of EM iterations
#         # inline void setEMIterations (int iterations)
#         # /** \brief Get the number of EM iterations. */
#         # inline int getEMIterations () const { return (iterations_EM_); }
#         # protected:
#         # /** \brief Compute the median absolute deviation:
#         # * \f[
#         # * MAD = \sigma * median_i (| Xi - median_j(Xj) |)
#         # * \f]
#         # * \note Sigma needs to be chosen carefully (a good starting sigma value is 1.4826)
#         # * \param[in] cloud the point cloud data message
#         # * \param[in] indices the set of point indices to use
#         # * \param[in] sigma the sigma value
#         # double computeMedianAbsoluteDeviation (const PointCloudConstPtr &cloud, 
#         #                               const boost::shared_ptr <std::vector<int> > &indices, 
#         #                               double sigma);
#         # /** \brief Determine the minimum and maximum 3D bounding box coordinates for a given set of points
#         # * \param[in] cloud the point cloud message
#         # * \param[in] indices the set of point indices to use
#         # * \param[out] min_p the resultant minimum bounding box coordinates
#         # * \param[out] max_p the resultant maximum bounding box coordinates
#         # */
#         # void getMinMax (const PointCloudConstPtr &cloud, 
#         #          const boost::shared_ptr <std::vector<int> > &indices, 
#         #          Eigen::Vector4f &min_p, 
#         #          Eigen::Vector4f &max_p);
#         # /** \brief Compute the median value of a 3D point cloud using a given set point indices and return it as a Point32.
#         # * \param[in] cloud the point cloud data message
#         # * \param[in] indices the point indices
#         # * \param[out] median the resultant median value
#         # */
#         # void computeMedian (const PointCloudConstPtr &cloud, 
#         #              const boost::shared_ptr <std::vector<int> > &indices, 
#         #              Eigen::Vector4f &median);
# ###
# 
# # msac.h
# # namespace pcl
# #   template <typename PointT>
# #   class MEstimatorSampleConsensus : public SampleConsensus<PointT>
# cdef extern from "pcl/sample_consensus/msac.h" namespace "pcl":
#     cdef cppclass MEstimatorSampleConsensus[T](SampleConsensus[T]):
#         MEstimatorSampleConsensus ()
#         # MEstimatorSampleConsensus (const SampleConsensusModelPtr &model)
#         # MEstimatorSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
#         # using SampleConsensus<PointT>::max_iterations_;
#         # using SampleConsensus<PointT>::threshold_;
#         # using SampleConsensus<PointT>::iterations_;
#         # using SampleConsensus<PointT>::sac_model_;
#         # using SampleConsensus<PointT>::model_;
#         # using SampleConsensus<PointT>::model_coefficients_;
#         # using SampleConsensus<PointT>::inliers_;
#         # using SampleConsensus<PointT>::probability_;
#         # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
#         # public:
#         # /** \brief Compute the actual model and find the inliers
#         # * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
#         # */
#         # bool computeModel (int debug_verbosity_level = 0);
# ###
# 
# # prosac.h
# # namespace pcl
# # template<typename PointT>
# # class ProgressiveSampleConsensus : public SampleConsensus<PointT>
# cdef extern from "pcl/sample_consensus/prosac.h" namespace "pcl":
#     cdef cppclass ProgressiveSampleConsensus[T](SampleConsensus[T]):
#         ProgressiveSampleConsensus ()
#         # ProgressiveSampleConsensus (const SampleConsensusModelPtr &model) 
#         # ProgressiveSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
#         using SampleConsensus<PointT>::max_iterations_;
#         using SampleConsensus<PointT>::threshold_;
#         using SampleConsensus<PointT>::iterations_;
#         using SampleConsensus<PointT>::sac_model_;
#         using SampleConsensus<PointT>::model_;
#         using SampleConsensus<PointT>::model_coefficients_;
#         using SampleConsensus<PointT>::inliers_;
#         using SampleConsensus<PointT>::probability_;
#         typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
#         public:
#         /** \brief Compute the actual model and find the inliers
#         * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
#         bool computeModel (int debug_verbosity_level = 0)
# ###
# 
# # ransac.h
# # namespace pcl
# # template <typename PointT>
# # class RandomSampleConsensus : public SampleConsensus<PointT>
# cdef extern from "pcl/sample_consensus/prosac.h" namespace "pcl":
#     cdef cppclass RandomSampleConsensus[T](SampleConsensus[T]):
#         RandomSampleConsensus ()
#         # RandomSampleConsensus (const SampleConsensusModelPtr &model)
#         # RandomSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
#         using SampleConsensus<PointT>::max_iterations_;
#         using SampleConsensus<PointT>::threshold_;
#         using SampleConsensus<PointT>::iterations_;
#         using SampleConsensus<PointT>::sac_model_;
#         using SampleConsensus<PointT>::model_;
#         using SampleConsensus<PointT>::model_coefficients_;
#         using SampleConsensus<PointT>::inliers_;
#         using SampleConsensus<PointT>::probability_;
#         typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
#         public:
#         /** \brief Compute the actual model and find the inliers
#         * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
#         bool computeModel (int debug_verbosity_level = 0);
# ###
# 
# # rmsac.h
# # namespace pcl
# # template <typename PointT>
# # class RandomizedMEstimatorSampleConsensus : public SampleConsensus<PointT>
# cdef extern from "pcl/sample_consensus/prosac.h" namespace "pcl":
#     cdef cppclass RandomizedMEstimatorSampleConsensus[T](SampleConsensus[T]):
#         RandomizedMEstimatorSampleConsensus ()
#         # RandomizedMEstimatorSampleConsensus (const SampleConsensusModelPtr &model)
#         # RandomizedMEstimatorSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
#         # using SampleConsensus<PointT>::max_iterations_;
#         # using SampleConsensus<PointT>::threshold_;
#         # using SampleConsensus<PointT>::iterations_;
#         # using SampleConsensus<PointT>::sac_model_;
#         # using SampleConsensus<PointT>::model_;
#         # using SampleConsensus<PointT>::model_coefficients_;
#         # using SampleConsensus<PointT>::inliers_;
#         # using SampleConsensus<PointT>::probability_;
#         # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
#         # public:
#         # /** \brief Compute the actual model and find the inliers
#         # * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
#         # */
#         # bool computeModel (int debug_verbosity_level = 0);
#         # /** \brief Set the percentage of points to pre-test.
#         # * \param nr_pretest percentage of points to pre-test
#         # */
#         # inline void setFractionNrPretest (double nr_pretest)
#         # /** \brief Get the percentage of points to pre-test. */
#         # inline double getFractionNrPretest ()
# ###
# 
# # rransac.h
# # namespace pcl
# # template <typename PointT>
# # class RandomizedRandomSampleConsensus : public SampleConsensus<PointT>
# cdef extern from "pcl/sample_consensus/rransac.h" namespace "pcl":
#     cdef cppclass RandomizedRandomSampleConsensus[T](SampleConsensus[T]):
#         RandomizedRandomSampleConsensus ()
#         # RandomizedRandomSampleConsensus (const SampleConsensusModelPtr &model)
#         # RandomizedRandomSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
#         # using SampleConsensus<PointT>::max_iterations_;
#         # using SampleConsensus<PointT>::threshold_;
#         # using SampleConsensus<PointT>::iterations_;
#         # using SampleConsensus<PointT>::sac_model_;
#         # using SampleConsensus<PointT>::model_;
#         # using SampleConsensus<PointT>::model_coefficients_;
#         # using SampleConsensus<PointT>::inliers_;
#         # using SampleConsensus<PointT>::probability_;
#         # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
#         # public:
#         # /** \brief RANSAC (Randomized RAndom SAmple Consensus) main constructor
#         # * \param model a Sample Consensus model
#         # */
#         # /** \brief RRANSAC (RAndom SAmple Consensus) main constructor
#         # * \param model a Sample Consensus model
#         # * \param threshold distance to model threshold
#         # /** \brief Compute the actual model and find the inliers
#         # * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
#         # */
#         bool computeModel (int debug_verbosity_level = 0)
#         # /** \brief Set the percentage of points to pre-test.
#         # * \param nr_pretest percentage of points to pre-test
#         # */
#         inline void setFractionNrPretest (double nr_pretest)
#         # /** \brief Get the percentage of points to pre-test. */
#         inline double getFractionNrPretest ()
# ###

# # sac_model_circle.h
# # namespace pcl
# # template <typename PointT>
# # class SampleConsensusModelCircle2D : public SampleConsensusModel<PointT>
# cdef extern from "pcl/sample_consensus/sac_model_circle.h" namespace "pcl":
#     cdef cppclass SampleConsensusModelCircle2D[T](SampleConsensusModel[T]):
#         SampleConsensusModelCircle2D ()
#         # SampleConsensusModelCircle2D (const PointCloudConstPtr &cloud)
#         # SampleConsensusModelCircle2D (const PointCloudConstPtr &cloud, const std::vector<int> &indices)
#         # SampleConsensusModelCircle2D (const SampleConsensusModelCircle2D &source) :
#         # inline SampleConsensusModelCircle2D& operator = (const SampleConsensusModelCircle2D &source)
#         # using SampleConsensusModel<PointT>::input_;
#         # using SampleConsensusModel<PointT>::indices_;
#         # using SampleConsensusModel<PointT>::radius_min_;
#         # using SampleConsensusModel<PointT>::radius_max_;
#         # public:
#         # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
#         # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
#         # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
#         # typedef boost::shared_ptr<SampleConsensusModelCircle2D> Ptr;
#         # /** \brief Check whether the given index samples can form a valid 2D circle model, compute the model coefficients
#         # * from these samples and store them in model_coefficients. The circle coefficients are: x, y, R.
#         # * \param[in] samples the point indices found as possible good candidates for creating a valid model
#         # * \param[out] model_coefficients the resultant model coefficients
#         # bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);
#         # /** \brief Compute all distances from the cloud data to a given 2D circle model.
#         # * \param[in] model_coefficients the coefficients of a 2D circle model that we need to compute distances to
#         # * \param[out] distances the resultant estimated distances
#         # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
#         # /** \brief Compute all distances from the cloud data to a given 2D circle model.
#         # * \param[in] model_coefficients the coefficients of a 2D circle model that we need to compute distances to
#         # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         # * \param[out] inliers the resultant model inliers
#         # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
#         #                     const double threshold, 
#         #                     std::vector<int> &inliers);
#         # /** \brief Count all the points which respect the given model coefficients as inliers. 
#         # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
#         # * \return the resultant number of inliers
#         # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, 
#         #                    const double threshold);
#         # /** \brief Recompute the 2d circle coefficients using the given inlier set and return them to the user.
#         # * @note: these are the coefficients of the 2d circle model after refinement (eg. after SVD)
#         # * \param[in] inliers the data inliers found as supporting the model
#         # * \param[in] model_coefficients the initial guess for the optimization
#         # * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
#         # void optimizeModelCoefficients (const std::vector<int> &inliers, 
#         #                          const Eigen::VectorXf &model_coefficients, 
#         #                          Eigen::VectorXf &optimized_coefficients);
#         # /** \brief Create a new point cloud with inliers projected onto the 2d circle model.
#         # * \param[in] inliers the data inliers that we want to project on the 2d circle model
#         # * \param[in] model_coefficients the coefficients of a 2d circle model
#         # * \param[out] projected_points the resultant projected points
#         # * \param[in] copy_data_fields set to true if we need to copy the other data fields
#         # void projectPoints (const std::vector<int> &inliers, 
#         #              const Eigen::VectorXf &model_coefficients, 
#         #              PointCloud &projected_points, 
#         #              bool copy_data_fields = true);
#         # /** \brief Verify whether a subset of indices verifies the given 2d circle model coefficients.
#         # * \param[in] indices the data indices that need to be tested against the 2d circle model
#         # * \param[in] model_coefficients the 2d circle model coefficients
#         # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         # bool doSamplesVerifyModel (const std::set<int> &indices, 
#         #                     const Eigen::VectorXf &model_coefficients, 
#         #                     const double threshold);
#         # /** \brief Return an unique id for this model (SACMODEL_CIRCLE2D). */
#         # inline pcl::SacModel getModelType () const
#         # protected:
#         # /** \brief Check whether a model is valid given the user constraints.
#         # * \param[in] model_coefficients the set of model coefficients
#         # bool isModelValid (const Eigen::VectorXf &model_coefficients);
#         # /** \brief Check if a sample of indices results in a good sample of points indices.
#         # * \param[in] samples the resultant index samples
#         # bool isSampleGood(const std::vector<int> &samples) const;
# ###
# 
# # namespace pcl
# # struct OptimizationFunctor : pcl::Functor<float>
# #         OptimizationFunctor (int m_data_points, pcl::SampleConsensusModelCircle2D<PointT> *model) : 
# # 
# #         /** Cost function to be minimized
# #           * \param[in] x the variables array
# #           * \param[out] fvec the resultant functions evaluations
# #           * \return 0
# #           */
# #         int operator() (const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
# #         pcl::SampleConsensusModelCircle2D<PointT> *model_;
# ###
# 
# # sac_model_cone.h
# # namespace pcl
# # template <typename PointT, typename PointNT>
# # class SampleConsensusModelCone : public SampleConsensusModel<PointT>, public SampleConsensusModelFromNormals<PointT, PointNT>
# #    cdef cppclass SampleConsensusModelCone[T, NT](SampleConsensusModel[T])(SampleConsensusModelFromNormals[T, NT]):
# cdef extern from "pcl/sample_consensus/sac_model_cone.h" namespace "pcl":
#     cdef cppclass SampleConsensusModelCone[T, NT]:
#         # Nothing
#         # SampleConsensusModelCone ()
#         # Use
#         # SampleConsensusModelCone (const PointCloudConstPtr &cloud)
#         # SampleConsensusModelCone (const PointCloudConstPtr &cloud, const std::vector<int> &indices)
#         # SampleConsensusModelCone (const SampleConsensusModelCone &source)
#         # inline SampleConsensusModelCone& operator = (const SampleConsensusModelCone &source)
#         # using SampleConsensusModel<PointT>::input_;
#         # using SampleConsensusModel<PointT>::indices_;
#         # using SampleConsensusModel<PointT>::radius_min_;
#         # using SampleConsensusModel<PointT>::radius_max_;
#         # using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
#         # using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
#         # public:
#         # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
#         # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
#         # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
#         # typedef boost::shared_ptr<SampleConsensusModelCone> Ptr;
#         # /** \brief Set the angle epsilon (delta) threshold.
#         # * \param[in] ea the maximum allowed difference between the cone's axis and the given axis.
#         # inline void setEpsAngle (double ea)
#         # /** \brief Get the angle epsilon (delta) threshold. */
#         # inline double getEpsAngle () const
#         # /** \brief Set the axis along which we need to search for a cone direction.
#         # * \param[in] ax the axis along which we need to search for a cone direction
#         # inline void setAxis (const Eigen::Vector3f &ax)
#         # /** \brief Get the axis along which we need to search for a cone direction. */
#         # inline Eigen::Vector3f getAxis () const
#         # /** \brief Set the minimum and maximum allowable opening angle for a cone model
#         # * given from a user.
#         # * \param[in] min_angle the minimum allwoable opening angle of a cone model
#         # * \param[in] max_angle the maximum allwoable opening angle of a cone model
#         # inline void setMinMaxOpeningAngle (const double &min_angle, const double &max_angle)
#         # /** \brief Get the opening angle which we need minumum to validate a cone model.
#         # * \param[out] min_angle the minimum allwoable opening angle of a cone model
#         # * \param[out] max_angle the maximum allwoable opening angle of a cone model
#         # inline void getMinMaxOpeningAngle (double &min_angle, double &max_angle) const
#         # /** \brief Check whether the given index samples can form a valid cone model, compute the model coefficients
#         # * from these samples and store them in model_coefficients. The cone coefficients are: apex,
#         # * axis_direction, opening_angle.
#         # * \param[in] samples the point indices found as possible good candidates for creating a valid model
#         # * \param[out] model_coefficients the resultant model coefficients
#         # bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);
#         # /** \brief Compute all distances from the cloud data to a given cone model.
#         # * \param[in] model_coefficients the coefficients of a cone model that we need to compute distances to
#         # * \param[out] distances the resultant estimated distances
#         # void getDistancesToModel (const Eigen::VectorXf &model_coefficients,  std::vector<double> &distances);
#         # /** \brief Select all the points which respect the given model coefficients as inliers.
#         # * \param[in] model_coefficients the coefficients of a cone model that we need to compute distances to
#         # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         # * \param[out] inliers the resultant model inliers
#         # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
#         #                     const double threshold, std::vector<int> &inliers);
#         # /** \brief Count all the points which respect the given model coefficients as inliers. 
#         # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
#         # * \return the resultant number of inliers
#         # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
#         # /** \brief Recompute the cone coefficients using the given inlier set and return them to the user.
#         # * @note: these are the coefficients of the cone model after refinement (eg. after SVD)
#         # * \param[in] inliers the data inliers found as supporting the model
#         # * \param[in] model_coefficients the initial guess for the optimization
#         # * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
#         # void optimizeModelCoefficients (const std::vector<int> &inliers, 
#         #                          const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients);
#         # /** \brief Create a new point cloud with inliers projected onto the cone model.
#         # * \param[in] inliers the data inliers that we want to project on the cone model
#         # * \param[in] model_coefficients the coefficients of a cone model
#         # * \param[out] projected_points the resultant projected points
#         # * \param[in] copy_data_fields set to true if we need to copy the other data fields
#         # void projectPoints (const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, 
#         #              PointCloud &projected_points, bool copy_data_fields = true);
#         # /** \brief Verify whether a subset of indices verifies the given cone model coefficients.
#         # * \param[in] indices the data indices that need to be tested against the cone model
#         # * \param[in] model_coefficients the cone model coefficients
#         # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         # bool doSamplesVerifyModel (const std::set<int> &indices, 
#         #                     const Eigen::VectorXf &model_coefficients, const double threshold);
#         # /** \brief Return an unique id for this model (SACMODEL_CONE). */
#         # inline pcl::SacModel getModelType () const
#         # protected:
#         # /** \brief Get the distance from a point to a line (represented by a point and a direction)
#         # * \param[in] pt a point
#         # * \param[in] model_coefficients the line coefficients (a point on the line, line direction)
#         # double pointToAxisDistance (const Eigen::Vector4f &pt, const Eigen::VectorXf &model_coefficients);
#         # /** \brief Get a string representation of the name of this class. */
#         # std::string getName () const { return ("SampleConsensusModelCone"); }
#         # protected:
#         # /** \brief Check whether a model is valid given the user constraints.
#         # * \param[in] model_coefficients the set of model coefficients
#         # bool isModelValid (const Eigen::VectorXf &model_coefficients);
#         # /** \brief Check if a sample of indices results in a good sample of points
#         # * indices. Pure virtual.
#         # * \param[in] samples the resultant index samples
#         # bool isSampleGood (const std::vector<int> &samples) const;
# ###
# 
# # namespace pcl
# # /** \brief Functor for the optimization function */
# # struct OptimizationFunctor : pcl::Functor<float>
# # cdef extern from "pcl/sample_consensus/sac_model_cone.h" namespace "pcl":
# #     cdef cppclass OptimizationFunctor(Functor[float]):
# #         OptimizationFunctor (int m_data_points, pcl::SampleConsensusModelCone<PointT, PointNT> *model) : 
# #         int operator() (const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
# #         pcl::SampleConsensusModelCone<PointT, PointNT> *model_;
# ###
# 
# # sac_model_cylinder.h
# # namespace pcl
# # {
# #   /** \brief @b SampleConsensusModelCylinder defines a model for 3D cylinder segmentation.
# #     * The model coefficients are defined as:
# #     *   - \b point_on_axis.x  : the X coordinate of a point located on the cylinder axis
# #     *   - \b point_on_axis.y  : the Y coordinate of a point located on the cylinder axis
# #     *   - \b point_on_axis.z  : the Z coordinate of a point located on the cylinder axis
# #     *   - \b axis_direction.x : the X coordinate of the cylinder's axis direction
# #     *   - \b axis_direction.y : the Y coordinate of the cylinder's axis direction
# #     *   - \b axis_direction.z : the Z coordinate of the cylinder's axis direction
# #     *   - \b radius           : the cylinder's radius
# #     * 
# #     * \author Radu Bogdan Rusu
# #     * \ingroup sample_consensus
# #     */
# #   template <typename PointT, typename PointNT>
# #   class SampleConsensusModelCylinder : public SampleConsensusModel<PointT>, public SampleConsensusModelFromNormals<PointT, PointNT>
# #   {
# #     using SampleConsensusModel<PointT>::input_;
# #     using SampleConsensusModel<PointT>::indices_;
# #     using SampleConsensusModel<PointT>::radius_min_;
# #     using SampleConsensusModel<PointT>::radius_max_;
# #     using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
# #     using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
# # 
# #     public:
# #       typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
# #       typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
# #       typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
# # 
# #       typedef boost::shared_ptr<SampleConsensusModelCylinder> Ptr;
# # 
# #       /** \brief Constructor for base SampleConsensusModelCylinder.
# #         * \param[in] cloud the input point cloud dataset
# #         */
# #       SampleConsensusModelCylinder (const PointCloudConstPtr &cloud) : 
# #         SampleConsensusModel<PointT> (cloud), 
# #         axis_ (Eigen::Vector3f::Zero ()),
# #         eps_angle_ (0),
# #         tmp_inliers_ ()
# #       {
# #       }
# # 
# #       /** \brief Constructor for base SampleConsensusModelCylinder.
# #         * \param[in] cloud the input point cloud dataset
# #         * \param[in] indices a vector of point indices to be used from \a cloud
# #         */
# #       SampleConsensusModelCylinder (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
# #         SampleConsensusModel<PointT> (cloud, indices), 
# #         axis_ (Eigen::Vector3f::Zero ()),
# #         eps_angle_ (0),
# #         tmp_inliers_ ()
# #       {
# #       }
# # 
# #       /** \brief Copy constructor.
# #         * \param[in] source the model to copy into this
# #         */
# #       SampleConsensusModelCylinder (const SampleConsensusModelCylinder &source) :
# #         SampleConsensusModel<PointT> (),
# #         axis_ (Eigen::Vector3f::Zero ()),
# #         eps_angle_ (0),
# #         tmp_inliers_ ()
# #       {
# #         *this = source;
# #       }
# # 
# #       /** \brief Copy constructor.
# #         * \param[in] source the model to copy into this
# #         */
# #       inline SampleConsensusModelCylinder&
# #       operator = (const SampleConsensusModelCylinder &source)
# #       {
# #         SampleConsensusModel<PointT>::operator=(source);
# #         axis_ = source.axis_;
# #         eps_angle_ = source.eps_angle_;
# #         tmp_inliers_ = source.tmp_inliers_;
# #         return (*this);
# #       }
# # 
# #       /** \brief Set the angle epsilon (delta) threshold.
# #         * \param[in] ea the maximum allowed difference between the cyilinder axis and the given axis.
# #         */
# #       inline void 
# #       setEpsAngle (const double ea) { eps_angle_ = ea; }
# # 
# #       /** \brief Get the angle epsilon (delta) threshold. */
# #       inline double 
# #       getEpsAngle () { return (eps_angle_); }
# # 
# #       /** \brief Set the axis along which we need to search for a cylinder direction.
# #         * \param[in] ax the axis along which we need to search for a cylinder direction
# #         */
# #       inline void 
# #       setAxis (const Eigen::Vector3f &ax) { axis_ = ax; }
# # 
# #       /** \brief Get the axis along which we need to search for a cylinder direction. */
# #       inline Eigen::Vector3f 
# #       getAxis ()  { return (axis_); }
# # 
# #       /** \brief Check whether the given index samples can form a valid cylinder model, compute the model coefficients
# #         * from these samples and store them in model_coefficients. The cylinder coefficients are: point_on_axis,
# #         * axis_direction, cylinder_radius_R
# #         * \param[in] samples the point indices found as possible good candidates for creating a valid model
# #         * \param[out] model_coefficients the resultant model coefficients
# #         */
# #       bool 
# #       computeModelCoefficients (const std::vector<int> &samples, 
# #                                 Eigen::VectorXf &model_coefficients);
# # 
# #       /** \brief Compute all distances from the cloud data to a given cylinder model.
# #         * \param[in] model_coefficients the coefficients of a cylinder model that we need to compute distances to
# #         * \param[out] distances the resultant estimated distances
# #         */
# #       void 
# #       getDistancesToModel (const Eigen::VectorXf &model_coefficients, 
# #                            std::vector<double> &distances);
# # 
# #       /** \brief Select all the points which respect the given model coefficients as inliers.
# #         * \param[in] model_coefficients the coefficients of a cylinder model that we need to compute distances to
# #         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
# #         * \param[out] inliers the resultant model inliers
# #         */
# #       void 
# #       selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
# #                             const double threshold, 
# #                             std::vector<int> &inliers);
# # 
# #       /** \brief Count all the points which respect the given model coefficients as inliers. 
# #         * 
# #         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
# #         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
# #         * \return the resultant number of inliers
# #         */
# #       virtual int
# #       countWithinDistance (const Eigen::VectorXf &model_coefficients, 
# #                            const double threshold);
# # 
# #       /** \brief Recompute the cylinder coefficients using the given inlier set and return them to the user.
# #         * @note: these are the coefficients of the cylinder model after refinement (eg. after SVD)
# #         * \param[in] inliers the data inliers found as supporting the model
# #         * \param[in] model_coefficients the initial guess for the optimization
# #         * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
# #         */
# #       void 
# #       optimizeModelCoefficients (const std::vector<int> &inliers, 
# #                                  const Eigen::VectorXf &model_coefficients, 
# #                                  Eigen::VectorXf &optimized_coefficients);
# # 
# # 
# #       /** \brief Create a new point cloud with inliers projected onto the cylinder model.
# #         * \param[in] inliers the data inliers that we want to project on the cylinder model
# #         * \param[in] model_coefficients the coefficients of a cylinder model
# #         * \param[out] projected_points the resultant projected points
# #         * \param[in] copy_data_fields set to true if we need to copy the other data fields
# #         */
# #       void 
# #       projectPoints (const std::vector<int> &inliers, 
# #                      const Eigen::VectorXf &model_coefficients, 
# #                      PointCloud &projected_points, 
# #                      bool copy_data_fields = true);
# # 
# #       /** \brief Verify whether a subset of indices verifies the given cylinder model coefficients.
# #         * \param[in] indices the data indices that need to be tested against the cylinder model
# #         * \param[in] model_coefficients the cylinder model coefficients
# #         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
# #         */
# #       bool 
# #       doSamplesVerifyModel (const std::set<int> &indices, 
# #                             const Eigen::VectorXf &model_coefficients, 
# #                             const double threshold);
# # 
# #       /** \brief Return an unique id for this model (SACMODEL_CYLINDER). */
# #       inline pcl::SacModel 
# #       getModelType () const { return (SACMODEL_CYLINDER); }
# # 
# #     protected:
# #       /** \brief Get the distance from a point to a line (represented by a point and a direction)
# #         * \param[in] pt a point
# #         * \param[in] model_coefficients the line coefficients (a point on the line, line direction)
# #         */
# #       double 
# #       pointToLineDistance (const Eigen::Vector4f &pt, const Eigen::VectorXf &model_coefficients);
# # 
# #       /** \brief Project a point onto a line given by a point and a direction vector
# #         * \param[in] pt the input point to project
# #         * \param[in] line_pt the point on the line (make sure that line_pt[3] = 0 as there are no internal checks!)
# #         * \param[in] line_dir the direction of the line (make sure that line_dir[3] = 0 as there are no internal checks!)
# #         * \param[out] pt_proj the resultant projected point
# #         */
# #       inline void
# #       projectPointToLine (const Eigen::Vector4f &pt, 
# #                           const Eigen::Vector4f &line_pt, 
# #                           const Eigen::Vector4f &line_dir,
# #                           Eigen::Vector4f &pt_proj)
# #       {
# #         float k = (pt.dot (line_dir) - line_pt.dot (line_dir)) / line_dir.dot (line_dir);
# #         // Calculate the projection of the point on the line
# #         pt_proj = line_pt + k * line_dir;
# #       }
# # 
# #       /** \brief Project a point onto a cylinder given by its model coefficients (point_on_axis, axis_direction,
# #         * cylinder_radius_R)
# #         * \param[in] pt the input point to project
# #         * \param[in] model_coefficients the coefficients of the cylinder (point_on_axis, axis_direction, cylinder_radius_R)
# #         * \param[out] pt_proj the resultant projected point
# #         */
# #       void 
# #       projectPointToCylinder (const Eigen::Vector4f &pt, 
# #                               const Eigen::VectorXf &model_coefficients, 
# #                               Eigen::Vector4f &pt_proj);
# # 
# #       /** \brief Get a string representation of the name of this class. */
# #       std::string 
# #       getName () const { return ("SampleConsensusModelCylinder"); }
# # 
# #     protected:
# #       /** \brief Check whether a model is valid given the user constraints.
# #         * \param[in] model_coefficients the set of model coefficients
# #         */
# #       bool 
# #       isModelValid (const Eigen::VectorXf &model_coefficients);
# # 
# #       /** \brief Check if a sample of indices results in a good sample of points
# #         * indices. Pure virtual.
# #         * \param[in] samples the resultant index samples
# #         */
# #       bool
# #       isSampleGood (const std::vector<int> &samples) const;
# # 
# #     private:
# #       /** \brief The axis along which we need to search for a plane perpendicular to. */
# #       Eigen::Vector3f axis_;
# #     
# #       /** \brief The maximum allowed difference between the plane normal and the given axis. */
# #       double eps_angle_;
# # 
# #       /** \brief temporary pointer to a list of given indices for optimizeModelCoefficients () */
# #       const std::vector<int> *tmp_inliers_;
# # 
# # #if defined BUILD_Maintainer && defined __GNUC__ && __GNUC__ == 4 && __GNUC_MINOR__ > 3
# # #pragma GCC diagnostic ignored "-Weffc++"
# # #endif
# #       /** \brief Functor for the optimization function */
# #       struct OptimizationFunctor : pcl::Functor<float>
# #       {
# #         /** Functor constructor
# #           * \param[in] m_data_points the number of data points to evaluate
# #           * \param[in] estimator pointer to the estimator object
# #           * \param[in] distance distance computation function pointer
# #           */
# #         OptimizationFunctor (int m_data_points, pcl::SampleConsensusModelCylinder<PointT, PointNT> *model) : 
# #           pcl::Functor<float> (m_data_points), model_ (model) {}
# # 
# #         /** Cost function to be minimized
# #           * \param[in] x variables array
# #           * \param[out] fvec resultant functions evaluations
# #           * \return 0
# #           */
# #         int 
# #         operator() (const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
# #         {
# #           Eigen::Vector4f line_pt  (x[0], x[1], x[2], 0);
# #           Eigen::Vector4f line_dir (x[3], x[4], x[5], 0);
# #           
# #           for (int i = 0; i < values (); ++i)
# #           {
# #             // dist = f - r
# #             Eigen::Vector4f pt (model_->input_->points[(*model_->tmp_inliers_)[i]].x,
# #                                 model_->input_->points[(*model_->tmp_inliers_)[i]].y,
# #                                 model_->input_->points[(*model_->tmp_inliers_)[i]].z, 0);
# # 
# #             fvec[i] = static_cast<float> (pcl::sqrPointToLineDistance (pt, line_pt, line_dir) - x[6]*x[6]);
# #           }
# #           return (0);
# #         }
# # 
# #         pcl::SampleConsensusModelCylinder<PointT, PointNT> *model_;
# #       };
# ###
# 
# # sac_model_line.h
# # namespace pcl
# # {
# #   /** \brief SampleConsensusModelLine defines a model for 3D line segmentation.
# #     * The model coefficients are defined as:
# #     *   - \b point_on_line.x  : the X coordinate of a point on the line
# #     *   - \b point_on_line.y  : the Y coordinate of a point on the line
# #     *   - \b point_on_line.z  : the Z coordinate of a point on the line
# #     *   - \b line_direction.x : the X coordinate of a line's direction
# #     *   - \b line_direction.y : the Y coordinate of a line's direction
# #     *   - \b line_direction.z : the Z coordinate of a line's direction
# #     *
# #     * \author Radu B. Rusu
# #     * \ingroup sample_consensus
# #     */
# #   template <typename PointT>
# #   class SampleConsensusModelLine : public SampleConsensusModel<PointT>
# #   {
# #     using SampleConsensusModel<PointT>::input_;
# #     using SampleConsensusModel<PointT>::indices_;
# # 
# #     public:
# #       typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
# #       typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
# #       typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
# # 
# #       typedef boost::shared_ptr<SampleConsensusModelLine> Ptr;
# # 
# #       /** \brief Constructor for base SampleConsensusModelLine.
# #         * \param[in] cloud the input point cloud dataset
# #         */
# #       SampleConsensusModelLine (const PointCloudConstPtr &cloud) : SampleConsensusModel<PointT> (cloud) {};
# # 
# #       /** \brief Constructor for base SampleConsensusModelLine.
# #         * \param[in] cloud the input point cloud dataset
# #         * \param[in] indices a vector of point indices to be used from \a cloud
# #         */
# #       SampleConsensusModelLine (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : SampleConsensusModel<PointT> (cloud, indices) {};
# # 
# #       /** \brief Check whether the given index samples can form a valid line model, compute the model coefficients from
# #         * these samples and store them internally in model_coefficients_. The line coefficients are represented by a
# #         * point and a line direction
# #         * \param[in] samples the point indices found as possible good candidates for creating a valid model
# #         * \param[out] model_coefficients the resultant model coefficients
# #         */
# #       bool 
# #       computeModelCoefficients (const std::vector<int> &samples, 
# #                                 Eigen::VectorXf &model_coefficients);
# # 
# #       /** \brief Compute all squared distances from the cloud data to a given line model.
# #         * \param[in] model_coefficients the coefficients of a line model that we need to compute distances to
# #         * \param[out] distances the resultant estimated squared distances
# #         */
# #       void 
# #       getDistancesToModel (const Eigen::VectorXf &model_coefficients, 
# #                            std::vector<double> &distances);
# # 
# #       /** \brief Select all the points which respect the given model coefficients as inliers.
# #         * \param[in] model_coefficients the coefficients of a line model that we need to compute distances to
# #         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
# #         * \param[out] inliers the resultant model inliers
# #         */
# #       void 
# #       selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
# #                             const double threshold, 
# #                             std::vector<int> &inliers);
# # 
# #       /** \brief Count all the points which respect the given model coefficients as inliers. 
# #         * 
# #         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
# #         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
# #         * \return the resultant number of inliers
# #         */
# #       virtual int
# #       countWithinDistance (const Eigen::VectorXf &model_coefficients, 
# #                            const double threshold);
# # 
# #       /** \brief Recompute the line coefficients using the given inlier set and return them to the user.
# #         * @note: these are the coefficients of the line model after refinement (eg. after SVD)
# #         * \param[in] inliers the data inliers found as supporting the model
# #         * \param[in] model_coefficients the initial guess for the model coefficients
# #         * \param[out] optimized_coefficients the resultant recomputed coefficients after optimization
# #         */
# #       void 
# #       optimizeModelCoefficients (const std::vector<int> &inliers, 
# #                                  const Eigen::VectorXf &model_coefficients, 
# #                                  Eigen::VectorXf &optimized_coefficients);
# # 
# #       /** \brief Create a new point cloud with inliers projected onto the line model.
# #         * \param[in] inliers the data inliers that we want to project on the line model
# #         * \param[in] model_coefficients the *normalized* coefficients of a line model
# #         * \param[out] projected_points the resultant projected points
# #         * \param[in] copy_data_fields set to true if we need to copy the other data fields
# #         */
# #       void 
# #       projectPoints (const std::vector<int> &inliers, 
# #                      const Eigen::VectorXf &model_coefficients, 
# #                      PointCloud &projected_points, 
# #                      bool copy_data_fields = true);
# # 
# #       /** \brief Verify whether a subset of indices verifies the given line model coefficients.
# #         * \param[in] indices the data indices that need to be tested against the line model
# #         * \param[in] model_coefficients the line model coefficients
# #         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
# #         */
# #       bool 
# #       doSamplesVerifyModel (const std::set<int> &indices, 
# #                             const Eigen::VectorXf &model_coefficients, 
# #                             const double threshold);
# # 
# #       /** \brief Return an unique id for this model (SACMODEL_LINE). */
# #       inline pcl::SacModel 
# #       getModelType () const { return (SACMODEL_LINE); }
# # 
# #     protected:
# #       /** \brief Check whether a model is valid given the user constraints.
# #         * \param[in] model_coefficients the set of model coefficients
# #         */
# #       inline bool 
# #       isModelValid (const Eigen::VectorXf &model_coefficients)
# #       {
# #         if (model_coefficients.size () != 6)
# #         {
# #           PCL_ERROR ("[pcl::SampleConsensusModelLine::selectWithinDistance] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
# #           return (false);
# #         }
# # 
# #         return (true);
# #       }
# # 
# #       /** \brief Check if a sample of indices results in a good sample of points
# #         * indices.
# #         * \param[in] samples the resultant index samples
# #         */
# #       bool
# #       isSampleGood (const std::vector<int> &samples) const;
# # };
# ###
# 
# # sac_model_normal_parallel_plane.h
# # namespace pcl
# # {
# #   /** \brief SampleConsensusModelNormalParallelPlane defines a model for 3D
# #     * plane segmentation using additional surface normal constraints. Basically
# #     * this means that checking for inliers will not only involve a "distance to
# #     * model" criterion, but also an additional "maximum angular deviation"
# #     * between the plane's normal and the inlier points normals. In addition,
# #     * the plane normal must lie parallel to an user-specified axis.
# #     *
# #     * The model coefficients are defined as:
# #     *   - \b a : the X coordinate of the plane's normal (normalized)
# #     *   - \b b : the Y coordinate of the plane's normal (normalized)
# #     *   - \b c : the Z coordinate of the plane's normal (normalized)
# #     *   - \b d : the fourth <a href="http://mathworld.wolfram.com/HessianNormalForm.html">Hessian component</a> of the plane's equation
# #     *
# #     * To set the influence of the surface normals in the inlier estimation
# #     * process, set the normal weight (0.0-1.0), e.g.:
# #     * \code
# #     * SampleConsensusModelNormalPlane<pcl::PointXYZ, pcl::Normal> sac_model;
# #     * ...
# #     * sac_model.setNormalDistanceWeight (0.1);
# #     * ...
# #     * \endcode
# #     *
# #     * In addition, the user can specify more constraints, such as:
# #     * 
# #     *   - an axis along which we need to search for a plane perpendicular to (\ref setAxis);
# #     *   - an angle \a tolerance threshold between the plane's normal and the above given axis (\ref setEpsAngle);
# #     *   - a distance we expect the plane to be from the origin (\ref setDistanceFromOrigin);
# #     *   - a distance \a tolerance as the maximum allowed deviation from the above given distance from the origin (\ref setEpsDist).
# #     *
# #     * \note Please remember that you need to specify an angle > 0 in order to activate the axis-angle constraint!
# #     *
# #     * \author Radu B. Rusu and Jared Glover and Nico Blodow
# #     * \ingroup sample_consensus
# #     */
# #   template <typename PointT, typename PointNT>
# #   class SampleConsensusModelNormalParallelPlane : public SampleConsensusModelPlane<PointT>, public SampleConsensusModelFromNormals<PointT, PointNT>
# #   {
# #     using SampleConsensusModel<PointT>::input_;
# #     using SampleConsensusModel<PointT>::indices_;
# #     using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
# #     using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
# # 
# #     public:
# # 
# #       typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
# #       typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
# #       typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
# # 
# #       typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNPtr PointCloudNPtr;
# #       typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNConstPtr PointCloudNConstPtr;
# # 
# #       typedef boost::shared_ptr<SampleConsensusModelNormalParallelPlane> Ptr;
# # 
# #       /** \brief Constructor for base SampleConsensusModelNormalParallelPlane.
# #         * \param[in] cloud the input point cloud dataset
# #         */
# #       SampleConsensusModelNormalParallelPlane (const PointCloudConstPtr &cloud) : 
# #         SampleConsensusModelPlane<PointT> (cloud),
# #         axis_ (Eigen::Vector4f::Zero ()),
# #         distance_from_origin_ (0),
# #         eps_angle_ (-1.0), cos_angle_ (-1.0), eps_dist_ (0.0)
# #       {
# #       }
# # 
# #       /** \brief Constructor for base SampleConsensusModelNormalParallelPlane.
# #         * \param[in] cloud the input point cloud dataset
# #         * \param[in] indices a vector of point indices to be used from \a cloud
# #         */
# #       SampleConsensusModelNormalParallelPlane (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
# #         SampleConsensusModelPlane<PointT> (cloud, indices),
# #         axis_ (Eigen::Vector4f::Zero ()),
# #         distance_from_origin_ (0),
# #         eps_angle_ (-1.0), cos_angle_ (-1.0), eps_dist_ (0.0)
# #       {
# #       }
# # 
# #       /** \brief Set the axis along which we need to search for a plane perpendicular to.
# #         * \param[in] ax the axis along which we need to search for a plane perpendicular to
# #         */
# #       inline void
# #       setAxis (const Eigen::Vector3f &ax) { axis_.head<3> () = ax; axis_.normalize ();}
# # 
# #       /** \brief Get the axis along which we need to search for a plane perpendicular to. */
# #       inline Eigen::Vector3f
# #       getAxis () { return (axis_.head<3> ()); }
# # 
# #       /** \brief Set the angle epsilon (delta) threshold.
# #         * \param[in] ea the maximum allowed deviation from 90 degrees between the plane normal and the given axis.
# #         * \note You need to specify an angle > 0 in order to activate the axis-angle constraint!
# #         */
# #       inline void
# #       setEpsAngle (const double ea) { eps_angle_ = ea; cos_angle_ = fabs (cos (ea));}
# # 
# #       /** \brief Get the angle epsilon (delta) threshold. */
# #       inline double
# #       getEpsAngle () { return (eps_angle_); }
# # 
# #       /** \brief Set the distance we expect the plane to be from the origin
# #         * \param[in] d distance from the template plane to the origin
# #         */
# #       inline void
# #       setDistanceFromOrigin (const double d) { distance_from_origin_ = d; }
# # 
# #       /** \brief Get the distance of the plane from the origin. */
# #       inline double
# #       getDistanceFromOrigin () { return (distance_from_origin_); }
# # 
# #       /** \brief Set the distance epsilon (delta) threshold.
# #         * \param[in] delta the maximum allowed deviation from the template distance from the origin
# #         */
# #       inline void
# #       setEpsDist (const double delta) { eps_dist_ = delta; }
# # 
# #       /** \brief Get the distance epsilon (delta) threshold. */
# #       inline double
# #       getEpsDist () { return (eps_dist_); }
# # 
# #       /** \brief Select all the points which respect the given model coefficients as inliers.
# #         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
# #         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
# #         * \param[out] inliers the resultant model inliers
# #         */
# #       void
# #       selectWithinDistance (const Eigen::VectorXf &model_coefficients,
# #                             const double threshold,
# #                             std::vector<int> &inliers);
# # 
# #       /** \brief Count all the points which respect the given model coefficients as inliers.
# #         *
# #         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
# #         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
# #         * \return the resultant number of inliers
# #         */
# #       virtual int
# #       countWithinDistance (const Eigen::VectorXf &model_coefficients,
# #                            const double threshold);
# # 
# #       /** \brief Compute all distances from the cloud data to a given plane model.
# #         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
# #         * \param[out] distances the resultant estimated distances
# #         */
# #       void
# #       getDistancesToModel (const Eigen::VectorXf &model_coefficients,
# #                            std::vector<double> &distances);
# # 
# #       /** \brief Return an unique id for this model (SACMODEL_NORMAL_PARALLEL_PLANE). */
# #       inline pcl::SacModel
# #       getModelType () const { return (SACMODEL_NORMAL_PARALLEL_PLANE); }
# # 
# #     	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
# # 
# #     protected:
# #       /** \brief Check whether a model is valid given the user constraints.
# #         * \param[in] model_coefficients the set of model coefficients
# #         */
# #       bool
# #       isModelValid (const Eigen::VectorXf &model_coefficients);
# # 
# #    private:
# #       /** \brief The axis along which we need to search for a plane perpendicular to. */
# #       Eigen::Vector4f axis_;
# # 
# #       /** \brief The distance from the template plane to the origin. */
# #       double distance_from_origin_;
# # 
# #       /** \brief The maximum allowed difference between the plane normal and the given axis.  */
# #       double eps_angle_;
# # 
# #       /** \brief The cosine of the angle*/
# #       double cos_angle_;
# #       /** \brief The maximum allowed deviation from the template distance from the origin. */
# #       double eps_dist_;
# ###
# 
# # sac_model_normal_plane.h
# # namespace pcl
# # {
# #   /** \brief SampleConsensusModelNormalPlane defines a model for 3D plane
# #     * segmentation using additional surface normal constraints. Basically this
# #     * means that checking for inliers will not only involve a "distance to
# #     * model" criterion, but also an additional "maximum angular deviation"
# #     * between the plane's normal and the inlier points normals.
# #     *
# #     * The model coefficients are defined as:
# #     *   - \b a : the X coordinate of the plane's normal (normalized)
# #     *   - \b b : the Y coordinate of the plane's normal (normalized)
# #     *   - \b c : the Z coordinate of the plane's normal (normalized)
# #     *   - \b d : the fourth <a href="http://mathworld.wolfram.com/HessianNormalForm.html">Hessian component</a> of the plane's equation
# #     *
# #     * To set the influence of the surface normals in the inlier estimation
# #     * process, set the normal weight (0.0-1.0), e.g.:
# #     * \code
# #     * SampleConsensusModelNormalPlane<pcl::PointXYZ, pcl::Normal> sac_model;
# #     * ...
# #     * sac_model.setNormalDistanceWeight (0.1);
# #     * ...
# #     * \endcode
# #     *
# #     * \author Radu B. Rusu and Jared Glover
# #     * \ingroup sample_consensus
# #     */
# #   template <typename PointT, typename PointNT>
# #   class SampleConsensusModelNormalPlane : public SampleConsensusModelPlane<PointT>, public SampleConsensusModelFromNormals<PointT, PointNT>
# #   {
# #     using SampleConsensusModel<PointT>::input_;
# #     using SampleConsensusModel<PointT>::indices_;
# #     using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
# #     using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
# # 
# #     public:
# # 
# #       typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
# #       typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
# #       typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
# # 
# #       typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNPtr PointCloudNPtr;
# #       typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNConstPtr PointCloudNConstPtr;
# # 
# #       typedef boost::shared_ptr<SampleConsensusModelNormalPlane> Ptr;
# # 
# #       /** \brief Constructor for base SampleConsensusModelNormalPlane.
# #         * \param[in] cloud the input point cloud dataset
# #         */
# #       SampleConsensusModelNormalPlane (const PointCloudConstPtr &cloud) : SampleConsensusModelPlane<PointT> (cloud)
# #       {
# #       }
# # 
# #       /** \brief Constructor for base SampleConsensusModelNormalPlane.
# #         * \param[in] cloud the input point cloud dataset
# #         * \param[in] indices a vector of point indices to be used from \a cloud
# #         */
# #       SampleConsensusModelNormalPlane (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : SampleConsensusModelPlane<PointT> (cloud, indices)
# #       {
# #       }
# # 
# #       /** \brief Select all the points which respect the given model coefficients as inliers.
# #         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
# #         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
# #         * \param[out] inliers the resultant model inliers
# #         */
# #       void 
# #       selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
# #                             const double threshold, 
# #                             std::vector<int> &inliers);
# # 
# #       /** \brief Count all the points which respect the given model coefficients as inliers. 
# #         * 
# #         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
# #         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
# #         * \return the resultant number of inliers
# #         */
# #       virtual int
# #       countWithinDistance (const Eigen::VectorXf &model_coefficients, 
# #                            const double threshold);
# # 
# #       /** \brief Compute all distances from the cloud data to a given plane model.
# #         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
# #         * \param[out] distances the resultant estimated distances
# #         */
# #       void 
# #       getDistancesToModel (const Eigen::VectorXf &model_coefficients, 
# #                            std::vector<double> &distances);
# # 
# #       /** \brief Return an unique id for this model (SACMODEL_NORMAL_PLANE). */
# #       inline pcl::SacModel 
# #       getModelType () const { return (SACMODEL_NORMAL_PLANE); }
# # 
# #     	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
# # 
# #     protected:
# #       /** \brief Check whether a model is valid given the user constraints.
# #         * \param[in] model_coefficients the set of model coefficients
# #         */
# #       bool 
# #       isModelValid (const Eigen::VectorXf &model_coefficients);
# # 
# ###
# 
# sac_model_normal_sphere.h
# namespace pcl
# {
#   /** \brief @b SampleConsensusModelNormalSphere defines a model for 3D sphere
#     * segmentation using additional surface normal constraints. Basically this
#     * means that checking for inliers will not only involve a "distance to
#     * model" criterion, but also an additional "maximum angular deviation"
#     * between the sphere's normal and the inlier points normals.
#     *
#     * The model coefficients are defined as:
#     * <ul>
#     * <li><b>a</b> : the X coordinate of the plane's normal (normalized)
#     * <li><b>b</b> : the Y coordinate of the plane's normal (normalized)
#     * <li><b>c</b> : the Z coordinate of the plane's normal (normalized)
#     * <li><b>d</b> : radius of the sphere
#     * </ul>
#     *
#     * \author Stefan Schrandt
#     * \ingroup sample_consensus
#     */
#   template <typename PointT, typename PointNT>
#   class SampleConsensusModelNormalSphere : public SampleConsensusModelSphere<PointT>, public SampleConsensusModelFromNormals<PointT, PointNT>
#   {
#     using SampleConsensusModel<PointT>::input_;
#     using SampleConsensusModel<PointT>::indices_;
#     using SampleConsensusModel<PointT>::radius_min_;
#     using SampleConsensusModel<PointT>::radius_max_;
#     using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
#     using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
# 
#     public:
# 
#       typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
#       typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
#       typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
# 
#       typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNPtr PointCloudNPtr;
#       typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNConstPtr PointCloudNConstPtr;
# 
#       typedef boost::shared_ptr<SampleConsensusModelNormalSphere> Ptr;
# 
#       /** \brief Constructor for base SampleConsensusModelNormalSphere.
#         * \param[in] cloud the input point cloud dataset
#         */
#       SampleConsensusModelNormalSphere (const PointCloudConstPtr &cloud) : SampleConsensusModelSphere<PointT> (cloud)
#       {
#       }
# 
#       /** \brief Constructor for base SampleConsensusModelNormalSphere.
#         * \param[in] cloud the input point cloud dataset
#         * \param[in] indices a vector of point indices to be used from \a cloud
#         */
#       SampleConsensusModelNormalSphere (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : SampleConsensusModelSphere<PointT> (cloud, indices)
#       {
#       }
# 
#       /** \brief Select all the points which respect the given model coefficients as inliers.
#         * \param[in] model_coefficients the coefficients of a sphere model that we need to compute distances to
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         * \param[out] inliers the resultant model inliers
#         */
#       void 
#       selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                             const double threshold, 
#                             std::vector<int> &inliers);
# 
#       /** \brief Count all the points which respect the given model coefficients as inliers. 
#         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
#         * \return the resultant number of inliers
#         */
#       virtual int
#       countWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                            const double threshold);
# 
#       /** \brief Compute all distances from the cloud data to a given sphere model.
#         * \param[in] model_coefficients the coefficients of a sphere model that we need to compute distances to
#         * \param[out] distances the resultant estimated distances
#         */
#       void 
#       getDistancesToModel (const Eigen::VectorXf &model_coefficients, 
#                            std::vector<double> &distances);
# 
#       /** \brief Return an unique id for this model (SACMODEL_NORMAL_SPHERE). */
#       inline pcl::SacModel 
#       getModelType () const { return (SACMODEL_NORMAL_SPHERE); }
# 
#     	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
# 
#     protected:
#       /** \brief Check whether a model is valid given the user constraints.
#         * \param[in] model_coefficients the set of model coefficients
#         */
#       bool 
#       isModelValid (const Eigen::VectorXf &model_coefficients);
# 
###

# sac_model_parallel_line.h
# namespace pcl
# {
#   /** \brief SampleConsensusModelParallelLine defines a model for 3D line segmentation using additional angular
#     * constraints.
#     * The model coefficients are defined as:
#     *   - \b point_on_line.x  : the X coordinate of a point on the line
#     *   - \b point_on_line.y  : the Y coordinate of a point on the line
#     *   - \b point_on_line.z  : the Z coordinate of a point on the line
#     *   - \b line_direction.x : the X coordinate of a line's direction
#     *   - \b line_direction.y : the Y coordinate of a line's direction
#     *   - \b line_direction.z : the Z coordinate of a line's direction
#     * 
#     * \author Radu B. Rusu
#     * \ingroup sample_consensus
#     */
#   template <typename PointT>
#   class SampleConsensusModelParallelLine : public SampleConsensusModelLine<PointT>
#   {
#     public:
#       typedef typename SampleConsensusModelLine<PointT>::PointCloud PointCloud;
#       typedef typename SampleConsensusModelLine<PointT>::PointCloudPtr PointCloudPtr;
#       typedef typename SampleConsensusModelLine<PointT>::PointCloudConstPtr PointCloudConstPtr;
# 
#       typedef boost::shared_ptr<SampleConsensusModelParallelLine> Ptr;
# 
#       /** \brief Constructor for base SampleConsensusModelParallelLine.
#         * \param[in] cloud the input point cloud dataset
#         */
#       SampleConsensusModelParallelLine (const PointCloudConstPtr &cloud) : 
#         SampleConsensusModelLine<PointT> (cloud),
#         axis_ (Eigen::Vector3f::Zero ()),
#         eps_angle_ (0.0)
#       {
#       }
# 
#       /** \brief Constructor for base SampleConsensusModelParallelLine.
#         * \param[in] cloud the input point cloud dataset
#         * \param[in] indices a vector of point indices to be used from \a cloud
#         */
#       SampleConsensusModelParallelLine (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
#         SampleConsensusModelLine<PointT> (cloud, indices),
#         axis_ (Eigen::Vector3f::Zero ()),
#         eps_angle_ (0.0)
#       {
#       }
# 
#       /** \brief Set the axis along which we need to search for a plane perpendicular to.
#         * \param[in] ax the axis along which we need to search for a plane perpendicular to
#         */
#       inline void 
#       setAxis (const Eigen::Vector3f &ax) { axis_ = ax; axis_.normalize (); }
# 
#       /** \brief Get the axis along which we need to search for a plane perpendicular to. */
#       inline Eigen::Vector3f 
#       getAxis ()  { return (axis_); }
# 
#       /** \brief Set the angle epsilon (delta) threshold.
#         * \param[in] ea the maximum allowed difference between the plane normal and the given axis.
#         */
#       inline void 
#       setEpsAngle (const double ea) { eps_angle_ = ea; }
# 
#       /** \brief Get the angle epsilon (delta) threshold. */
#       inline double getEpsAngle () { return (eps_angle_); }
# 
#       /** \brief Select all the points which respect the given model coefficients as inliers.
#         * \param[in] model_coefficients the coefficients of a line model that we need to compute distances to
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         * \param[out] inliers the resultant model inliers
#         */
#       void 
#       selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                             const double threshold, 
#                             std::vector<int> &inliers);
# 
#       /** \brief Count all the points which respect the given model coefficients as inliers. 
#         * 
#         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
#         * \return the resultant number of inliers
#         */
#       virtual int
#       countWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                            const double threshold);
# 
#       /** \brief Compute all squared distances from the cloud data to a given line model.
#         * \param[in] model_coefficients the coefficients of a line model that we need to compute distances to
#         * \param[out] distances the resultant estimated squared distances
#         */
#       void 
#       getDistancesToModel (const Eigen::VectorXf &model_coefficients, 
#                            std::vector<double> &distances);
# 
#       /** \brief Return an unique id for this model (SACMODEL_PARALLEL_LINE). */
#       inline pcl::SacModel 
#       getModelType () const { return (SACMODEL_PARALLEL_LINE); }
# 
#     protected:
#       /** \brief Check whether a model is valid given the user constraints.
#         * \param[in] model_coefficients the set of model coefficients
#         */
#       bool 
#       isModelValid (const Eigen::VectorXf &model_coefficients);
# 
#     protected:
#       /** \brief The axis along which we need to search for a plane perpendicular to. */
#       Eigen::Vector3f axis_;
# 
#       /** \brief The maximum allowed difference between the plane normal and the given axis. */
#       double eps_angle_;
#   };
# 
###

# sac_model_parallel_plane.h
# namespace pcl
# {
#   /** \brief @b SampleConsensusModelParallelPlane defines a model for 3D plane segmentation using additional
#     * angular constraints. The plane must be parallel to a user-specified axis
#     * (\ref setAxis) within an user-specified angle threshold (\ref setEpsAngle).
#     *
#     * Code example for a plane model, parallel (within a 15 degrees tolerance) with the Z axis:
#     * \code
#     * SampleConsensusModelParallelPlane<pcl::PointXYZ> model (cloud);
#     * model.setAxis (Eigen::Vector3f (0.0, 0.0, 1.0));
#     * model.setEpsAngle (pcl::deg2rad (15));
#     * \endcode
#     *
#     * \note Please remember that you need to specify an angle > 0 in order to activate the axis-angle constraint!
#     *
#     * \author Radu Bogdan Rusu, Nico Blodow
#     * \ingroup sample_consensus
#     */
#   template <typename PointT>
#   class SampleConsensusModelParallelPlane : public SampleConsensusModelPlane<PointT>
#   {
#     public:
#       typedef typename SampleConsensusModelPlane<PointT>::PointCloud PointCloud;
#       typedef typename SampleConsensusModelPlane<PointT>::PointCloudPtr PointCloudPtr;
#       typedef typename SampleConsensusModelPlane<PointT>::PointCloudConstPtr PointCloudConstPtr;
# 
#       typedef boost::shared_ptr<SampleConsensusModelParallelPlane> Ptr;
# 
#       /** \brief Constructor for base SampleConsensusModelParallelPlane.
#         * \param[in] cloud the input point cloud dataset
#         */
#       SampleConsensusModelParallelPlane (const PointCloudConstPtr &cloud) : 
#         SampleConsensusModelPlane<PointT> (cloud),
#         axis_ (Eigen::Vector3f::Zero ()),
#         eps_angle_ (0.0), sin_angle_ (-1.0)
#       {
#       }
# 
#       /** \brief Constructor for base SampleConsensusModelParallelPlane.
#         * \param[in] cloud the input point cloud dataset
#         * \param[in] indices a vector of point indices to be used from \a cloud
#         */
#       SampleConsensusModelParallelPlane (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
#         SampleConsensusModelPlane<PointT> (cloud, indices),
#         axis_ (Eigen::Vector3f::Zero ()),
#         eps_angle_ (0.0), sin_angle_ (-1.0)
#       {
#       }
# 
#       /** \brief Set the axis along which we need to search for a plane perpendicular to.
#         * \param[in] ax the axis along which we need to search for a plane perpendicular to
#         */
#       inline void
#       setAxis (const Eigen::Vector3f &ax) { axis_ = ax; }
# 
#       /** \brief Get the axis along which we need to search for a plane perpendicular to. */
#       inline Eigen::Vector3f
#       getAxis ()  { return (axis_); }
# 
#       /** \brief Set the angle epsilon (delta) threshold.
#         * \param[in] ea the maximum allowed difference between the plane normal and the given axis.
#         * \note You need to specify an angle > 0 in order to activate the axis-angle constraint!
#         */
#       inline void
#       setEpsAngle (const double ea) { eps_angle_ = ea; sin_angle_ = fabs (sin (ea));}
# 
#       /** \brief Get the angle epsilon (delta) threshold. */
#       inline double
#       getEpsAngle () { return (eps_angle_); }
# 
#       /** \brief Select all the points which respect the given model coefficients as inliers.
#         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         * \param[out] inliers the resultant model inliers
#         */
#       void
#       selectWithinDistance (const Eigen::VectorXf &model_coefficients,
#                             const double threshold,
#                             std::vector<int> &inliers);
# 
#       /** \brief Count all the points which respect the given model coefficients as inliers.
#         *
#         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
#         * \return the resultant number of inliers
#         */
#       virtual int
#       countWithinDistance (const Eigen::VectorXf &model_coefficients,
#                            const double threshold);
# 
#       /** \brief Compute all distances from the cloud data to a given plane model.
#         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
#         * \param[out] distances the resultant estimated distances
#         */
#       void
#       getDistancesToModel (const Eigen::VectorXf &model_coefficients,
#                            std::vector<double> &distances);
# 
#       /** \brief Return an unique id for this model (SACMODEL_PARALLEL_PLANE). */
#       inline pcl::SacModel
#       getModelType () const { return (SACMODEL_PARALLEL_PLANE); }
# 
#     protected:
#       /** \brief Check whether a model is valid given the user constraints.
#         * \param[in] model_coefficients the set of model coefficients
#         */
#       bool
#       isModelValid (const Eigen::VectorXf &model_coefficients);
# 
#       /** \brief The axis along which we need to search for a plane perpendicular to. */
#       Eigen::Vector3f axis_;
# 
#       /** \brief The maximum allowed difference between the plane and the given axis. */
#       double eps_angle_;
# 
#       /** \brief The sine of the angle*/
#       double sin_angle_;
###

# sac_model_perpendicular_plane.h
# namespace pcl
# {
#   /** \brief SampleConsensusModelPerpendicularPlane defines a model for 3D plane segmentation using additional
#     * angular constraints. The plane must be perpendicular to an user-specified axis (\ref setAxis), up to an user-specified angle threshold (\ref setEpsAngle).
#     * The model coefficients are defined as:
#     *   - \b a : the X coordinate of the plane's normal (normalized)
#     *   - \b b : the Y coordinate of the plane's normal (normalized)
#     *   - \b c : the Z coordinate of the plane's normal (normalized)
#     *   - \b d : the fourth <a href="http://mathworld.wolfram.com/HessianNormalForm.html">Hessian component</a> of the plane's equation
#     * 
#     * 
#     * Code example for a plane model, perpendicular (within a 15 degrees tolerance) with the Z axis:
#     * \code
#     * SampleConsensusModelPerpendicularPlane<pcl::PointXYZ> model (cloud);
#     * model.setAxis (Eigen::Vector3f (0.0, 0.0, 1.0));
#     * model.setEpsAngle (pcl::deg2rad (15));
#     * \endcode
#     *
#     * \note Please remember that you need to specify an angle > 0 in order to activate the axis-angle constraint!
#     *
#     * \author Radu B. Rusu
#     * \ingroup sample_consensus
#     */
#   template <typename PointT>
#   class SampleConsensusModelPerpendicularPlane : public SampleConsensusModelPlane<PointT>
#   {
#     public:
#       typedef typename SampleConsensusModelPlane<PointT>::PointCloud PointCloud;
#       typedef typename SampleConsensusModelPlane<PointT>::PointCloudPtr PointCloudPtr;
#       typedef typename SampleConsensusModelPlane<PointT>::PointCloudConstPtr PointCloudConstPtr;
# 
#       typedef boost::shared_ptr<SampleConsensusModelPerpendicularPlane> Ptr;
# 
#       /** \brief Constructor for base SampleConsensusModelPerpendicularPlane.
#         * \param[in] cloud the input point cloud dataset
#         */
#       SampleConsensusModelPerpendicularPlane (const PointCloudConstPtr &cloud) : 
#         SampleConsensusModelPlane<PointT> (cloud), 
#         axis_ (Eigen::Vector3f::Zero ()),
#         eps_angle_ (0.0)
#       {
#       }
# 
#       /** \brief Constructor for base SampleConsensusModelPerpendicularPlane.
#         * \param[in] cloud the input point cloud dataset
#         * \param[in] indices a vector of point indices to be used from \a cloud
#         */
#       SampleConsensusModelPerpendicularPlane (const PointCloudConstPtr &cloud, 
#                                               const std::vector<int> &indices) : 
#         SampleConsensusModelPlane<PointT> (cloud, indices), 
#         axis_ (Eigen::Vector3f::Zero ()),
#         eps_angle_ (0.0)
#       {
#       }
# 
#       /** \brief Set the axis along which we need to search for a plane perpendicular to.
#         * \param[in] ax the axis along which we need to search for a plane perpendicular to
#         */
#       inline void 
#       setAxis (const Eigen::Vector3f &ax) { axis_ = ax; }
# 
#       /** \brief Get the axis along which we need to search for a plane perpendicular to. */
#       inline Eigen::Vector3f 
#       getAxis ()  { return (axis_); }
# 
#       /** \brief Set the angle epsilon (delta) threshold.
#         * \param[in] ea the maximum allowed difference between the plane normal and the given axis.
#         * \note You need to specify an angle > 0 in order to activate the axis-angle constraint!
#         */
#       inline void 
#       setEpsAngle (const double ea) { eps_angle_ = ea; }
# 
#       /** \brief Get the angle epsilon (delta) threshold. */
#       inline double 
#       getEpsAngle () { return (eps_angle_); }
# 
#       /** \brief Select all the points which respect the given model coefficients as inliers.
#         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         * \param[out] inliers the resultant model inliers
#         */
#       void 
#       selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                             const double threshold, 
#                             std::vector<int> &inliers);
# 
#       /** \brief Count all the points which respect the given model coefficients as inliers. 
#         * 
#         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
#         * \return the resultant number of inliers
#         */
#       virtual int
#       countWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                            const double threshold);
# 
#       /** \brief Compute all distances from the cloud data to a given plane model.
#         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
#         * \param[out] distances the resultant estimated distances
#         */
#       void 
#       getDistancesToModel (const Eigen::VectorXf &model_coefficients, 
#                            std::vector<double> &distances);
# 
#       /** \brief Return an unique id for this model (SACMODEL_PERPENDICULAR_PLANE). */
#       inline pcl::SacModel 
#       getModelType () const { return (SACMODEL_PERPENDICULAR_PLANE); }
# 
#     protected:
#       /** \brief Check whether a model is valid given the user constraints.
#         * \param[in] model_coefficients the set of model coefficients
#         */
#       bool 
#       isModelValid (const Eigen::VectorXf &model_coefficients);
# 
#       /** \brief The axis along which we need to search for a plane perpendicular to. */
#       Eigen::Vector3f axis_;
# 
#       /** \brief The maximum allowed difference between the plane normal and the given axis. */
#       double eps_angle_;
#   };
###

# sac_model_plane.h
# namespace pcl
# {
# 
#   /** \brief Project a point on a planar model given by a set of normalized coefficients
#     * \param[in] p the input point to project
#     * \param[in] model_coefficients the coefficients of the plane (a, b, c, d) that satisfy ax+by+cz+d=0
#     * \param[out] q the resultant projected point
#     */
#   template <typename Point> inline void
#   projectPoint (const Point &p, const Eigen::Vector4f &model_coefficients, Point &q)
#   {
#     // Calculate the distance from the point to the plane
#     Eigen::Vector4f pp (p.x, p.y, p.z, 1);
#     // use normalized coefficients to calculate the scalar projection 
#     float distance_to_plane = pp.dot(model_coefficients);
# 
# 
#     //TODO: Why doesn't getVector4Map work here?
#     //Eigen::Vector4f q_e = q.getVector4fMap ();
#     //q_e = pp - model_coefficients * distance_to_plane;
#     
#     Eigen::Vector4f q_e = pp - distance_to_plane * model_coefficients;
#     q.x = q_e[0];
#     q.y = q_e[1];
#     q.z = q_e[2];
#   }
# 
#   /** \brief Get the distance from a point to a plane (signed) defined by ax+by+cz+d=0
#     * \param p a point
#     * \param a the normalized <i>a</i> coefficient of a plane
#     * \param b the normalized <i>b</i> coefficient of a plane
#     * \param c the normalized <i>c</i> coefficient of a plane
#     * \param d the normalized <i>d</i> coefficient of a plane
#     * \ingroup sample_consensus
#     */
#   template <typename Point> inline double
#   pointToPlaneDistanceSigned (const Point &p, double a, double b, double c, double d)
#   {
#     return (a * p.x + b * p.y + c * p.z + d);
#   }
# 
#   /** \brief Get the distance from a point to a plane (signed) defined by ax+by+cz+d=0
#     * \param p a point
#     * \param plane_coefficients the normalized coefficients (a, b, c, d) of a plane
#     * \ingroup sample_consensus
#     */
#   template <typename Point> inline double
#   pointToPlaneDistanceSigned (const Point &p, const Eigen::Vector4f &plane_coefficients)
#   {
#     return ( plane_coefficients[0] * p.x + plane_coefficients[1] * p.y + plane_coefficients[2] * p.z + plane_coefficients[3] );
#   }
# 
#   /** \brief Get the distance from a point to a plane (unsigned) defined by ax+by+cz+d=0
#     * \param p a point
#     * \param a the normalized <i>a</i> coefficient of a plane
#     * \param b the normalized <i>b</i> coefficient of a plane
#     * \param c the normalized <i>c</i> coefficient of a plane
#     * \param d the normalized <i>d</i> coefficient of a plane
#     * \ingroup sample_consensus
#     */
#   template <typename Point> inline double
#   pointToPlaneDistance (const Point &p, double a, double b, double c, double d)
#   {
#     return (fabs (pointToPlaneDistanceSigned (p, a, b, c, d)) );
#   }
# 
#   /** \brief Get the distance from a point to a plane (unsigned) defined by ax+by+cz+d=0
#     * \param p a point
#     * \param plane_coefficients the normalized coefficients (a, b, c, d) of a plane
#     * \ingroup sample_consensus
#     */
#   template <typename Point> inline double
#   pointToPlaneDistance (const Point &p, const Eigen::Vector4f &plane_coefficients)
#   {
#     return ( fabs (pointToPlaneDistanceSigned (p, plane_coefficients)) );
#   }
# 
#   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#   /** \brief SampleConsensusModelPlane defines a model for 3D plane segmentation.
#     * The model coefficients are defined as:
#     *   - \b a : the X coordinate of the plane's normal (normalized)
#     *   - \b b : the Y coordinate of the plane's normal (normalized)
#     *   - \b c : the Z coordinate of the plane's normal (normalized)
#     *   - \b d : the fourth <a href="http://mathworld.wolfram.com/HessianNormalForm.html">Hessian component</a> of the plane's equation
#     * 
#     * \author Radu B. Rusu
#     * \ingroup sample_consensus
#     */
#   template <typename PointT>
#   class SampleConsensusModelPlane : public SampleConsensusModel<PointT>
#   {
#     public:
#       using SampleConsensusModel<PointT>::input_;
#       using SampleConsensusModel<PointT>::indices_;
# 
#       typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
#       typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
#       typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
# 
#       typedef boost::shared_ptr<SampleConsensusModelPlane> Ptr;
# 
#       /** \brief Constructor for base SampleConsensusModelPlane.
#         * \param[in] cloud the input point cloud dataset
#         */
#       SampleConsensusModelPlane (const PointCloudConstPtr &cloud) : SampleConsensusModel<PointT> (cloud) {};
# 
#       /** \brief Constructor for base SampleConsensusModelPlane.
#         * \param[in] cloud the input point cloud dataset
#         * \param[in] indices a vector of point indices to be used from \a cloud
#         */
#       SampleConsensusModelPlane (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : SampleConsensusModel<PointT> (cloud, indices) {};
# 
#       /** \brief Check whether the given index samples can form a valid plane model, compute the model coefficients from
#         * these samples and store them internally in model_coefficients_. The plane coefficients are:
#         * a, b, c, d (ax+by+cz+d=0)
#         * \param[in] samples the point indices found as possible good candidates for creating a valid model
#         * \param[out] model_coefficients the resultant model coefficients
#         */
#       bool 
#       computeModelCoefficients (const std::vector<int> &samples, 
#                                 Eigen::VectorXf &model_coefficients);
# 
#       /** \brief Compute all distances from the cloud data to a given plane model.
#         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
#         * \param[out] distances the resultant estimated distances
#         */
#       void 
#       getDistancesToModel (const Eigen::VectorXf &model_coefficients, 
#                            std::vector<double> &distances);
# 
#       /** \brief Select all the points which respect the given model coefficients as inliers.
#         * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         * \param[out] inliers the resultant model inliers
#         */
#       void 
#       selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                             const double threshold, 
#                             std::vector<int> &inliers);
# 
#       /** \brief Count all the points which respect the given model coefficients as inliers. 
#         * 
#         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
#         * \return the resultant number of inliers
#         */
#       virtual int
#       countWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                            const double threshold);
# 
#       /** \brief Recompute the plane coefficients using the given inlier set and return them to the user.
#         * @note: these are the coefficients of the plane model after refinement (eg. after SVD)
#         * \param[in] inliers the data inliers found as supporting the model
#         * \param[in] model_coefficients the initial guess for the model coefficients
#         * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
#         */
#       void 
#       optimizeModelCoefficients (const std::vector<int> &inliers, 
#                                  const Eigen::VectorXf &model_coefficients, 
#                                  Eigen::VectorXf &optimized_coefficients);
# 
#       /** \brief Create a new point cloud with inliers projected onto the plane model.
#         * \param[in] inliers the data inliers that we want to project on the plane model
#         * \param[in] model_coefficients the *normalized* coefficients of a plane model
#         * \param[out] projected_points the resultant projected points
#         * \param[in] copy_data_fields set to true if we need to copy the other data fields
#         */
#       void 
#       projectPoints (const std::vector<int> &inliers, 
#                      const Eigen::VectorXf &model_coefficients, 
#                      PointCloud &projected_points, 
#                      bool copy_data_fields = true);
# 
#       /** \brief Verify whether a subset of indices verifies the given plane model coefficients.
#         * \param[in] indices the data indices that need to be tested against the plane model
#         * \param[in] model_coefficients the plane model coefficients
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         */
#       bool 
#       doSamplesVerifyModel (const std::set<int> &indices, 
#                             const Eigen::VectorXf &model_coefficients, 
#                             const double threshold);
# 
#       /** \brief Return an unique id for this model (SACMODEL_PLANE). */
#       inline pcl::SacModel 
#       getModelType () const { return (SACMODEL_PLANE); }
# 
#     protected:
#       /** \brief Check whether a model is valid given the user constraints.
#         * \param[in] model_coefficients the set of model coefficients
#         */
#       inline bool 
#       isModelValid (const Eigen::VectorXf &model_coefficients)
#       {
#         // Needs a valid model coefficients
#         if (model_coefficients.size () != 4)
#         {
#           PCL_ERROR ("[pcl::SampleConsensusModelPlane::isModelValid] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
#           return (false);
#         }
#         return (true);
#       }
###

# sac_model_registration.h
# namespace pcl
# {
#   /** \brief SampleConsensusModelRegistration defines a model for Point-To-Point registration outlier rejection.
#     * \author Radu Bogdan Rusu
#     * \ingroup sample_consensus
#     */
#   template <typename PointT>
#   class SampleConsensusModelRegistration : public SampleConsensusModel<PointT>
#   {
#     using SampleConsensusModel<PointT>::input_;
#     using SampleConsensusModel<PointT>::indices_;
# 
#     public:
#       typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
#       typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
#       typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
# 
#       typedef boost::shared_ptr<SampleConsensusModelRegistration> Ptr;
# 
#       /** \brief Constructor for base SampleConsensusModelRegistration.
#         * \param[in] cloud the input point cloud dataset
#         */
#       SampleConsensusModelRegistration (const PointCloudConstPtr &cloud) : 
#         SampleConsensusModel<PointT> (cloud),
#         target_ (),
#         indices_tgt_ (),
#         correspondences_ (),
#         sample_dist_thresh_ (0)
#       {
#         // Call our own setInputCloud
#         setInputCloud (cloud);
#       }
# 
#       /** \brief Constructor for base SampleConsensusModelRegistration.
#         * \param[in] cloud the input point cloud dataset
#         * \param[in] indices a vector of point indices to be used from \a cloud
#         */
#       SampleConsensusModelRegistration (const PointCloudConstPtr &cloud,
#                                         const std::vector<int> &indices) :
#         SampleConsensusModel<PointT> (cloud, indices),
#         target_ (),
#         indices_tgt_ (),
#         correspondences_ (),
#         sample_dist_thresh_ (0)
#       {
#         computeOriginalIndexMapping ();
#         computeSampleDistanceThreshold (cloud, indices);
#       }
# 
#       /** \brief Provide a pointer to the input dataset
#         * \param[in] cloud the const boost shared pointer to a PointCloud message
#         */
#       inline virtual void
#       setInputCloud (const PointCloudConstPtr &cloud)
#       {
#         SampleConsensusModel<PointT>::setInputCloud (cloud);
#         computeOriginalIndexMapping ();
#         computeSampleDistanceThreshold (cloud);
#       }
# 
#       /** \brief Set the input point cloud target.
#         * \param target the input point cloud target
#         */
#       inline void
#       setInputTarget (const PointCloudConstPtr &target)
#       {
#         target_ = target;
#         indices_tgt_.reset (new std::vector<int>);
#         // Cache the size and fill the target indices
#         int target_size = static_cast<int> (target->size ());
#         indices_tgt_->resize (target_size);
# 
#         for (int i = 0; i < target_size; ++i)
#           (*indices_tgt_)[i] = i;
#         computeOriginalIndexMapping ();
#       }
# 
#       /** \brief Set the input point cloud target.
#         * \param[in] target the input point cloud target
#         * \param[in] indices_tgt a vector of point indices to be used from \a target
#         */
#       inline void
#       setInputTarget (const PointCloudConstPtr &target, const std::vector<int> &indices_tgt)
#       {
#         target_ = target;
#         indices_tgt_.reset (new std::vector<int> (indices_tgt));
#         computeOriginalIndexMapping ();
#       }
# 
#       /** \brief Compute a 4x4 rigid transformation matrix from the samples given
#         * \param[in] samples the indices found as good candidates for creating a valid model
#         * \param[out] model_coefficients the resultant model coefficients
#         */
#       bool
#       computeModelCoefficients (const std::vector<int> &samples,
#                                 Eigen::VectorXf &model_coefficients);
# 
#       /** \brief Compute all distances from the transformed points to their correspondences
#         * \param[in] model_coefficients the 4x4 transformation matrix
#         * \param[out] distances the resultant estimated distances
#         */
#       void
#       getDistancesToModel (const Eigen::VectorXf &model_coefficients,
#                            std::vector<double> &distances);
# 
#       /** \brief Select all the points which respect the given model coefficients as inliers.
#         * \param[in] model_coefficients the 4x4 transformation matrix
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         * \param[out] inliers the resultant model inliers
#         */
#       void
#       selectWithinDistance (const Eigen::VectorXf &model_coefficients,
#                             const double threshold,
#                             std::vector<int> &inliers);
# 
#       /** \brief Count all the points which respect the given model coefficients as inliers.
#         *
#         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
#         * \return the resultant number of inliers
#         */
#       virtual int
#       countWithinDistance (const Eigen::VectorXf &model_coefficients,
#                            const double threshold);
# 
#       /** \brief Recompute the 4x4 transformation using the given inlier set
#         * \param[in] inliers the data inliers found as supporting the model
#         * \param[in] model_coefficients the initial guess for the optimization
#         * \param[out] optimized_coefficients the resultant recomputed transformation
#         */
#       void
#       optimizeModelCoefficients (const std::vector<int> &inliers,
#                                  const Eigen::VectorXf &model_coefficients,
#                                  Eigen::VectorXf &optimized_coefficients);
# 
#       void
#       projectPoints (const std::vector<int> &,
#                      const Eigen::VectorXf &,
#                      PointCloud &, bool = true)
#       {
#       };
# 
#       bool
#       doSamplesVerifyModel (const std::set<int> &,
#                             const Eigen::VectorXf &,
#                             const double)
#       {
#         return (false);
#       }
# 
#       /** \brief Return an unique id for this model (SACMODEL_REGISTRATION). */
#       inline pcl::SacModel
#       getModelType () const { return (SACMODEL_REGISTRATION); }
# 
#     protected:
#       /** \brief Check whether a model is valid given the user constraints.
#         * \param[in] model_coefficients the set of model coefficients
#         */
#       inline bool
#       isModelValid (const Eigen::VectorXf &model_coefficients)
#       {
#         // Needs a valid model coefficients
#         if (model_coefficients.size () != 16)
#           return (false);
# 
#         return (true);
#       }
# 
#       /** \brief Check if a sample of indices results in a good sample of points
#         * indices.
#         * \param[in] samples the resultant index samples
#         */
#       bool
#       isSampleGood (const std::vector<int> &samples) const;
# 
#       /** \brief Computes an "optimal" sample distance threshold based on the
#         * principal directions of the input cloud.
#         * \param[in] cloud the const boost shared pointer to a PointCloud message
#         */
#       inline void
#       computeSampleDistanceThreshold (const PointCloudConstPtr &cloud)
#       {
#         // Compute the principal directions via PCA
#         Eigen::Vector4f xyz_centroid;
#         Eigen::Matrix3f covariance_matrix = Eigen::Matrix3f::Zero ();
# 
#         computeMeanAndCovarianceMatrix (*cloud, covariance_matrix, xyz_centroid);
# 
#         // Check if the covariance matrix is finite or not.
#         for (int i = 0; i < 3; ++i)
#           for (int j = 0; j < 3; ++j)
#             if (!pcl_isfinite (covariance_matrix.coeffRef (i, j)))
#               PCL_ERROR ("[pcl::SampleConsensusModelRegistration::computeSampleDistanceThreshold] Covariance matrix has NaN values! Is the input cloud finite?\n");
# 
#         Eigen::Vector3f eigen_values;
#         pcl::eigen33 (covariance_matrix, eigen_values);
# 
#         // Compute the distance threshold for sample selection
#         sample_dist_thresh_ = eigen_values.array ().sqrt ().sum () / 3.0;
#         sample_dist_thresh_ *= sample_dist_thresh_;
#         PCL_DEBUG ("[pcl::SampleConsensusModelRegistration::setInputCloud] Estimated a sample selection distance threshold of: %f\n", sample_dist_thresh_);
#       }
# 
#       /** \brief Computes an "optimal" sample distance threshold based on the
#         * principal directions of the input cloud.
#         * \param[in] cloud the const boost shared pointer to a PointCloud message
#         */
#       inline void
#       computeSampleDistanceThreshold (const PointCloudConstPtr &cloud,
#                                       const std::vector<int> &indices)
#       {
#         // Compute the principal directions via PCA
#         Eigen::Vector4f xyz_centroid;
#         Eigen::Matrix3f covariance_matrix;
#         computeMeanAndCovarianceMatrix (*cloud, indices, covariance_matrix, xyz_centroid);
# 
#         // Check if the covariance matrix is finite or not.
#         for (int i = 0; i < 3; ++i)
#           for (int j = 0; j < 3; ++j)
#             if (!pcl_isfinite (covariance_matrix.coeffRef (i, j)))
#               PCL_ERROR ("[pcl::SampleConsensusModelRegistration::computeSampleDistanceThreshold] Covariance matrix has NaN values! Is the input cloud finite?\n");
# 
#         Eigen::Vector3f eigen_values;
#         pcl::eigen33 (covariance_matrix, eigen_values);
# 
#         // Compute the distance threshold for sample selection
#         sample_dist_thresh_ = eigen_values.array ().sqrt ().sum () / 3.0;
#         sample_dist_thresh_ *= sample_dist_thresh_;
#         PCL_DEBUG ("[pcl::SampleConsensusModelRegistration::setInputCloud] Estimated a sample selection distance threshold of: %f\n", sample_dist_thresh_);
#       }
# 
#     public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
###

# sac_model_sphere.h
# namespace pcl
# {
#   /** \brief SampleConsensusModelSphere defines a model for 3D sphere segmentation.
#     * The model coefficients are defined as:
#     *   - \b center.x : the X coordinate of the sphere's center
#     *   - \b center.y : the Y coordinate of the sphere's center
#     *   - \b center.z : the Z coordinate of the sphere's center
#     *   - \b radius   : the sphere's radius
#     *
#     * \author Radu B. Rusu
#     * \ingroup sample_consensus
#     */
#   template <typename PointT>
#   class SampleConsensusModelSphere : public SampleConsensusModel<PointT>
#   {
#     public:
#       using SampleConsensusModel<PointT>::input_;
#       using SampleConsensusModel<PointT>::indices_;
#       using SampleConsensusModel<PointT>::radius_min_;
#       using SampleConsensusModel<PointT>::radius_max_;
# 
#     
#       typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
#       typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
#       typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
# 
#       typedef boost::shared_ptr<SampleConsensusModelSphere> Ptr;
# 
#       /** \brief Constructor for base SampleConsensusModelSphere.
#         * \param[in] cloud the input point cloud dataset
#         */
#       SampleConsensusModelSphere (const PointCloudConstPtr &cloud) : 
#         SampleConsensusModel<PointT> (cloud), tmp_inliers_ ()
#       {}
# 
#       /** \brief Constructor for base SampleConsensusModelSphere.
#         * \param[in] cloud the input point cloud dataset
#         * \param[in] indices a vector of point indices to be used from \a cloud
#         */
#       SampleConsensusModelSphere (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
#         SampleConsensusModel<PointT> (cloud, indices), tmp_inliers_ ()
#       {}
# 
#       /** \brief Copy constructor.
#         * \param[in] source the model to copy into this
#         */
#       SampleConsensusModelSphere (const SampleConsensusModelSphere &source) :
#         SampleConsensusModel<PointT> (), tmp_inliers_ () 
#       {
#         *this = source;
#       }
# 
#       /** \brief Copy constructor.
#         * \param[in] source the model to copy into this
#         */
#       inline SampleConsensusModelSphere&
#       operator = (const SampleConsensusModelSphere &source)
#       {
#         SampleConsensusModel<PointT>::operator=(source);
#         tmp_inliers_ = source.tmp_inliers_;
#         return (*this);
#       }
# 
#       /** \brief Check whether the given index samples can form a valid sphere model, compute the model 
#         * coefficients from these samples and store them internally in model_coefficients. 
#         * The sphere coefficients are: x, y, z, R.
#         * \param[in] samples the point indices found as possible good candidates for creating a valid model
#         * \param[out] model_coefficients the resultant model coefficients
#         */
#       bool 
#       computeModelCoefficients (const std::vector<int> &samples, 
#                                 Eigen::VectorXf &model_coefficients);
# 
#       /** \brief Compute all distances from the cloud data to a given sphere model.
#         * \param[in] model_coefficients the coefficients of a sphere model that we need to compute distances to
#         * \param[out] distances the resultant estimated distances
#         */
#       void 
#       getDistancesToModel (const Eigen::VectorXf &model_coefficients, 
#                            std::vector<double> &distances);
# 
#       /** \brief Select all the points which respect the given model coefficients as inliers.
#         * \param[in] model_coefficients the coefficients of a sphere model that we need to compute distances to
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         * \param[out] inliers the resultant model inliers
#         */
#       void 
#       selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                             const double threshold, 
#                             std::vector<int> &inliers);
# 
#       /** \brief Count all the points which respect the given model coefficients as inliers. 
#         * 
#         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
#         * \return the resultant number of inliers
#         */
#       virtual int
#       countWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                            const double threshold);
# 
#       /** \brief Recompute the sphere coefficients using the given inlier set and return them to the user.
#         * @note: these are the coefficients of the sphere model after refinement (eg. after SVD)
#         * \param[in] inliers the data inliers found as supporting the model
#         * \param[in] model_coefficients the initial guess for the optimization
#         * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
#         */
#       void 
#       optimizeModelCoefficients (const std::vector<int> &inliers, 
#                                  const Eigen::VectorXf &model_coefficients, 
#                                  Eigen::VectorXf &optimized_coefficients);
# 
#       /** \brief Create a new point cloud with inliers projected onto the sphere model.
#         * \param[in] inliers the data inliers that we want to project on the sphere model
#         * \param[in] model_coefficients the coefficients of a sphere model
#         * \param[out] projected_points the resultant projected points
#         * \param[in] copy_data_fields set to true if we need to copy the other data fields
#         * \todo implement this.
#         */
#       void 
#       projectPoints (const std::vector<int> &inliers, 
#                      const Eigen::VectorXf &model_coefficients, 
#                      PointCloud &projected_points, 
#                      bool copy_data_fields = true);
# 
#       /** \brief Verify whether a subset of indices verifies the given sphere model coefficients.
#         * \param[in] indices the data indices that need to be tested against the sphere model
#         * \param[in] model_coefficients the sphere model coefficients
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         */
#       bool 
#       doSamplesVerifyModel (const std::set<int> &indices, 
#                             const Eigen::VectorXf &model_coefficients, 
#                             const double threshold);
# 
#       /** \brief Return an unique id for this model (SACMODEL_SPHERE). */
#       inline pcl::SacModel getModelType () const { return (SACMODEL_SPHERE); }
# 
#     protected:
#       /** \brief Check whether a model is valid given the user constraints.
#         * \param[in] model_coefficients the set of model coefficients
#         */
#       inline bool 
#       isModelValid (const Eigen::VectorXf &model_coefficients)
#       {
#         // Needs a valid model coefficients
#         if (model_coefficients.size () != 4)
#         {
#           PCL_ERROR ("[pcl::SampleConsensusModelSphere::isModelValid] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
#           return (false);
#         }
# 
#         if (radius_min_ != -std::numeric_limits<double>::max() && model_coefficients[3] < radius_min_)
#           return (false);
#         if (radius_max_ != std::numeric_limits<double>::max() && model_coefficients[3] > radius_max_)
#           return (false);
# 
#         return (true);
#       }
# 
#       /** \brief Check if a sample of indices results in a good sample of points
#         * indices.
#         * \param[in] samples the resultant index samples
#         */
#       bool
#       isSampleGood(const std::vector<int> &samples) const;
# 
#     private:
#       /** \brief Temporary pointer to a list of given indices for optimizeModelCoefficients () */
#       const std::vector<int> *tmp_inliers_;
# 
#       struct OptimizationFunctor : pcl::Functor<float>
#       {
#         /** Functor constructor
#           * \param[in] m_data_points the number of data points to evaluate
#           * \param[in] estimator pointer to the estimator object
#           * \param[in] distance distance computation function pointer
#           */
#         OptimizationFunctor (int m_data_points, pcl::SampleConsensusModelSphere<PointT> *model) : 
#           pcl::Functor<float>(m_data_points), model_ (model) {}
# 
#         /** Cost function to be minimized
#           * \param[in] x the variables array
#           * \param[out] fvec the resultant functions evaluations
#           * \return 0
#           */
#         int 
#         operator() (const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
#         {
#           Eigen::Vector4f cen_t;
#           cen_t[3] = 0;
#           for (int i = 0; i < values (); ++i)
#           {
#             // Compute the difference between the center of the sphere and the datapoint X_i
#             cen_t[0] = model_->input_->points[(*model_->tmp_inliers_)[i]].x - x[0];
#             cen_t[1] = model_->input_->points[(*model_->tmp_inliers_)[i]].y - x[1];
#             cen_t[2] = model_->input_->points[(*model_->tmp_inliers_)[i]].z - x[2];
#             
#             // g = sqrt ((x-a)^2 + (y-b)^2 + (z-c)^2) - R
#             fvec[i] = sqrtf (cen_t.dot (cen_t)) - x[3];
#           }
#           return (0);
#         }
#         
#         pcl::SampleConsensusModelSphere<PointT> *model_;
#       };
#if defined BUILD_Maintainer && defined __GNUC__ && __GNUC__ == 4 && __GNUC_MINOR__ > 3
#pragma GCC diagnostic warning "-Weffc++"
#endif
###

# sac_model_stick.h
# namespace pcl
# {
#   /** \brief SampleConsensusModelStick defines a model for 3D stick segmentation. 
#     * A stick is a line with an user given minimum/maximum width.
#     * The model coefficients are defined as:
#     *   - \b point_on_line.x  : the X coordinate of a point on the line
#     *   - \b point_on_line.y  : the Y coordinate of a point on the line
#     *   - \b point_on_line.z  : the Z coordinate of a point on the line
#     *   - \b line_direction.x : the X coordinate of a line's direction
#     *   - \b line_direction.y : the Y coordinate of a line's direction
#     *   - \b line_direction.z : the Z coordinate of a line's direction
#     *   - \b line_width       : the width of the line
#     * \author Radu B. Rusu
#     * \ingroup sample_consensus
#     */
#   template <typename PointT>
#   class SampleConsensusModelStick : public SampleConsensusModel<PointT>
#   {
#     using SampleConsensusModel<PointT>::input_;
#     using SampleConsensusModel<PointT>::indices_;
#     using SampleConsensusModel<PointT>::radius_min_;
#     using SampleConsensusModel<PointT>::radius_max_;
# 
#     public:
#       typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
#       typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
#       typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
# 
#       typedef boost::shared_ptr<SampleConsensusModelStick> Ptr;
# 
#       /** \brief Constructor for base SampleConsensusModelStick.
#         * \param[in] cloud the input point cloud dataset
#         */
#       SampleConsensusModelStick (const PointCloudConstPtr &cloud) : SampleConsensusModel<PointT> (cloud) {};
# 
#       /** \brief Constructor for base SampleConsensusModelStick.
#         * \param[in] cloud the input point cloud dataset
#         * \param[in] indices a vector of point indices to be used from \a cloud
#         */
#       SampleConsensusModelStick (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : SampleConsensusModel<PointT> (cloud, indices) {};
# 
#       /** \brief Check whether the given index samples can form a valid stick model, compute the model coefficients from
#         * these samples and store them internally in model_coefficients_. The stick coefficients are represented by a
#         * point and a line direction
#         * \param[in] samples the point indices found as possible good candidates for creating a valid model
#         * \param[out] model_coefficients the resultant model coefficients
#         */
#       bool 
#       computeModelCoefficients (const std::vector<int> &samples, 
#                                 Eigen::VectorXf &model_coefficients);
# 
#       /** \brief Compute all squared distances from the cloud data to a given stick model.
#         * \param[in] model_coefficients the coefficients of a stick model that we need to compute distances to
#         * \param[out] distances the resultant estimated squared distances
#         */
#       void 
#       getDistancesToModel (const Eigen::VectorXf &model_coefficients, 
#                            std::vector<double> &distances);
# 
#       /** \brief Select all the points which respect the given model coefficients as inliers.
#         * \param[in] model_coefficients the coefficients of a stick model that we need to compute distances to
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         * \param[out] inliers the resultant model inliers
#         */
#       void 
#       selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                             const double threshold, 
#                             std::vector<int> &inliers);
# 
#       /** \brief Count all the points which respect the given model coefficients as inliers. 
#         * 
#         * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
#         * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
#         * \return the resultant number of inliers
#         */
#       virtual int
#       countWithinDistance (const Eigen::VectorXf &model_coefficients, 
#                            const double threshold);
# 
#       /** \brief Recompute the stick coefficients using the given inlier set and return them to the user.
#         * @note: these are the coefficients of the stick model after refinement (eg. after SVD)
#         * \param[in] inliers the data inliers found as supporting the model
#         * \param[in] model_coefficients the initial guess for the model coefficients
#         * \param[out] optimized_coefficients the resultant recomputed coefficients after optimization
#         */
#       void 
#       optimizeModelCoefficients (const std::vector<int> &inliers, 
#                                  const Eigen::VectorXf &model_coefficients, 
#                                  Eigen::VectorXf &optimized_coefficients);
# 
#       /** \brief Create a new point cloud with inliers projected onto the stick model.
#         * \param[in] inliers the data inliers that we want to project on the stick model
#         * \param[in] model_coefficients the *normalized* coefficients of a stick model
#         * \param[out] projected_points the resultant projected points
#         * \param[in] copy_data_fields set to true if we need to copy the other data fields
#         */
#       void 
#       projectPoints (const std::vector<int> &inliers, 
#                      const Eigen::VectorXf &model_coefficients, 
#                      PointCloud &projected_points, 
#                      bool copy_data_fields = true);
# 
#       /** \brief Verify whether a subset of indices verifies the given stick model coefficients.
#         * \param[in] indices the data indices that need to be tested against the plane model
#         * \param[in] model_coefficients the plane model coefficients
#         * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
#         */
#       bool 
#       doSamplesVerifyModel (const std::set<int> &indices, 
#                             const Eigen::VectorXf &model_coefficients, 
#                             const double threshold);
# 
#       /** \brief Return an unique id for this model (SACMODEL_STACK). */
#       inline pcl::SacModel 
#       getModelType () const { return (SACMODEL_STICK); }
# 
#     protected:
#       /** \brief Check whether a model is valid given the user constraints.
#         * \param[in] model_coefficients the set of model coefficients
#         */
#       inline bool 
#       isModelValid (const Eigen::VectorXf &model_coefficients)
#       {
#         if (model_coefficients.size () != 7)
#         {
#           PCL_ERROR ("[pcl::SampleConsensusModelStick::selectWithinDistance] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
#           return (false);
#         }
# 
#         return (true);
#       }
# 
#       /** \brief Check if a sample of indices results in a good sample of points
#         * indices.
#         * \param[in] samples the resultant index samples
#         */
#       bool
#       isSampleGood (const std::vector<int> &samples) const;
#   };
# 
###

###############################################################################
# Enum
###############################################################################

# method_types.h
cdef extern from "pcl/sample_consensus/method_types.h" namespace "pcl":
    cdef enum:
        SAC_RANSAC = 0
        SAC_LMEDS = 1
        SAC_MSAC = 2
        SAC_RRANSAC = 3
        SAC_RMSAC = 4
        SAC_MLESAC = 5
        SAC_PROSAC = 6
###

# model_types.h
cdef extern from "pcl/sample_consensus/model_types.h" namespace "pcl":
    cdef enum SacModel:
        SACMODEL_PLANE
        SACMODEL_LINE
        SACMODEL_CIRCLE2D
        SACMODEL_CIRCLE3D
        SACMODEL_SPHERE
        SACMODEL_CYLINDER
        SACMODEL_CONE
        SACMODEL_TORUS
        SACMODEL_PARALLEL_LINE
        SACMODEL_PERPENDICULAR_PLANE
        SACMODEL_PARALLEL_LINES
        SACMODEL_NORMAL_PLANE
        SACMODEL_NORMAL_SPHERE        # Version 1.6
        SACMODEL_REGISTRATION
        SACMODEL_PARALLEL_PLANE
        SACMODEL_NORMAL_PARALLEL_PLANE
        SACMODEL_STICK

# // Define the number of samples in SacModel needs
# typedef std::map<pcl::SacModel, unsigned int>::value_type SampleSizeModel;
# const static SampleSizeModel sample_size_pairs[] = {SampleSizeModel (pcl::SACMODEL_PLANE, 3),
#                                                     SampleSizeModel (pcl::SACMODEL_LINE, 2),
#                                                     SampleSizeModel (pcl::SACMODEL_CIRCLE2D, 3),
#                                                     //SampleSizeModel (pcl::SACMODEL_CIRCLE3D, 3),
#                                                     SampleSizeModel (pcl::SACMODEL_SPHERE, 4),
#                                                     SampleSizeModel (pcl::SACMODEL_CYLINDER, 2),
#                                                     SampleSizeModel (pcl::SACMODEL_CONE, 3),
#                                                     //SampleSizeModel (pcl::SACMODEL_TORUS, 2),
#                                                     SampleSizeModel (pcl::SACMODEL_PARALLEL_LINE, 2),
#                                                     SampleSizeModel (pcl::SACMODEL_PERPENDICULAR_PLANE, 3),
#                                                     //SampleSizeModel (pcl::PARALLEL_LINES, 2),
#                                                     SampleSizeModel (pcl::SACMODEL_NORMAL_PLANE, 3),
#                                                     SampleSizeModel (pcl::SACMODEL_NORMAL_SPHERE, 4),
#                                                     SampleSizeModel (pcl::SACMODEL_REGISTRATION, 3),
#                                                     SampleSizeModel (pcl::SACMODEL_PARALLEL_PLANE, 3),
#                                                     SampleSizeModel (pcl::SACMODEL_NORMAL_PARALLEL_PLANE, 3),
#                                                     SampleSizeModel (pcl::SACMODEL_STICK, 2)};
# 
# namespace pcl
# {
#   const static std::map<pcl::SacModel, unsigned int> SAC_SAMPLE_SIZE (sample_size_pairs, sample_size_pairs
#       + sizeof (sample_size_pairs) / sizeof (SampleSizeModel));
# }
###

###############################################################################
# Activation
###############################################################################
