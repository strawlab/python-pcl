# -*- coding: utf-8 -*-

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# import
cimport pcl_defs as cpp
from boost_shared_ptr cimport shared_ptr

###############################################################################
# Types
###############################################################################

### base class ###

# sac_model.h
# namespace pcl
# template<class T> class ProgressiveSampleConsensus;

# sac_model.h
# namespace pcl
# template <typename PointT>
# class SampleConsensusModel
cdef extern from "pcl/sample_consensus/sac_model.h" namespace "pcl":
    cdef cppclass SampleConsensusModel[T]:
        SampleConsensusModel()
        # SampleConsensusModel (bool random = false) 
        # SampleConsensusModel (const PointCloudConstPtr &cloud, bool random = false)
        # SampleConsensusModel (const PointCloudConstPtr &cloud, const std::vector<int> &indices, bool random = false)
        # public:
        # typedef typename pcl::PointCloud<PointT> PointCloud;
        # typedef typename pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
        # typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
        # typedef typename pcl::search::Search<PointT>::Ptr SearchPtr;
        # typedef boost::shared_ptr<SampleConsensusModel> Ptr;
        # typedef boost::shared_ptr<const SampleConsensusModel> ConstPtr;
        # public:
        # /** \brief Get a set of random data samples and return them as point
        # * indices. Pure virtual.  
        # * \param[out] iterations the internal number of iterations used by SAC methods
        # * \param[out] samples the resultant model samples
        # */
        # void getSamples (int &iterations, std::vector<int> &samples)
        void getSamples (int &iterations, vector[int] &samples)
        
        # /** \brief Check whether the given index samples can form a valid model,
        # * compute the model coefficients from these samples and store them
        # * in model_coefficients. Pure virtual.
        # * \param[in] samples the point indices found as possible good candidates
        # * for creating a valid model 
        # * \param[out] model_coefficients the computed model coefficients
        # */
        # virtual bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients) = 0;
        
        # /** \brief Recompute the model coefficients using the given inlier set
        # * and return them to the user. Pure virtual.
        # * @note: these are the coefficients of the model after refinement
        # * (e.g., after a least-squares optimization)
        # * \param[in] inliers the data inliers supporting the model
        # * \param[in] model_coefficients the initial guess for the model coefficients
        # * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
        # */
        # virtual void optimizeModelCoefficients (const std::vector<int> &inliers,  const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients) = 0;
        
        # /** \brief Compute all distances from the cloud data to a given model. Pure virtual.
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to 
        # * \param[out] distances the resultant estimated distances
        # virtual void  getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances) = 0;
        # /** \brief Select all the points which respect the given model
        # * coefficients as inliers. Pure virtual.
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from 
        # * the outliers
        # * \param[out] inliers the resultant model inliers
        # virtual void selectWithinDistance (const Eigen::VectorXf &model_coefficients,  const double threshold, std::vector<int> &inliers) = 0;
        
        # /** \brief Count all the points which respect the given model
        # * coefficients as inliers. Pure virtual.
        # * \param[in] model_coefficients the coefficients of a model that we need to
        # * compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for
        # * determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold) = 0;
        
        # /** \brief Create a new point cloud with inliers projected onto the model. Pure virtual.
        # * \param[in] inliers the data inliers that we want to project on the model
        # * \param[in] model_coefficients the coefficients of a model
        # * \param[out] projected_points the resultant projected points
        # * \param[in] copy_data_fields set to true (default) if we want the \a
        # * projected_points cloud to be an exact copy of the input dataset minus
        # * the point projections on the plane model
        # virtual void projectPoints (const std::vector<int> &inliers, 
        #              const Eigen::VectorXf &model_coefficients,
        #              PointCloud &projected_points, 
        #              bool copy_data_fields = true) = 0;
        
        # /** \brief Verify whether a subset of indices verifies a given set of
        # * model coefficients. Pure virtual.
        # * \param[in] indices the data indices that need to be tested against the model
        # * \param[in] model_coefficients the set of model coefficients
        # * \param[in] threshold a maximum admissible distance threshold for
        # * determining the inliers from the outliers
        # virtual bool doSamplesVerifyModel (const std::set<int> &indices, 
        #                     const Eigen::VectorXf &model_coefficients, 
        #                     const double threshold) = 0;
        
        # /** \brief Provide a pointer to the input dataset
        # * \param[in] cloud the const boost shared pointer to a PointCloud message
        # inline virtual void setInputCloud (const PointCloudConstPtr &cloud)
        
        # /** \brief Get a pointer to the input point cloud dataset. */
        # inline PointCloudConstPtr getInputCloud () const
        
        # /** \brief Provide a pointer to the vector of indices that represents the input data.
        # * \param[in] indices a pointer to the vector of indices that represents the input data.
        # inline void setIndices (const boost::shared_ptr <std::vector<int> > &indices) 
        
        # /** \brief Provide the vector of indices that represents the input data.
        # * \param[out] indices the vector of indices that represents the input data.
        # inline void setIndices (const std::vector<int> &indices) 
        
        # /** \brief Get a pointer to the vector of indices used. */
        # inline boost::shared_ptr <std::vector<int> > getIndices () const
        
        # /** \brief Return an unique id for each type of model employed. */
        # virtual SacModel getModelType () const = 0;
        
        # /** \brief Return the size of a sample from which a model is computed */
        # inline unsigned int getSampleSize () const 
        
        # /** \brief Set the minimum and maximum allowable radius limits for the
        # * model (applicable to models that estimate a radius)
        # * \param[in] min_radius the minimum radius model
        # * \param[in] max_radius the maximum radius model
        # * \todo change this to set limits on the entire model
        # inline void setRadiusLimits (const double &min_radius, const double &max_radius)
        
        # /** \brief Get the minimum and maximum allowable radius limits for the
        # * model as set by the user.
        # * \param[out] min_radius the resultant minimum radius model
        # * \param[out] max_radius the resultant maximum radius model
        # inline void getRadiusLimits (double &min_radius, double &max_radius)
        
        # /** \brief Set the maximum distance allowed when drawing random samples
        # * \param[in] radius the maximum distance (L2 norm)
        # inline void setSamplesMaxDist (const double &radius, SearchPtr search)
        
        # /** \brief Get maximum distance allowed when drawing random samples
        # * \param[out] radius the maximum distance (L2 norm)
        # inline void getSamplesMaxDist (double &radius)


ctypedef SampleConsensusModel[cpp.PointXYZ] SampleConsensusModel_t
ctypedef SampleConsensusModel[cpp.PointXYZI] SampleConsensusModel_PointXYZI_t
ctypedef SampleConsensusModel[cpp.PointXYZRGB] SampleConsensusModel_PointXYZRGB_t
ctypedef SampleConsensusModel[cpp.PointXYZRGBA] SampleConsensusModel_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModel[cpp.PointXYZ]] SampleConsensusModelPtr_t
ctypedef shared_ptr[SampleConsensusModel[cpp.PointXYZI]] SampleConsensusModel_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModel[cpp.PointXYZRGB]] SampleConsensusModel_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModel[cpp.PointXYZRGBA]] SampleConsensusModel_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModel[cpp.PointXYZ]] SampleConsensusModelConstPtr_t
ctypedef shared_ptr[const SampleConsensusModel[cpp.PointXYZI]] SampleConsensusModel_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModel[cpp.PointXYZRGB]] SampleConsensusModel_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModel[cpp.PointXYZRGBA]] SampleConsensusModel_PointXYZRGBA_ConstPtr_t
###

# sac_model.h
# template <typename PointT, typename PointNT>
# class SampleConsensusModelFromNormals
cdef extern from "pcl/sample_consensus/sac_model.h" namespace "pcl":
    cdef cppclass SampleConsensusModelFromNormals[T, NT]:
        SampleConsensusModelFromNormals ()
        # public:
        # typedef typename pcl::PointCloud<PointNT>::ConstPtr PointCloudNConstPtr;
        # typedef typename pcl::PointCloud<PointNT>::Ptr PointCloudNPtr;
        # typedef boost::shared_ptr<SampleConsensusModelFromNormals> Ptr;
        # typedef boost::shared_ptr<const SampleConsensusModelFromNormals> ConstPtr;
        # /** \brief Set the normal angular distance weight.
        # * \param[in] w the relative weight (between 0 and 1) to give to the angular
        # * distance (0 to pi/2) between point normals and the plane normal.
        # * (The Euclidean distance will have weight 1-w.)
        # */
        # inline void setNormalDistanceWeight (const double w) 
        void setNormalDistanceWeight (const double w) 
        
        # /** \brief Get the normal angular distance weight. */
        # inline double getNormalDistanceWeight ()
        double getNormalDistanceWeight ()
        
        # /** \brief Provide a pointer to the input dataset that contains the point
        # * normals of the XYZ dataset.
        # * \param[in] normals the const boost shared pointer to a PointCloud message
        # inline void setInputNormals (const PointCloudNConstPtr &normals) 
        void setInputNormals (shared_ptr[cpp.PointCloud[NT]] normals) 
        
        # /** \brief Get a pointer to the normals of the input XYZ point cloud dataset. */
        # inline PointCloudNConstPtr getInputNormals ()
        shared_ptr[cpp.PointCloud[NT]] getInputNormals () 


# ctypedef SampleConsensusModelFromNormals[cpp.PointXYZ, cpp.Normal] SampleConsensusModelFromNormals_t
# ctypedef SampleConsensusModelFromNormals[cpp.PointXYZI, cpp.Normal] SampleConsensusModelFromNormals_PointXYZI_t
# ctypedef SampleConsensusModelFromNormals[cpp.PointXYZRGB, cpp.Normal] SampleConsensusModelFromNormals_PointXYZRGB_t
# ctypedef SampleConsensusModelFromNormals[cpp.PointXYZRGBA, cpp.Normal] SampleConsensusModelFromNormals_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelFromNormals[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelFromNormalsPtr_t
ctypedef shared_ptr[SampleConsensusModelFromNormals[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelFromNormals_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelFromNormals[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelFromNormals_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelFromNormals[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelFromNormals_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelFromNormals[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelFromNormalsConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelFromNormals[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelFromNormals_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelFromNormals[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelFromNormals_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelFromNormals[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelFromNormals_PointXYZRGBA_ConstPtr_t
###

# sac.h
# namespace pcl
# template <typename T>
# class SampleConsensus
cdef extern from "pcl/sample_consensus/sac.h" namespace "pcl":
    cdef cppclass SampleConsensus[T]:
        # SampleConsensus (const SampleConsensusModelPtr &model, bool random = false)
        # SampleConsensus (const SampleConsensusModelPtr &model, double threshold, bool random = false) : 
        # \brief Constructor for base SAC.
        # \param[in] model a Sample Consensus model
        # \param[in] random if true set the random seed to the current time, else set to 12345 (default: false)
        SampleConsensus (const SampleConsensusModelPtr_t &model)
        SampleConsensus (const SampleConsensusModel_PointXYZI_Ptr_t &model)
        SampleConsensus (const SampleConsensusModel_PointXYZRGB_Ptr_t &model)
        SampleConsensus (const SampleConsensusModel_PointXYZRGBA_Ptr_t &model)
        
        # public:
        # typedef boost::shared_ptr<SampleConsensus> Ptr;
        # typedef boost::shared_ptr<const SampleConsensus> ConstPtr;
        # \brief Set the distance to model threshold.
        # \param[in] threshold distance to model threshold
        # inline void setDistanceThreshold (double threshold)
        void setDistanceThreshold (double threshold)
        
        # /** \brief Get the distance to model threshold, as set by the user. */
        # inline double getDistanceThreshold ()
        double getDistanceThreshold ()
        
        # /** \brief Set the maximum number of iterations.
        # * \param[in] max_iterations maximum number of iterations
        # inline void setMaxIterations (int max_iterations)
        void setMaxIterations (int max_iterations)
        
        # /** \brief Get the maximum number of iterations, as set by the user. */
        # inline int getMaxIterations ()
        int getMaxIterations ()
        
        # /** \brief Set the desired probability of choosing at least one sample free from outliers.
        # * \param[in] probability the desired probability of choosing at least one sample free from outliers
        # * \note internally, the probability is set to 99% (0.99) by default.
        # inline void setProbability (double probability)
        void setProbability (double probability)
        
        # /** \brief Obtain the probability of choosing at least one sample free from outliers, as set by the user. */
        # inline double getProbability ()
        double getProbability ()
        
        # /** \brief Compute the actual model. Pure virtual. */
        # virtual bool computeModel (int debug_verbosity_level = 0) = 0;
        
        # /** \brief Get a set of randomly selected indices.
        # * \param[in] indices the input indices vector
        # * \param[in] nr_samples the desired number of point indices to randomly select
        # * \param[out] indices_subset the resultant output set of randomly selected indices
        # inline void getRandomSamples (const boost::shared_ptr <std::vector<int> > &indices,  size_t nr_samples, std::set<int> &indices_subset)
        # void getRandomSamples (shared_ptr [vector[int]] &indices,  size_t nr_samples, set[int] &indices_subset)
        
        # /** \brief Return the best model found so far. 
        # * \param[out] model the resultant model
        # inline void getModel (std::vector<int> &model)
        void getModel (vector[int] &model)
        
        # /** \brief Return the best set of inliers found so far for this model. 
        # * \param[out] inliers the resultant set of inliers
        # inline void getInliers (std::vector<int> &inliers)
        void getInliers (vector[int] &inliers)
        
        # /** \brief Return the model coefficients of the best model found so far. 
        # * \param[out] model_coefficients the resultant model coefficients
        # inline void  getModelCoefficients (Eigen::VectorXf &model_coefficients)


ctypedef SampleConsensus[cpp.PointXYZ] SampleConsensus_t
ctypedef SampleConsensus[cpp.PointXYZI] SampleConsensus_PointXYZI_t
ctypedef SampleConsensus[cpp.PointXYZRGB] SampleConsensus_PointXYZRGB_t
ctypedef SampleConsensus[cpp.PointXYZRGBA] SampleConsensus_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensus[cpp.PointXYZ]] SampleConsensusPtr_t
ctypedef shared_ptr[SampleConsensus[cpp.PointXYZI]] SampleConsensus_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensus[cpp.PointXYZRGB]] SampleConsensus_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensus[cpp.PointXYZRGBA]] SampleConsensus_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensus[cpp.PointXYZ]] SampleConsensusConstPtr_t
ctypedef shared_ptr[const SampleConsensus[cpp.PointXYZI]] SampleConsensus_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensus[cpp.PointXYZRGB]] SampleConsensus_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensus[cpp.PointXYZRGBA]] SampleConsensus_PointXYZRGBA_ConstPtr_t
###


# template<typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
# struct Functor
cdef extern from "pcl/sample_consensus/rransac.h" namespace "pcl":
    cdef cppclass Functor[_Scalar]:
        Functor ()
        # Functor (int m_data_points)
        # typedef _Scalar Scalar;
        # enum 
        # {
        #   InputsAtCompileTime = NX,
        #   ValuesAtCompileTime = NY
        # };
        # typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
        # typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
        # typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
        # /** \brief Get the number of values. */ 
        # int values () const


###

# sac_model_plane.h
# namespace pcl
# /** \brief Project a point on a planar model given by a set of normalized coefficients
#   * \param[in] p the input point to project
#   * \param[in] model_coefficients the coefficients of the plane (a, b, c, d) that satisfy ax+by+cz+d=0
#   * \param[out] q the resultant projected point
#   */
# template <typename Point> inline void
# projectPoint (const Point &p, const Eigen::Vector4f &model_coefficients, Point &q)
# {
#   // Calculate the distance from the point to the plane
#   Eigen::Vector4f pp (p.x, p.y, p.z, 1);
#   // use normalized coefficients to calculate the scalar projection 
#   float distance_to_plane = pp.dot(model_coefficients);
#  
#   //TODO: Why doesn't getVector4Map work here?
#   //Eigen::Vector4f q_e = q.getVector4fMap ();
#   //q_e = pp - model_coefficients * distance_to_plane;
#   
#   Eigen::Vector4f q_e = pp - distance_to_plane * model_coefficients;
#   q.x = q_e[0];
#   q.y = q_e[1];
#   q.z = q_e[2];
# }
# 
# sac_model_plane.h
# namespace pcl
# /** \brief Get the distance from a point to a plane (signed) defined by ax+by+cz+d=0
#   * \param p a point
#   * \param a the normalized <i>a</i> coefficient of a plane
#   * \param b the normalized <i>b</i> coefficient of a plane
#   * \param c the normalized <i>c</i> coefficient of a plane
#   * \param d the normalized <i>d</i> coefficient of a plane
#   * \ingroup sample_consensus
#   */
# template <typename Point> inline double
# pointToPlaneDistanceSigned (const Point &p, double a, double b, double c, double d)
# 
# sac_model_plane.h
# namespace pcl
# /** \brief Get the distance from a point to a plane (signed) defined by ax+by+cz+d=0
#   * \param p a point
#   * \param plane_coefficients the normalized coefficients (a, b, c, d) of a plane
#   * \ingroup sample_consensus
#   */
# template <typename Point> inline double
# pointToPlaneDistanceSigned (const Point &p, const Eigen::Vector4f &plane_coefficients)
# 
# sac_model_plane.h
# namespace pcl
# /** \brief Get the distance from a point to a plane (unsigned) defined by ax+by+cz+d=0
#   * \param p a point
#   * \param a the normalized <i>a</i> coefficient of a plane
#   * \param b the normalized <i>b</i> coefficient of a plane
#   * \param c the normalized <i>c</i> coefficient of a plane
#   * \param d the normalized <i>d</i> coefficient of a plane
#   * \ingroup sample_consensus
#   */
# template <typename Point> inline double
# pointToPlaneDistance (const Point &p, double a, double b, double c, double d)
# 
# sac_model_plane.h
# namespace pcl
# /** \brief Get the distance from a point to a plane (unsigned) defined by ax+by+cz+d=0
#   * \param p a point
#   * \param plane_coefficients the normalized coefficients (a, b, c, d) of a plane
#   * \ingroup sample_consensus
#   */
# template <typename Point> inline double
# pointToPlaneDistance (const Point &p, const Eigen::Vector4f &plane_coefficients)
###

# sac_model_plane.h
# namespace pcl
# /** \brief SampleConsensusModelPlane defines a model for 3D plane segmentation.
#   * The model coefficients are defined as:
#   *   - \b a : the X coordinate of the plane's normal (normalized)
#   *   - \b b : the Y coordinate of the plane's normal (normalized)
#   *   - \b c : the Z coordinate of the plane's normal (normalized)
#   *   - \b d : the fourth <a href="http://mathworld.wolfram.com/HessianNormalForm.html">Hessian component</a> of the plane's equation
#   * \author Radu B. Rusu
#   * \ingroup sample_consensus
#   */
# template <typename PointT>
# class SampleConsensusModelPlane : public SampleConsensusModel<PointT>
cdef extern from "pcl/sample_consensus/sac_model_plane.h" namespace "pcl":
    cdef cppclass SampleConsensusModelPlane[PointT](SampleConsensusModel[PointT]):
        SampleConsensusModelPlane()
        SampleConsensusModelPlane(shared_ptr[cpp.PointCloud[PointT]] cloud)
        # public:
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelPlane> Ptr;
        # 
        # /** \brief Constructor for base SampleConsensusModelPlane.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelPlane (const PointCloudConstPtr &cloud) : SampleConsensusModel<PointT> (cloud) {};
        # 
        # /** \brief Constructor for base SampleConsensusModelPlane.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelPlane (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : SampleConsensusModel<PointT> (cloud, indices) {};
        
        # /** \brief Check whether the given index samples can form a valid plane model, compute the model coefficients from
        # * these samples and store them internally in model_coefficients_. The plane coefficients are:
        # * a, b, c, d (ax+by+cz+d=0)
        # * \param[in] samples the point indices found as possible good candidates for creating a valid model
        # * \param[out] model_coefficients the resultant model coefficients
        # */
        # bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);
        # 
        # /** \brief Compute all distances from the cloud data to a given plane model.
        # * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
        # * \param[out] distances the resultant estimated distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers. 
        # * 
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Recompute the plane coefficients using the given inlier set and return them to the user.
        # * @note: these are the coefficients of the plane model after refinement (eg. after SVD)
        # * \param[in] inliers the data inliers found as supporting the model
        # * \param[in] model_coefficients the initial guess for the model coefficients
        # * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
        # */
        # void optimizeModelCoefficients (const std::vector<int> &inliers, 
        #                          const Eigen::VectorXf &model_coefficients, 
        #                          Eigen::VectorXf &optimized_coefficients);
        # 
        # /** \brief Create a new point cloud with inliers projected onto the plane model.
        # * \param[in] inliers the data inliers that we want to project on the plane model
        # * \param[in] model_coefficients the *normalized* coefficients of a plane model
        # * \param[out] projected_points the resultant projected points
        # * \param[in] copy_data_fields set to true if we need to copy the other data fields
        # */
        # void projectPoints (const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, PointCloud &projected_points, bool copy_data_fields = true);
        # 
        # /** \brief Verify whether a subset of indices verifies the given plane model coefficients.
        # * \param[in] indices the data indices that need to be tested against the plane model
        # * \param[in] model_coefficients the plane model coefficients
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # */
        # bool doSamplesVerifyModel (const std::set<int> &indices, 
        #                     const Eigen::VectorXf &model_coefficients, 
        #                     const double threshold);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_PLANE). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_PLANE); }


ctypedef SampleConsensusModelPlane[cpp.PointXYZ] SampleConsensusModelPlane_t
ctypedef SampleConsensusModelPlane[cpp.PointXYZI] SampleConsensusModelPlane_PointXYZI_t
ctypedef SampleConsensusModelPlane[cpp.PointXYZRGB] SampleConsensusModelPlane_PointXYZRGB_t
ctypedef SampleConsensusModelPlane[cpp.PointXYZRGBA] SampleConsensusModelPlane_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelPlane[cpp.PointXYZ]] SampleConsensusModelPlanePtr_t
ctypedef shared_ptr[SampleConsensusModelPlane[cpp.PointXYZI]] SampleConsensusModelPlane_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelPlane[cpp.PointXYZRGB]] SampleConsensusModelPlane_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelPlane[cpp.PointXYZRGBA]] SampleConsensusModelPlane_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelPlane[cpp.PointXYZ]] SampleConsensusModelPlaneConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelPlane[cpp.PointXYZI]] SampleConsensusModelPlane_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelPlane[cpp.PointXYZRGB]] SampleConsensusModelPlane_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelPlane[cpp.PointXYZRGBA]] SampleConsensusModelPlane_PointXYZRGBA_ConstPtr_t
###

# sac_model_sphere.h
# namespace pcl
# /** \brief SampleConsensusModelSphere defines a model for 3D sphere segmentation.
#   * The model coefficients are defined as:
#   *   - \b center.x : the X coordinate of the sphere's center
#   *   - \b center.y : the Y coordinate of the sphere's center
#   *   - \b center.z : the Z coordinate of the sphere's center
#   *   - \b radius   : the sphere's radius
#   * \author Radu B. Rusu
#   * \ingroup sample_consensus
#   */
# template <typename PointT>
# class SampleConsensusModelSphere : public SampleConsensusModel<PointT>
cdef extern from "pcl/sample_consensus/sac_model_sphere.h" namespace "pcl":
    cdef cppclass SampleConsensusModelSphere[PointT](SampleConsensusModel[PointT]):
        # SampleConsensusModelSphere()
        SampleConsensusModelSphere(shared_ptr[cpp.PointCloud[PointT]] cloud)
        # public:
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # using SampleConsensusModel<PointT>::radius_min_;
        # using SampleConsensusModel<PointT>::radius_max_;
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelSphere> Ptr;
        # 
        # /** \brief Constructor for base SampleConsensusModelSphere.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelSphere (const PointCloudConstPtr &cloud) : 
        # SampleConsensusModel<PointT> (cloud), tmp_inliers_ ()
        # 
        # /** \brief Constructor for base SampleConsensusModelSphere.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelSphere (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
        # SampleConsensusModel<PointT> (cloud, indices), tmp_inliers_ ()
        # 
        # /** \brief Copy constructor.
        # * \param[in] source the model to copy into this
        # */
        # SampleConsensusModelSphere (const SampleConsensusModelSphere &source) :
        # SampleConsensusModel<PointT> (), tmp_inliers_ () 
        # 
        # /** \brief Copy constructor.
        # * \param[in] source the model to copy into this
        # */
        # inline SampleConsensusModelSphere& operator = (const SampleConsensusModelSphere &source)
        # 
        # /** \brief Check whether the given index samples can form a valid sphere model, compute the model 
        # * coefficients from these samples and store them internally in model_coefficients. 
        # * The sphere coefficients are: x, y, z, R.
        # * \param[in] samples the point indices found as possible good candidates for creating a valid model
        # * \param[out] model_coefficients the resultant model coefficients
        # */
        # bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);
        # 
        # /** \brief Compute all distances from the cloud data to a given sphere model.
        # * \param[in] model_coefficients the coefficients of a sphere model that we need to compute distances to
        # * \param[out] distances the resultant estimated distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a sphere model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers. 
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Recompute the sphere coefficients using the given inlier set and return them to the user.
        # * @note: these are the coefficients of the sphere model after refinement (eg. after SVD)
        # * \param[in] inliers the data inliers found as supporting the model
        # * \param[in] model_coefficients the initial guess for the optimization
        # * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
        # */
        # void optimizeModelCoefficients (const std::vector<int> &inliers, 
        #                          const Eigen::VectorXf &model_coefficients, 
        #                          Eigen::VectorXf &optimized_coefficients);
        # 
        # /** \brief Create a new point cloud with inliers projected onto the sphere model.
        # * \param[in] inliers the data inliers that we want to project on the sphere model
        # * \param[in] model_coefficients the coefficients of a sphere model
        # * \param[out] projected_points the resultant projected points
        # * \param[in] copy_data_fields set to true if we need to copy the other data fields
        # * \todo implement this.
        # */
        # void projectPoints (const std::vector<int> &inliers, 
        #              const Eigen::VectorXf &model_coefficients, 
        #              PointCloud &projected_points, 
        #              bool copy_data_fields = true);
        # 
        # /** \brief Verify whether a subset of indices verifies the given sphere model coefficients.
        # * \param[in] indices the data indices that need to be tested against the sphere model
        # * \param[in] model_coefficients the sphere model coefficients
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # */
        # bool doSamplesVerifyModel (const std::set<int> &indices, 
        #                     const Eigen::VectorXf &model_coefficients, 
        #                     const double threshold);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_SPHERE). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_SPHERE); }


ctypedef SampleConsensusModelSphere[cpp.PointXYZ] SampleConsensusModelSphere_t
ctypedef SampleConsensusModelSphere[cpp.PointXYZI] SampleConsensusModelSphere_PointXYZI_t
ctypedef SampleConsensusModelSphere[cpp.PointXYZRGB] SampleConsensusModelSphere_PointXYZRGB_t
ctypedef SampleConsensusModelSphere[cpp.PointXYZRGBA] SampleConsensusModelSphere_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelSphere[cpp.PointXYZ]] SampleConsensusModelSpherePtr_t
ctypedef shared_ptr[SampleConsensusModelSphere[cpp.PointXYZI]] SampleConsensusModelSphere_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelSphere[cpp.PointXYZRGB]] SampleConsensusModelSphere_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelSphere[cpp.PointXYZRGBA]] SampleConsensusModelSphere_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelSphere[cpp.PointXYZ]] SampleConsensusModelSphereConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelSphere[cpp.PointXYZI]] SampleConsensusModelSphere_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelSphere[cpp.PointXYZRGB]] SampleConsensusModelSphere_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelSphere[cpp.PointXYZRGBA]] SampleConsensusModelSphere_PointXYZRGBA_ConstPtr_t
###

### Inheritance class ###

# lmeds.h
# namespace pcl
# template <typename PointT>
# class LeastMedianSquares : public SampleConsensus<PointT>
cdef extern from "pcl/sample_consensus/lmeds.h" namespace "pcl":
    cdef cppclass LeastMedianSquares[T](SampleConsensus[T]):
        # LeastMedianSquares ()
        LeastMedianSquares (shared_ptr[SampleConsensusModel[T]] model)
        # LeastMedianSquares (const SampleConsensusModelPtr &model)
        # LeastMedianSquares (const SampleConsensusModelPtr &model, double threshold)
        # using SampleConsensus<PointT>::max_iterations_;
        # using SampleConsensus<PointT>::threshold_;
        # using SampleConsensus<PointT>::iterations_;
        # using SampleConsensus<PointT>::sac_model_;
        # using SampleConsensus<PointT>::model_;
        # using SampleConsensus<PointT>::model_coefficients_;
        # using SampleConsensus<PointT>::inliers_;
        # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
        # public:
        # /** \brief Compute the actual model and find the inliers
        #   * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
        #   */
        # bool computeModel (int debug_verbosity_level = 0)
        bool computeModel (int debug_verbosity_level = 0)


###

# mlesac.h
# namespace pcl
# template <typename PointT>
# class MaximumLikelihoodSampleConsensus : public SampleConsensus<PointT>
cdef extern from "pcl/sample_consensus/mlesac.h" namespace "pcl":
    cdef cppclass MaximumLikelihoodSampleConsensus[T](SampleConsensus[T]):
        MaximumLikelihoodSampleConsensus ()
        MaximumLikelihoodSampleConsensus (shared_ptr[SampleConsensusModel[T]] model)
        # \brief MLESAC (Maximum Likelihood Estimator SAmple Consensus) main constructor
        # \param[in] model a Sample Consensus model
        # MaximumLikelihoodSampleConsensus (const SampleConsensusModelPtr &model)
        # MaximumLikelihoodSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
        # using SampleConsensus<PointT>::max_iterations_;
        # using SampleConsensus<PointT>::threshold_;
        # using SampleConsensus<PointT>::iterations_;
        # using SampleConsensus<PointT>::sac_model_;
        # using SampleConsensus<PointT>::model_;
        # using SampleConsensus<PointT>::model_coefficients_;
        # using SampleConsensus<PointT>::inliers_;
        # using SampleConsensus<PointT>::probability_;
        # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr; 
        # public:
        # \brief Compute the actual model and find the inliers
        # \param[in] debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
        # bool computeModel (int debug_verbosity_level = 0);
        
        # /** \brief Set the number of EM iterations.
        # * \param[in] iterations the number of EM iterations
        # inline void setEMIterations (int iterations)
        
        # /** \brief Get the number of EM iterations. */
        # inline int getEMIterations () const { return (iterations_EM_); }


###

# msac.h
# namespace pcl
#   template <typename PointT>
#   class MEstimatorSampleConsensus : public SampleConsensus<PointT>
cdef extern from "pcl/sample_consensus/msac.h" namespace "pcl":
    cdef cppclass MEstimatorSampleConsensus[T](SampleConsensus[T]):
        MEstimatorSampleConsensus ()
        MEstimatorSampleConsensus (shared_ptr[SampleConsensusModel[T]] model)
        # MEstimatorSampleConsensus (const SampleConsensusModelPtr &model)
        # MEstimatorSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
        # using SampleConsensus<PointT>::max_iterations_;
        # using SampleConsensus<PointT>::threshold_;
        # using SampleConsensus<PointT>::iterations_;
        # using SampleConsensus<PointT>::sac_model_;
        # using SampleConsensus<PointT>::model_;
        # using SampleConsensus<PointT>::model_coefficients_;
        # using SampleConsensus<PointT>::inliers_;
        # using SampleConsensus<PointT>::probability_;
        # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
        # public:
        # \brief Compute the actual model and find the inliers
        # \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
        # bool computeModel (int debug_verbosity_level = 0);
        bool computeModel (int debug_verbosity_level)


###

# prosac.h
# namespace pcl
# template<typename PointT>
# class ProgressiveSampleConsensus : public SampleConsensus<PointT>
cdef extern from "pcl/sample_consensus/prosac.h" namespace "pcl":
    cdef cppclass ProgressiveSampleConsensus[T](SampleConsensus[T]):
        ProgressiveSampleConsensus ()
        # ProgressiveSampleConsensus (const SampleConsensusModelPtr &model) 
        # ProgressiveSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
        # using SampleConsensus<PointT>::max_iterations_;
        # using SampleConsensus<PointT>::threshold_;
        # using SampleConsensus<PointT>::iterations_;
        # using SampleConsensus<PointT>::sac_model_;
        # using SampleConsensus<PointT>::model_;
        # using SampleConsensus<PointT>::model_coefficients_;
        # using SampleConsensus<PointT>::inliers_;
        # using SampleConsensus<PointT>::probability_;
        # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
        # public:
        # /** \brief Compute the actual model and find the inliers
        # * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
        # bool computeModel (int debug_verbosity_level = 0)
        bool computeModel (int debug_verbosity_level)


###

# ransac.h
# namespace pcl
# template <typename PointT>
# class RandomSampleConsensus : public SampleConsensus<PointT>
cdef extern from "pcl/sample_consensus/ransac.h" namespace "pcl":
    cdef cppclass RandomSampleConsensus[T](SampleConsensus[T]):
        # RandomSampleConsensus ()
        RandomSampleConsensus (shared_ptr[SampleConsensusModel[T]] model)
        
        # RandomSampleConsensus (shared_ptr[SampleConsensus[T]] model)
        # RandomSampleConsensus (const SampleConsensusModelPtr &model)
        # RandomSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
        # using SampleConsensus<PointT>::max_iterations_;
        # using SampleConsensus<PointT>::threshold_;
        # using SampleConsensus<PointT>::iterations_;
        # using SampleConsensus<PointT>::sac_model_;
        # using SampleConsensus<PointT>::model_;
        # using SampleConsensus<PointT>::model_coefficients_;
        # using SampleConsensus<PointT>::inliers_;
        # using SampleConsensus<PointT>::probability_;
        # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
        # public:
        # /** \brief Compute the actual model and find the inliers
        # * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
        # bool computeModel (int debug_verbosity_level = 0);
        bool computeModel (int debug_verbosity_level)


ctypedef RandomSampleConsensus[cpp.PointXYZ] RandomSampleConsensus_t
ctypedef RandomSampleConsensus[cpp.PointXYZI] RandomSampleConsensus_PointXYZI_t
ctypedef RandomSampleConsensus[cpp.PointXYZRGB] RandomSampleConsensus_PointXYZRGB_t
ctypedef RandomSampleConsensus[cpp.PointXYZRGBA] RandomSampleConsensus_PointXYZRGBA_t
ctypedef shared_ptr[RandomSampleConsensus[cpp.PointXYZ]] RandomSampleConsensusPtr_t
ctypedef shared_ptr[RandomSampleConsensus[cpp.PointXYZI]] RandomSampleConsensus_PointXYZI_Ptr_t
ctypedef shared_ptr[RandomSampleConsensus[cpp.PointXYZRGB]] RandomSampleConsensus_PointXYZRGB_Ptr_t
ctypedef shared_ptr[RandomSampleConsensus[cpp.PointXYZRGBA]] RandomSampleConsensus_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const RandomSampleConsensus[cpp.PointXYZ]] RandomSampleConsensusConstPtr_t
ctypedef shared_ptr[const RandomSampleConsensus[cpp.PointXYZI]] RandomSampleConsensus_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const RandomSampleConsensus[cpp.PointXYZRGB]] RandomSampleConsensus_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const RandomSampleConsensus[cpp.PointXYZRGBA]] RandomSampleConsensus_PointXYZRGBA_ConstPtr_t
###

# rmsac.h
# namespace pcl
# template <typename PointT>
# class RandomizedMEstimatorSampleConsensus : public SampleConsensus<PointT>
cdef extern from "pcl/sample_consensus/rmsac.h" namespace "pcl":
    cdef cppclass RandomizedMEstimatorSampleConsensus[T](SampleConsensus[T]):
        RandomizedMEstimatorSampleConsensus ()
        # RandomizedMEstimatorSampleConsensus (const SampleConsensusModelPtr &model)
        # RandomizedMEstimatorSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
        RandomizedMEstimatorSampleConsensus (shared_ptr[SampleConsensusModel[T]] model)
        
        # using SampleConsensus<PointT>::max_iterations_;
        # using SampleConsensus<PointT>::threshold_;
        # using SampleConsensus<PointT>::iterations_;
        # using SampleConsensus<PointT>::sac_model_;
        # using SampleConsensus<PointT>::model_;
        # using SampleConsensus<PointT>::model_coefficients_;
        # using SampleConsensus<PointT>::inliers_;
        # using SampleConsensus<PointT>::probability_;
        # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
        # public:
        # /** \brief Compute the actual model and find the inliers
        # * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
        # */
        # bool computeModel (int debug_verbosity_level = 0);
        bool computeModel (int debug_verbosity_level)
        
        # /** \brief Set the percentage of points to pre-test.
        # * \param nr_pretest percentage of points to pre-test
        # */
        # inline void setFractionNrPretest (double nr_pretest)
        void setFractionNrPretest (double nr_pretest)
        
        # /** \brief Get the percentage of points to pre-test. */
        # inline double getFractionNrPretest ()
        double getFractionNrPretest ()


ctypedef RandomizedMEstimatorSampleConsensus[cpp.PointXYZ] RandomizedMEstimatorSampleConsensus_t
ctypedef RandomizedMEstimatorSampleConsensus[cpp.PointXYZI] RandomizedMEstimatorSampleConsensus_PointXYZI_t
ctypedef RandomizedMEstimatorSampleConsensus[cpp.PointXYZRGB] RandomizedMEstimatorSampleConsensus_PointXYZRGB_t
ctypedef RandomizedMEstimatorSampleConsensus[cpp.PointXYZRGBA] RandomizedMEstimatorSampleConsensus_PointXYZRGBA_t
ctypedef shared_ptr[RandomizedMEstimatorSampleConsensus[cpp.PointXYZ]] RandomizedMEstimatorSampleConsensusPtr_t
ctypedef shared_ptr[RandomizedMEstimatorSampleConsensus[cpp.PointXYZI]] RandomizedMEstimatorSampleConsensus_PointXYZI_Ptr_t
ctypedef shared_ptr[RandomizedMEstimatorSampleConsensus[cpp.PointXYZRGB]] RandomizedMEstimatorSampleConsensus_PointXYZRGB_Ptr_t
ctypedef shared_ptr[RandomizedMEstimatorSampleConsensus[cpp.PointXYZRGBA]] RandomizedMEstimatorSampleConsensus_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const RandomizedMEstimatorSampleConsensus[cpp.PointXYZ]] RandomizedMEstimatorSampleConsensusConstPtr_t
ctypedef shared_ptr[const RandomizedMEstimatorSampleConsensus[cpp.PointXYZI]] RandomizedMEstimatorSampleConsensus_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const RandomizedMEstimatorSampleConsensus[cpp.PointXYZRGB]] RandomizedMEstimatorSampleConsensus_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const RandomizedMEstimatorSampleConsensus[cpp.PointXYZRGBA]] RandomizedMEstimatorSampleConsensus_PointXYZRGBA_ConstPtr_t
###

# rransac.h
# namespace pcl
# template <typename PointT>
# class RandomizedRandomSampleConsensus : public SampleConsensus<PointT>
cdef extern from "pcl/sample_consensus/rransac.h" namespace "pcl":
    cdef cppclass RandomizedRandomSampleConsensus[T](SampleConsensus[T]):
        RandomizedRandomSampleConsensus ()
        RandomizedRandomSampleConsensus (shared_ptr[SampleConsensusModel[T]] model)
        # RandomizedRandomSampleConsensus (const SampleConsensusModelPtr &model)
        # RandomizedRandomSampleConsensus (const SampleConsensusModelPtr &model, double threshold)
        # using SampleConsensus<PointT>::max_iterations_;
        # using SampleConsensus<PointT>::threshold_;
        # using SampleConsensus<PointT>::iterations_;
        # using SampleConsensus<PointT>::sac_model_;
        # using SampleConsensus<PointT>::model_;
        # using SampleConsensus<PointT>::model_coefficients_;
        # using SampleConsensus<PointT>::inliers_;
        # using SampleConsensus<PointT>::probability_;
        # typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
        # public:
        # /** \brief RRANSAC (RAndom SAmple Consensus) main constructor
        # * \param model a Sample Consensus model
        # * \param threshold distance to model threshold
        # /** \brief Compute the actual model and find the inliers
        # * \param debug_verbosity_level enable/disable on-screen debug information and set the verbosity level
        # */
        # bool computeModel (int debug_verbosity_level = 0)
        bool computeModel (int debug_verbosity_level)
        
        # \brief Set the percentage of points to pre-test.
        # \param nr_pretest percentage of points to pre-test
        # inline void setFractionNrPretest (double nr_pretest)
        void setFractionNrPretest (double nr_pretest)
        
        # /** \brief Get the percentage of points to pre-test. */
        # inline double getFractionNrPretest ()
        double getFractionNrPretest ()


ctypedef RandomizedRandomSampleConsensus[cpp.PointXYZ] RandomizedRandomSampleConsensus_t
ctypedef RandomizedRandomSampleConsensus[cpp.PointXYZI] RandomizedRandomSampleConsensus_PointXYZI_t
ctypedef RandomizedRandomSampleConsensus[cpp.PointXYZRGB] RandomizedRandomSampleConsensus_PointXYZRGB_t
ctypedef RandomizedRandomSampleConsensus[cpp.PointXYZRGBA] RandomizedRandomSampleConsensus_PointXYZRGBA_t
ctypedef shared_ptr[RandomizedRandomSampleConsensus[cpp.PointXYZ]] RandomizedRandomSampleConsensusPtr_t
ctypedef shared_ptr[RandomizedRandomSampleConsensus[cpp.PointXYZI]] RandomizedRandomSampleConsensus_PointXYZI_Ptr_t
ctypedef shared_ptr[RandomizedRandomSampleConsensus[cpp.PointXYZRGB]] RandomizedRandomSampleConsensus_PointXYZRGB_Ptr_t
ctypedef shared_ptr[RandomizedRandomSampleConsensus[cpp.PointXYZRGBA]] RandomizedRandomSampleConsensus_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const RandomizedRandomSampleConsensus[cpp.PointXYZ]] RandomizedRandomSampleConsensusConstPtr_t
ctypedef shared_ptr[const RandomizedRandomSampleConsensus[cpp.PointXYZI]] RandomizedRandomSampleConsensus_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const RandomizedRandomSampleConsensus[cpp.PointXYZRGB]] RandomizedRandomSampleConsensus_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const RandomizedRandomSampleConsensus[cpp.PointXYZRGBA]] RandomizedRandomSampleConsensus_PointXYZRGBA_ConstPtr_t
###

# sac_model_circle.h
# namespace pcl
# template <typename PointT>
# class SampleConsensusModelCircle2D : public SampleConsensusModel<PointT>
cdef extern from "pcl/sample_consensus/sac_model_circle.h" namespace "pcl":
    cdef cppclass SampleConsensusModelCircle2D[T](SampleConsensusModel[T]):
        SampleConsensusModelCircle2D ()
        # SampleConsensusModelCircle2D (const PointCloudConstPtr &cloud)
        # SampleConsensusModelCircle2D (const PointCloudConstPtr &cloud, const std::vector<int> &indices)
        # SampleConsensusModelCircle2D (const SampleConsensusModelCircle2D &source) :
        # inline SampleConsensusModelCircle2D& operator = (const SampleConsensusModelCircle2D &source)
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # using SampleConsensusModel<PointT>::radius_min_;
        # using SampleConsensusModel<PointT>::radius_max_;
        # public:
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelCircle2D> Ptr;
        # /** \brief Check whether the given index samples can form a valid 2D circle model, compute the model coefficients
        # * from these samples and store them in model_coefficients. The circle coefficients are: x, y, R.
        # * \param[in] samples the point indices found as possible good candidates for creating a valid model
        # * \param[out] model_coefficients the resultant model coefficients
        # bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);
        # /** \brief Compute all distances from the cloud data to a given 2D circle model.
        # * \param[in] model_coefficients the coefficients of a 2D circle model that we need to compute distances to
        # * \param[out] distances the resultant estimated distances
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # /** \brief Compute all distances from the cloud data to a given 2D circle model.
        # * \param[in] model_coefficients the coefficients of a 2D circle model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
        #                     const double threshold, 
        #                     std::vector<int> &inliers);
        # /** \brief Count all the points which respect the given model coefficients as inliers. 
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, 
        #                    const double threshold);
        # /** \brief Recompute the 2d circle coefficients using the given inlier set and return them to the user.
        # * @note: these are the coefficients of the 2d circle model after refinement (eg. after SVD)
        # * \param[in] inliers the data inliers found as supporting the model
        # * \param[in] model_coefficients the initial guess for the optimization
        # * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
        # void optimizeModelCoefficients (const std::vector<int> &inliers, 
        #                          const Eigen::VectorXf &model_coefficients, 
        #                          Eigen::VectorXf &optimized_coefficients);
        # /** \brief Create a new point cloud with inliers projected onto the 2d circle model.
        # * \param[in] inliers the data inliers that we want to project on the 2d circle model
        # * \param[in] model_coefficients the coefficients of a 2d circle model
        # * \param[out] projected_points the resultant projected points
        # * \param[in] copy_data_fields set to true if we need to copy the other data fields
        # void projectPoints (const std::vector<int> &inliers, 
        #              const Eigen::VectorXf &model_coefficients, 
        #              PointCloud &projected_points, 
        #              bool copy_data_fields = true);
        # /** \brief Verify whether a subset of indices verifies the given 2d circle model coefficients.
        # * \param[in] indices the data indices that need to be tested against the 2d circle model
        # * \param[in] model_coefficients the 2d circle model coefficients
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # bool doSamplesVerifyModel (const std::set<int> &indices, 
        #                     const Eigen::VectorXf &model_coefficients, 
        #                     const double threshold);
        # /** \brief Return an unique id for this model (SACMODEL_CIRCLE2D). */
        # inline pcl::SacModel getModelType () const


ctypedef SampleConsensusModelCircle2D[cpp.PointXYZ] SampleConsensusModelCircle2D_t
ctypedef SampleConsensusModelCircle2D[cpp.PointXYZI] SampleConsensusModelCircle2D_PointXYZI_t
ctypedef SampleConsensusModelCircle2D[cpp.PointXYZRGB] SampleConsensusModelCircle2D_PointXYZRGB_t
ctypedef SampleConsensusModelCircle2D[cpp.PointXYZRGBA] SampleConsensusModelCircle2D_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelCircle2D[cpp.PointXYZ]] SampleConsensusModelCircle2DPtr_t
ctypedef shared_ptr[SampleConsensusModelCircle2D[cpp.PointXYZI]] SampleConsensusModelCircle2D_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelCircle2D[cpp.PointXYZRGB]] SampleConsensusModelCircle2D_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelCircle2D[cpp.PointXYZRGBA]] SampleConsensusModelCircle2D_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelCircle2D[cpp.PointXYZ]] SampleConsensusModelCircle2DConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelCircle2D[cpp.PointXYZI]] SampleConsensusModelCircle2D_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelCircle2D[cpp.PointXYZRGB]] SampleConsensusModelCircle2D_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelCircle2D[cpp.PointXYZRGBA]] SampleConsensusModelCircle2D_PointXYZRGBA_ConstPtr_t
###

# namespace pcl
# struct OptimizationFunctor : pcl::Functor<float>
#         OptimizationFunctor (int m_data_points, pcl::SampleConsensusModelCircle2D<PointT> *model) : 
# 
#         /** Cost function to be minimized
#           * \param[in] x the variables array
#           * \param[out] fvec the resultant functions evaluations
#           * \return 0
#           */
#         int operator() (const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
#         pcl::SampleConsensusModelCircle2D<PointT> *model_;
###

# sac_model_cone.h
# namespace pcl
# template <typename PointT, typename PointNT>
# class SampleConsensusModelCone : public SampleConsensusModel<PointT>, public SampleConsensusModelFromNormals<PointT, PointNT>
cdef extern from "pcl/sample_consensus/sac_model_cone.h" namespace "pcl":
    # cdef cppclass SampleConsensusModelCone[T, NT](SampleConsensusModel[T])(SampleConsensusModelFromNormals[T, NT]):
    cdef cppclass SampleConsensusModelCone[T, NT]:
        SampleConsensusModelCone ()
        # Nothing
        # SampleConsensusModelCone ()
        # Use
        # SampleConsensusModelCone (const PointCloudConstPtr &cloud)
        # SampleConsensusModelCone (const PointCloudConstPtr &cloud, const std::vector<int> &indices)
        # SampleConsensusModelCone (const SampleConsensusModelCone &source)
        # inline SampleConsensusModelCone& operator = (const SampleConsensusModelCone &source)
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # using SampleConsensusModel<PointT>::radius_min_;
        # using SampleConsensusModel<PointT>::radius_max_;
        # using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
        # using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
        # public:
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelCone> Ptr;
        # /** \brief Set the angle epsilon (delta) threshold.
        # * \param[in] ea the maximum allowed difference between the cone's axis and the given axis.
        # inline void setEpsAngle (double ea)
        # /** \brief Get the angle epsilon (delta) threshold. */
        # inline double getEpsAngle () const
        # /** \brief Set the axis along which we need to search for a cone direction.
        # * \param[in] ax the axis along which we need to search for a cone direction
        # inline void setAxis (const Eigen::Vector3f &ax)
        # /** \brief Get the axis along which we need to search for a cone direction. */
        # inline Eigen::Vector3f getAxis () const
        # /** \brief Set the minimum and maximum allowable opening angle for a cone model
        # * given from a user.
        # * \param[in] min_angle the minimum allwoable opening angle of a cone model
        # * \param[in] max_angle the maximum allwoable opening angle of a cone model
        # inline void setMinMaxOpeningAngle (const double &min_angle, const double &max_angle)
        # /** \brief Get the opening angle which we need minumum to validate a cone model.
        # * \param[out] min_angle the minimum allwoable opening angle of a cone model
        # * \param[out] max_angle the maximum allwoable opening angle of a cone model
        # inline void getMinMaxOpeningAngle (double &min_angle, double &max_angle) const
        # /** \brief Check whether the given index samples can form a valid cone model, compute the model coefficients
        # * from these samples and store them in model_coefficients. The cone coefficients are: apex,
        # * axis_direction, opening_angle.
        # * \param[in] samples the point indices found as possible good candidates for creating a valid model
        # * \param[out] model_coefficients the resultant model coefficients
        # bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);
        # /** \brief Compute all distances from the cloud data to a given cone model.
        # * \param[in] model_coefficients the coefficients of a cone model that we need to compute distances to
        # * \param[out] distances the resultant estimated distances
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients,  std::vector<double> &distances);
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a cone model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, 
        #                     const double threshold, std::vector<int> &inliers);
        # /** \brief Count all the points which respect the given model coefficients as inliers. 
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # /** \brief Recompute the cone coefficients using the given inlier set and return them to the user.
        # * @note: these are the coefficients of the cone model after refinement (eg. after SVD)
        # * \param[in] inliers the data inliers found as supporting the model
        # * \param[in] model_coefficients the initial guess for the optimization
        # * \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
        # void optimizeModelCoefficients (const std::vector<int> &inliers, 
        #                          const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients);
        # /** \brief Create a new point cloud with inliers projected onto the cone model.
        # * \param[in] inliers the data inliers that we want to project on the cone model
        # * \param[in] model_coefficients the coefficients of a cone model
        # * \param[out] projected_points the resultant projected points
        # * \param[in] copy_data_fields set to true if we need to copy the other data fields
        # void projectPoints (const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, 
        #              PointCloud &projected_points, bool copy_data_fields = true);
        # /** \brief Verify whether a subset of indices verifies the given cone model coefficients.
        # * \param[in] indices the data indices that need to be tested against the cone model
        # * \param[in] model_coefficients the cone model coefficients
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # bool doSamplesVerifyModel (const std::set<int> &indices, 
        #                     const Eigen::VectorXf &model_coefficients, const double threshold);
        # /** \brief Return an unique id for this model (SACMODEL_CONE). */
        # inline pcl::SacModel getModelType () const
        # protected:
        # /** \brief Get the distance from a point to a line (represented by a point and a direction)
        # * \param[in] pt a point
        # * \param[in] model_coefficients the line coefficients (a point on the line, line direction)
        # double pointToAxisDistance (const Eigen::Vector4f &pt, const Eigen::VectorXf &model_coefficients);
        # /** \brief Get a string representation of the name of this class. */
        # std::string getName () const { return ("SampleConsensusModelCone"); }
        # protected:
        # /** \brief Check whether a model is valid given the user constraints.
        # * \param[in] model_coefficients the set of model coefficients
        # bool isModelValid (const Eigen::VectorXf &model_coefficients);
        # /** \brief Check if a sample of indices results in a good sample of points
        # * indices. Pure virtual.
        # * \param[in] samples the resultant index samples
        # bool isSampleGood (const std::vector<int> &samples) const;


ctypedef SampleConsensusModelCone[cpp.PointXYZ, cpp.Normal] SampleConsensusModelCone_t
ctypedef SampleConsensusModelCone[cpp.PointXYZI, cpp.Normal] SampleConsensusModelCone_PointXYZI_t
ctypedef SampleConsensusModelCone[cpp.PointXYZRGB, cpp.Normal] SampleConsensusModelCone_PointXYZRGB_t
ctypedef SampleConsensusModelCone[cpp.PointXYZRGBA, cpp.Normal] SampleConsensusModelCone_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelCone[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelConePtr_t
ctypedef shared_ptr[SampleConsensusModelCone[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelCone_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelCone[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelCone_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelCone[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelCone_PointXYZRGBA_Ptr_t
###

# namespace pcl
# /** \brief Functor for the optimization function */
# struct OptimizationFunctor : pcl::Functor<float>
# cdef extern from "pcl/sample_consensus/sac_model_cone.h" namespace "pcl":
#     cdef cppclass OptimizationFunctor(Functor[float]):
#         OptimizationFunctor (int m_data_points, pcl::SampleConsensusModelCone<PointT, PointNT> *model) : 
#         int operator() (const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
#         pcl::SampleConsensusModelCone<PointT, PointNT> *model_;
###


# sac_model_cylinder.h
# namespace pcl
# \brief @b SampleConsensusModelCylinder defines a model for 3D cylinder segmentation.
# The model coefficients are defined as:
# \b point_on_axis.x  : the X coordinate of a point located on the cylinder axis
# \b point_on_axis.y  : the Y coordinate of a point located on the cylinder axis
# \b point_on_axis.z  : the Z coordinate of a point located on the cylinder axis
# \b axis_direction.x : the X coordinate of the cylinder's axis direction
# \b axis_direction.y : the Y coordinate of the cylinder's axis direction
# \b axis_direction.z : the Z coordinate of the cylinder's axis direction
# \b radius           : the cylinder's radius
# \author Radu Bogdan Rusu
# \ingroup sample_consensus
# template <typename PointT, typename PointNT>
# class SampleConsensusModelCylinder : public SampleConsensusModel<PointT>, public SampleConsensusModelFromNormals<PointT, PointNT>
# Multi Inheritance NG
# cdef cppclass SampleConsensusModelCylinder[PointT](SampleConsensusModel[PointT])(SampleConsensusModelFromNormals[PointT, PointNT]):
cdef extern from "pcl/sample_consensus/sac_model_cylinder.h" namespace "pcl":
    cdef cppclass SampleConsensusModelCylinder[PointT, PointNT]:
        SampleConsensusModelCylinder()
        SampleConsensusModelCylinder(shared_ptr[cpp.PointCloud[PointT]] cloud)
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # using SampleConsensusModel<PointT>::radius_min_;
        # using SampleConsensusModel<PointT>::radius_max_;
        # using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
        # using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
        # public:
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelCylinder> Ptr;
        # 
        # \brief Constructor for base SampleConsensusModelCylinder.
        # \param[in] cloud the input point cloud dataset
        # SampleConsensusModelCylinder (const PointCloudConstPtr &cloud) : 
        # SampleConsensusModel<PointT> (cloud), 
        # axis_ (Eigen::Vector3f::Zero ()),
        # eps_angle_ (0),
        # tmp_inliers_ ()
        # 
        # \brief Constructor for base SampleConsensusModelCylinder.
        # \param[in] cloud the input point cloud dataset
        # \param[in] indices a vector of point indices to be used from \a cloud
        # SampleConsensusModelCylinder (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
        # SampleConsensusModel<PointT> (cloud, indices), 
        # axis_ (Eigen::Vector3f::Zero ()),
        # eps_angle_ (0),
        # tmp_inliers_ ()
        # 
        # \brief Copy constructor.
        # \param[in] source the model to copy into this
        # SampleConsensusModelCylinder (const SampleConsensusModelCylinder &source) :
        # SampleConsensusModel<PointT> (),
        # axis_ (Eigen::Vector3f::Zero ()),
        # eps_angle_ (0),
        # tmp_inliers_ ()
        # 
        # \brief Copy constructor.
        # \param[in] source the model to copy into this
        # inline SampleConsensusModelCylinder& operator = (const SampleConsensusModelCylinder &source)
        # 
        # \brief Set the angle epsilon (delta) threshold.
        # \param[in] ea the maximum allowed difference between the cyilinder axis and the given axis.
        # inline void setEpsAngle (const double ea) { eps_angle_ = ea; }
        # 
        # \brief Get the angle epsilon (delta) threshold.
        # inline double getEpsAngle () { return (eps_angle_); }
        # 
        # \brief Set the axis along which we need to search for a cylinder direction.
        # \param[in] ax the axis along which we need to search for a cylinder direction
        # inline void setAxis (const Eigen::Vector3f &ax) { axis_ = ax; }
        # 
        # \brief Get the axis along which we need to search for a cylinder direction.
        # inline Eigen::Vector3f getAxis ()  { return (axis_); }
        # 
        # \brief Check whether the given index samples can form a valid cylinder model, compute the model coefficients
        #  from these samples and store them in model_coefficients. The cylinder coefficients are: point_on_axis,
        #  axis_direction, cylinder_radius_R
        # \param[in] samples the point indices found as possible good candidates for creating a valid model
        # \param[out] model_coefficients the resultant model coefficients
        # bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);
        # 
        # \brief Compute all distances from the cloud data to a given cylinder model.
        # \param[in] model_coefficients the coefficients of a cylinder model that we need to compute distances to
        # \param[out] distances the resultant estimated distances
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # \brief Select all the points which respect the given model coefficients as inliers.
        # \param[in] model_coefficients the coefficients of a cylinder model that we need to compute distances to
        # \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # \param[out] inliers the resultant model inliers
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # \brief Count all the points which respect the given model coefficients as inliers. 
        # \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # \return the resultant number of inliers
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # \brief Recompute the cylinder coefficients using the given inlier set and return them to the user.
        #  @note: these are the coefficients of the cylinder model after refinement (eg. after SVD)
        # \param[in] inliers the data inliers found as supporting the model
        # \param[in] model_coefficients the initial guess for the optimization
        # \param[out] optimized_coefficients the resultant recomputed coefficients after non-linear optimization
        # void optimizeModelCoefficients (const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients);
        # 
        # \brief Create a new point cloud with inliers projected onto the cylinder model.
        # \param[in] inliers the data inliers that we want to project on the cylinder model
        # \param[in] model_coefficients the coefficients of a cylinder model
        # \param[out] projected_points the resultant projected points
        # \param[in] copy_data_fields set to true if we need to copy the other data fields
        # void projectPoints (const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients,  PointCloud &projected_points, bool copy_data_fields = true);
        # 
        # \brief Verify whether a subset of indices verifies the given cylinder model coefficients.
        # \param[in] indices the data indices that need to be tested against the cylinder model
        # \param[in] model_coefficients the cylinder model coefficients
        # \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # bool doSamplesVerifyModel (const std::set<int> &indices, const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_CYLINDER). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_CYLINDER); }


ctypedef SampleConsensusModelCylinder[cpp.PointXYZ, cpp.Normal] SampleConsensusModelCylinder_t
ctypedef SampleConsensusModelCylinder[cpp.PointXYZI, cpp.Normal] SampleConsensusModelCylinder_PointXYZI_t
ctypedef SampleConsensusModelCylinder[cpp.PointXYZRGB, cpp.Normal] SampleConsensusModelCylinder_PointXYZRGB_t
ctypedef SampleConsensusModelCylinder[cpp.PointXYZRGBA, cpp.Normal] SampleConsensusModelCylinder_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelCylinder[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelCylinderPtr_t
ctypedef shared_ptr[SampleConsensusModelCylinder[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelCylinder_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelCylinder[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelCylinder_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelCylinder[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelCylinder_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelCylinder[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelCylinderConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelCylinder[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelCylinder_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelCylinder[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelCylinder_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelCylinder[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelCylinder_PointXYZRGBA_ConstPtr_t
###

# sac_model_line.h
# namespace pcl
# /** \brief SampleConsensusModelLine defines a model for 3D line segmentation.
#   * The model coefficients are defined as:
#   *   - \b point_on_line.x  : the X coordinate of a point on the line
#   *   - \b point_on_line.y  : the Y coordinate of a point on the line
#   *   - \b point_on_line.z  : the Z coordinate of a point on the line
#   *   - \b line_direction.x : the X coordinate of a line's direction
#   *   - \b line_direction.y : the Y coordinate of a line's direction
#   *   - \b line_direction.z : the Z coordinate of a line's direction
#   *
#   * \author Radu B. Rusu
#   * \ingroup sample_consensus
#   */
# template <typename PointT>
# class SampleConsensusModelLine : public SampleConsensusModel<PointT>
cdef extern from "pcl/sample_consensus/sac_model_line.h" namespace "pcl":
    cdef cppclass SampleConsensusModelLine[PointT](SampleConsensusModel[PointT]):
        SampleConsensusModelLine()
        SampleConsensusModelLine(shared_ptr[cpp.PointCloud[PointT]] cloud)
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # public:
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelLine> Ptr;
        # 
        # /** \brief Constructor for base SampleConsensusModelLine.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelLine (const PointCloudConstPtr &cloud) : SampleConsensusModel<PointT> (cloud) {};
        # 
        # /** \brief Constructor for base SampleConsensusModelLine.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelLine (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : SampleConsensusModel<PointT> (cloud, indices) {};
        # 
        # /** \brief Check whether the given index samples can form a valid line model, compute the model coefficients from
        # * these samples and store them internally in model_coefficients_. The line coefficients are represented by a
        # * point and a line direction
        # * \param[in] samples the point indices found as possible good candidates for creating a valid model
        # * \param[out] model_coefficients the resultant model coefficients
        # */
        # bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);
        # 
        # /** \brief Compute all squared distances from the cloud data to a given line model.
        # * \param[in] model_coefficients the coefficients of a line model that we need to compute distances to
        # * \param[out] distances the resultant estimated squared distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a line model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers. 
        # * 
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Recompute the line coefficients using the given inlier set and return them to the user.
        # * @note: these are the coefficients of the line model after refinement (eg. after SVD)
        # * \param[in] inliers the data inliers found as supporting the model
        # * \param[in] model_coefficients the initial guess for the model coefficients
        # * \param[out] optimized_coefficients the resultant recomputed coefficients after optimization
        # */
        # void optimizeModelCoefficients (const std::vector<int> &inliers, 
        #                          const Eigen::VectorXf &model_coefficients, 
        #                          Eigen::VectorXf &optimized_coefficients);
        # 
        # /** \brief Create a new point cloud with inliers projected onto the line model.
        # * \param[in] inliers the data inliers that we want to project on the line model
        # * \param[in] model_coefficients the *normalized* coefficients of a line model
        # * \param[out] projected_points the resultant projected points
        # * \param[in] copy_data_fields set to true if we need to copy the other data fields
        # */
        # void projectPoints (const std::vector<int> &inliers, 
        #              const Eigen::VectorXf &model_coefficients, 
        #              PointCloud &projected_points, 
        #              bool copy_data_fields = true);
        # 
        # /** \brief Verify whether a subset of indices verifies the given line model coefficients.
        # * \param[in] indices the data indices that need to be tested against the line model
        # * \param[in] model_coefficients the line model coefficients
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # */
        # bool doSamplesVerifyModel (const std::set<int> &indices, 
        #                     const Eigen::VectorXf &model_coefficients, 
        #                     const double threshold);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_LINE). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_LINE); }


ctypedef SampleConsensusModelLine[cpp.PointXYZ] SampleConsensusModelLine_t
ctypedef SampleConsensusModelLine[cpp.PointXYZI] SampleConsensusModelLine_PointXYZI_t
ctypedef SampleConsensusModelLine[cpp.PointXYZRGB] SampleConsensusModelLine_PointXYZRGB_t
ctypedef SampleConsensusModelLine[cpp.PointXYZRGBA] SampleConsensusModelLine_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelLine[cpp.PointXYZ]] SampleConsensusModelLinePtr_t
ctypedef shared_ptr[SampleConsensusModelLine[cpp.PointXYZI]] SampleConsensusModelLine_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelLine[cpp.PointXYZRGB]] SampleConsensusModelLine_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelLine[cpp.PointXYZRGBA]] SampleConsensusModelLine_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelLine[cpp.PointXYZ]] SampleConsensusModelLineConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelLine[cpp.PointXYZI]] SampleConsensusModelLine_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelLine[cpp.PointXYZRGB]] SampleConsensusModelLine_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelLine[cpp.PointXYZRGBA]] SampleConsensusModelLine_PointXYZRGBA_ConstPtr_t
###

# sac_model_normal_parallel_plane.h
# namespace pcl
# /** \brief SampleConsensusModelNormalParallelPlane defines a model for 3D
#   * plane segmentation using additional surface normal constraints. Basically
#   * this means that checking for inliers will not only involve a "distance to
#   * model" criterion, but also an additional "maximum angular deviation"
#   * between the plane's normal and the inlier points normals. In addition,
#   * the plane normal must lie parallel to an user-specified axis.
#   * The model coefficients are defined as:
#   *   - \b a : the X coordinate of the plane's normal (normalized)
#   *   - \b b : the Y coordinate of the plane's normal (normalized)
#   *   - \b c : the Z coordinate of the plane's normal (normalized)
#   *   - \b d : the fourth <a href="http://mathworld.wolfram.com/HessianNormalForm.html">Hessian component</a> of the plane's equation
#   * To set the influence of the surface normals in the inlier estimation
#   * process, set the normal weight (0.0-1.0), e.g.:
#   * \code
#   * SampleConsensusModelNormalPlane<pcl::PointXYZ, pcl::Normal> sac_model;
#   * ...
#   * sac_model.setNormalDistanceWeight (0.1);
#   * ...
#   * \endcode
#   * In addition, the user can specify more constraints, such as:
#   * 
#   *   - an axis along which we need to search for a plane perpendicular to (\ref setAxis);
#   *   - an angle \a tolerance threshold between the plane's normal and the above given axis (\ref setEpsAngle);
#   *   - a distance we expect the plane to be from the origin (\ref setDistanceFromOrigin);
#   *   - a distance \a tolerance as the maximum allowed deviation from the above given distance from the origin (\ref setEpsDist).
#   *
#   * \note Please remember that you need to specify an angle > 0 in order to activate the axis-angle constraint!
#   * \author Radu B. Rusu and Jared Glover and Nico Blodow
#   * \ingroup sample_consensus
#   */
# template <typename PointT, typename PointNT>
# class SampleConsensusModelNormalParallelPlane : public SampleConsensusModelPlane<PointT>, public SampleConsensusModelFromNormals<PointT, PointNT>
cdef extern from "pcl/sample_consensus/sac_model_normal_parallel_plane.h" namespace "pcl":
    # cdef cppclass SampleConsensusModelNormalParallelPlane[PointT](SampleConsensusModelPlane[PointT])(SampleConsensusModelFromNormals[PointT, PointNT]):
    cdef cppclass SampleConsensusModelNormalParallelPlane[PointT, PointNT]:
        SampleConsensusModelNormalParallelPlane()
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
        # using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
        # public:
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNPtr PointCloudNPtr;
        # typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNConstPtr PointCloudNConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelNormalParallelPlane> Ptr;
        # 
        # /** \brief Constructor for base SampleConsensusModelNormalParallelPlane.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelNormalParallelPlane (const PointCloudConstPtr &cloud) : 
        # SampleConsensusModelPlane<PointT> (cloud),
        # axis_ (Eigen::Vector4f::Zero ()),
        # distance_from_origin_ (0),
        # eps_angle_ (-1.0), cos_angle_ (-1.0), eps_dist_ (0.0)
        # 
        # /** \brief Constructor for base SampleConsensusModelNormalParallelPlane.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelNormalParallelPlane (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
        # SampleConsensusModelPlane<PointT> (cloud, indices),
        # axis_ (Eigen::Vector4f::Zero ()),
        # distance_from_origin_ (0),
        # eps_angle_ (-1.0), cos_angle_ (-1.0), eps_dist_ (0.0)
        # 
        # /** \brief Set the axis along which we need to search for a plane perpendicular to.
        # * \param[in] ax the axis along which we need to search for a plane perpendicular to
        # */
        # inline void setAxis (const Eigen::Vector3f &ax) { axis_.head<3> () = ax; axis_.normalize ();}
        # 
        # /** \brief Get the axis along which we need to search for a plane perpendicular to. */
        # inline Eigen::Vector3f getAxis () { return (axis_.head<3> ()); }
        # 
        # /** \brief Set the angle epsilon (delta) threshold.
        # * \param[in] ea the maximum allowed deviation from 90 degrees between the plane normal and the given axis.
        # * \note You need to specify an angle > 0 in order to activate the axis-angle constraint!
        # */
        # inline void setEpsAngle (const double ea) { eps_angle_ = ea; cos_angle_ = fabs (cos (ea));}
        # 
        # /** \brief Get the angle epsilon (delta) threshold. */
        # inline double getEpsAngle () { return (eps_angle_); }
        # 
        # /** \brief Set the distance we expect the plane to be from the origin
        # * \param[in] d distance from the template plane to the origin
        # */
        # inline void setDistanceFromOrigin (const double d) { distance_from_origin_ = d; }
        # 
        # /** \brief Get the distance of the plane from the origin. */
        # inline double getDistanceFromOrigin () { return (distance_from_origin_); }
        # 
        # /** \brief Set the distance epsilon (delta) threshold.
        # * \param[in] delta the maximum allowed deviation from the template distance from the origin
        # */
        # inline void setEpsDist (const double delta) { eps_dist_ = delta; }
        # 
        # /** \brief Get the distance epsilon (delta) threshold. */
        # inline double getEpsDist () { return (eps_dist_); }
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers.
        # *
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Compute all distances from the cloud data to a given plane model.
        # * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
        # * \param[out] distances the resultant estimated distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_NORMAL_PARALLEL_PLANE). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_NORMAL_PARALLEL_PLANE); }


ctypedef SampleConsensusModelNormalParallelPlane[cpp.PointXYZ, cpp.Normal] SampleConsensusModelNormalParallelPlane_t
ctypedef SampleConsensusModelNormalParallelPlane[cpp.PointXYZI, cpp.Normal] SampleConsensusModelNormalParallelPlane_PointXYZI_t
ctypedef SampleConsensusModelNormalParallelPlane[cpp.PointXYZRGB, cpp.Normal] SampleConsensusModelNormalParallelPlane_PointXYZRGB_t
ctypedef SampleConsensusModelNormalParallelPlane[cpp.PointXYZRGBA, cpp.Normal] SampleConsensusModelNormalParallelPlane_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelNormalParallelPlane[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelNormalParallelPlanePtr_t
ctypedef shared_ptr[SampleConsensusModelNormalParallelPlane[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelNormalParallelPlane_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelNormalParallelPlane[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelNormalParallelPlane_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelNormalParallelPlane[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelNormalParallelPlane_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelNormalParallelPlane[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelNormalParallelPlaneConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelNormalParallelPlane[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelNormalParallelPlane_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelNormalParallelPlane[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelNormalParallelPlane_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelNormalParallelPlane[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelNormalParallelPlane_PointXYZRGBA_ConstPtr_t
###

# sac_model_normal_plane.h
# namespace pcl
# /** \brief SampleConsensusModelNormalPlane defines a model for 3D plane
#   * segmentation using additional surface normal constraints. Basically this
#   * means that checking for inliers will not only involve a "distance to
#   * model" criterion, but also an additional "maximum angular deviation"
#   * between the plane's normal and the inlier points normals.
#   *
#   * The model coefficients are defined as:
#   *   - \b a : the X coordinate of the plane's normal (normalized)
#   *   - \b b : the Y coordinate of the plane's normal (normalized)
#   *   - \b c : the Z coordinate of the plane's normal (normalized)
#   *   - \b d : the fourth <a href="http://mathworld.wolfram.com/HessianNormalForm.html">Hessian component</a> of the plane's equation
#   * To set the influence of the surface normals in the inlier estimation
#   * process, set the normal weight (0.0-1.0), e.g.:
#   * \code
#   * SampleConsensusModelNormalPlane<pcl::PointXYZ, pcl::Normal> sac_model;
#   * ...
#   * sac_model.setNormalDistanceWeight (0.1);
#   * ...
#   * \endcode
#   * \author Radu B. Rusu and Jared Glover
#   * \ingroup sample_consensus
#   */
# template <typename PointT, typename PointNT>
# class SampleConsensusModelNormalPlane : public SampleConsensusModelPlane<PointT>, public SampleConsensusModelFromNormals<PointT, PointNT>
cdef extern from "pcl/sample_consensus/sac_model_normal_plane.h" namespace "pcl":
    # cdef cppclass SampleConsensusModelNormalPlane[PointT, PointNT](SampleConsensusModelPlane[PointT])(SampleConsensusModelFromNormals[PointT, PointNT]):
    cdef cppclass SampleConsensusModelNormalPlane[PointT, PointNT]:
        SampleConsensusModelNormalPlane()
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
        # using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
        # public:
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNPtr PointCloudNPtr;
        # typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNConstPtr PointCloudNConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelNormalPlane> Ptr;
        # 
        # /** \brief Constructor for base SampleConsensusModelNormalPlane.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelNormalPlane (const PointCloudConstPtr &cloud) : SampleConsensusModelPlane<PointT> (cloud)
        # 
        # /** \brief Constructor for base SampleConsensusModelNormalPlane.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelNormalPlane (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : SampleConsensusModelPlane<PointT> (cloud, indices)
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers. 
        # * 
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Compute all distances from the cloud data to a given plane model.
        # * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
        # * \param[out] distances the resultant estimated distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_NORMAL_PLANE). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_NORMAL_PLANE); }


ctypedef SampleConsensusModelNormalPlane[cpp.PointXYZ, cpp.Normal] SampleConsensusModelNormalPlane_t
ctypedef SampleConsensusModelNormalPlane[cpp.PointXYZI, cpp.Normal] SampleConsensusModelNormalPlane_PointXYZI_t
ctypedef SampleConsensusModelNormalPlane[cpp.PointXYZRGB, cpp.Normal] SampleConsensusModelNormalPlane_PointXYZRGB_t
ctypedef SampleConsensusModelNormalPlane[cpp.PointXYZRGBA, cpp.Normal] SampleConsensusModelNormalPlane_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelNormalPlane[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelNormalPlanePtr_t
ctypedef shared_ptr[SampleConsensusModelNormalPlane[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelNormalPlane_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelNormalPlane[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelNormalPlane_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelNormalPlane[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelNormalPlane_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelNormalPlane[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelNormalPlaneConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelNormalPlane[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelNormalPlane_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelNormalPlane[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelNormalPlane_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelNormalPlane[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelNormalPlane_PointXYZRGBA_ConstPtr_t
###

# sac_model_normal_sphere.h
# namespace pcl
# /** \brief @b SampleConsensusModelNormalSphere defines a model for 3D sphere
#   * segmentation using additional surface normal constraints. Basically this
#   * means that checking for inliers will not only involve a "distance to
#   * model" criterion, but also an additional "maximum angular deviation"
#   * between the sphere's normal and the inlier points normals.
#   * The model coefficients are defined as:
#   * <ul>
#   * <li><b>a</b> : the X coordinate of the plane's normal (normalized)
#   * <li><b>b</b> : the Y coordinate of the plane's normal (normalized)
#   * <li><b>c</b> : the Z coordinate of the plane's normal (normalized)
#   * <li><b>d</b> : radius of the sphere
#   * </ul>
#   * \author Stefan Schrandt
#   * \ingroup sample_consensus
#   */
# template <typename PointT, typename PointNT>
# class SampleConsensusModelNormalSphere : public SampleConsensusModelSphere<PointT>, public SampleConsensusModelFromNormals<PointT, PointNT>
cdef extern from "pcl/sample_consensus/sac_model_normal_sphere.h" namespace "pcl":
    # cdef cppclass SampleConsensusModelNormalSphere[PointT, PointNT](SampleConsensusModelSphere[PointT])(SampleConsensusModelFromNormals[PointT, PointNT]):
    cdef cppclass SampleConsensusModelNormalSphere[PointT, PointNT]:
        SampleConsensusModelNormalSphere()
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # using SampleConsensusModel<PointT>::radius_min_;
        # using SampleConsensusModel<PointT>::radius_max_;
        # using SampleConsensusModelFromNormals<PointT, PointNT>::normals_;
        # using SampleConsensusModelFromNormals<PointT, PointNT>::normal_distance_weight_;
        # public:
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNPtr PointCloudNPtr;
        # typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::PointCloudNConstPtr PointCloudNConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelNormalSphere> Ptr;
        # 
        # /** \brief Constructor for base SampleConsensusModelNormalSphere.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelNormalSphere (const PointCloudConstPtr &cloud) : SampleConsensusModelSphere<PointT> (cloud)
        # 
        # /** \brief Constructor for base SampleConsensusModelNormalSphere.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelNormalSphere (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : SampleConsensusModelSphere<PointT> (cloud, indices)
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a sphere model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers. 
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Compute all distances from the cloud data to a given sphere model.
        # * \param[in] model_coefficients the coefficients of a sphere model that we need to compute distances to
        # * \param[out] distances the resultant estimated distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_NORMAL_SPHERE). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_NORMAL_SPHERE); }


ctypedef SampleConsensusModelNormalSphere[cpp.PointXYZ, cpp.Normal] SampleConsensusModelNormalSphere_t
ctypedef SampleConsensusModelNormalSphere[cpp.PointXYZI, cpp.Normal] SampleConsensusModelNormalSphere_PointXYZI_t
ctypedef SampleConsensusModelNormalSphere[cpp.PointXYZRGB, cpp.Normal] SampleConsensusModelNormalSphere_PointXYZRGB_t
ctypedef SampleConsensusModelNormalSphere[cpp.PointXYZRGBA, cpp.Normal] SampleConsensusModelNormalSphere_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelNormalSphere[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelNormalSpherePtr_t
ctypedef shared_ptr[SampleConsensusModelNormalSphere[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelNormalSphere_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelNormalSphere[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelNormalSphere_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelNormalSphere[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelNormalSphere_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelNormalSphere[cpp.PointXYZ, cpp.Normal]] SampleConsensusModelNormalSphereConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelNormalSphere[cpp.PointXYZI, cpp.Normal]] SampleConsensusModelNormalSphere_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelNormalSphere[cpp.PointXYZRGB, cpp.Normal]] SampleConsensusModelNormalSphere_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelNormalSphere[cpp.PointXYZRGBA, cpp.Normal]] SampleConsensusModelNormalSphere_PointXYZRGBA_ConstPtr_t
###

# sac_model_parallel_line.h
# namespace pcl
# /** \brief SampleConsensusModelParallelLine defines a model for 3D line segmentation using additional angular
#   * constraints.
#   * The model coefficients are defined as:
#   *   - \b point_on_line.x  : the X coordinate of a point on the line
#   *   - \b point_on_line.y  : the Y coordinate of a point on the line
#   *   - \b point_on_line.z  : the Z coordinate of a point on the line
#   *   - \b line_direction.x : the X coordinate of a line's direction
#   *   - \b line_direction.y : the Y coordinate of a line's direction
#   *   - \b line_direction.z : the Z coordinate of a line's direction
#   * \author Radu B. Rusu
#   * \ingroup sample_consensus
#   */
# template <typename PointT>
# class SampleConsensusModelParallelLine : public SampleConsensusModelLine<PointT>
cdef extern from "pcl/sample_consensus/sac_model_parallel_line.h" namespace "pcl":
    # cdef cppclass SampleConsensusModelParallelLine[PointT](SampleConsensusModelLine[PointT]):
    cdef cppclass SampleConsensusModelParallelLine[PointT]:
        SampleConsensusModelParallelLine()
        # public:
        # typedef typename SampleConsensusModelLine<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModelLine<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModelLine<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelParallelLine> Ptr;
        # /** \brief Constructor for base SampleConsensusModelParallelLine.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelParallelLine (const PointCloudConstPtr &cloud) : 
        # SampleConsensusModelLine<PointT> (cloud),
        # axis_ (Eigen::Vector3f::Zero ()),
        # eps_angle_ (0.0)
        # 
        # /** \brief Constructor for base SampleConsensusModelParallelLine.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelParallelLine (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
        # SampleConsensusModelLine<PointT> (cloud, indices),
        # axis_ (Eigen::Vector3f::Zero ()),
        # eps_angle_ (0.0)
        # 
        # /** \brief Set the axis along which we need to search for a plane perpendicular to.
        # * \param[in] ax the axis along which we need to search for a plane perpendicular to
        # */
        # inline void setAxis (const Eigen::Vector3f &ax) { axis_ = ax; axis_.normalize (); }
        # 
        # /** \brief Get the axis along which we need to search for a plane perpendicular to. */
        # inline Eigen::Vector3f getAxis ()  { return (axis_); }
        # 
        # /** \brief Set the angle epsilon (delta) threshold.
        # * \param[in] ea the maximum allowed difference between the plane normal and the given axis.
        # */
        # inline void setEpsAngle (const double ea) { eps_angle_ = ea; }
        # 
        # /** \brief Get the angle epsilon (delta) threshold. */
        # inline double getEpsAngle () { return (eps_angle_); }
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a line model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers. 
        # * 
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Compute all squared distances from the cloud data to a given line model.
        # * \param[in] model_coefficients the coefficients of a line model that we need to compute distances to
        # * \param[out] distances the resultant estimated squared distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_PARALLEL_LINE). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_PARALLEL_LINE); }


ctypedef SampleConsensusModelParallelLine[cpp.PointXYZ] SampleConsensusModelParallelLine_t
ctypedef SampleConsensusModelParallelLine[cpp.PointXYZI] SampleConsensusModelParallelLine_PointXYZI_t
ctypedef SampleConsensusModelParallelLine[cpp.PointXYZRGB] SampleConsensusModelParallelLine_PointXYZRGB_t
ctypedef SampleConsensusModelParallelLine[cpp.PointXYZRGBA] SampleConsensusModelParallelLine_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelParallelLine[cpp.PointXYZ]] SampleConsensusModelParallelLinePtr_t
ctypedef shared_ptr[SampleConsensusModelParallelLine[cpp.PointXYZI]] SampleConsensusModelParallelLine_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelParallelLine[cpp.PointXYZRGB]] SampleConsensusModelParallelLine_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelParallelLine[cpp.PointXYZRGBA]] SampleConsensusModelParallelLine_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelParallelLine[cpp.PointXYZ]] SampleConsensusModelParallelLineConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelParallelLine[cpp.PointXYZI]] SampleConsensusModelParallelLine_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelParallelLine[cpp.PointXYZRGB]] SampleConsensusModelParallelLine_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelParallelLine[cpp.PointXYZRGBA]] SampleConsensusModelParallelLine_PointXYZRGBA_ConstPtr_t
###

# sac_model_parallel_plane.h
# namespace pcl
# /** \brief @b SampleConsensusModelParallelPlane defines a model for 3D plane segmentation using additional
#   * angular constraints. The plane must be parallel to a user-specified axis
#   * (\ref setAxis) within an user-specified angle threshold (\ref setEpsAngle).
#   * Code example for a plane model, parallel (within a 15 degrees tolerance) with the Z axis:
#   * \code
#   * SampleConsensusModelParallelPlane<pcl::PointXYZ> model (cloud);
#   * model.setAxis (Eigen::Vector3f (0.0, 0.0, 1.0));
#   * model.setEpsAngle (pcl::deg2rad (15));
#   * \endcode
#   * \note Please remember that you need to specify an angle > 0 in order to activate the axis-angle constraint!
#   * \author Radu Bogdan Rusu, Nico Blodow
#   * \ingroup sample_consensus
#   */
# template <typename PointT>
# class SampleConsensusModelParallelPlane : public SampleConsensusModelPlane<PointT>
cdef extern from "pcl/sample_consensus/sac_model_parallel_plane.h" namespace "pcl":
    cdef cppclass SampleConsensusModelParallelPlane[PointT](SampleConsensusModelPlane[PointT]):
        SampleConsensusModelParallelLine()
        # public:
        # typedef typename SampleConsensusModelPlane<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModelPlane<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModelPlane<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelParallelPlane> Ptr;
        # 
        # /** \brief Constructor for base SampleConsensusModelParallelPlane.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelParallelPlane (const PointCloudConstPtr &cloud) : 
        # SampleConsensusModelPlane<PointT> (cloud),
        # axis_ (Eigen::Vector3f::Zero ()),
        # eps_angle_ (0.0), sin_angle_ (-1.0)
        # 
        # /** \brief Constructor for base SampleConsensusModelParallelPlane.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelParallelPlane (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
        # SampleConsensusModelPlane<PointT> (cloud, indices),
        # axis_ (Eigen::Vector3f::Zero ()),
        # eps_angle_ (0.0), sin_angle_ (-1.0)
        # 
        # /** \brief Set the axis along which we need to search for a plane perpendicular to.
        # * \param[in] ax the axis along which we need to search for a plane perpendicular to
        # */
        # inline void setAxis (const Eigen::Vector3f &ax) { axis_ = ax; }
        # 
        # /** \brief Get the axis along which we need to search for a plane perpendicular to. */
        # inline Eigen::Vector3f getAxis ()  { return (axis_); }
        # 
        # /** \brief Set the angle epsilon (delta) threshold.
        # * \param[in] ea the maximum allowed difference between the plane normal and the given axis.
        # * \note You need to specify an angle > 0 in order to activate the axis-angle constraint!
        # */
        # inline void setEpsAngle (const double ea) { eps_angle_ = ea; sin_angle_ = fabs (sin (ea));}
        # 
        # /** \brief Get the angle epsilon (delta) threshold. */
        # inline double getEpsAngle () { return (eps_angle_); }
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Compute all distances from the cloud data to a given plane model.
        # * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
        # * \param[out] distances the resultant estimated distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_PARALLEL_PLANE). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_PARALLEL_PLANE); }


ctypedef SampleConsensusModelParallelPlane[cpp.PointXYZ] SampleConsensusModelParallelPlane_t
ctypedef SampleConsensusModelParallelPlane[cpp.PointXYZI] SampleConsensusModelParallelPlane_PointXYZI_t
ctypedef SampleConsensusModelParallelPlane[cpp.PointXYZRGB] SampleConsensusModelParallelPlane_PointXYZRGB_t
ctypedef SampleConsensusModelParallelPlane[cpp.PointXYZRGBA] SampleConsensusModelParallelPlane_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelParallelPlane[cpp.PointXYZ]] SampleConsensusModelParallelPlanePtr_t
ctypedef shared_ptr[SampleConsensusModelParallelPlane[cpp.PointXYZI]] SampleConsensusModelParallelPlane_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelParallelPlane[cpp.PointXYZRGB]] SampleConsensusModelParallelPlane_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelParallelPlane[cpp.PointXYZRGBA]] SampleConsensusModelParallelPlane_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelParallelPlane[cpp.PointXYZ]] SampleConsensusModelParallelPlaneConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelParallelPlane[cpp.PointXYZI]] SampleConsensusModelParallelPlane_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelParallelPlane[cpp.PointXYZRGB]] SampleConsensusModelParallelPlane_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelParallelPlane[cpp.PointXYZRGBA]] SampleConsensusModelParallelPlane_PointXYZRGBA_ConstPtr_t
###

# sac_model_perpendicular_plane.h
# namespace pcl
# /** \brief SampleConsensusModelPerpendicularPlane defines a model for 3D plane segmentation using additional
#   * angular constraints. The plane must be perpendicular to an user-specified axis (\ref setAxis), up to an user-specified angle threshold (\ref setEpsAngle).
#   * The model coefficients are defined as:
#   *   - \b a : the X coordinate of the plane's normal (normalized)
#   *   - \b b : the Y coordinate of the plane's normal (normalized)
#   *   - \b c : the Z coordinate of the plane's normal (normalized)
#   *   - \b d : the fourth <a href="http://mathworld.wolfram.com/HessianNormalForm.html">Hessian component</a> of the plane's equation
#   * Code example for a plane model, perpendicular (within a 15 degrees tolerance) with the Z axis:
#   * \code
#   * SampleConsensusModelPerpendicularPlane<pcl::PointXYZ> model (cloud);
#   * model.setAxis (Eigen::Vector3f (0.0, 0.0, 1.0));
#   * model.setEpsAngle (pcl::deg2rad (15));
#   * \endcode
#   * \note Please remember that you need to specify an angle > 0 in order to activate the axis-angle constraint!
#   * \author Radu B. Rusu
#   * \ingroup sample_consensus
#   */
# template <typename PointT>
# class SampleConsensusModelPerpendicularPlane : public SampleConsensusModelPlane<PointT>
cdef extern from "pcl/sample_consensus/sac_model_perpendicular_plane.h" namespace "pcl":
    cdef cppclass SampleConsensusModelPerpendicularPlane[PointT](SampleConsensusModelPlane[PointT]):
        SampleConsensusModelPerpendicularPlane()
        # public:
        # typedef typename SampleConsensusModelPlane<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModelPlane<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModelPlane<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelPerpendicularPlane> Ptr;
        # 
        # /** \brief Constructor for base SampleConsensusModelPerpendicularPlane.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelPerpendicularPlane (const PointCloudConstPtr &cloud) : 
        # SampleConsensusModelPlane<PointT> (cloud), 
        # axis_ (Eigen::Vector3f::Zero ()),
        # eps_angle_ (0.0)
        # 
        # /** \brief Constructor for base SampleConsensusModelPerpendicularPlane.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelPerpendicularPlane (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : 
        # SampleConsensusModelPlane<PointT> (cloud, indices), 
        # axis_ (Eigen::Vector3f::Zero ()),
        # eps_angle_ (0.0)
        # 
        # /** \brief Set the axis along which we need to search for a plane perpendicular to.
        # * \param[in] ax the axis along which we need to search for a plane perpendicular to
        # */
        # inline void setAxis (const Eigen::Vector3f &ax) { axis_ = ax; }
        # 
        # /** \brief Get the axis along which we need to search for a plane perpendicular to. */
        # inline Eigen::Vector3f getAxis ()  { return (axis_); }
        # 
        # /** \brief Set the angle epsilon (delta) threshold.
        # * \param[in] ea the maximum allowed difference between the plane normal and the given axis.
        # * \note You need to specify an angle > 0 in order to activate the axis-angle constraint!
        # */
        # inline void setEpsAngle (const double ea) { eps_angle_ = ea; }
        # 
        # /** \brief Get the angle epsilon (delta) threshold. */
        # inline double getEpsAngle () { return (eps_angle_); }
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers. 
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Compute all distances from the cloud data to a given plane model.
        # * \param[in] model_coefficients the coefficients of a plane model that we need to compute distances to
        # * \param[out] distances the resultant estimated distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_PERPENDICULAR_PLANE). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_PERPENDICULAR_PLANE); }


ctypedef SampleConsensusModelPerpendicularPlane[cpp.PointXYZ] SampleConsensusModelPerpendicularPlane_t
ctypedef SampleConsensusModelPerpendicularPlane[cpp.PointXYZI] SampleConsensusModelPerpendicularPlane_PointXYZI_t
ctypedef SampleConsensusModelPerpendicularPlane[cpp.PointXYZRGB] SampleConsensusModelPerpendicularPlane_PointXYZRGB_t
ctypedef SampleConsensusModelPerpendicularPlane[cpp.PointXYZRGBA] SampleConsensusModelPerpendicularPlane_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelPerpendicularPlane[cpp.PointXYZ]] SampleConsensusModelPerpendicularPlanePtr_t
ctypedef shared_ptr[SampleConsensusModelPerpendicularPlane[cpp.PointXYZI]] SampleConsensusModelPerpendicularPlane_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelPerpendicularPlane[cpp.PointXYZRGB]] SampleConsensusModelPerpendicularPlane_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelPerpendicularPlane[cpp.PointXYZRGBA]] SampleConsensusModelPerpendicularPlane_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelPerpendicularPlane[cpp.PointXYZ]] SampleConsensusModelPerpendicularPlaneConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelPerpendicularPlane[cpp.PointXYZI]] SampleConsensusModelPerpendicularPlane_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelPerpendicularPlane[cpp.PointXYZRGB]] SampleConsensusModelPerpendicularPlane_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelPerpendicularPlane[cpp.PointXYZRGBA]] SampleConsensusModelPerpendicularPlane_PointXYZRGBA_ConstPtr_t
###

# sac_model_registration.h
# namespace pcl
# /** \brief SampleConsensusModelRegistration defines a model for Point-To-Point registration outlier rejection.
#   * \author Radu Bogdan Rusu
#   * \ingroup sample_consensus
#   */
# template <typename PointT>
# class SampleConsensusModelRegistration : public SampleConsensusModel<PointT>
cdef extern from "pcl/sample_consensus/sac_model_registration.h" namespace "pcl":
    cdef cppclass SampleConsensusModelRegistration[PointT](SampleConsensusModel[PointT]):
        SampleConsensusModelRegistration()
        SampleConsensusModelRegistration(shared_ptr[cpp.PointCloud[PointT]] cloud)
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # public:
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelRegistration> Ptr;
        # 
        # /** \brief Constructor for base SampleConsensusModelRegistration.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelRegistration (const PointCloudConstPtr &cloud) : 
        # SampleConsensusModel<PointT> (cloud),
        # target_ (),
        # indices_tgt_ (),
        # correspondences_ (),
        # sample_dist_thresh_ (0)
        # 
        # /** \brief Constructor for base SampleConsensusModelRegistration.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelRegistration (const PointCloudConstPtr &cloud, const std::vector<int> &indices) :
        # SampleConsensusModel<PointT> (cloud, indices),
        # target_ (),
        # indices_tgt_ (),
        # correspondences_ (),
        # sample_dist_thresh_ (0)
        # 
        # /** \brief Provide a pointer to the input dataset
        # * \param[in] cloud the const boost shared pointer to a PointCloud message
        # */
        # inline virtual void setInputCloud (const PointCloudConstPtr &cloud)
        # 
        # /** \brief Set the input point cloud target.
        # * \param target the input point cloud target
        # */
        # inline void setInputTarget (const PointCloudConstPtr &target)
        # 
        # /** \brief Set the input point cloud target.
        # * \param[in] target the input point cloud target
        # * \param[in] indices_tgt a vector of point indices to be used from \a target
        # */
        # inline void setInputTarget (const PointCloudConstPtr &target, const std::vector<int> &indices_tgt)
        # 
        # /** \brief Compute a 4x4 rigid transformation matrix from the samples given
        # * \param[in] samples the indices found as good candidates for creating a valid model
        # * \param[out] model_coefficients the resultant model coefficients
        # */
        # bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);
        # 
        # /** \brief Compute all distances from the transformed points to their correspondences
        # * \param[in] model_coefficients the 4x4 transformation matrix
        # * \param[out] distances the resultant estimated distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the 4x4 transformation matrix
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Recompute the 4x4 transformation using the given inlier set
        # * \param[in] inliers the data inliers found as supporting the model
        # * \param[in] model_coefficients the initial guess for the optimization
        # * \param[out] optimized_coefficients the resultant recomputed transformation
        # */
        # void optimizeModelCoefficients (const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients);
        # 
        # void projectPoints (const std::vector<int> &, const Eigen::VectorXf &, PointCloud &, bool = true)
        # 
        # bool doSamplesVerifyModel (const std::set<int> &, const Eigen::VectorXf &, const double)
        # 
        # /** \brief Return an unique id for this model (SACMODEL_REGISTRATION). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_REGISTRATION); }


ctypedef SampleConsensusModelRegistration[cpp.PointXYZ] SampleConsensusModelRegistration_t
ctypedef SampleConsensusModelRegistration[cpp.PointXYZI] SampleConsensusModelRegistration_PointXYZI_t
ctypedef SampleConsensusModelRegistration[cpp.PointXYZRGB] SampleConsensusModelRegistration_PointXYZRGB_t
ctypedef SampleConsensusModelRegistration[cpp.PointXYZRGBA] SampleConsensusModelRegistration_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelRegistration[cpp.PointXYZ]] SampleConsensusModelRegistrationPtr_t
ctypedef shared_ptr[SampleConsensusModelRegistration[cpp.PointXYZI]] SampleConsensusModelRegistration_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelRegistration[cpp.PointXYZRGB]] SampleConsensusModelRegistration_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelRegistration[cpp.PointXYZRGBA]] SampleConsensusModelRegistration_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelRegistration[cpp.PointXYZ]] SampleConsensusModelRegistrationConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelRegistration[cpp.PointXYZI]] SampleConsensusModelRegistration_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelRegistration[cpp.PointXYZRGB]] SampleConsensusModelRegistration_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelRegistration[cpp.PointXYZRGBA]] SampleConsensusModelRegistration_PointXYZRGBA_ConstPtr_t
###

# sac_model_stick.h
# namespace pcl
# /** \brief SampleConsensusModelStick defines a model for 3D stick segmentation. 
#   * A stick is a line with an user given minimum/maximum width.
#   * The model coefficients are defined as:
#   *   - \b point_on_line.x  : the X coordinate of a point on the line
#   *   - \b point_on_line.y  : the Y coordinate of a point on the line
#   *   - \b point_on_line.z  : the Z coordinate of a point on the line
#   *   - \b line_direction.x : the X coordinate of a line's direction
#   *   - \b line_direction.y : the Y coordinate of a line's direction
#   *   - \b line_direction.z : the Z coordinate of a line's direction
#   *   - \b line_width       : the width of the line
#   * \author Radu B. Rusu
#   * \ingroup sample_consensus
#   */
# template <typename PointT>
# class SampleConsensusModelStick : public SampleConsensusModel<PointT>
cdef extern from "pcl/sample_consensus/sac_model_stick.h" namespace "pcl":
    cdef cppclass SampleConsensusModelStick[PointT](SampleConsensusModel[PointT]):
        SampleConsensusModelStick()
        SampleConsensusModelStick(shared_ptr[cpp.PointCloud[PointT]] cloud)
        # using SampleConsensusModel<PointT>::input_;
        # using SampleConsensusModel<PointT>::indices_;
        # using SampleConsensusModel<PointT>::radius_min_;
        # using SampleConsensusModel<PointT>::radius_max_;
        # public:
        # typedef typename SampleConsensusModel<PointT>::PointCloud PointCloud;
        # typedef typename SampleConsensusModel<PointT>::PointCloudPtr PointCloudPtr;
        # typedef typename SampleConsensusModel<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<SampleConsensusModelStick> Ptr;
        # 
        # /** \brief Constructor for base SampleConsensusModelStick.
        # * \param[in] cloud the input point cloud dataset
        # */
        # SampleConsensusModelStick (const PointCloudConstPtr &cloud) : SampleConsensusModel<PointT> (cloud) {};
        # 
        # /** \brief Constructor for base SampleConsensusModelStick.
        # * \param[in] cloud the input point cloud dataset
        # * \param[in] indices a vector of point indices to be used from \a cloud
        # */
        # SampleConsensusModelStick (const PointCloudConstPtr &cloud, const std::vector<int> &indices) : SampleConsensusModel<PointT> (cloud, indices) {};
        # 
        # /** \brief Check whether the given index samples can form a valid stick model, compute the model coefficients from
        # * these samples and store them internally in model_coefficients_. The stick coefficients are represented by a
        # * point and a line direction
        # * \param[in] samples the point indices found as possible good candidates for creating a valid model
        # * \param[out] model_coefficients the resultant model coefficients
        # */
        # bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);
        # 
        # /** \brief Compute all squared distances from the cloud data to a given stick model.
        # * \param[in] model_coefficients the coefficients of a stick model that we need to compute distances to
        # * \param[out] distances the resultant estimated squared distances
        # */
        # void getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);
        # 
        # /** \brief Select all the points which respect the given model coefficients as inliers.
        # * \param[in] model_coefficients the coefficients of a stick model that we need to compute distances to
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # * \param[out] inliers the resultant model inliers
        # */
        # void selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers);
        # 
        # /** \brief Count all the points which respect the given model coefficients as inliers. 
        # * 
        # * \param[in] model_coefficients the coefficients of a model that we need to compute distances to
        # * \param[in] threshold maximum admissible distance threshold for determining the inliers from the outliers
        # * \return the resultant number of inliers
        # */
        # virtual int countWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Recompute the stick coefficients using the given inlier set and return them to the user.
        # * @note: these are the coefficients of the stick model after refinement (eg. after SVD)
        # * \param[in] inliers the data inliers found as supporting the model
        # * \param[in] model_coefficients the initial guess for the model coefficients
        # * \param[out] optimized_coefficients the resultant recomputed coefficients after optimization
        # */
        # void optimizeModelCoefficients (const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients);
        # 
        # /** \brief Create a new point cloud with inliers projected onto the stick model.
        # * \param[in] inliers the data inliers that we want to project on the stick model
        # * \param[in] model_coefficients the *normalized* coefficients of a stick model
        # * \param[out] projected_points the resultant projected points
        # * \param[in] copy_data_fields set to true if we need to copy the other data fields
        # */
        # void projectPoints (const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, PointCloud &projected_points, bool copy_data_fields = true);
        # 
        # /** \brief Verify whether a subset of indices verifies the given stick model coefficients.
        # * \param[in] indices the data indices that need to be tested against the plane model
        # * \param[in] model_coefficients the plane model coefficients
        # * \param[in] threshold a maximum admissible distance threshold for determining the inliers from the outliers
        # */
        # bool doSamplesVerifyModel (const std::set<int> &indices, const Eigen::VectorXf &model_coefficients, const double threshold);
        # 
        # /** \brief Return an unique id for this model (SACMODEL_STACK). */
        # inline pcl::SacModel getModelType () const { return (SACMODEL_STICK); }


ctypedef SampleConsensusModelStick[cpp.PointXYZ] SampleConsensusModelStick_t
ctypedef SampleConsensusModelStick[cpp.PointXYZI] SampleConsensusModelStick_PointXYZI_t
ctypedef SampleConsensusModelStick[cpp.PointXYZRGB] SampleConsensusModelStick_PointXYZRGB_t
ctypedef SampleConsensusModelStick[cpp.PointXYZRGBA] SampleConsensusModelStick_PointXYZRGBA_t
ctypedef shared_ptr[SampleConsensusModelStick[cpp.PointXYZ]] SampleConsensusModelStickPtr_t
ctypedef shared_ptr[SampleConsensusModelStick[cpp.PointXYZI]] SampleConsensusModelStick_PointXYZI_Ptr_t
ctypedef shared_ptr[SampleConsensusModelStick[cpp.PointXYZRGB]] SampleConsensusModelStick_PointXYZRGB_Ptr_t
ctypedef shared_ptr[SampleConsensusModelStick[cpp.PointXYZRGBA]] SampleConsensusModelStick_PointXYZRGBA_Ptr_t
ctypedef shared_ptr[const SampleConsensusModelStick[cpp.PointXYZ]] SampleConsensusModelStickConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelStick[cpp.PointXYZI]] SampleConsensusModelStick_PointXYZI_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelStick[cpp.PointXYZRGB]] SampleConsensusModelStick_PointXYZRGB_ConstPtr_t
ctypedef shared_ptr[const SampleConsensusModelStick[cpp.PointXYZRGBA]] SampleConsensusModelStick_PointXYZRGBA_ConstPtr_t
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
###

# cdef extern from "pcl/sample_consensus/rransac.h" namespace "pcl":
#     cdef cppclass Functor[_Scalar]:
#         # enum 
#         # {
#         #   InputsAtCompileTime = NX,
#         #   ValuesAtCompileTime = NY
#         # };


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
#   const static std::map<pcl::SacModel, unsigned int> SAC_SAMPLE_SIZE (sample_size_pairs, sample_size_pairs + sizeof (sample_size_pairs) / sizeof (SampleSizeModel));
# }
###

###############################################################################
# Activation
###############################################################################

