# -*- coding: utf-8 -*-

from libcpp cimport bool
from libcpp.string cimport string

# main
cimport pcl_defs as cpp
from boost_shared_ptr cimport shared_ptr

# Cython - limits.pxd
# from libcpp cimport numeric_limits

# base
from eigen cimport Matrix4f

# registration.h
# Version 1.7.2
# cdef extern from "pcl/registration/registration.h" namespace "pcl" nogil:
#     cdef cppclass Registration[Source, Target]:
#         cppclass Matrix4:
#             float *data()
#         void align(cpp.PointCloud[Source] &) except +
#         Matrix4f getFinalTransformation() except +
#         double getFitnessScore() except +
#         bool hasConverged() except +
#         void setInputSource(cpp.PointCloudPtr_t) except +
#         void setInputTarget(cpp.PointCloudPtr_t) except +
#         void setMaximumIterations(int) except +
###

# template <typename PointSource, typename PointTarget>
# class Registration : public PCLBase<PointSource>
# cdef cppclass Registration[Source, Target](cpp.PCLBase[Source]):
cdef extern from "pcl/registration/registration.h" namespace "pcl" nogil:
    cdef cppclass Registration[Source, Target]:
        # Registration()
        # override?
        void setInputCloud(cpp.PointCloudPtr_t ptcloud) except +
        # void setInputSource(cpp.PointCloudPtr_t ptcloud) except +
        # cdef void setInputSource(cpp.PointCloudPtr_t ptcloud) except +
        #   cpp.PCLBase.setInputSource(ptcloud)
        # void setInputSource(cpp.PointCloudPtr2_t pt2cloud) except +
        # public:
        # using PCLBase<PointSource>::initCompute;
        # using PCLBase<PointSource>::deinitCompute;
        # using PCLBase<PointSource>::input_;
        # using PCLBase<PointSource>::indices_;
        # ctypedef boost::shared_ptr< Registration<PointSource, PointTarget> > Ptr;
        # ctypedef boost::shared_ptr< const Registration<PointSource, PointTarget> > ConstPtr;
        # ctypedef typename pcl::KdTree<PointTarget> KdTree;
        # ctypedef typename pcl::KdTree<PointTarget>::Ptr KdTreePtr;
        # ctypedef pcl::PointCloud<PointSource> PointCloudSource;
        # ctypedef typename PointCloudSource::Ptr PointCloudSourcePtr;
        # ctypedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;
        # ctypedef pcl::PointCloud<PointTarget> PointCloudTarget;
        # ctypedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
        # ctypedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;
        # ctypedef typename KdTree::PointRepresentationConstPtr PointRepresentationConstPtr;
        # ctypedef typename pcl::registration::TransformationEstimation<PointSource, PointTarget> TransformationEstimation;
        # ctypedef typename TransformationEstimation::Ptr TransformationEstimationPtr;
        # ctypedef typename TransformationEstimation::ConstPtr TransformationEstimationConstPtr;
        # brief Provide a pointer to the transformation estimation object. (e.g., SVD, point to plane etc.) 
        # param te is the pointer to the corresponding transformation estimation object
        # void setTransformationEstimation (const TransformationEstimationPtr &te)
        # virtual inline void setInputTarget (const PointCloudTargetConstPtr &cloud);
        void setInputTarget(cpp.PointCloudPtr_t ptcloud) except +
        void setInputTarget2(cpp.PointCloudPtr_t pt2cloud) except +
        # /** \brief Get a pointer to the input point cloud dataset target. */
        # inline PointCloudTargetConstPtr const getInputTarget ()
        cpp.PointCloudPtr_t getInputTarget ()

        # brief Get the final transformation matrix estimated by the registration method.
        Matrix4f getFinalTransformation ()
        # /** \brief Get the last incremental transformation matrix estimated by the registration method. */
        Matrix4f getLastIncrementalTransformation ()
        # Set the maximum number of iterations the internal optimization should run for.
        # param nr_iterations the maximum number of iterations the internal optimization should run for
        void setMaximumIterations (int nr_iterations) except +
        # /** \brief Get the maximum number of iterations the internal optimization should run for, as set by the user. */
        int getMaximumIterations () 
        # /** \brief Set the number of iterations RANSAC should run for.
        #   * \param ransac_iterations is the number of iterations RANSAC should run for
        void setRANSACIterations (int ransac_iterations)
        # /** \brief Get the number of iterations RANSAC should run for, as set by the user. */
        double getRANSACIterations ()
        # brief Set the inlier distance threshold for the internal RANSAC outlier rejection loop.
        # The method considers a point to be an inlier, if the distance between the target data index and the transformed 
        # source index is smaller than the given inlier distance threshold. 
        # The value is set by default to 0.05m.
        # \param inlier_threshold the inlier distance threshold for the internal RANSAC outlier rejection loop
        void setRANSACOutlierRejectionThreshold (double inlier_threshold)
        # brief Get the inlier distance threshold for the internal outlier rejection loop as set by the user. */
        double getRANSACOutlierRejectionThreshold ()
        # brief Set the maximum distance threshold between two correspondent points in source <-> target. If the 
        #   distance is larger than this threshold, the points will be ignored in the alignment process.
        # param distance_threshold the maximum distance threshold between a point and its nearest neighbor 
        # correspondent in order to be considered in the alignment process
        void setMaxCorrespondenceDistance (double distance_threshold)
        # brief Get the maximum distance threshold between two correspondent points in source <-> target. If the 
        # distance is larger than this threshold, the points will be ignored in the alignment process.
        double getMaxCorrespondenceDistance ()
        # brief Set the transformation epsilon (maximum allowable difference between two consecutive 
        # transformations) in order for an optimization to be considered as having converged to the final 
        # solution.
        # param epsilon the transformation epsilon in order for an optimization to be considered as having 
        # converged to the final solution.
        void setTransformationEpsilon (double epsilon)
        # brief Get the transformation epsilon (maximum allowable difference between two consecutive 
        # transformations) as set by the user.
        double getTransformationEpsilon ()
        # brief Set the maximum allowed Euclidean error between two consecutive steps in the ICP loop, before 
        # the algorithm is considered to have converged. 
        # The error is estimated as the sum of the differences between correspondences in an Euclidean sense, 
        # divided by the number of correspondences.
        # param epsilon the maximum allowed distance error before the algorithm will be considered to have
        # converged
        void setEuclideanFitnessEpsilon (double epsilon)
        # brief Get the maximum allowed distance error before the algorithm will be considered to have converged,
        # as set by the user. See \ref setEuclideanFitnessEpsilon
        double getEuclideanFitnessEpsilon ()
        # brief Provide a boost shared pointer to the PointRepresentation to be used when comparing points
        # param point_representation the PointRepresentation to be used by the k-D tree
        # void setPointRepresentation (const PointRepresentationConstPtr &point_representation)
        # brief Register the user callback function which will be called from registration thread
        # in order to update point cloud obtained after each iteration
        # param[in] visualizerCallback reference of the user callback function
        # template<typename FunctionSignature>  bool
        # registerVisualizationCallback (boost::function<FunctionSignature> &visualizerCallback)
        # /** \brief Obtain the Euclidean fitness score (e.g., sum of squared distances from the source to the target)
        #   * \param max_range maximum allowable distance between a point and its correspondence in the target 
        #   * (default: double::max)
        #   */
        #  double getFitnessScore (double max_range = numeric_limits[double]::max ());
        double getFitnessScore() except +
        # /** \brief Obtain the Euclidean fitness score (e.g., sum of squared distances from the source to the target)
        #   * from two sets of correspondence distances (distances between source and target points)
        #   * \param[in] distances_a the first set of distances between correspondences
        #   * \param[in] distances_b the second set of distances between correspondences
        #   */
        #  double getFitnessScore (const std::vector<float> &distances_a, const std::vector<float> &distances_b);
        # /** \brief Return the state of convergence after the last align run */
        #  bool hasConverged ()
        bool hasConverged() except +
        # brief Call the registration algorithm which estimates the transformation and returns the transformed source 
        # (input) as \a output.
        # param output the resultant input transfomed point cloud dataset
        #  void align (PointCloudSource &output);
        void align(cpp.PointCloud[Source] &) except +
        # brief Call the registration algorithm which estimates the transformation and returns the transformed source 
        # (input) as \a output.
        # param output the resultant input transfomed point cloud dataset
        # param guess the initial gross estimation of the transformation
        # void align (PointCloudSource &output, const Eigen::Matrix4f& guess);
        # /** \brief Abstract class get name method. */
        const string getClassName ()
###

# warp_point_rigid.h
# template <class PointSourceT, class PointTargetT>
# class WarpPointRigid
cdef extern from "pcl/registration/warp_point_rigid.h" namespace "pcl" nogil:
    cdef cppclass WarpPointRigid[Source, Target]:
        WarpPointRigid (int nr_dim)
        # public:
        # virtual void 
        # setParam (const Eigen::VectorXf& p) = 0;
        # void warpPoint (const PointSourceT& pnt_in, PointSourceT& pnt_out) const
        # int getDimension () const {return nr_dim_;}
        # const Eigen::Matrix4f& 
        # getTransform () const { return transform_matrix_; }
        # protected:
        # int nr_dim_;
        # Eigen::Matrix4f transform_matrix_;

###

# correspondence_rejection.h
# class CorrespondenceRejector
cdef extern from "pcl/registration/correspondence_estimation_normal_shooting.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceRejector:
        CorrespondenceRejector()
#         /** \brief Provide a pointer to the vector of the input correspondences.
#           * \param[in] correspondences the const boost shared pointer to a correspondence vector
#         virtual inline void setInputCorrespondences (const CorrespondencesConstPtr &correspondences) 
#         /** \brief Get a pointer to the vector of the input correspondences.
#           * \return correspondences the const boost shared pointer to a correspondence vector
#         inline CorrespondencesConstPtr getInputCorrespondences ()
#         /** \brief Run correspondence rejection
#           * \param[out] correspondences Vector of correspondences that have not been rejected.
#         inline void getCorrespondences (pcl::Correspondences &correspondences)
#         /** \brief Get a list of valid correspondences after rejection from the original set of correspondences.
#           * Pure virtual. Compared to \a getCorrespondences this function is
#           * stateless, i.e., input correspondences do not need to be provided beforehand,
#           * but are directly provided in the function call.
#           * \param[in] original_correspondences the set of initial correspondences given
#           * \param[out] remaining_correspondences the resultant filtered set of remaining correspondences
#         virtual inline void 
#         getRemainingCorrespondences (const pcl::Correspondences& original_correspondences, 
#                                      pcl::Correspondences& remaining_correspondences) = 0;
#         /** \brief Determine the indices of query points of
#           * correspondences that have been rejected, i.e., the difference
#           * between the input correspondences (set via \a setInputCorrespondences)
#           * and the given correspondence vector.
#           * \param[in] correspondences Vector of correspondences after rejection
#           * \param[out] indices Vector of query point indices of those correspondences
#           * that have been rejected.
#           */
#         inline void 
#         getRejectedQueryIndices (const pcl::Correspondences &correspondences, 
#                                  std::vector<int>& indices)
# 
#     /** @b DataContainerInterface provides a generic interface for computing correspondence scores between correspondent
#       * points in the input and target clouds
#       * \ingroup registration
#       */
#     class DataContainerInterface
#     {
#       public:
#         virtual ~DataContainerInterface () {}
#         virtual double getCorrespondenceScore (int index) = 0;
#         virtual double getCorrespondenceScore (const pcl::Correspondence &) = 0;
#     };
# 
#     /** @b DataContainer is a container for the input and target point clouds and implements the interface 
#       * to compute correspondence scores between correspondent points in the input and target clouds
#       * \ingroup registration
#       */
#     template <typename PointT, typename NormalT=pcl::PointNormal>
#       class DataContainer : public DataContainerInterface
#     {
#       typedef typename pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
#       typedef typename pcl::KdTree<PointT>::Ptr KdTreePtr;
#       typedef typename pcl::PointCloud<NormalT>::ConstPtr NormalsPtr;
# 
#       public:
# 
#       /** \brief Empty constructor. */
#       DataContainer ()
# 
#       /** \brief Provide a source point cloud dataset (must contain XYZ
#        * data!), used to compute the correspondence distance.  
#        * \param[in] cloud a cloud containing XYZ data
#        */
#       inline void setInputCloud (const PointCloudConstPtr &cloud)
#       /** \brief Provide a target point cloud dataset (must contain XYZ
#        * data!), used to compute the correspondence distance.  
#        * \param[in] target a cloud containing XYZ data
#        */
#       inline void setInputTarget (const PointCloudConstPtr &target)
#       /** \brief Set the normals computed on the input point cloud
#         * \param[in] normals the normals computed for the input cloud
#         */
#       inline void setInputNormals (const NormalsPtr &normals)
#       /** \brief Set the normals computed on the target point cloud
#         * \param[in] normals the normals computed for the input cloud
#         */
#       inline void setTargetNormals (const NormalsPtr &normals)
#       /** \brief Get the normals computed on the input point cloud */
#       inline NormalsPtr getInputNormals ()
#       /** \brief Get the normals computed on the target point cloud */
#       inline NormalsPtr getTargetNormals ()
#       /** \brief Get the correspondence score for a point in the input cloud
#        *  \param[index] index of the point in the input cloud
#        */
#       inline double getCorrespondenceScore (int index)
#       /** \brief Get the correspondence score for a given pair of correspondent points
#        *  \param[corr] Correspondent points
#        */
#       inline double getCorrespondenceScore (const pcl::Correspondence &corr)
#       /** \brief Get the correspondence score for a given pair of correspondent points based on 
#         * the angle betweeen the normals. The normmals for the in put and target clouds must be 
#         * set before using this function
#         * \param[in] corr Correspondent points
#         */
#       double getCorrespondenceScoreFromNormals (const pcl::Correspondence &corr)
###

# Inheritance

###
# bfgs.h
# template< typename _Scalar >
# class PolynomialSolver<_Scalar,2> : public PolynomialSolverBase<_Scalar,2>
# cdef extern from "pcl/registration/bfgs.h" namespace "Eigen" nogil:
#		cdef cppclass PolynomialSolver[_Scalar, 2](PolynomialSolverBase[_Scalar, 2]):
#       PolynomialSolver (int nr_dim)
#     	public:
#       typedef PolynomialSolverBase<_Scalar,2>    PS_Base;
#       EIGEN_POLYNOMIAL_SOLVER_BASE_INHERITED_TYPES( PS_Base )
#     	public:
#       template< typename OtherPolynomial >
#       inline PolynomialSolver( const OtherPolynomial& poly, bool& hasRealRoot )
#       /** Computes the complex roots of a new polynomial. */
#       template< typename OtherPolynomial >
#       void compute( const OtherPolynomial& poly, bool& hasRealRoot)
#       template< typename OtherPolynomial >
#       void compute( const OtherPolynomial& poly)
# 

# template<typename _Scalar, int NX=Eigen::Dynamic>
# struct BFGSDummyFunctor
# cdef extern from "pcl/registration/bfgs.h" nogil:
#   cdef struct BFGSDummyFunctor[_Scalar, NX]:
#       BFGSDummyFunctor ()
#       BFGSDummyFunctor(int inputs)
#   	typedef _Scalar Scalar;
#   	enum { InputsAtCompileTime = NX };
#   	typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> VectorType;
#   	const int m_inputs;
#   	int inputs() const { return m_inputs; }
#   	virtual double operator() (const VectorType &x) = 0;
#   	virtual void  df(const VectorType &x, VectorType &df) = 0;
#   	virtual void fdf(const VectorType &x, Scalar &f, VectorType &df) = 0;
# 
# namespace BFGSSpace {
#   enum Status {
#     NegativeGradientEpsilon = -3,
#     NotStarted = -2,
#     Running = -1,
#     Success = 0,
#     NoProgress = 1
#   };
# }
# 
# /**
#  * BFGS stands for Broyden窶擢letcher窶敵oldfarb窶鉄hanno (BFGS) method for solving 
#  * unconstrained nonlinear optimization problems. 
#  * For further details please visit: http://en.wikipedia.org/wiki/BFGS_method
#  * The method provided here is almost similar to the one provided by GSL.
#  * It reproduces Fletcher's original algorithm in Practical Methods of Optimization
#  * algorithms : 2.6.2 and 2.6.4 and uses the same politics in GSL with cubic 
#  * interpolation whenever it is possible else falls to quadratic interpolation for 
#  * alpha parameter.
#  */
# template<typename FunctorType>
# class BFGS
# cdef extern from "pcl/registration/bfgs.h" nogil:
#     cdef cppclass BFGS[FunctorType]:
#         # BFGS (FunctorType &_functor) 
# public:
#   typedef typename FunctorType::Scalar Scalar;
#   typedef typename FunctorType::VectorType FVectorType;
# 
#   typedef Eigen::DenseIndex Index;
# 
#   struct Parameters {
#     Parameters()
#     : max_iters(400)
#       , bracket_iters(100)
#       , section_iters(100)
#       , rho(0.01)
#       , sigma(0.01)
#       , tau1(9)
#       , tau2(0.05)
#       , tau3(0.5)
#       , step_size(1)
#       , order(3) {}
#     Index max_iters;   // maximum number of function evaluation
#     Index bracket_iters;
#     Index section_iters;
#     Scalar rho;
#     Scalar sigma;
#     Scalar tau1;
#     Scalar tau2;
#     Scalar tau3;
#     Scalar step_size;
#     Index order;
#   };
# 
#   BFGSSpace::Status minimize(FVectorType &x);
#   BFGSSpace::Status minimizeInit(FVectorType &x);
#   BFGSSpace::Status minimizeOneStep(FVectorType &x);
#   BFGSSpace::Status testGradient(Scalar epsilon);
#   void resetParameters(void) { parameters = Parameters(); }
#   
#   Parameters parameters;
#   Scalar f;
#   FVectorType gradient;
# 
#
# template<typename FunctorType> void
# BFGS<FunctorType>::checkExtremum(const Eigen::Matrix<Scalar, 4, 1>& coefficients, Scalar x, Scalar& xmin, Scalar& fmin)
# 
# template<typename FunctorType> void
# BFGS<FunctorType>::moveTo(Scalar alpha)
# 
# template<typename FunctorType> typename BFGS<FunctorType>::Scalar
# BFGS<FunctorType>::slope()
# 
# template<typename FunctorType> typename BFGS<FunctorType>::Scalar
# BFGS<FunctorType>::applyF(Scalar alpha)
# 
# template<typename FunctorType> typename BFGS<FunctorType>::Scalar
# BFGS<FunctorType>::applyDF(Scalar alpha)
# 
# template<typename FunctorType> void
# BFGS<FunctorType>::applyFDF(Scalar alpha, Scalar& f, Scalar& df)
# 
# template<typename FunctorType> void
# BFGS<FunctorType>::updatePosition (Scalar alpha, FVectorType &x, Scalar &f, FVectorType &g)
#
# template<typename FunctorType> void
# BFGS<FunctorType>::changeDirection ()
# 
# template<typename FunctorType> BFGSSpace::Status
# BFGS<FunctorType>::minimize(FVectorType  &x)
# 
# template<typename FunctorType> BFGSSpace::Status
# BFGS<FunctorType>::minimizeInit(FVectorType  &x)
# 
# template<typename FunctorType> BFGSSpace::Status
# BFGS<FunctorType>::minimizeOneStep(FVectorType  &x)
# 
# template<typename FunctorType> typename BFGSSpace::Status 
# BFGS<FunctorType>::testGradient(Scalar epsilon)
# 
# template<typename FunctorType> typename BFGS<FunctorType>::Scalar 
# BFGS<FunctorType>::interpolate (Scalar a, Scalar fa, Scalar fpa,
#                                 Scalar b, Scalar fb, Scalar fpb, 
#                                 Scalar xmin, Scalar xmax,
#                                 int order)
# 
# template<typename FunctorType> BFGSSpace::Status 
# BFGS<FunctorType>::lineSearch(Scalar rho, Scalar sigma, 
#                               Scalar tau1, Scalar tau2, Scalar tau3,
#                               int order, Scalar alpha1, Scalar &alpha_new)
###

# correspondence_estimation.h
# template <typename PointSource, typename PointTarget>
# class CorrespondenceEstimation : public PCLBase<PointSource>
cdef extern from "pcl/registration/correspondence_estimation.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceEstimation[Source, Target](cpp.PCLBase[Source]):
        CorrespondenceEstimation()
#       public:
#         using PCLBase<PointSource>::initCompute;
#         using PCLBase<PointSource>::deinitCompute;
#         using PCLBase<PointSource>::input_;
#         using PCLBase<PointSource>::indices_;
#         typedef typename pcl::KdTree<PointTarget> KdTree;
#         typedef typename pcl::KdTree<PointTarget>::Ptr KdTreePtr;
#         typedef pcl::PointCloud<PointSource> PointCloudSource;
#         typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
#         typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;
#         typedef pcl::PointCloud<PointTarget> PointCloudTarget;
#         typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
#         typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;
#         typedef typename KdTree::PointRepresentationConstPtr PointRepresentationConstPtr;
# 
#         /** \brief Provide a pointer to the input target (e.g., the point cloud that we want to align the 
#           * input source to)
#           * \param[in] cloud the input point cloud target
#           */
#         virtual inline void 
#         setInputTarget (const PointCloudTargetConstPtr &cloud);
# 
#         /** \brief Get a pointer to the input point cloud dataset target. */
#         inline PointCloudTargetConstPtr const 
#         getInputTarget () { return (target_ ); }
# 
#         /** \brief Provide a boost shared pointer to the PointRepresentation to be used when comparing points
#           * \param[in] point_representation the PointRepresentation to be used by the k-D tree
#           */
#         inline void
#         setPointRepresentation (const PointRepresentationConstPtr &point_representation)
# 
#         /** \brief Determine the correspondences between input and target cloud.
#           * \param[out] correspondences the found correspondences (index of query point, index of target point, distance)
#           * \param[in] max_distance maximum distance between correspondences
#           */
#         virtual void 
#         determineCorrespondences (pcl::Correspondences &correspondences,
#                                   float max_distance = std::numeric_limits<float>::max ());
# 
#         /** \brief Determine the correspondences between input and target cloud.
#           * \param[out] correspondences the found correspondences (index of query and target point, distance)
#           */
#         virtual void 
#         determineReciprocalCorrespondences (pcl::Correspondences &correspondences);
# 
# ###
# 
# # correspondence_estimation_normal_shooting.h
# template <typename PointSource, typename PointTarget, typename NormalT>
# class CorrespondenceEstimationNormalShooting : public CorrespondenceEstimation <PointSource, PointTarget>
cdef extern from "pcl/registration/correspondence_estimation_normal_shooting.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceEstimationNormalShooting[Source, Target, NormalT](CorrespondenceEstimation[Source, Target]):
        CorrespondenceEstimationNormalShooting()
#       public:
#         using PCLBase<PointSource>::initCompute;
#         using PCLBase<PointSource>::deinitCompute;
#         using PCLBase<PointSource>::input_;
#         using PCLBase<PointSource>::indices_;
#         using CorrespondenceEstimation<PointSource, PointTarget>::getClassName;
#         typedef typename pcl::KdTree<PointTarget> KdTree;
#         typedef typename pcl::KdTree<PointTarget>::Ptr KdTreePtr;
#         typedef pcl::PointCloud<PointSource> PointCloudSource;
#         typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
#         typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;
#         typedef pcl::PointCloud<PointTarget> PointCloudTarget;
#         typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
#         typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;
#         typedef typename KdTree::PointRepresentationConstPtr PointRepresentationConstPtr;
#         typedef typename pcl::PointCloud<NormalT>::Ptr NormalsPtr;
#
#         /** \brief Set the normals computed on the input point cloud
#           * \param[in] normals the normals computed for the input cloud
#           */
#         inline void
#         setSourceNormals (const NormalsPtr &normals)
# 
#         /** \brief Get the normals of the input point cloud
#           */
#         inline NormalsPtr
#         getSourceNormals () const
# 
#         /** \brief Determine the correspondences between input and target cloud.
#           * \param[out] correspondences the found correspondences (index of query point, index of target point, distance)
#           * \param[in] max_distance maximum distance between the normal on the source point cloud and the corresponding point in the target
#           * point cloud
#           */
#         void 
#         determineCorrespondences (pcl::Correspondences &correspondences,
#                                   float max_distance = std::numeric_limits<float>::max ());
# 
#         /** \brief Set the number of nearest neighbours to be considered in the target point cloud
#           * \param[in] k the number of nearest neighbours to be considered
#           */
#         inline void
#         setKSearch (unsigned int k) { k_ = k; }
# 
#         /** \brief Get the number of nearest neighbours considered in the target point cloud for computing correspondence
#           */
#         inline void
#         getKSearch () const { return (k_); }
# 
# ###

# 
# correspondence_rejection_distance.h
# class CorrespondenceRejectorDistance: public CorrespondenceRejector
cdef extern from "pcl/registration/correspondence_rejection_distance.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceRejectorDistance(CorrespondenceRejector):
        CorrespondenceRejectorDistance()
#       using CorrespondenceRejector::input_correspondences_;
#       using CorrespondenceRejector::rejection_name_;
#       using CorrespondenceRejector::getClassName;
#       public:
#         /** \brief Get a list of valid correspondences after rejection from the original set of correspondences.
#           * \param[in] original_correspondences the set of initial correspondences given
#           * \param[out] remaining_correspondences the resultant filtered set of remaining correspondences
#           */
#         inline void 
#         getRemainingCorrespondences (const pcl::Correspondences& original_correspondences, 
#                                      pcl::Correspondences& remaining_correspondences);
# 
#         /** \brief Set the maximum distance used for thresholding in correspondence rejection.
#           * \param[in] distance Distance to be used as maximum distance between correspondences. 
#           * Correspondences with larger distances are rejected.
#           * \note Internally, the distance will be stored squared.
#           */
#         virtual inline void 
#         setMaximumDistance (float distance) { max_distance_ = distance * distance; };
# 
#         /** \brief Get the maximum distance used for thresholding in correspondence rejection. */
#         inline float 
#         getMaximumDistance () { return std::sqrt (max_distance_); };
# 
#         /** \brief Provide a source point cloud dataset (must contain XYZ
#           * data!), used to compute the correspondence distance.  
#           * \param[in] cloud a cloud containing XYZ data
#           */
#         template <typename PointT> inline void 
#         setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud)
# 
#         /** \brief Provide a target point cloud dataset (must contain XYZ
#           * data!), used to compute the correspondence distance.  
#           * \param[in] target a cloud containing XYZ data
#           */
#         template <typename PointT> inline void 
#         setInputTarget (const typename pcl::PointCloud<PointT>::ConstPtr &target)
#         {
#           if (!data_container_)
#             data_container_.reset (new DataContainer<PointT>);
#           boost::static_pointer_cast<DataContainer<PointT> > (data_container_)->setInputTarget (target);
#         }
# 
###

# correspondence_rejection_features.h
# class CorrespondenceRejectorFeatures: public CorrespondenceRejector
cdef extern from "pcl/registration/correspondence_rejection_distance.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceRejectorFeatures(CorrespondenceRejector):
        CorrespondenceRejectorFeatures()
#       using CorrespondenceRejector::input_correspondences_;
#       using CorrespondenceRejector::rejection_name_;
#       using CorrespondenceRejector::getClassName;
#         /** \brief Get a list of valid correspondences after rejection from the original set of correspondences
#           * \param[in] original_correspondences the set of initial correspondences given
#           * \param[out] remaining_correspondences the resultant filtered set of remaining correspondences
#           */
#         void 
#         getRemainingCorrespondences (const pcl::Correspondences& original_correspondences, 
#                                      pcl::Correspondences& remaining_correspondences);
# 
#         /** \brief Provide a pointer to a cloud of feature descriptors associated with the source point cloud
#           * \param[in] source_feature a cloud of feature descriptors associated with the source point cloud
#           * \param[in] key a string that uniquely identifies the feature
#           */
#         template <typename FeatureT> inline void 
#         setSourceFeature (const typename pcl::PointCloud<FeatureT>::ConstPtr &source_feature, 
#                           const std::string &key);
# 
#         /** \brief Get a pointer to the source cloud's feature descriptors, specified by the given \a key
#           * \param[in] key a string that uniquely identifies the feature (must match the key provided by setSourceFeature)
#           */
#         template <typename FeatureT> inline typename pcl::PointCloud<FeatureT>::ConstPtr 
#         getSourceFeature (const std::string &key);
# 
#         /** \brief Provide a pointer to a cloud of feature descriptors associated with the target point cloud
#           * \param[in] target_feature a cloud of feature descriptors associated with the target point cloud
#           * \param[in] key a string that uniquely identifies the feature
#           */
#         template <typename FeatureT> inline void 
#         setTargetFeature (const typename pcl::PointCloud<FeatureT>::ConstPtr &target_feature, 
#                           const std::string &key);
# 
#         /** \brief Get a pointer to the source cloud's feature descriptors, specified by the given \a key
#           * \param[in] key a string that uniquely identifies the feature (must match the key provided by setTargetFeature)
#           */
#         template <typename FeatureT> inline typename pcl::PointCloud<FeatureT>::ConstPtr 
#         getTargetFeature (const std::string &key);
# 
#         /** \brief Set a hard distance threshold in the feature \a FeatureT space, between source and target
#           * features. Any feature correspondence that is above this threshold will be considered bad and will be
#           * filtered out.
#           * \param[in] thresh the distance threshold
#           * \param[in] key a string that uniquely identifies the feature
#           */
#         template <typename FeatureT> inline void 
#         setDistanceThreshold (double thresh, const std::string &key);
# 
#         /** \brief Test that all features are valid (i.e., does each key have a valid source cloud, target cloud, 
#           * and search method)
#           */
#         inline bool 
#         hasValidFeatures ();
# 
#         /** \brief Provide a boost shared pointer to a PointRepresentation to be used when comparing features
#           * \param[in] key a string that uniquely identifies the feature
#           * \param[in] fr the point feature representation to be used 
#           */
#         template <typename FeatureT> inline void
#         setFeatureRepresentation (const typename pcl::PointRepresentation<FeatureT>::ConstPtr &fr,
#                                   const std::string &key);
# 
#       protected:
# 
#         /** \brief Apply the rejection algorithm.
#           * \param[out] correspondences the set of resultant correspondences.
#           */
#         inline void 
#         applyRejection (pcl::Correspondences &correspondences)
# 
#         class FeatureContainerInterface
#         {
#           public:
#             virtual bool isValid () = 0;
#             virtual double getCorrespondenceScore (int index) = 0;
#             virtual bool isCorrespondenceValid (int index) = 0;
#         };
# 
#         typedef boost::unordered_map<std::string, boost::shared_ptr<FeatureContainerInterface> > FeaturesMap;
# 
#         /** \brief An STL map containing features to use when performing the correspondence search.*/
#         FeaturesMap features_map_;
# 
#         /** \brief An inner class containing pointers to the source and target feature clouds 
#           * and the parameters needed to perform the correspondence search.  This class extends 
#           * FeatureContainerInterface, which contains abstract methods for any methods that do not depend on the 
#           * FeatureT --- these methods can thus be called from a pointer to FeatureContainerInterface without 
#           * casting to the derived class.
#           */
#         template <typename FeatureT>
#         class FeatureContainer : public pcl::registration::CorrespondenceRejectorFeatures::FeatureContainerInterface
#           public:
#             typedef typename pcl::PointCloud<FeatureT>::ConstPtr FeatureCloudConstPtr;
#             typedef boost::function<int (const pcl::PointCloud<FeatureT> &, int, std::vector<int> &, 
#                                           std::vector<float> &)> SearchMethod;
#             
#             typedef typename pcl::PointRepresentation<FeatureT>::ConstPtr PointRepresentationConstPtr;
#             FeatureContainer () : thresh_(std::numeric_limits<double>::max ()), feature_representation_()
#
#             inline void 
#             setSourceFeature (const FeatureCloudConstPtr &source_features)
#             {
#               source_features_ = source_features;
#             }
#             
#             inline FeatureCloudConstPtr 
#             getSourceFeature ()
#             {
#               return (source_features_);
#             }
#             
#             inline void 
#             setTargetFeature (const FeatureCloudConstPtr &target_features)
#             {
#               target_features_ = target_features;
#             }
#             
#             inline FeatureCloudConstPtr 
#             getTargetFeature ()
#             {
#               return (target_features_);
#             }
#             
#             inline void 
#             setDistanceThreshold (double thresh)
#             {
#               thresh_ = thresh;
#             }
# 
#             virtual inline bool 
#             isValid ()
#             {
#               if (!source_features_ || !target_features_)
#                 return (false);
#               else
#                 return (source_features_->points.size () > 0 && 
#                         target_features_->points.size () > 0);
#             }
# 
#             /** \brief Provide a boost shared pointer to a PointRepresentation to be used when comparing features
#               * \param[in] fr the point feature representation to be used
#               */
#             inline void
#             setFeatureRepresentation (const PointRepresentationConstPtr &fr)
#             {
#               feature_representation_ = fr;
#             }
# 
#             /** \brief Obtain a score between a pair of correspondences.
#               * \param[in] the index to check in the list of correspondences
#               * \return score the resultant computed score
#               */
#             virtual inline double
#             getCorrespondenceScore (int index)
#             {
#               // If no feature representation was given, reset to the default implementation for FeatureT
#               if (!feature_representation_)
#                 feature_representation_.reset (new DefaultFeatureRepresentation<FeatureT>);
# 
#               // Get the source and the target feature from the list
#               const FeatureT &feat_src = source_features_->points[index];
#               const FeatureT &feat_tgt = target_features_->points[index];
# 
#               // Check if the representations are valid
#               if (!feature_representation_->isValid (feat_src) || !feature_representation_->isValid (feat_tgt))
#               {
#                 PCL_ERROR ("[pcl::registration::CorrespondenceRejectorFeatures::getCorrespondenceScore] Invalid feature representation given!\n");
#                 return (std::numeric_limits<double>::max ());
#               }
# 
#               // Set the internal feature point representation of choice
#               Eigen::VectorXf feat_src_ptr = Eigen::VectorXf::Zero (feature_representation_->getNumberOfDimensions ());
#               feature_representation_->vectorize (FeatureT (feat_src), feat_src_ptr);
#               Eigen::VectorXf feat_tgt_ptr = Eigen::VectorXf::Zero (feature_representation_->getNumberOfDimensions ());
#               feature_representation_->vectorize (FeatureT (feat_tgt), feat_tgt_ptr);
# 
#               // Compute the L2 norm
#               return ((feat_src_ptr - feat_tgt_ptr).squaredNorm ());
#             }
# 
#             /** \brief Check whether the correspondence pair at the given index is valid
#               * by computing the score and testing it against the user given threshold 
#               * \param[in] the index to check in the list of correspondences
#               * \return true if the correspondence is good, false otherwise
#               */
#             virtual inline bool
#             isCorrespondenceValid (int index)
#             {
#               if (getCorrespondenceScore (index) < thresh_ * thresh_)
#                 return (true);
#               else
#                 return (false);
#             }
#         };
#     };
###

# correspondence_rejection_median_distance.h
#     class CorrespondenceRejectorMedianDistance: public CorrespondenceRejector
cdef extern from "pcl/registration/correspondence_rejection_median_distance.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceRejectorMedianDistance(CorrespondenceRejector):
        CorrespondenceRejectorMedianDistance()
#       using CorrespondenceRejector::input_correspondences_;
#       using CorrespondenceRejector::rejection_name_;
#       using CorrespondenceRejector::getClassName;
#       public:
#         /** \brief Get a list of valid correspondences after rejection from the original set of correspondences.
#           * \param[in] original_correspondences the set of initial correspondences given
#           * \param[out] remaining_correspondences the resultant filtered set of remaining correspondences
#           */
#         inline void 
#         getRemainingCorrespondences (const pcl::Correspondences& original_correspondences, 
#                                      pcl::Correspondences& remaining_correspondences);
# 
#         /** \brief Get the median distance used for thresholding in correspondence rejection. */
#         inline double getMedianDistance () const
# 
#         /** \brief Provide a source point cloud dataset (must contain XYZ
#           * data!), used to compute the correspondence distance.  
#           * \param[in] cloud a cloud containing XYZ data
#           */
#         template <typename PointT> inline void 
#         setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud)
# 
#         /** \brief Provide a target point cloud dataset (must contain XYZ
#           * data!), used to compute the correspondence distance.  
#           * \param[in] target a cloud containing XYZ data
#           */
#         template <typename PointT> inline void 
#         setInputTarget (const typename pcl::PointCloud<PointT>::ConstPtr &target)
# 
#         /** \brief Set the factor for correspondence rejection. Points with distance greater than median times factor
#          *  will be rejected
#          *  \param[in] factor value
#          */
#         inline void setMedianFactor (double factor)
# 
#         /** \brief Get the factor used for thresholding in correspondence rejection. */
#         inline double getMedianFactor () const { return factor_; };
# 
###

# correspondence_rejection_one_to_one.h
# class CorrespondenceRejectorOneToOne: public CorrespondenceRejector
cdef extern from "pcl/registration/correspondence_rejection_one_to_one.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceRejectorOneToOne(CorrespondenceRejector):
        CorrespondenceRejectorOneToOne()
#       using CorrespondenceRejector::input_correspondences_;
#       using CorrespondenceRejector::rejection_name_;
#       using CorrespondenceRejector::getClassName;
#       public:
#         /** \brief Get a list of valid correspondences after rejection from the original set of correspondences.
#           * \param[in] original_correspondences the set of initial correspondences given
#           * \param[out] remaining_correspondences the resultant filtered set of remaining correspondences
#           */
#         inline void 
#         getRemainingCorrespondences (const pcl::Correspondences& original_correspondences, 
#                                      pcl::Correspondences& remaining_correspondences);
# 
#       protected:
#         /** \brief Apply the rejection algorithm.
#           * \param[out] correspondences the set of resultant correspondences.
#           */
#         inline void 
#         applyRejection (pcl::Correspondences &correspondences)
#         {
#           getRemainingCorrespondences (*input_correspondences_, correspondences);
#         }
#     };

# 
###
 
# correspondence_rejection_sample_consensus.h
# template <typename PointT>
# class CorrespondenceRejectorSampleConsensus: public CorrespondenceRejector
cdef extern from "pcl/registration/correspondence_rejection_sample_consensus.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceRejectorSampleConsensus[T](CorrespondenceRejector):
        CorrespondenceRejectorSampleConsensus()
#       using CorrespondenceRejector::input_correspondences_;
#       using CorrespondenceRejector::rejection_name_;
#       using CorrespondenceRejector::getClassName;
#       typedef pcl::PointCloud<PointT> PointCloud;
#       typedef typename PointCloud::Ptr PointCloudPtr;
#       typedef typename PointCloud::ConstPtr PointCloudConstPtr;
#       public:
#         /** \brief Get a list of valid correspondences after rejection from the original set of correspondences.
#           * \param[in] original_correspondences the set of initial correspondences given
#           * \param[out] remaining_correspondences the resultant filtered set of remaining correspondences
#           */
#         inline void 
#         getRemainingCorrespondences (const pcl::Correspondences& original_correspondences, 
#                                      pcl::Correspondences& remaining_correspondences);
# 
#         /** \brief Provide a source point cloud dataset (must contain XYZ data!)
#           * \param[in] cloud a cloud containing XYZ data
#           */
#         virtual inline void 
#         setInputCloud (const PointCloudConstPtr &cloud) { input_ = cloud; }
# 
#         /** \brief Provide a target point cloud dataset (must contain XYZ data!)
#           * \param[in] cloud a cloud containing XYZ data
#           */
#         virtual inline void 
#         setTargetCloud (const PointCloudConstPtr &cloud) { target_ = cloud; }
# 
#         /** \brief Set the maximum distance between corresponding points.
#           * Correspondences with distances below the threshold are considered as inliers.
#           * \param[in] threshold Distance threshold in the same dimension as source and target data sets.
#           */
#         inline void 
#         setInlierThreshold (double threshold) { inlier_threshold_ = threshold; };
# 
#         /** \brief Get the maximum distance between corresponding points.
#           * \return Distance threshold in the same dimension as source and target data sets.
#           */
#         inline double 
#         getInlierThreshold() { return inlier_threshold_; };
# 
#         /** \brief Set the maximum number of iterations.
#           * \param[in] max_iterations Maximum number if iterations to run
#           */
#         inline void 
#         setMaxIterations (int max_iterations) {max_iterations_ = std::max(max_iterations, 0); };
# 
#         /** \brief Get the maximum number of iterations.
#           * \return max_iterations Maximum number if iterations to run
#           */
#         inline int 
#         getMaxIterations () { return max_iterations_; };
# 
#         /** \brief Get the best transformation after RANSAC rejection.
#           * \return The homogeneous 4x4 transformation yielding the largest number of inliers.
#           */
#         inline Eigen::Matrix4f 
#         getBestTransformation () { return best_transformation_; };
# 
#       protected:
# 
#         /** \brief Apply the rejection algorithm.
#           * \param[out] correspondences the set of resultant correspondences.
#           */
#         inline void 
#         applyRejection (pcl::Correspondences &correspondences)
#         {
#           getRemainingCorrespondences (*input_correspondences_, correspondences);
#         }
# 
#         PointCloudConstPtr input_;
#         PointCloudConstPtr target_;
# 
#         Eigen::Matrix4f best_transformation_;
#       public:
#         EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#     };

###

# correspondence_rejection_surface_normal.h
# class CorrespondenceRejectorSurfaceNormal : public CorrespondenceRejector
cdef extern from "pcl/registration/correspondence_rejection_surface_normal.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceRejectorSurfaceNormal(CorrespondenceRejector):
        CorrespondenceRejectorSurfaceNormal()
#       using CorrespondenceRejector::input_correspondences_;
#       using CorrespondenceRejector::rejection_name_;
#       using CorrespondenceRejector::getClassName;
#       public:
#         /** \brief Get a list of valid correspondences after rejection from the original set of correspondences.
#           * \param[in] original_correspondences the set of initial correspondences given
#           * \param[out] remaining_correspondences the resultant filtered set of remaining correspondences
#           */
#         inline void 
#         getRemainingCorrespondences (const pcl::Correspondences& original_correspondences, 
#                                      pcl::Correspondences& remaining_correspondences);
# 
#         /** \brief Set the thresholding angle between the normals for correspondence rejection. 
#           * \param[in] threshold cosine of the thresholding angle between the normals for rejection
#           */
#         inline void
#         setThreshold (double threshold) { threshold_ = threshold; };
# 
#         /** \brief Get the thresholding angle between the normals for correspondence rejection. */
#         inline double
#         getThreshold () const { return threshold_; };
# 
#         /** \brief Initialize the data container object for the point type and the normal type
#           */
#         template <typename PointT, typename NormalT> inline void 
#         initializeDataContainer ()
#         {
#             data_container_.reset (new DataContainer<PointT, NormalT>);
#         }
#         /** \brief Provide a source point cloud dataset (must contain XYZ
#           * data!), used to compute the correspondence distance.  
#           * \param[in] cloud a cloud containing XYZ data
#           */
#         template <typename PointT> inline void 
#         setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr &input)
# 
#         /** \brief Provide a target point cloud dataset (must contain XYZ
#           * data!), used to compute the correspondence distance.  
#           * \param[in] target a cloud containing XYZ data
#           */
#         template <typename PointT> inline void 
#         setInputTarget (const typename pcl::PointCloud<PointT>::ConstPtr &target)
#         {
#           assert (data_container_ && "Initilize the data container object by calling intializeDataContainer () before using this function");
#           boost::static_pointer_cast<DataContainer<PointT> > (data_container_)->setInputTarget (target);
#         }
# 
#         /** \brief Set the normals computed on the input point cloud
#           * \param[in] normals the normals computed for the input cloud
#           */
#         template <typename PointT, typename NormalT> inline void 
#         setInputNormals (const typename pcl::PointCloud<NormalT>::ConstPtr &normals)
#         {
#           assert (data_container_ && "Initilize the data container object by calling intializeDataContainer () before using this function");
#           boost::static_pointer_cast<DataContainer<PointT, NormalT> > (data_container_)->setInputNormals (normals);
#         }
# 
#         /** \brief Set the normals computed on the target point cloud
#           * \param[in] normals the normals computed for the input cloud
#           */
#         template <typename PointT, typename NormalT> inline void 
#         setTargetNormals (const typename pcl::PointCloud<NormalT>::ConstPtr &normals)
#         {
#           assert (data_container_ && "Initilize the data container object by calling intializeDataContainer () before using this function");
#           boost::static_pointer_cast<DataContainer<PointT, NormalT> > (data_container_)->setTargetNormals (normals);
#         }
# 
#         /** \brief Get the normals computed on the input point cloud */
#         template <typename NormalT> inline typename pcl::PointCloud<NormalT>::Ptr
#         getInputNormals () const { return boost::static_pointer_cast<DataContainer<pcl::PointXYZ, NormalT> > (data_container_)->getInputNormals (); }
# 
#         /** \brief Get the normals computed on the target point cloud */
#         template <typename NormalT> inline typename pcl::PointCloud<NormalT>::Ptr
#         getTargetNormals () const { return boost::static_pointer_cast<DataContainer<pcl::PointXYZ, NormalT> > (data_container_)->getTargetNormals (); }
# 
#       protected:
#         /** \brief Apply the rejection algorithm.
#           * \param[out] correspondences the set of resultant correspondences.
#           */
#         inline void 
#         applyRejection (pcl::Correspondences &correspondences)
#         {
#           getRemainingCorrespondences (*input_correspondences_, correspondences);
#         }
# 
#         /** \brief The median distance threshold between two correspondent points in source <-> target.
#           */
#         double threshold_;
# 
#         typedef boost::shared_ptr<DataContainerInterface> DataContainerPtr;
# 
#         /** \brief A pointer to the DataContainer object containing the input and target point clouds */
#         DataContainerPtr data_container_;
###

# correspondence_rejection_trimmed.h
#     class CorrespondenceRejectorTrimmed: public CorrespondenceRejector
cdef extern from "pcl/registration/correspondence_rejection_trimmed.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceRejectorTrimmed(CorrespondenceRejector):
        CorrespondenceRejectorTrimmed()
#       using CorrespondenceRejector::input_correspondences_;
#       using CorrespondenceRejector::rejection_name_;
#       using CorrespondenceRejector::getClassName;
#       public:
#         /** \brief Set the expected ratio of overlap between point clouds (in
#           * terms of correspondences).
#           * \param[in] ratio ratio of overlap between 0 (no overlap, no
#           * correspondences) and 1 (full overlap, all correspondences)
#           */
#         virtual inline void setOverlapRadio (float ratio)
# 
#         /** \brief Get the maximum distance used for thresholding in correspondence rejection. */
#         inline float getOverlapRadio ()
# 
#         /** \brief Set a minimum number of correspondences. If the specified overlap ratio causes to have
#           * less correspondences,  \a CorrespondenceRejectorTrimmed will try to return at least
#           * \a nr_min_correspondences_ correspondences (or all correspondences in case \a nr_min_correspondences_
#           * is less than the number of given correspondences). 
#           * \param[in] min_correspondences the minimum number of correspondences
#           */
#         inline void setMinCorrespondences (unsigned int min_correspondences) { nr_min_correspondences_ = min_correspondences; };
# 
#         /** \brief Get the minimum number of correspondences. */
#         inline unsigned int getMinCorrespondences ()
#
#         /** \brief Get a list of valid correspondences after rejection from the original set of correspondences.
#           * \param[in] original_correspondences the set of initial correspondences given
#           * \param[out] remaining_correspondences the resultant filtered set of remaining correspondences
#           */
#         inline void
#         getRemainingCorrespondences (const pcl::Correspondences& original_correspondences,
#                                      pcl::Correspondences& remaining_correspondences);
#       protected:
#         /** \brief Apply the rejection algorithm.
#           * \param[out] correspondences the set of resultant correspondences.
#           */
#         inline void 
#         applyRejection (pcl::Correspondences &correspondences)
#         {
#           getRemainingCorrespondences (*input_correspondences_, correspondences);
#         }
# 
#         /** Overlap Ratio in [0..1] */
#         float overlap_ratio_;
# 
#         /** Minimum number of correspondences. */
#         unsigned int nr_min_correspondences_;
###

# correspondence_rejection_var_trimmed.h
#     class CorrespondenceRejectorVarTrimmed: public CorrespondenceRejector
cdef extern from "pcl/registration/correspondence_rejection_var_trimmed.h" namespace "pcl::registration" nogil:
    cdef cppclass CorrespondenceRejectorVarTrimmed(CorrespondenceRejector):
        CorrespondenceRejectorVarTrimmed()
#       using CorrespondenceRejector::input_correspondences_;
#       using CorrespondenceRejector::rejection_name_;
#       using CorrespondenceRejector::getClassName;
#       public:
#         /** \brief Get a list of valid correspondences after rejection from the original set of correspondences.
#           * \param[in] original_correspondences the set of initial correspondences given
#           * \param[out] remaining_correspondences the resultant filtered set of remaining correspondences
#           */
#         inline void 
#         getRemainingCorrespondences (const pcl::Correspondences& original_correspondences, 
#                                      pcl::Correspondences& remaining_correspondences);
# 
#         /** \brief Get the trimmed distance used for thresholding in correspondence rejection. */
#         inline double
#         getTrimmedDistance () const { return trimmed_distance_; };
# 
#         /** \brief Provide a source point cloud dataset (must contain XYZ
#           * data!), used to compute the correspondence distance.  
#           * \param[in] cloud a cloud containing XYZ data
#           */
#         template <typename PointT> inline void 
#         setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr &cloud)
# 
#         /** \brief Provide a target point cloud dataset (must contain XYZ
#           * data!), used to compute the correspondence distance.  
#           * \param[in] target a cloud containing XYZ data
#           */
#         template <typename PointT> inline void 
#         setInputTarget (const typename pcl::PointCloud<PointT>::ConstPtr &target)
#         {
#           if (!data_container_)
#             data_container_.reset (new DataContainer<PointT>);
#           boost::static_pointer_cast<DataContainer<PointT> > (data_container_)->setInputTarget (target);
#         }
# 
#         /** \brief Get the computed inlier ratio used for thresholding in correspondence rejection. */
#         inline double
#         getTrimFactor () const { return factor_; }
# 
#         /** brief set the minimum overlap ratio
#           * \param[in] ratio the overlap ratio [0..1]
#           */
#         inline void
#         setMinRatio (double ratio) { min_ratio_ = ratio; }
# 
#         /** brief get the minimum overlap ratio
#           */
#         inline double
#         getMinRatio () const { return min_ratio_; }
# 
#         /** brief set the maximum overlap ratio
#           * \param[in] ratio the overlap ratio [0..1]
#           */
#         inline void
#         setMaxRatio (double ratio) { max_ratio_ = ratio; }
# 
#         /** brief get the maximum overlap ratio
#           */
#         inline double
#         getMaxRatio () const { return max_ratio_; }
#       protected:
#         /** \brief Apply the rejection algorithm.
#           * \param[out] correspondences the set of resultant correspondences.
#           */
#         inline void 
#         applyRejection (pcl::Correspondences &correspondences)
#         {
#           getRemainingCorrespondences (*input_correspondences_, correspondences);
#         }
# 
#         /** \brief The inlier distance threshold (based on the computed trim factor) between two correspondent points in source <-> target.
#           */
#         double trimmed_distance_;
# 
#         /** \brief The factor for correspondence rejection. Only factor times the total points sorted based on 
#          *  the correspondence distances will be considered as inliers. Remaining points are rejected. This factor is
#          *  computed internally 
#          */
#         double factor_;
# 
#         /** \brief The minimum overlap ratio between the input and target clouds
#          */
#         double min_ratio_;
# 
#         /** \brief The maximum overlap ratio between the input and target clouds
#          */
#         double max_ratio_;
# 
#                 /** \brief part of the term that balances the root mean square difference. This is an internal parameter
#          */
#         double lambda_;
# 
#         typedef boost::shared_ptr<DataContainerInterface> DataContainerPtr;
# 
#         /** \brief A pointer to the DataContainer object containing the input and target point clouds */
#         DataContainerPtr data_container_;
# 
###

# correspondence_sorting.h
#     /** @b sortCorrespondencesByQueryIndex : a functor for sorting correspondences by query index
#       * \author Dirk Holz
#       * \ingroup registration
#       */
#     struct sortCorrespondencesByQueryIndex : public std::binary_function<pcl::Correspondence, pcl::Correspondence, bool>
#     {
#       bool
#       operator()( pcl::Correspondence a, pcl::Correspondence b)
#       {
#         return (a.index_query < b.index_query);
#       }
#     };
# 
#     /** @b sortCorrespondencesByMatchIndex : a functor for sorting correspondences by match index
#       * \author Dirk Holz
#       * \ingroup registration
#       */
#     struct sortCorrespondencesByMatchIndex : public std::binary_function<pcl::Correspondence, pcl::Correspondence, bool>
#     {
#       bool 
#       operator()( pcl::Correspondence a, pcl::Correspondence b)
#       {
#         return (a.index_match < b.index_match);
#       }
#     };
# 
#     /** @b sortCorrespondencesByDistance : a functor for sorting correspondences by distance
#       * \author Dirk Holz
#       * \ingroup registration
#       */
#     struct sortCorrespondencesByDistance : public std::binary_function<pcl::Correspondence, pcl::Correspondence, bool>
#     {
#       bool 
#       operator()( pcl::Correspondence a, pcl::Correspondence b)
#       {
#         return (a.distance < b.distance);
#       }
#     };
# 
#     /** @b sortCorrespondencesByQueryIndexAndDistance : a functor for sorting correspondences by query index _and_ distance
#       * \author Dirk Holz
#       * \ingroup registration
#       */
#     struct sortCorrespondencesByQueryIndexAndDistance : public std::binary_function<pcl::Correspondence, pcl::Correspondence, bool>
#     {
#       inline bool 
#       operator()( pcl::Correspondence a, pcl::Correspondence b)
#       {
#         if (a.index_query < b.index_query)
#           return (true);
#         else if ( (a.index_query == b.index_query) && (a.distance < b.distance) )
#           return (true);
#         return (false);
#       }
#     };
# 
#     /** @b sortCorrespondencesByMatchIndexAndDistance : a functor for sorting correspondences by match index _and_ distance
#       * \author Dirk Holz
#       * \ingroup registration
#       */
#     struct sortCorrespondencesByMatchIndexAndDistance : public std::binary_function<pcl::Correspondence, pcl::Correspondence, bool>
#     {
#       inline bool 
#       operator()( pcl::Correspondence a, pcl::Correspondence b)
#       {
#         if (a.index_match < b.index_match)
#           return (true);
#         else if ( (a.index_match == b.index_match) && (a.distance < b.distance) )
#           return (true);
#         return (false);
#       }
#     };

# 
###

# correspondence_types.h
#     /** \brief calculates the mean and standard deviation of descriptor distances from correspondences
#       * \param[in] correspondences list of correspondences
#       * \param[out] mean the mean descriptor distance of correspondences
#       * \param[out] stddev the standard deviation of descriptor distances.
#       * \note The sample varaiance is used to determine the standard deviation
#       */
#     inline void 
#     getCorDistMeanStd (const pcl::Correspondences& correspondences, double &mean, double &stddev);
# 
#     /** \brief extracts the query indices
#       * \param[in] correspondences list of correspondences
#       * \param[out] indices array of extracted indices.
#       * \note order of indices corresponds to input list of descriptor correspondences
#       */
#     inline void 
#     getQueryIndices (const pcl::Correspondences& correspondences, std::vector<int>& indices);
# 
#     /** \brief extracts the match indices
#       * \param[in] correspondences list of correspondences
#       * \param[out] indices array of extracted indices.
#       * \note order of indices corresponds to input list of descriptor correspondences
#       */
#     inline void 
#     getMatchIndices (const pcl::Correspondences& correspondences, std::vector<int>& indices);

# 
###

# distances.h
#     /** \brief Compute the median value from a set of doubles
#       * \param[in] fvec the set of doubles
#       * \param[in] m the number of doubles in the set
#       */
#     inline double 
#     computeMedian (double *fvec, int m)
#     {
#       // Copy the values to vectors for faster sorting
#       std::vector<double> data (m);
#       memcpy (&data[0], fvec, sizeof (double) * m);
#       
#       std::nth_element(data.begin(), data.begin() + (data.size () >> 1), data.end());
#       return (data[data.size () >> 1]);
#     }
# 
#     /** \brief Use a Huber kernel to estimate the distance between two vectors
#       * \param[in] p_src the first eigen vector
#       * \param[in] p_tgt the second eigen vector
#       * \param[in] sigma the sigma value
#       */
#     inline double
#     huber (const Eigen::Vector4f &p_src, const Eigen::Vector4f &p_tgt, double sigma) 
#     {
#       Eigen::Array4f diff = (p_tgt.array () - p_src.array ()).abs ();
#       double norm = 0.0;
#       for (int i = 0; i < 3; ++i)
#       {
#         if (diff[i] < sigma)
#           norm += diff[i] * diff[i];
#         else
#           norm += 2.0 * sigma * diff[i] - sigma * sigma;
#       }
#       return (norm);
#     }
# 
#     /** \brief Use a Huber kernel to estimate the distance between two vectors
#       * \param[in] diff the norm difference between two vectors
#       * \param[in] sigma the sigma value
#       */
#     inline double
#     huber (double diff, double sigma) 
#     {
#       double norm = 0.0;
#       if (diff < sigma)
#         norm += diff * diff;
#       else
#         norm += 2.0 * sigma * diff - sigma * sigma;
#       return (norm);
#     }
# 
#     /** \brief Use a Gedikli kernel to estimate the distance between two vectors
#       * (for more information, see 
#       * \param[in] val the norm difference between two vectors
#       * \param[in] clipping the clipping value
#       * \param[in] slope the slope. Default: 4
#       */
#     inline double
#     gedikli (double val, double clipping, double slope = 4) 
#     {
#       return (1.0 / (1.0 + pow (fabs(val) / clipping, slope)));
#     }
# 
#     /** \brief Compute the Manhattan distance between two eigen vectors.
#       * \param[in] p_src the first eigen vector
#       * \param[in] p_tgt the second eigen vector
#       */
#     inline double
#     l1 (const Eigen::Vector4f &p_src, const Eigen::Vector4f &p_tgt) 
#     {
#       return ((p_src.array () - p_tgt.array ()).abs ().sum ());
#     }
# 
#     /** \brief Compute the Euclidean distance between two eigen vectors.
#       * \param[in] p_src the first eigen vector
#       * \param[in] p_tgt the second eigen vector
#       */
#     inline double
#     l2 (const Eigen::Vector4f &p_src, const Eigen::Vector4f &p_tgt) 
#     {
#       return ((p_src - p_tgt).norm ());
#     }
# 
#     /** \brief Compute the squared Euclidean distance between two eigen vectors.
#       * \param[in] p_src the first eigen vector
#       * \param[in] p_tgt the second eigen vector
#       */
#     inline double
#     l2Sqr (const Eigen::Vector4f &p_src, const Eigen::Vector4f &p_tgt) 
#     {
#       return ((p_src - p_tgt).squaredNorm ());
#     }

# 
# ###
# 
# # eigen.h
# # 
# #include <Eigen/Core>
# #include <Eigen/Geometry>
# #include <unsupported/Eigen/Polynomials>
# #include <Eigen/Dense>
# ###

# elch.h
# template <typename PointT>
# class ELCH : public PCLBase<PointT>
cdef extern from "pcl/registration/elch.h" namespace "pcl::registration" nogil:
    cdef cppclass ELCH[T](cpp.PCLBase[T]):
        ELCH()
#       public:
#         typedef boost::shared_ptr< ELCH<PointT> > Ptr;
#         typedef boost::shared_ptr< const ELCH<PointT> > ConstPtr;
#         typedef pcl::PointCloud<PointT> PointCloud;
#         typedef typename PointCloud::Ptr PointCloudPtr;
#         typedef typename PointCloud::ConstPtr PointCloudConstPtr;
#         struct Vertex
#         {
#           Vertex () : cloud () {}
#           PointCloudPtr cloud;
#         };
# 
#         /** \brief graph structure to hold the SLAM graph */
#         typedef boost::adjacency_list<
#           boost::listS, boost::vecS, boost::undirectedS,
#           Vertex,
#           boost::no_property>
#         LoopGraph;
#         typedef boost::shared_ptr< LoopGraph > LoopGraphPtr;
#         typedef typename pcl::Registration<PointT, PointT> Registration;
#         typedef typename Registration::Ptr RegistrationPtr;
#         typedef typename Registration::ConstPtr RegistrationConstPtr;
#
#         /** \brief Add a new point cloud to the internal graph.
#          * \param[in] cloud the new point cloud
#          */
#         inline void
#         addPointCloud (PointCloudPtr cloud)
# 
#         /** \brief Getter for the internal graph. */
#         inline LoopGraphPtr
#         getLoopGraph ()
# 
#         /** \brief Setter for a new internal graph.
#          * \param[in] loop_graph the new graph
#          */
#         inline void
#         setLoopGraph (LoopGraphPtr loop_graph)
# 
#         /** \brief Getter for the first scan of a loop. */
#         inline typename boost::graph_traits<LoopGraph>::vertex_descriptor
#         getLoopStart ()
#
#         /** \brief Setter for the first scan of a loop.
#          * \param[in] loop_start the scan that starts the loop
#          */
#         inline void
#         setLoopStart (const typename boost::graph_traits<LoopGraph>::vertex_descriptor &loop_start)
#
#         /** \brief Getter for the last scan of a loop. */
#         inline typename boost::graph_traits<LoopGraph>::vertex_descriptor
#         getLoopEnd ()
# 
#         /** \brief Setter for the last scan of a loop.
#          * \param[in] loop_end the scan that ends the loop
#          */
#         inline void
#         setLoopEnd (const typename boost::graph_traits<LoopGraph>::vertex_descriptor &loop_end)
# 
#         /** \brief Getter for the registration algorithm. */
#         inline RegistrationPtr
#         getReg ()
# 
#         /** \brief Setter for the registration algorithm.
#          * \param[in] reg the registration algorithm used to compute the transformation between the start and the end of the loop
#          */
#         inline void setReg (RegistrationPtr reg)
# 
#         /** \brief Getter for the transformation between the first and the last scan. */
#         inline Eigen::Matrix4f getLoopTransform ()
# 
#         /** \brief Setter for the transformation between the first and the last scan.
#          * \param[in] loop_transform the transformation between the first and the last scan
#          */
#         inline void setLoopTransform (const Eigen::Matrix4f &loop_transform)
# 
#         /** \brief Computes now poses for all point clouds by closing the loop
#          * between start and end point cloud. This will transform all given point
#          * clouds for now!
#          */
#         void compute ();
#       protected:
#         using PCLBase<PointT>::deinitCompute;
# 
#         /** \brief This method should get called before starting the actual computation. */
#         virtual bool initCompute ();
#       public:
#         EIGEN_MAKE_ALIGNED_OPERATOR_NEW
###
# 
# # exceptions.h
# # pcl/exceptions
# #  /** \class SolverDidntConvergeException
# #     * \brief An exception that is thrown when the non linear solver didn't converge
# #     */
# #   class PCL_EXPORTS SolverDidntConvergeException : public PCLException
# #   {
# #     public:
# #     
# #     SolverDidntConvergeException (const std::string& error_description,
# #                                   const std::string& file_name = "",
# #                                   const std::string& function_name = "" ,
# #                                   unsigned line_number = 0) throw ()
# #       : pcl::PCLException (error_description, file_name, function_name, line_number) { }
# #   } ;
# # 
# #  /** \class NotEnoughPointsException
# #     * \brief An exception that is thrown when the number of correspondants is not equal
# #     * to the minimum required
# #     */
# #   class PCL_EXPORTS NotEnoughPointsException : public PCLException
# #   {
# #     public:
# #     
# #     NotEnoughPointsException (const std::string& error_description,
# #                               const std::string& file_name = "",
# #                               const std::string& function_name = "" ,
# #                               unsigned line_number = 0) throw ()
# #       : pcl::PCLException (error_description, file_name, function_name, line_number) { }
# #   } ;
# # 
# ###

# gicp.h
cdef extern from "pcl/registration/gicp.h" namespace "pcl" nogil:
    cdef cppclass GeneralizedIterativeClosestPoint[Source, Target](Registration[Source, Target]):
        GeneralizedIterativeClosestPoint() except +
        # using IterativeClosestPoint<PointSource, PointTarget>::reg_name_;
        # using IterativeClosestPoint<PointSource, PointTarget>::getClassName;
        # using IterativeClosestPoint<PointSource, PointTarget>::indices_;
        # using IterativeClosestPoint<PointSource, PointTarget>::target_;
        # using IterativeClosestPoint<PointSource, PointTarget>::input_;
        # using IterativeClosestPoint<PointSource, PointTarget>::tree_;
        # using IterativeClosestPoint<PointSource, PointTarget>::nr_iterations_;
        # using IterativeClosestPoint<PointSource, PointTarget>::max_iterations_;
        # using IterativeClosestPoint<PointSource, PointTarget>::previous_transformation_;
        # using IterativeClosestPoint<PointSource, PointTarget>::final_transformation_;
        # using IterativeClosestPoint<PointSource, PointTarget>::transformation_;
        # using IterativeClosestPoint<PointSource, PointTarget>::transformation_epsilon_;
        # using IterativeClosestPoint<PointSource, PointTarget>::converged_;
        # using IterativeClosestPoint<PointSource, PointTarget>::corr_dist_threshold_;
        # using IterativeClosestPoint<PointSource, PointTarget>::inlier_threshold_;
        # using IterativeClosestPoint<PointSource, PointTarget>::min_number_correspondences_;
        # using IterativeClosestPoint<PointSource, PointTarget>::update_visualizer_;
        # typedef pcl::PointCloud<PointSource> PointCloudSource;
        # typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
        # typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;
        # typedef pcl::PointCloud<PointTarget> PointCloudTarget;
        # typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
        # typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;
        # typedef PointIndices::Ptr PointIndicesPtr;
        # typedef PointIndices::ConstPtr PointIndicesConstPtr;
        # typedef typename pcl::KdTree<PointSource> InputKdTree;
        # typedef typename pcl::KdTree<PointSource>::Ptr InputKdTreePtr;
        # typedef Eigen::Matrix<double, 6, 1> Vector6d;
        # public:
        # /** \brief Provide a pointer to the input dataset
        #     * \param cloud the const boost shared pointer to a PointCloud message
        #     */
        # void setInputCloud (cpp.PointCloudPtr_t ptcloud)
    	# 
        # /** \brief Provide a pointer to the input target (e.g., the point cloud that we want to align the input source to)
        #     * \param[in] target the input point cloud target
        #     */
        # inline void setInputTarget (const PointCloudTargetConstPtr &target)
    	# 
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using an iterative
        #     * non-linear Levenberg-Marquardt approach.
        #     * \param[in] cloud_src the source point cloud dataset
        #     * \param[in] indices_src the vector of indices describing the points of interest in \a cloud_src
        #     * \param[in] cloud_tgt the target point cloud dataset
        #     * \param[in] indices_tgt the vector of indices describing the correspondences of the interst points from \a indices_src
        #     * \param[out] transformation_matrix the resultant transformation matrix
        #     */
        # void estimateRigidTransformationBFGS (
        #                                 const PointCloudSource &cloud_src,
        #                                 const std::vector<int> &indices_src,
        #                                 const PointCloudTarget &cloud_tgt,
        #                                 const std::vector<int> &indices_tgt,
        #                                 Eigen::Matrix4f &transformation_matrix);
        #   
        # /** \brief \return Mahalanobis distance matrix for the given point index */
        # inline const Eigen::Matrix3d& mahalanobis(size_t index) const
    	# 
        # /** \brief Computes rotation matrix derivative.
        #     * rotation matrix is obtainded from rotation angles x[3], x[4] and x[5]
        #     * \return d/d_rx, d/d_ry and d/d_rz respectively in g[3], g[4] and g[5]
        #     * param x array representing 3D transformation
        #     * param R rotation matrix
        #     * param g gradient vector
        #     */
        # void computeRDerivative(const Vector6d &x, const Eigen::Matrix3d &R, Vector6d &g) const;
    	# 
        # /** \brief Set the rotation epsilon (maximum allowable difference between two 
        #     * consecutive rotations) in order for an optimization to be considered as having 
        #     * converged to the final solution.
        #     * \param epsilon the rotation epsilon
        #     */
        # inline void setRotationEpsilon (double epsilon)
    	# 
        # /** \brief Get the rotation epsilon (maximum allowable difference between two 
        #     * consecutive rotations) as set by the user.
        #     */
        # inline double getRotationEpsilon ()
    	# 
        # /** \brief Set the number of neighbors used when selecting a point neighbourhood
        #     * to compute covariances. 
        #     * A higher value will bring more accurate covariance matrix but will make 
        #     * covariances computation slower.
        #     * \param k the number of neighbors to use when computing covariances
        #     */
        # void setCorrespondenceRandomness (int k)
    	# 
        # /** \brief Get the number of neighbors used when computing covariances as set by 
        #     * the user 
        #     */
        # int getCorrespondenceRandomness ()
    	# 
        # /** set maximum number of iterations at the optimization step
        #     * \param[in] max maximum number of iterations for the optimizer
        #     */
        # void setMaximumOptimizerIterations (int max)
    	# 
        # ///\return maximum number of iterations at the optimization step
        # int getMaximumOptimizerIterations ()
###

# ia_ransac.h
# template <typename PointSource, typename PointTarget, typename FeatureT>
# class SampleConsensusInitialAlignment : public Registration<PointSource, PointTarget>
cdef extern from "pcl/registration/ia_ransac.h" namespace "pcl" nogil:
    cdef cppclass SampleConsensusInitialAlignment[Source, Target, Feature](Registration[Source, Target]):
        SampleConsensusInitialAlignment() except +
        # public:
        # using Registration<PointSource, PointTarget>::reg_name_;
        # using Registration<PointSource, PointTarget>::input_;
        # using Registration<PointSource, PointTarget>::indices_;
        # using Registration<PointSource, PointTarget>::target_;
        # using Registration<PointSource, PointTarget>::final_transformation_;
        # using Registration<PointSource, PointTarget>::transformation_;
        # using Registration<PointSource, PointTarget>::corr_dist_threshold_;
        # using Registration<PointSource, PointTarget>::min_number_correspondences_;
        # using Registration<PointSource, PointTarget>::max_iterations_;
        # using Registration<PointSource, PointTarget>::tree_;
        # using Registration<PointSource, PointTarget>::transformation_estimation_;
        # using Registration<PointSource, PointTarget>::getClassName;
        # ctypedef typename Registration<PointSource, PointTarget>::PointCloudSource PointCloudSource;
        # ctypedef typename PointCloudSource::Ptr PointCloudSourcePtr;
        # ctypedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;
        # ctypedef typename Registration<PointSource, PointTarget>::PointCloudTarget PointCloudTarget;
        # ctypedef PointIndices::Ptr PointIndicesPtr;
        # ctypedef PointIndices::ConstPtr PointIndicesConstPtr;
        # ctypedef pcl::PointCloud<FeatureT> FeatureCloud;
        # ctypedef typename FeatureCloud::Ptr FeatureCloudPtr;
        # ctypedef typename FeatureCloud::ConstPtr FeatureCloudConstPtr;
        # cdef cppclass ErrorFunctor
        # {
        #   public:
        #     virtual ~ErrorFunctor () {}
        #     virtual float operator () (float d) const = 0;
        # };
        # class HuberPenalty : public ErrorFunctor
        # cdef cppclass HuberPenalty(ErrorFunctor)
        #     HuberPenalty ()
        #   public:
        #     HuberPenalty (float threshold)
        #     virtual float operator () (float e) const
        #     { 
        #       if (e <= threshold_)
        #         return (0.5 * e*e); 
        #       else
        #         return (0.5 * threshold_ * (2.0 * fabs (e) - threshold_));
        #     }
        #   protected:
        #     float threshold_;
        # };
        # class TruncatedError : public ErrorFunctor
        # cdef cppclass TruncatedError(ErrorFunctor)
        #     TruncatedError ()
        #   public:
        #     virtual ~TruncatedError () {}
        #     TruncatedError (float threshold) : threshold_ (threshold) {}
        #     virtual float operator () (float e) const
        #     { 
        #       if (e <= threshold_)
        #         return (e / threshold_);
        #       else
        #         return (1.0);
        #     }
        #   protected:
        #     float threshold_;
        # };
        # typedef typename KdTreeFLANN<FeatureT>::Ptr FeatureKdTreePtr; 
        # /** \brief Provide a boost shared pointer to the source point cloud's feature descriptors
        #   * \param features the source point cloud's features
        #   */
        # void 
        # setSourceFeatures (const FeatureCloudConstPtr &features);
        # /** \brief Get a pointer to the source point cloud's features */
        # inline FeatureCloudConstPtr const 
        # getSourceFeatures () { return (input_features_); }
        # /** \brief Provide a boost shared pointer to the target point cloud's feature descriptors
        #   * \param features the target point cloud's features
        #   */
        # void 
        # setTargetFeatures (const FeatureCloudConstPtr &features);
        # /** \brief Get a pointer to the target point cloud's features */
        # inline FeatureCloudConstPtr const 
        # getTargetFeatures () { return (target_features_); }
        # /** \brief Set the minimum distances between samples
        #   * \param min_sample_distance the minimum distances between samples
        #   */
        # void 
        # setMinSampleDistance (float min_sample_distance) { min_sample_distance_ = min_sample_distance; }
        # /** \brief Get the minimum distances between samples, as set by the user */
        # float 
        # getMinSampleDistance () { return (min_sample_distance_); }
        # /** \brief Set the number of samples to use during each iteration
        #   * \param nr_samples the number of samples to use during each iteration
        #   */
        # void 
        # setNumberOfSamples (int nr_samples) { nr_samples_ = nr_samples; }
        # /** \brief Get the number of samples to use during each iteration, as set by the user */
        # int 
        # getNumberOfSamples () { return (nr_samples_); }
        # /** \brief Set the number of neighbors to use when selecting a random feature correspondence.  A higher value will
        #   * add more randomness to the feature matching.
        #   * \param k the number of neighbors to use when selecting a random feature correspondence.
        #   */
        # void
        # setCorrespondenceRandomness (int k) { k_correspondences_ = k; }
        # /** \brief Get the number of neighbors used when selecting a random feature correspondence, as set by the user */
        # int
        # getCorrespondenceRandomness () { return (k_correspondences_); }
        # /** \brief Specify the error function to minimize
        #  * \note This call is optional.  TruncatedError will be used by default
        #  * \param[in] error_functor a shared pointer to a subclass of SampleConsensusInitialAlignment::ErrorFunctor
        #  */
        # void
        # setErrorFunction (const boost::shared_ptr<ErrorFunctor> & error_functor) { error_functor_ = error_functor; }
        # /** \brief Get a shared pointer to the ErrorFunctor that is to be minimized  
        #  * \return A shared pointer to a subclass of SampleConsensusInitialAlignment::ErrorFunctor
        #  */
        # boost::shared_ptr<ErrorFunctor>
        # getErrorFunction () { return (error_functor_); }
        # protected:
        # /** \brief Choose a random index between 0 and n-1
        #   * \param n the number of possible indices to choose from
        #   */
        # inline int 
        # getRandomIndex (int n) { return (static_cast<int> (n * (rand () / (RAND_MAX + 1.0)))); };
        # /** \brief Select \a nr_samples sample points from cloud while making sure that their pairwise distances are 
        #   * greater than a user-defined minimum distance, \a min_sample_distance.
        #   * \param cloud the input point cloud
        #   * \param nr_samples the number of samples to select
        #   * \param min_sample_distance the minimum distance between any two samples
        #   * \param sample_indices the resulting sample indices
        #   */
        # void 
        # selectSamples (const PointCloudSource &cloud, int nr_samples, float min_sample_distance, 
        #                std::vector<int> &sample_indices);
        # /** \brief For each of the sample points, find a list of points in the target cloud whose features are similar to 
        #   * the sample points' features. From these, select one randomly which will be considered that sample point's 
        #   * correspondence. 
        #   * \param input_features a cloud of feature descriptors
        #   * \param sample_indices the indices of each sample point
        #   * \param corresponding_indices the resulting indices of each sample's corresponding point in the target cloud
        #   */
        # void 
        # findSimilarFeatures (const FeatureCloud &input_features, const std::vector<int> &sample_indices, 
        #                      std::vector<int> &corresponding_indices);
        # /** \brief An error metric for that computes the quality of the alignment between the given cloud and the target.
        #   * \param cloud the input cloud
        #   * \param threshold distances greater than this value are capped
        #   */
        # float 
        # computeErrorMetric (const PointCloudSource &cloud, float threshold);
        # /** \brief Rigid transformation computation method.
        #   * \param output the transformed input point cloud dataset using the rigid transformation found
        #   */
        # virtual void 
        # computeTransformation (PointCloudSource &output, const Eigen::Matrix4f& guess);
        # /** \brief The source point cloud's feature descriptors. */
        # FeatureCloudConstPtr input_features_;
        # /** \brief The target point cloud's feature descriptors. */
        # FeatureCloudConstPtr target_features_;  
        # /** \brief The number of samples to use during each iteration. */
        # int nr_samples_;
        # /** \brief The minimum distances between samples. */
        # float min_sample_distance_;
        # /** \brief The number of neighbors to use when selecting a random feature correspondence. */
        # int k_correspondences_;
        # /** \brief The KdTree used to compare feature descriptors. */
        # FeatureKdTreePtr feature_tree_;               
        # /** */
        # boost::shared_ptr<ErrorFunctor> error_functor_;
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
###

# icp.h
# template <typename PointSource, typename PointTarget>
# class IterativeClosestPoint : public Registration<PointSource, PointTarget>
cdef extern from "pcl/registration/icp.h" namespace "pcl" nogil:
    cdef cppclass IterativeClosestPoint[Source, Target](Registration[Source, Target]):
        IterativeClosestPoint() except +
        # ctypedef typename Registration<PointSource, PointTarget>::PointCloudSource PointCloudSource;
        # ctypedef typename PointCloudSource::Ptr PointCloudSourcePtr;
        # ctypedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;
        # ctypedef typename Registration<PointSource, PointTarget>::PointCloudTarget PointCloudTarget;
        # ctypedef PointIndices::Ptr PointIndicesPtr;
        # ctypedef PointIndices::ConstPtr PointIndicesConstPtr;
###

# icp_nl.h
# template <typename PointSource, typename PointTarget>
# class IterativeClosestPointNonLinear : public IterativeClosestPoint<PointSource, PointTarget>
#   cdef cppclass IterativeClosestPointNonLinear[Source, Target](Registration[Source, Target]):
cdef extern from "pcl/registration/icp_nl.h" namespace "pcl" nogil:
    cdef cppclass IterativeClosestPointNonLinear[Source, Target](IterativeClosestPoint[Source, Target]):
        IterativeClosestPointNonLinear() except +
###

# ppf_registration.h
# template <typename PointSource, typename PointTarget>
# class PPFRegistration : public Registration<PointSource, PointTarget>
cdef extern from "pcl/registration/ppf_registration.h" namespace "pcl" nogil:
    cdef cppclass PPFRegistration[Source, Target](Registration[Source, Target]):
        PPFRegistration() except +
        # public:
        # cdef struct PoseWithVotes
        #   PoseWithVotes(Eigen::Affine3f &a_pose, unsigned int &a_votes)
        #   Eigen::Affine3f pose;
        #   unsigned int votes;
        # ctypedef std::vector<PoseWithVotes, Eigen::aligned_allocator<PoseWithVotes> > PoseWithVotesList;
        # /// input_ is the model cloud
        # using Registration<PointSource, PointTarget>::input_;
        # /// target_ is the scene cloud
        # using Registration<PointSource, PointTarget>::target_;
        # using Registration<PointSource, PointTarget>::converged_;
        # using Registration<PointSource, PointTarget>::final_transformation_;
        # using Registration<PointSource, PointTarget>::transformation_;
        # ctypedef pcl::PointCloud<PointSource> PointCloudSource;
        # ctypedef typename PointCloudSource::Ptr PointCloudSourcePtr;
        # ctypedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;
        # ctypedef pcl::PointCloud<PointTarget> PointCloudTarget;
        # ctypedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
        # ctypedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;
        # /** \brief Method for setting the position difference clustering parameter
        #  * \param clustering_position_diff_threshold distance threshold below which two poses are
        #  * considered close enough to be in the same cluster (for the clustering phase of the algorithm)
        #  */
        #   inline void
        # setPositionClusteringThreshold (float clustering_position_diff_threshold)
        # /** \brief Returns the parameter defining the position difference clustering parameter -
        #  * distance threshold below which two poses are considered close enough to be in the same cluster
        #  * (for the clustering phase of the algorithm)
        #  */
        # inline float
        # getPositionClusteringThreshold ()
        # /** \brief Method for setting the rotation clustering parameter
        #  * \param clustering_rotation_diff_threshold rotation difference threshold below which two
        #  * poses are considered to be in the same cluster (for the clustering phase of the algorithm)
        #  */
        # inline void
        # setRotationClusteringThreshold (float clustering_rotation_diff_threshold)
        # /** \brief Returns the parameter defining the rotation clustering threshold
        #  */
        # inline float
        # getRotationClusteringThreshold ()
        # /** \brief Method for setting the scene reference point sampling rate
        #  * \param scene_reference_point_sampling_rate sampling rate for the scene reference point
        #  */
        # inline void setSceneReferencePointSamplingRate (unsigned int scene_reference_point_sampling_rate) { scene_reference_point_sampling_rate_ = scene_reference_point_sampling_rate; }
        # /** \brief Returns the parameter for the scene reference point sampling rate of the algorithm */
        # inline unsigned int
        # getSceneReferencePointSamplingRate ()
        # /** \brief Function that sets the search method for the algorithm
        #  * \note Right now, the only available method is the one initially proposed by
        #  * the authors - by using a hash map with discretized feature vectors
        #  * \param search_method smart pointer to the search method to be set
        #  */
        # inline void setSearchMethod (PPFHashMapSearch::Ptr search_method)
        # /** \brief Getter function for the search method of the class */
        # inline PPFHashMapSearch::Ptr getSearchMethod ()
        # /** \brief Provide a pointer to the input target (e.g., the point cloud that we want to align the input source to)
        #  * \param cloud the input point cloud target
        #  */
        # void setInputTarget (const PointCloudTargetConstPtr &cloud);

###

# pyramid_feature_matching.h
# template <typename PointFeature>
# class PyramidFeatureHistogram : public PCLBase<PointFeature>
# cdef cppclass PyramidFeatureHistogram[PointFeature](PCLBase[PointFeature]):
cdef extern from "pcl/registration/pyramid_feature_matching.h" namespace "pcl" nogil:
    cdef cppclass PyramidFeatureHistogram[PointFeature]:
        PyramidFeatureHistogram() except +
        # public:
        # using PCLBase<PointFeature>::input_;
        # ctypedef boost::shared_ptr<PyramidFeatureHistogram<PointFeature> > Ptr;
        # ctypedef Ptr PyramidFeatureHistogramPtr;
        # ctypedef boost::shared_ptr<const pcl::PointRepresentation<PointFeature> > FeatureRepresentationConstPtr;
        # /** \brief Method for setting the input dimension range parameter.
        #  * \note Please check the PyramidHistogram class description for more details about this parameter.
        #  */
        # inline void setInputDimensionRange (std::vector<std::pair<float, float> > &dimension_range_input)
        # /** \brief Method for retrieving the input dimension range vector */
        # inline std::vector<std::pair<float, float> > getInputDimensionRange () { return dimension_range_input_; }
        # /** \brief Method to set the target dimension range parameter.
        #  * \note Please check the PyramidHistogram class description for more details about this parameter.
        #  */
        # inline void
        # setTargetDimensionRange (std::vector<std::pair<float, float> > &dimension_range_target)
        # /** \brief Method for retrieving the target dimension range vector */
        # inline std::vector<std::pair<float, float> >
        # getTargetDimensionRange () { return dimension_range_target_; }
        # /** \brief Provide a pointer to the feature representation to use to convert features to k-D vectors.
        #  * \param feature_representation the const boost shared pointer to a PointRepresentation
        #  */
        # inline void
        # setPointRepresentation (const FeatureRepresentationConstPtr& feature_representation) { feature_representation_ = feature_representation; }
        # /** \brief Get a pointer to the feature representation used when converting features into k-D vectors. */
        # inline FeatureRepresentationConstPtr const
        # getPointRepresentation () { return feature_representation_; }
        # /** \brief The central method for inserting the feature set inside the pyramid and obtaining the complete pyramid */
        # void
        # compute ();
        # /** \brief Checks whether the pyramid histogram has been computed */
        # inline bool
        # isComputed () { return is_computed_; }
        # /** \brief Static method for comparing two pyramid histograms that returns a floating point value between 0 and 1,
        #  * representing the similiarity between the feature sets on which the two pyramid histograms are based.
        #  * \param pyramid_a Pointer to the first pyramid to be compared (needs to be computed already).
        #  * \param pyramid_b Pointer to the second pyramid to be compared (needs to be computed already).
        #  */
        # static float
        # comparePyramidFeatureHistograms (const PyramidFeatureHistogramPtr &pyramid_a,
        #                                  const PyramidFeatureHistogramPtr &pyramid_b);

###

# transformation_estimation.h
# template <typename PointSource, typename PointTarget>
# class TransformationEstimation
cdef extern from "pcl/registration/transformation_estimation.h" namespace "pcl" nogil:
    cdef cppclass TransformationEstimation[Source, Target](Registration[Source, Target]):
        TransformationEstimation() except +
        # public:
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # virtual void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     Eigen::Matrix4f &transformation_matrix) = 0;
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] indices_src the vector of indices describing the points of interest in \a cloud_src
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # virtual void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const std::vector<int> &indices_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     Eigen::Matrix4f &transformation_matrix) = 0;
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] indices_src the vector of indices describing the points of interest in \a cloud_src
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[in] indices_tgt the vector of indices describing the correspondences of the interst points from \a indices_src
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # virtual void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const std::vector<int> &indices_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     const std::vector<int> &indices_tgt,
        #     Eigen::Matrix4f &transformation_matrix) = 0;
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[in] correspondences the vector of correspondences between source and target point cloud
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # virtual void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     const pcl::Correspondences &correspondences,
        #     Eigen::Matrix4f &transformation_matrix) = 0;

# ctypedef shared_ptr[TransformationEstimation<PointSource, PointTarget> > Ptr;
# ctypedef shared_ptr[const TransformationEstimation<PointSource, PointTarget> > ConstPtr;

###

# transformation_estimation_lm.h

# template <typename PointSource, typename PointTarget>
# class TransformationEstimationLM : public TransformationEstimation<PointSource, PointTarget>
cdef extern from "pcl/registration/transformation_estimation_lm.h" namespace "pcl" nogil:
    cdef cppclass TransformationEstimationLM[Source, Target](TransformationEstimation[Source, Target]):
        TransformationEstimationLM() except +
        # ctypedef pcl::PointCloud<PointSource> PointCloudSource;
        # ctypedef typename PointCloudSource::Ptr PointCloudSourcePtr;
        # ctypedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;
        # ctypedef pcl::PointCloud<PointTarget> PointCloudTarget;
        # ctypedef PointIndices::Ptr PointIndicesPtr;
        # ctypedef PointIndices::ConstPtr PointIndicesConstPtr;
        # public:
        # TransformationEstimationLM (const TransformationEstimationLM &src)
        # TransformationEstimationLM& operator = (const TransformationEstimationLM &src)
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using LM.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # inline void estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     Eigen::Matrix4f &transformation_matrix);
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using LM.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] indices_src the vector of indices describing the points of interest in \a cloud_src
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # inline void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const std::vector<int> &indices_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     Eigen::Matrix4f &transformation_matrix);
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using LM.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] indices_src the vector of indices describing the points of interest in \a cloud_src
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[in] indices_tgt the vector of indices describing the correspondences of the interst points from 
        #   * \a indices_src
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # inline void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const std::vector<int> &indices_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     const std::vector<int> &indices_tgt,
        #     Eigen::Matrix4f &transformation_matrix);
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using LM.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[in] correspondences the vector of correspondences between source and target point cloud
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # inline void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     const pcl::Correspondences &correspondences,
        #     Eigen::Matrix4f &transformation_matrix);

        # /** \brief Set the function we use to warp points. Defaults to rigid 6D warp.
        #   * \param[in] warp_fcn a shared pointer to an object that warps points
        #   */
        # void
        # setWarpFunction (const boost::shared_ptr<WarpPointRigid<PointSource, PointTarget> > &warp_fcn)

        # /** Base functor all the models that need non linear optimization must
        #   * define their own one and implement operator() (const Eigen::VectorXd& x, Eigen::VectorXd& fvec)
        #   * or operator() (const Eigen::VectorXf& x, Eigen::VectorXf& fvec) dependening on the choosen _Scalar
        #   */
        # template<typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
        # struct Functor
        # {
        #   typedef _Scalar Scalar;
        #   enum 
        #   {
        #     InputsAtCompileTime = NX,
        #     ValuesAtCompileTime = NY
        #   };
        #   typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
        #   typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
        #   typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
        # 
        #   /** \brief Empty Construtor. */
        #   Functor () : m_data_points_ (ValuesAtCompileTime) {}
        #   /** \brief Constructor
        #     * \param[in] m_data_points number of data points to evaluate.
        #     */
        #   Functor (int m_data_points) : m_data_points_ (m_data_points) {}
        # 
        #   /** \brief Destructor. */
        #   virtual ~Functor () {}
        # 
        #   /** \brief Get the number of values. */ 
        #   int
        #   values () const { return (m_data_points_); }
        # 
        #   protected:
        #     int m_data_points_;
        # };
        # struct OptimizationFunctor : public Functor<double>
        # {
        #   using Functor<double>::values;
        # /** Functor constructor
        #     * \param[in] m_data_points the number of data points to evaluate
        #     * \param[in,out] estimator pointer to the estimator object
        #     */
        #   OptimizationFunctor (int m_data_points, TransformationEstimationLM<PointSource, PointTarget> *estimator) : 
        #     Functor<double> (m_data_points), estimator_ (estimator) {}
        #   /** Copy constructor
        #     * \param[in] the optimization functor to copy into this
        #     */
        #   inline OptimizationFunctor (const OptimizationFunctor &src) : 
        #     Functor<double> (src.m_data_points_), estimator_ ()
        #   {
        #     *this = src;
        #   }
        #   /** Copy operator
        #     * \param[in] the optimization functor to copy into this
        #     */
        #   inline OptimizationFunctor& 
        #   operator = (const OptimizationFunctor &src) 
        #   { 
        #     Functor<double>::operator=(src);
        #     estimator_ = src.estimator_; 
        #     return (*this); 
        #   }
        #   /** \brief Destructor. */
        #   virtual ~OptimizationFunctor () {}
        #   /** Fill fvec from x. For the current state vector x fill the f values
        #     * \param[in] x state vector
        #     * \param[out] fvec f values vector
        #     */
        #   int 
        #   operator () (const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;
        # 
        #   TransformationEstimationLM<PointSource, PointTarget> *estimator_;
        # };
        # struct OptimizationFunctorWithIndices : public Functor<double>
        # {
        #   using Functor<double>::values;
        #   /** Functor constructor
        #     * \param[in] m_data_points the number of data points to evaluate
        #     * \param[in,out] estimator pointer to the estimator object
        #     */
        #   OptimizationFunctorWithIndices (int m_data_points, TransformationEstimationLM *estimator) :
        #     Functor<double> (m_data_points), estimator_ (estimator) {}
        #   /** Copy constructor
        #     * \param[in] the optimization functor to copy into this
        #     */
        #   inline OptimizationFunctorWithIndices (const OptimizationFunctorWithIndices &src) : 
        #     Functor<double> (src.m_data_points_), estimator_ ()
        #   {
        #     *this = src;
        #   }
        #   /** Copy operator
        #     * \param[in] the optimization functor to copy into this
        #     */
        #   inline OptimizationFunctorWithIndices& 
        #   operator = (const OptimizationFunctorWithIndices &src) 
        #   { 
        #     Functor<double>::operator=(src);
        #     estimator_ = src.estimator_; 
        #     return (*this); 
        #   }
        # 
        #   /** \brief Destructor. */
        #   virtual ~OptimizationFunctorWithIndices () {}
        # 
        #   /** Fill fvec from x. For the current state vector x fill the f values
        #     * \param[in] x state vector
        #     * \param[out] fvec f values vector
        #     */
        #   int 
        #   operator () (const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;
        #   TransformationEstimationLM<PointSource, PointTarget> *estimator_;
        # };
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW

###

# transformation_estimation_point_to_plane.h
# template <typename PointSource, typename PointTarget>
# class TransformationEstimationPointToPlane : public TransformationEstimationLM<PointSource, PointTarget>
cdef extern from "pcl/registration/transformation_estimation_point_to_plane.h" namespace "pcl" nogil:
    cdef cppclass TransformationEstimationPointToPlane[Source, Target](TransformationEstimationLM[Source, Target]):
        TransformationEstimationPointToPlane ()
        # public:
        # ctypedef boost::shared_ptr<TransformationEstimationPointToPlane<PointSource, PointTarget> > Ptr;
        # ctypedef pcl::PointCloud<PointSource> PointCloudSource;
        # ctypedef typename PointCloudSource::Ptr PointCloudSourcePtr;
        # ctypedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;
        # ctypedef pcl::PointCloud<PointTarget> PointCloudTarget;
        # ctypedef PointIndices::Ptr PointIndicesPtr;
        # ctypedef PointIndices::ConstPtr PointIndicesConstPtr;
###

# transformation_estimation_point_to_plane_lls.h
# template <typename PointSource, typename PointTarget>
# class TransformationEstimationPointToPlaneLLS : public TransformationEstimation<PointSource, PointTarget>

cdef extern from "pcl/registration/transformation_estimation_point_to_plane_lls.h" namespace "pcl" nogil:
    cdef cppclass TransformationEstimationPointToPlaneLLS[Source, Target](TransformationEstimation[Source, Target]):
        TransformationEstimationPointToPlaneLLS ()
        # inline void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     Eigen::Matrix4f &transformation_matrix);
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using SVD.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] indices_src the vector of indices describing the points of interest in \a cloud_src
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # inline void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const std::vector<int> &indices_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     Eigen::Matrix4f &transformation_matrix);
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using SVD.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] indices_src the vector of indices describing the points of interest in \a cloud_src
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[in] indices_tgt the vector of indices describing the correspondences of the interst points from \a indices_src
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # inline void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const std::vector<int> &indices_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     const std::vector<int> &indices_tgt,
        #     Eigen::Matrix4f &transformation_matrix);
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using SVD.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[in] correspondences the vector of correspondences between source and target point cloud
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # inline void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     const pcl::Correspondences &correspondences,
        #     Eigen::Matrix4f &transformation_matrix);

###

# transformation_estimation_svd.h
# template <typename PointSource, typename PointTarget>
# class TransformationEstimationSVD : public TransformationEstimation<PointSource, PointTarget>
cdef extern from "pcl/registration/transformation_estimation_svd.h" namespace "pcl" nogil:
    cdef cppclass TransformationEstimationSVD[Source, Target](TransformationEstimation[Source, Target]):
        TransformationEstimationSVD ()
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using SVD.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # inline void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     Eigen::Matrix4f &transformation_matrix);
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using SVD.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] indices_src the vector of indices describing the points of interest in \a cloud_src
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # inline void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const std::vector<int> &indices_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     Eigen::Matrix4f &transformation_matrix);
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using SVD.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] indices_src the vector of indices describing the points of interest in \a cloud_src
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[in] indices_tgt the vector of indices describing the correspondences of the interst points from \a indices_src
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # inline void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const std::vector<int> &indices_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     const std::vector<int> &indices_tgt,
        #     Eigen::Matrix4f &transformation_matrix);
        # /** \brief Estimate a rigid rotation transformation between a source and a target point cloud using SVD.
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[in] correspondences the vector of correspondences between source and target point cloud
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   */
        # void
        # estimateRigidTransformation (
        #     const pcl::PointCloud<PointSource> &cloud_src,
        #     const pcl::PointCloud<PointTarget> &cloud_tgt,
        #     const pcl::Correspondences &correspondences,
        #     Eigen::Matrix4f &transformation_matrix);
        # protected:
        # /** \brief Obtain a 4x4 rigid transformation matrix from a correlation matrix H = src * tgt'
        #   * \param[in] cloud_src_demean the input source cloud, demeaned, in Eigen format
        #   * \param[in] centroid_src the input source centroid, in Eigen format
        #   * \param[in] cloud_tgt_demean the input target cloud, demeaned, in Eigen format
        #   * \param[in] centroid_tgt the input target cloud, in Eigen format
        #   * \param[out] transformation_matrix the resultant 4x4 rigid transformation matrix
        #   */ 
        # void
        # getTransformationFromCorrelation (const Eigen::MatrixXf &cloud_src_demean,
        #                                   const Eigen::Vector4f &centroid_src,
        #                                   const Eigen::MatrixXf &cloud_tgt_demean,
        #                                   const Eigen::Vector4f &centroid_tgt,
        #                                   Eigen::Matrix4f &transformation_matrix);
###

# transformation_validation.h
# template <typename PointSource, typename PointTarget>
# class TransformationValidation
cdef extern from "pcl/registration/transformation_validation.h" namespace "pcl" nogil:
    cdef cppclass TransformationValidation[Source, Target]:
        TransformationValidation ()
        # public:
        # ctypedef pcl::PointCloud<PointSource> PointCloudSource;
        # ctypedef typename PointCloudSource::Ptr PointCloudSourcePtr;
        # ctypedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;
        # ctypedef pcl::PointCloud<PointTarget> PointCloudTarget;
        # ctypedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
        # ctypedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;
        # /** \brief Validate the given transformation with respect to the input cloud data, and return a score.
        #   *
        #   * \param[in] cloud_src the source point cloud dataset
        #   * \param[in] cloud_tgt the target point cloud dataset
        #   * \param[out] transformation_matrix the resultant transformation matrix
        #   *
        #   * \return the score or confidence measure for the given
        #   * transformation_matrix with respect to the input data
        #   */
        # virtual double validateTransformation (
        #    const cpp.PointCloudPtr_t &cloud_src,
        #    const cpp.PointCloudPtr_t &cloud_tgt,
        #    const Matrix4f &transformation_matrix) = 0;
        # ctypedef shared_ptr[TransformationValidation[PointSource, PointTarget] ] Ptr;
        # ctypedef shared_ptr[const TransformationValidation[PointSource, PointTarget] ] ConstPtr;
###

# transformation_validation_euclidean.h
# template <typename PointSource, typename PointTarget>
# class TransformationValidationEuclidean
cdef extern from "pcl/registration/transformation_validation_euclidean.h" namespace "pcl" nogil:
    cdef cppclass TransformationValidationEuclidean[Source, Target]:
        TransformationValidationEuclidean ()
        # public:
        # ctypedef boost::shared_ptr<TransformationValidation<PointSource, PointTarget> > Ptr;
        # ctypedef boost::shared_ptr<const TransformationValidation<PointSource, PointTarget> > ConstPtr;
        # ctypedef typename pcl::KdTree<PointTarget> KdTree;
        # ctypedef typename pcl::KdTree<PointTarget>::Ptr KdTreePtr;
        # ctypedef typename KdTree::PointRepresentationConstPtr PointRepresentationConstPtr;
        # ctypedef typename TransformationValidation<PointSource, PointTarget>::PointCloudSourceConstPtr PointCloudSourceConstPtr;
        # ctypedef typename TransformationValidation<PointSource, PointTarget>::PointCloudTargetConstPtr PointCloudTargetConstPtr;
        inline void setMaxRange (double max_range)
        double validateTransformation (
            const cpp.PointCloudPtr_t &cloud_src,
            const cpp.PointCloudPtr_t &cloud_tgt,
            const Matrix4f &transformation_matrix)
###

# transforms.h
# common/transforms.h
###

# warp_point_rigid_3d.h
# template <class PointSourceT, class PointTargetT>
# class WarpPointRigid3D : public WarpPointRigid<PointSourceT, PointTargetT>
cdef extern from "pcl/registration/warp_point_rigid_3d.h" namespace "pcl" nogil:
    cdef cppclass WarpPointRigid3D[Source, Target](WarpPointRigid[Source, Target]):
        WarpPointRigid3D ()
        # public:
        # virtual void setParam (const Eigen::VectorXf & p)
###

# warp_point_rigid_6d.h
# template <class PointSourceT, class PointTargetT>
# class WarpPointRigid6D : public WarpPointRigid<PointSourceT, PointTargetT>
cdef extern from "pcl/registration/warp_point_rigid_6d.h" namespace "pcl" nogil:
    cdef cppclass WarpPointRigid6D[Source, Target](WarpPointRigid[Source, Target]):
        WarpPointRigid6D ()
        # public:
        # virtual void setParam (const Eigen::VectorXf & p)

###

