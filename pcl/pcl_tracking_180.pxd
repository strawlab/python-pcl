from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

# boost
from boost_shared_ptr cimport shared_ptr

###############################################################################
# Types
###############################################################################

### base class ###

# class Tracker: public PCLBase<PointInT>
cdef extern from "pcl/tracking/tracker.h" namespace "pcl::tracking":
    cdef cppclass Tracker[T](cpp.PCLBase[T]):
        Tracker ()
        # using PCLBase<PointInT>::deinitCompute;
        # using PCLBase<PointInT>::indices_;
        # using PCLBase<PointInT>::input_;
        # ctypedef PCLBase<PointInT> BaseClass;
        # ctypedef boost::shared_ptr< Tracker<PointInT, StateT> > Ptr;
        # ctypedef boost::shared_ptr< const Tracker<PointInT, StateT> > ConstPtr;
        # ctypedef boost::shared_ptr<pcl::search::Search<PointInT> > SearchPtr;
        # ctypedef boost::shared_ptr<const pcl::search::Search<PointInT> > SearchConstPtr;
        # ctypedef pcl::PointCloud<PointInT> PointCloudIn;
        # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
        # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # ctypedef pcl::PointCloud<StateT> PointCloudState;
        # ctypedef typename PointCloudState::Ptr PointCloudStatePtr;
        # ctypedef typename PointCloudState::ConstPtr PointCloudStateConstPtr;
        # public:
        # brief Base method for tracking for all points given in 
        # <setInputCloud (), setIndices ()> using the indices in setIndices () 
        cdef void compute ()
        # protected:
        # brief The tracker name. 
        # std::string tracker_name_;
        # brief A pointer to the spatial search object. 
        # SearchPtr search_;
        # brief Get a string representation of the name of this class.
        # cdef inline const std::string& getClassName ()
        # brief This method should get called before starting the actual computation.
        # cdef bool initCompute ();
        # brief Provide a pointer to a dataset to add additional information
        # to estimate the features for every point in the input dataset.  This
        # is optional, if this is not set, it will only use the data in the
        # input cloud to estimate the features.  This is useful when you only
        # need to compute the features for a downsampled cloud.  
        # \param cloud a pointer to a PointCloud message
        # cdef void setSearchMethod (const SearchPtr &)
        # brief Get a pointer to the point cloud dataset.
        # inline SearchPtr getSearchMethod ()
        # brief Get an instance of the result of tracking.
        # virtual StateT getResult () const = 0;
###

cdef extern from "pcl/tracking/coherence.h" namespace "pcl::tracking":
    cdef cppclass PointCoherence[T]:
        PointCoherence ()
        # public:
        # ctypedef boost::shared_ptr< PointCoherence<PointInT> > Ptr;
        # ctypedef boost::shared_ptr< const PointCoherence<PointInT> > ConstPtr;
        # public:
        # cdef double compute (PointInT &source, PointInT &target);
        # protected:
        # std::string coherence_name_;
        # cdef double computeCoherence (PointInT &source, PointInT &target) = 0;
        # cdef const std::string& getClassName () const { return (coherence_name_);

###

cdef extern from "pcl/tracking/coherence.h" namespace "pcl::tracking":
    cdef cppclass PointCloudCoherence[T]:
        PointCloudCoherence ()
        # public:
        # ctypedef boost::shared_ptr< PointCloudCoherence<PointInT> > Ptr;
        # ctypedef boost::shared_ptr< const PointCloudCoherence<PointInT> > ConstPtr;
        # ctypedef pcl::PointCloud<PointInT> PointCloudIn;
        # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
        # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # ctypedef typename PointCoherence<PointInT>::Ptr PointCoherencePtr;
        cdef void compute (const PointCloudInConstPtr &cloud, const IndicesConstPtr &indices, float &w_i);
        # cdef vector[PointCoherencePtr] getPointCoherences ()
        cdef void setPointCoherences (std::vector<PointCoherencePtr> coherences)
        cdef bool initCompute ()
        cdef void addPointCoherence (PointCoherencePtr coherence)
        cdef void setTargetCloud (const PointCloudInConstPtr &cloud)
        # protected:
        # cdef void computeCoherence (const PointCloudInConstPtr &cloud, const IndicesConstPtr &indices, float &w_j) = 0;
        # cdef double calcPointCoherence (PointInT &source, PointInT &target);
        # cdef const std::string& getClassName () const { return (coherence_name_); }
        # std::string coherence_name_;
        # PointCloudInConstPtr target_input_;
        # std::vector<PointCoherencePtr> point_coherences_;
###

# class NearestPairPointCloudCoherence: public PointCloudCoherence<PointInT>
cdef extern from "pcl/tracking/nearest_pair_point_cloud_coherence.h" namespace "pcl::tracking":
    cdef cppclass NearestPairPointCloudCoherence[T](PointCoherence[T]):
        NearestPairPointCloudCoherence ()
        # public:
        # using PointCloudCoherence<PointInT>::getClassName;
        # using PointCloudCoherence<PointInT>::coherence_name_;
        # using PointCloudCoherence<PointInT>::target_input_;
        # ctypedef typename PointCloudCoherence<PointInT>::PointCoherencePtr PointCoherencePtr;
        # ctypedef typename PointCloudCoherence<PointInT>::PointCloudInConstPtr PointCloudInConstPtr;
        # ctypedef PointCloudCoherence<PointInT> BaseClass;
        # ctypedef boost::shared_ptr<NearestPairPointCloudCoherence<PointInT> > Ptr;
        # ctypedef boost::shared_ptr<const NearestPairPointCloudCoherence<PointInT> > ConstPtr;
        # ctypedef boost::shared_ptr<pcl::search::Search<PointInT> > SearchPtr;
        # ctypedef boost::shared_ptr<const pcl::search::Search<PointInT> > SearchConstPtr;
        # brief Provide a pointer to a dataset to add additional information
        # to estimate the features for every point in the input dataset.  This
        # is optional, if this is not set, it will only use the data in the
        # input cloud to estimate the features.  This is useful when you only
        # need to compute the features for a downsampled cloud. 
        # param cloud a pointer to a PointCloud message
        cdef void setSearchMethod (const SearchPtr &search)
        # brief Get a pointer to the point cloud dataset.
        # cdef SearchPtr getSearchMethod ()
        # brief add a PointCoherence to the PointCloudCoherence.
        # param coherence a pointer to PointCoherence.
        cdef void setTargetCloud (const PointCloudInConstPtr &cloud)
        # brief set maximum distance to be taken into account.
        # param maximum distance.
        cdef void setMaximumDistance (double )
        # protected:
        # using PointCloudCoherence<PointInT>::point_coherences_;
        # brief This method should get called before starting the actual computation.
        # virtual bool initCompute ();
        # brief A flag which is true if target_input_ is updated
        # bool new_target_;
        # brief A pointer to the spatial search object.
        # SearchPtr search_;
        # brief max of distance for points to be taken into account
        # double maximum_distance_;
        # brief compute the nearest pairs and compute coherence using point_coherences_ 
        # cdef void computeCoherence (const PointCloudInConstPtr &cloud, const IndicesConstPtr &indices, float &w_j);

###

# class ParticleFilterTracker: public Tracker<PointInT, StateT>
cdef extern from "pcl/tracking/particle_filter.h" namespace "pcl::tracking":
    cdef cppclass ParticleFilterTracker[T, S](Tracker[T]):
        ParticleFilterTracker ()
        # protected:
        # using Tracker<PointInT, StateT>::deinitCompute;
        # public:
        # using Tracker<PointInT, StateT>::tracker_name_;
        # using Tracker<PointInT, StateT>::search_;
        # using Tracker<PointInT, StateT>::input_;
        # using Tracker<PointInT, StateT>::indices_;
        # using Tracker<PointInT, StateT>::getClassName;
        # ctypedef Tracker<PointInT, StateT> BaseClass;
        # ctypedef typename Tracker<PointInT, StateT>::PointCloudIn PointCloudIn;
        # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
        # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # ctypedef typename Tracker<PointInT, StateT>::PointCloudState PointCloudState;
        # ctypedef typename PointCloudState::Ptr PointCloudStatePtr;
        # ctypedef typename PointCloudState::ConstPtr PointCloudStateConstPtr;
        # ctypedef PointCoherence<PointInT> Coherence;
        # ctypedef boost::shared_ptr< Coherence > CoherencePtr;
        # ctypedef boost::shared_ptr< const Coherence > CoherenceConstPtr;
        # ctypedef PointCloudCoherence<PointInT> CloudCoherence;
        # ctypedef boost::shared_ptr< CloudCoherence > CloudCoherencePtr;
        # ctypedef boost::shared_ptr< const CloudCoherence > CloudCoherenceConstPtr;
        # brief set the number of iteration.
        # param iteration_num the number of iteration.
        cdef void setIterationNum (int )
        # brief get the number of iteration.
        cdef int getIterationNum ()
        # brief set the number of the particles.
        # param particle_num the number of the particles.
        cdef void setParticleNum (const int )
        # brief get the number of the particles.
        cdef int getParticleNum ()
        # brief set a pointer to a reference dataset to be tracked.
        # param cloud a pointer to a PointCloud message
        cdef void setReferenceCloud (const PointCloudInConstPtr &ref)
        # brief get a pointer to a reference dataset to be tracked.
        cdef PointCloudInConstPtr const getReferenceCloud ()
        # brief set the PointCloudCoherence as likelihood.
        # param coherence a pointer to PointCloudCoherence.
        cdef void setCloudCoherence (const CloudCoherencePtr &coherence)
        # brief get the PointCloudCoherence to compute likelihood.
        cdef CloudCoherencePtr getCloudCoherence ()
        # brief set the covariance of step noise.
        # param step_noise_covariance the diagonal elements of covariance matrix of step noise.
        cdef void setStepNoiseCovariance (const std::vector<double> &step_noise_covariance)
        # brief set the covariance of the initial noise.
        # it will be used when initializing the particles.
        # param initial_noise_covariance the diagonal elements of covariance matrix of initial noise.
        cdef void setInitialNoiseCovariance (const std::vector<double> &initial_noise_covariance)
        # brief set the mean of the initial noise.
        # it will be used when initializing the particles.
        # param initial_noise_mean the mean values of initial noise.
        cdef void setInitialNoiseMean (const std::vector<double> &initial_noise_mean)
        # brief set the threshold to re-initialize the particles.
        # param resample_likelihood_thr threshold to re-initialize.
        cdef void setResampleLikelihoodThr (const double resample_likelihood_thr)
        # brief set the threshold of angle to be considered occlusion (default: pi/2).
        # ParticleFilterTracker does not take the occluded points into account according to the angle
        # between the normal and the position. 
        # param occlusion_angle_thr threshold of angle to be considered occlusion.
        cdef void setOcclusionAngleThe (const double occlusion_angle_thr)
        # brief set the minimum number of indices (default: 1).
        # ParticleFilterTracker does not take into account the hypothesis
        # whose the number of points is smaller than the minimum indices.
        # param min_indices the minimum number of indices.
        cdef void setMinIndices (const int min_indices)
        # brief set the transformation from the world coordinates to the frame of the particles.
        # param trans Affine transformation from the worldcoordinates to the frame of the particles.
        cdef void setTrans (const Eigen::Affine3f &trans)
        # brief get the transformation from the world coordinates to the frame of the particles.
        cdef Eigen::Affine3f getTrans () const { return trans_; }
        # brief Get an instance of the result of tracking.
        # cdef StateT getResult () const { return representative_state_; }
        # brief convert a state to affine transformation from the world coordinates frame.
        # param particle an instance of StateT.
        cdef Eigen::Affine3f toEigenMatrix (const StateT& particle)
        # brief get a pointer to a pointcloud of the particles.
        cdef PointCloudStatePtr getParticles ()
        # brief normalize the weight of a particle using
        # exp(1- alpha ( w - w_{min}) / (w_max - w_min)).
        # this method is described in [P.Azad et. al, ICRA11].
        # param w the weight to be normalized
        # param w_min the minimum weight of the particles
        # param w_max the maximum weight of the particles
        cdef double normalizeParticleWeight (double , double , double )
        # brief set the value of alpha.
        # param alpha the value of alpha
        cdef void setAlpha (double)
        # brief get the value of alpha.
        cdef double getAlpha ()
        # brief set the value of use_normal_.
        # param use_normal the value of use_normal_.
        cdef void setUseNormal (bool)
        # brief get the value of use_normal_.
        cdef bool getUseNormal ()
        # brief set the value of use_change_detector_.
        # param use_normal the value of use_change_detector_.
        cdef void setUseChangeDetector (bool )
        # brief get the value of use_change_detector_.
        cdef bool getUseChangeDetector ()
        # brief set the motion ratio
        # param motion_ratio the ratio of hypothesis to use motion model.
        cdef void setMotionRatio (double )
        # brief get the motion ratio
        cdef double getMotionRatio ()
        # brief set the number of interval frames to run change detection.
        # param change_detector_interval the number of interval frames.
        cdef void setIntervalOfChangeDetection (unsigned int )
        # brief get the number of interval frames to run change detection.
        cdef unsigned int getIntervalOfChangeDetection ()
        # brief set the minimum amount of points required within leaf node to become serialized in change detection
        # param change_detector_filter the minimum amount of points required within leaf node
        cdef void setMinPointsOfChangeDetection (unsigned int change_detector_filter)
        # brief set the resolution of change detection.
        # param resolution resolution of change detection octree
        cdef void setResolutionOfChangeDetection (double )
        # brief get the resolution of change detection.
        cdef double getResolutionOfChangeDetection ()
        # brief get the minimum amount of points required within leaf node to become serialized in change detection
        cdef unsigned int getMinPointsOfChangeDetection ()
        # brief get the adjustment ratio.
        cdef double getFitRatio()
        # brief reset the particles to restart tracking
        cdef void resetTracking ()
        ###
        # protected:
        # brief compute the parameters for the bounding box of 
        # hypothesis pointclouds.
        # param x_min the minimum value of x axis.
        # param x_max the maximum value of x axis.
        # param y_min the minimum value of y axis.
        # param y_max the maximum value of y axis.
        # param z_min the minimum value of z axis.
        # param z_max the maximum value of z axis.
        cdef void calcBoundingBox (double &x_min, double &x_max,
                              double &y_min, double &y_max,
                              double &z_min, double &z_max);
        # brief crop the pointcloud by the bounding box calculated
        #  from hypothesis and the reference pointcloud.
        # param cloud a pointer to pointcloud to be cropped.
        # param output a pointer to be assigned the cropped pointcloud.
        cdef void cropInputPointCloud (const PointCloudInConstPtr &cloud, PointCloudIn &output);

        # brief compute a reference pointcloud transformed to the pose that
        # hypothesis represents.
        # param hypothesis a particle which represents a hypothesis.
        # param indices the indices which should be taken into account.
        # param cloud the resultant point cloud model dataset which
        #             is transformed to hypothesis.
        cdef void computeTransformedPointCloud (const StateT& hypothesis,
                                           std::vector<int>& indices,
                                           PointCloudIn &cloud);
        # brief compute a reference pointcloud transformed to the pose that
        # hypothesis represents and calculate indices taking occlusion into \
        #  account.
        # param hypothesis a particle which represents a hypothesis.
        # param indices the indices which should be taken into account.
        # param cloud the resultant point cloud model dataset which
        #           is transformed to hypothesis.
        cdef void computeTransformedPointCloudWithNormal (const StateT& hypothesis,
                                           std::vector<int>& indices,
                                           PointCloudIn &cloud);
        # brief compute a reference pointcloud transformed to the pose that
        # hypothesis represents and calculate indices without taking
        # occlusion into account.
        # param hypothesis a particle which represents a hypothesis.
        # param cloud the resultant point cloud model dataset which
        #         is transformed to hypothesis.
        cdef void computeTransformedPointCloudWithoutNormal (const StateT& hypothesis, PointCloudIn &cloud);
        # brief This method should get called before starting the actual computation.
        cdef bool initCompute ()
        # brief weighting phase of particle filter method.
        # calculate the likelihood of all of the particles and set the weights.
        cdef void weight ()
        # brief resampling phase of particle filter method.
        # sampling the particles according to the weights calculated in weight method.
        # in particular, "sample with replacement" is archieved by walker's alias method.
        cdef void resample ()
        # brief calculate the weighted mean of the particles and set it as the result
        cdef void update ()
        # brief normalize the weights of all the particels.
        cdef void normalizeWeight ()
        # brief initialize the particles. initial_noise_covariance_ and initial_noise_mean_ are
        # used for gausiaan sampling.
        cdef void initParticles (bool reset)
        # brief track the pointcloud using particle filter method.
        cdef void computeTracking ()
        # brief implementation of "sample with replacement" using Walker's alias method.
        # about Walker's alias method, you can check the paper below:
        # param a an alias table, which generated by genAliasTable.
        # param q a table of weight, which generated by genAliasTable.
        cdef int sampleWithReplacement (const std::vector<int>& a, const std::vector<double>& q)
        # brief generate the tables for walker's alias method
        cdef void genAliasTable (std::vector<int> &a, std::vector<double> &q, const PointCloudStateConstPtr &particles)
        # brief resampling the particle with replacement
        cdef void resampleWithReplacement ()
        # brief resampling the particle in deterministic way
        cdef void resampleDeterministic ()
        # brief run change detection and return true if there is a change.
        # param input a pointer to the input pointcloud.
        cdef bool testChangeDetection (const PointCloudInConstPtr &input)
        # the number of iteration of particlefilter.
        # int iteration_num_;
        # brief the number of the particles.
        int particle_num_;
        # brief the minimum number of points which the hypothesis should have.
        int min_indices_;
        # brief adjustment of the particle filter.
        double fit_ratio_;
        # brief a pointer to reference point cloud.
        PointCloudInConstPtr ref_;
        # brief a pointer to the particles
        PointCloudStatePtr particles_;
        # brief a pointer to PointCloudCoherence.
        CloudCoherencePtr coherence_;
        # brief the diagonal elements of covariance matrix of the step noise. the covariance matrix is used
        #    at every resample method.
        std::vector<double> step_noise_covariance_;
        # brief the diagonal elements of covariance matrix of the initial noise. the covariance matrix is used
        # when initialize the particles.
        std::vector<double> initial_noise_covariance_;
        # brief the mean values of initial noise.
        std::vector<double> initial_noise_mean_;
        # brief the threshold for the particles to be re-initialized
        double resample_likelihood_thr_;
        # brief the threshold for the points to be considered as occluded
        double occlusion_angle_thr_;
        # brief the weight to be used in normalization
        #         of the weights of the particles 
        double alpha_;
        # brief the result of tracking.
        StateT representative_state_;
        # brief an affine transformation from the world coordinates frame to the origin of the particles
        Eigen::Affine3f trans_;
        # brief a flag to use normal or not. defaults to false
        bool use_normal_;
        # brief difference between the result in t and t-1
        StateT motion_;
        # brief ratio of hypothesis to use motion model
        double motion_ratio_;
        # brief pass through filter to crop the pointclouds within the hypothesis bounding box
        pcl::PassThrough<PointInT> pass_x_;
        # brief pass through filter to crop the pointclouds within the hypothesis bounding box
        pcl::PassThrough<PointInT> pass_y_;
        # brief pass through filter to crop the pointclouds within the hypothesis bounding box
        pcl::PassThrough<PointInT> pass_z_;
        # brief a list of the pointers to pointclouds
        std::vector<PointCloudInPtr> transed_reference_vector_;
        # brief change detector used as a trigger to track
        boost::shared_ptr<pcl::octree::OctreePointCloudChangeDetector<PointInT> > change_detector_;
        # brief a flag to be true when change of pointclouds is detected
        bool changed_;
        # brief a counter to skip change detection
        unsigned int change_counter_;
        # brief minimum points in a leaf when calling change detector. defaults to 10
        unsigned int change_detector_filter_;
        # brief the number of interval frame to run change detection. defaults to 10.
        unsigned int change_detector_interval_;
        # brief resolution of change detector. defaults to 0.01.
        double change_detector_resolution_;
        # brief the flag which will be true if using change detection
        bool use_change_detector_;
###

### Inheritance ###

# class ApproxNearestPairPointCloudCoherence: public NearestPairPointCloudCoherence<PointInT>
cdef extern from "pcl/tracking/approx_nearest_pair_point_cloud_coherence.h" namespace "pcl::tracking":
    cdef cppclass ApproxNearestPairPointCloudCoherence[T](NearestPairPointCloudCoherence[T]):
        ApproxNearestPairPointCloudCoherence ()
        # public:
        # ctypedef typename NearestPairPointCloudCoherence<PointInT>::PointCoherencePtr PointCoherencePtr;
        # ctypedef typename NearestPairPointCloudCoherence<PointInT>::PointCloudInConstPtr PointCloudInConstPtr;
        # using NearestPairPointCloudCoherence<PointInT>::maximum_distance_;
        # using NearestPairPointCloudCoherence<PointInT>::target_input_;
        # using NearestPairPointCloudCoherence<PointInT>::point_coherences_;
        # using NearestPairPointCloudCoherence<PointInT>::coherence_name_;
        # using NearestPairPointCloudCoherence<PointInT>::new_target_;
        # using NearestPairPointCloudCoherence<PointInT>::getClassName;

        # protected:
        # cdef bool initCompute ();
        # cdef void computeCoherence (const PointCloudInConstPtr &cloud, const IndicesConstPtr &indices, float &w_j);
        # typename boost::shared_ptr<pcl::search::Octree<PointInT> > search_;

###

# class DistanceCoherence: public PointCoherence<PointInT>
cdef extern from "pcl/tracking/distance_coherence.h" namespace "pcl::tracking":
    cdef cppclass DistanceCoherence[T](PointCoherence[T]):
        DistanceCoherence ()
        cdef void setWeight (double)
        cdef double getWeight ()
        # protected:
        # cdef double computeCoherence (PointInT &source, PointInT &target);
        # double weight_;
###

cdef extern from "pcl/tracking/hsv_color_coherence.h" namespace "pcl::tracking":
    cdef cppclass HSVColorCoherence[T]:
        HSVColorCoherence ()
        cdef void setWeight (double)
        cdef double getWeight ()
        # public:
        cdef void setWeight (double )
        cdef double getWeight ()
        cdef void setHWeight (double )
        cdef double getHWeight ()
        cdef void setSWeight (double )
        cdef double getSWeight ()
        cdef void setVWeight (double )
        cdef double getVWeight ()
        # protected:
        # cdef double computeCoherence (PointInT &source, PointInT &target);
        # double weight_;
        # double h_weight_;
        # double s_weight_;
        # double v_weight_;

###

# class KLDAdaptiveParticleFilterTracker: public ParticleFilterTracker<PointInT, StateT>
cdef extern from "pcl/tracking/kld_adaptive_particle_filter.h" namespace "pcl::tracking":
    cdef cppclass KLDAdaptiveParticleFilterTracker[T, S](ParticleFilterTracker[T, S]):
        KLDAdaptiveParticleFilterTracker ()
        # public:
        # using Tracker<PointInT, StateT>::tracker_name_;
        # using Tracker<PointInT, StateT>::search_;
        # using Tracker<PointInT, StateT>::input_;
        # using Tracker<PointInT, StateT>::getClassName;
        # using ParticleFilterTracker<PointInT, StateT>::transed_reference_vector_;
        # using ParticleFilterTracker<PointInT, StateT>::coherence_;
        # using ParticleFilterTracker<PointInT, StateT>::initParticles;
        # using ParticleFilterTracker<PointInT, StateT>::weight;
        # using ParticleFilterTracker<PointInT, StateT>::update;
        # using ParticleFilterTracker<PointInT, StateT>::iteration_num_;
        # using ParticleFilterTracker<PointInT, StateT>::particle_num_;
        # using ParticleFilterTracker<PointInT, StateT>::particles_;
        # using ParticleFilterTracker<PointInT, StateT>::use_normal_;
        # using ParticleFilterTracker<PointInT, StateT>::use_change_detector_;
        # using ParticleFilterTracker<PointInT, StateT>::change_detector_resolution_;
        # using ParticleFilterTracker<PointInT, StateT>::change_detector_;
        # using ParticleFilterTracker<PointInT, StateT>::motion_;
        # using ParticleFilterTracker<PointInT, StateT>::motion_ratio_;
        # using ParticleFilterTracker<PointInT, StateT>::step_noise_covariance_;
        # using ParticleFilterTracker<PointInT, StateT>::representative_state_;
        # using ParticleFilterTracker<PointInT, StateT>::sampleWithReplacement;
        # ctypedef Tracker<PointInT, StateT> BaseClass;
        # ctypedef typename Tracker<PointInT, StateT>::PointCloudIn PointCloudIn;
        # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
        # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # ctypedef typename Tracker<PointInT, StateT>::PointCloudState PointCloudState;
        # ctypedef typename PointCloudState::Ptr PointCloudStatePtr;
        # ctypedef typename PointCloudState::ConstPtr PointCloudStateConstPtr;
        # ctypedef PointCoherence<PointInT> Coherence;
        # ctypedef boost::shared_ptr< Coherence > CoherencePtr;
        # ctypedef boost::shared_ptr< const Coherence > CoherenceConstPtr;
        # ctypedef PointCloudCoherence<PointInT> CloudCoherence;
        # ctypedef boost::shared_ptr< CloudCoherence > CloudCoherencePtr;
        # ctypedef boost::shared_ptr< const CloudCoherence > CloudCoherenceConstPtr;
        # cdef void setBinSize (const StateT& bin_size) { bin_size_ = bin_size; }
        # cdef StateT getBinSize () const { return (bin_size_); }
        # cdef void setMaximumParticleNum (unsigned int nr) { maximum_particle_number_ = nr; }
        # cdef unsigned int getMaximumParticleNum () const { return (maximum_particle_number_); }
        # cdef void setEpsilon (double eps) { epsilon_ = eps; }
        # cdef double getEpsilon () const { return (epsilon_); }
        #cdef void setDelta (double delta) { delta_ = delta; }
        # brief get delta to be used in chi-squared distribution.
        cdef double getDelta () const { return (delta_); }
        # protected:
        # brief return true if the two bins are equal.
        # param a index of the bin
        # param b index of the bin
        # cdef bool equalBin (std::vector<int> a, std::vector<int> b)
        # brief return upper quantile of standard normal distribution.
        # param[in] u ratio of quantile.
        # double normalQuantile (double u)
        # brief calculate K-L boundary. K-L boundary follows 1/2e*chi(k-1, 1-d)^2.
        # param[in] k the number of bins and the first parameter of chi distribution.
        # cdef double calcKLBound (int k)
        # brief insert a bin into the set of the bins. if that bin is already registered,
        #   return false. if not, return true.
        # param bin a bin to be inserted.
        # param B a set of the bins
        # cdef bool insertIntoBins (std::vector<int> bin, std::vector<std::vector<int> > &B);
        # brief This method should get called before starting the actual computation.
        # cdef bool initCompute ();
        # brief resampling phase of particle filter method.
        #  sampling the particles according to the weights calculated in weight method.
        #  in particular, "sample with replacement" is archieved by walker's alias method.
        # cdef void resample ();
        # brief the maximum number of the particles.
        # unsigned int maximum_particle_number_;
        # brief error between K-L distance and MLE
        # double epsilon_;
        # brief probability of distance between K-L distance and MLE is less than epsilon_
        # double delta_;
        # brief the size of a bin.
        # StateT bin_size_;
###

# class KLDAdaptiveParticleFilterOMPTracker: public KLDAdaptiveParticleFilterTracker<PointInT, StateT>
cdef extern from "pcl/tracking/kld_adaptive_particle_filter_omp.h" namespace "pcl::tracking":
    cdef cppclass KLDAdaptiveParticleFilterOMPTracker[T, S](KLDAdaptiveParticleFilterTracker[T, S]):
        KLDAdaptiveParticleFilterOMPTracker ()
        KLDAdaptiveParticleFilterOMPTracker (unsigned int )
        # public:
        # using Tracker<PointInT, StateT>::tracker_name_;
        # using Tracker<PointInT, StateT>::search_;
        # using Tracker<PointInT, StateT>::input_;
        # using Tracker<PointInT, StateT>::indices_;
        # using Tracker<PointInT, StateT>::getClassName;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::particles_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::change_detector_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::change_counter_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::change_detector_interval_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::use_change_detector_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::pass_x_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::pass_y_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::pass_z_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::alpha_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::changed_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::coherence_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::use_normal_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::particle_num_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::change_detector_filter_;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::transed_reference_vector_;
        # //using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::calcLikelihood;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::normalizeWeight;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::normalizeParticleWeight;
        # using KLDAdaptiveParticleFilterTracker<PointInT, StateT>::calcBoundingBox;
        # ctypedef Tracker<PointInT, StateT> BaseClass;
        # ctypedef typename Tracker<PointInT, StateT>::PointCloudIn PointCloudIn;
        # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
        # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # ctypedef typename Tracker<PointInT, StateT>::PointCloudState PointCloudState;
        # ctypedef typename PointCloudState::Ptr PointCloudStatePtr;
        # ctypedef typename PointCloudState::ConstPtr PointCloudStateConstPtr;
        # ctypedef PointCoherence<PointInT> Coherence;
        # ctypedef boost::shared_ptr< Coherence > CoherencePtr;
        # ctypedef boost::shared_ptr< const Coherence > CoherenceConstPtr;
        # ctypedef PointCloudCoherence<PointInT> CloudCoherence;
        # ctypedef boost::shared_ptr< CloudCoherence > CloudCoherencePtr;
        # ctypedef boost::shared_ptr< const CloudCoherence > CloudCoherenceConstPtr;
        # brief Initialize the scheduler and set the number of threads to use.
        # param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
        void setNumberOfThreads (unsigned int nr_threads)
        # protected:
        # brief The number of threads the scheduler should use.
        # unsigned int threads_;
        # brief weighting phase of particle filter method.
        # calculate the likelihood of all of the particles and set the weights.
        void weight ();
###

# class NormalCoherence: public PointCoherence<PointInT>
cdef extern from "pcl/tracking/normal_coherence.h" namespace "pcl::tracking":
    cdef cppclass NormalCoherence[T](ParticleFilterTracker[T, S]):
        NormalCoherence ()
        # brief set the weight of coherence
        # param weight the weight of coherence
        cdef void setWeight (double )
        # brief get the weight of coherence
        cdef double getWeight ()
        # protected:
        # brief return the normal coherence between the two points.
        # param source instance of source point.
        # param target instance of target point.
        #
        # double computeCoherence (PointInT &source, PointInT &target);
        # double weight_;

###

# class ParticleFilterOMPTracker: public ParticleFilterTracker<PointInT, StateT>
cdef extern from "pcl/tracking/particle_filter_omp.h" namespace "pcl::tracking":
    cdef cppclass ParticleFilterOMPTracker[T, S](ParticleFilterTracker[T, S]):
        ParticleFilterOMPTracker ()
        # brief Initialize the scheduler and set the number of threads to use.
        # param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
        ParticleFilterOMPTracker (unsigned int )
        # public:
        # using Tracker<PointInT, StateT>::tracker_name_;
        # using Tracker<PointInT, StateT>::search_;
        # using Tracker<PointInT, StateT>::input_;
        # using Tracker<PointInT, StateT>::indices_;
        # using Tracker<PointInT, StateT>::getClassName;
        # using ParticleFilterTracker<PointInT, StateT>::particles_;
        # using ParticleFilterTracker<PointInT, StateT>::change_detector_;
        # using ParticleFilterTracker<PointInT, StateT>::change_counter_;
        # using ParticleFilterTracker<PointInT, StateT>::change_detector_interval_;
        # using ParticleFilterTracker<PointInT, StateT>::use_change_detector_;
        # using ParticleFilterTracker<PointInT, StateT>::alpha_;
        # using ParticleFilterTracker<PointInT, StateT>::changed_;
        # using ParticleFilterTracker<PointInT, StateT>::coherence_;
        # using ParticleFilterTracker<PointInT, StateT>::use_normal_;
        # using ParticleFilterTracker<PointInT, StateT>::particle_num_;
        # using ParticleFilterTracker<PointInT, StateT>::change_detector_filter_;
        # using ParticleFilterTracker<PointInT, StateT>::transed_reference_vector_;
        # //using ParticleFilterTracker<PointInT, StateT>::calcLikelihood;
        # using ParticleFilterTracker<PointInT, StateT>::normalizeWeight;
        # using ParticleFilterTracker<PointInT, StateT>::normalizeParticleWeight;
        # using ParticleFilterTracker<PointInT, StateT>::calcBoundingBox;
        # ctypedef Tracker<PointInT, StateT> BaseClass;
        # ctypedef typename Tracker<PointInT, StateT>::PointCloudIn PointCloudIn;
        # ctypedef typename PointCloudIn::Ptr PointCloudInPtr;
        # ctypedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # ctypedef typename Tracker<PointInT, StateT>::PointCloudState PointCloudState;
        # ctypedef typename PointCloudState::Ptr PointCloudStatePtr;
        # ctypedef typename PointCloudState::ConstPtr PointCloudStateConstPtr;
        # ctypedef PointCoherence<PointInT> Coherence;
        # ctypedef boost::shared_ptr< Coherence > CoherencePtr;
        # ctypedef boost::shared_ptr< const Coherence > CoherenceConstPtr;
        # ctypedef PointCloudCoherence<PointInT> CloudCoherence;
        # ctypedef boost::shared_ptr< CloudCoherence > CloudCoherencePtr;
        # ctypedef boost::shared_ptr< const CloudCoherence > CloudCoherenceConstPtr;
        # brief Initialize the scheduler and set the number of threads to use.
        # param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
        void setNumberOfThreads (unsigned int nr_threads)
        # protected:
        # brief The number of threads the scheduler should use.
        # unsigned int threads_;
        # brief weighting phase of particle filter method.
        # calculate the likelihood of all of the particles and set the weights.
        void weight ();

###

cdef extern from "pcl/tracking/tracking.h" namespace "pcl::tracking":
    # state definition
    cdef struct ParticleXYZRPY
    cdef struct ParticleXYR
    # brief return the value of normal distribution
    # mean
    # sigma
    cdef double sampleNormal (double , double);
###