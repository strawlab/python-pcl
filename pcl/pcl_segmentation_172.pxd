# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# NG
# from libcpp.memory cimport shared_ptr
from boost_shared_ptr cimport shared_ptr

# main
# cimport pcl_defs as cpp
# 
from pcl_defs cimport PointIndices
from pcl_defs cimport ModelCoefficients
from pcl_defs cimport PointCloud
from pcl_defs cimport PointXYZ
from pcl_defs cimport PointXYZI
from pcl_defs cimport PointXYZRGB
from pcl_defs cimport PointXYZRGBA
from pcl_defs cimport Normal
from pcl_defs cimport PCLBase

from pcl_sample_consensus cimport SacModel
# cimport pcl_defs as cpp
cimport pcl_surface as pclsf
cimport pcl_kdtree as pclkdt

##

from vector cimport vector as vector2

###############################################################################
# Types
###############################################################################

### base class ###

cdef extern from "pcl/segmentation/sac_segmentation.h" namespace "pcl":
    # cdef cppclass SACSegmentation[T](PCLBase[T]):
    cdef cppclass SACSegmentation[T]:
        void setOptimizeCoefficients (bool)
        void setModelType (SacModel)
        void setMethodType (int)
        void setDistanceThreshold (float)
        void setInputCloud (shared_ptr[PointCloud[T]])
        void setMaxIterations (int)
        void segment (PointIndices, ModelCoefficients)
        # Add
        # /** \brief Empty constructor. */
        # SACSegmentation () :  model_ (), sac_ (), model_type_ (-1), method_type_ (0), 
        #                       threshold_ (0), optimize_coefficients_ (true), 
        #                       radius_min_ (-std::numeric_limits<double>::max()), radius_max_ (std::numeric_limits<double>::max()), samples_radius_ (0.0), eps_angle_ (0.0),
        #                       axis_ (Eigen::Vector3f::Zero ()), max_iterations_ (50), probability_ (0.99)
        # {
        #   //srand ((unsigned)time (0)); // set a random seed
        # }
        # /** \brief Get the type of SAC model used. */
        # inline int getModelType () const { return (model_type_); }
        # /** \brief Get a pointer to the SAC method used. */
        # inline SampleConsensusPtr getMethod () const { return (sac_); }
        # /** \brief Get a pointer to the SAC model used. */
        # inline SampleConsensusModelPtr getModel () const { return (model_); }
        # /** \brief Get the type of sample consensus method used. */
        # inline int getMethodType () const { return (method_type_); }
        # /** \brief Get the distance to the model threshold. */
        # inline double getDistanceThreshold () const { return (threshold_); }
        # /** \brief Get maximum number of iterations before giving up. */
        # inline int getMaxIterations () const { return (max_iterations_); }
        # /** \brief Set the probability of choosing at least one sample free from outliers.
        #   * \param[in] probability the model fitting probability
        #   */
        # inline void setProbability (double probability) { probability_ = probability; }
        # 
        # /** \brief Get the probability of choosing at least one sample free from outliers. */
        # inline double getProbability () const { return (probability_); }
        # 
        # /** \brief Get the coefficient refinement internal flag. */
        # inline bool getOptimizeCoefficients () const { return (optimize_coefficients_); }
        # /** \brief Set the minimum and maximum allowable radius limits for the model (applicable to models that estimate a radius)
        #   * \param[in] min_radius the minimum radius model
        #   * \param[in] max_radius the maximum radius model
        #   */
        # inline void setRadiusLimits (const double &min_radius, const double &max_radius)
        # 
        # /** \brief Get the minimum and maximum allowable radius limits for the model as set by the user.
        #   * \param[out] min_radius the resultant minimum radius model
        #   * \param[out] max_radius the resultant maximum radius model
        #   */
        # inline void getRadiusLimits (double &min_radius, double &max_radius)
        # /** \brief Set the maximum distance allowed when drawing random samples
        #   * \param[in] radius the maximum distance (L2 norm)
        #   */
        # inline void setSamplesMaxDist (const double &radius, SearchPtr search)
        # 
        # /** \brief Get maximum distance allowed when drawing random samples
        #   * \param[out] radius the maximum distance (L2 norm)
        #   */
        # inline void getSamplesMaxDist (double &radius)
        # 
        # /** \brief Set the axis along which we need to search for a model perpendicular to.
        #   * \param[in] ax the axis along which we need to search for a model perpendicular to
        #   */
        # inline void setAxis (const Eigen::Vector3f &ax) { axis_ = ax; }
        # 
        # /** \brief Get the axis along which we need to search for a model perpendicular to. */
        # inline Eigen::Vector3f getAxis () const { return (axis_); }
        # /** \brief Set the angle epsilon (delta) threshold.
        #   * \param[in] ea the maximum allowed difference between the model normal and the given axis in radians.
        #   */
        # inline void setEpsAngle (double ea) { eps_angle_ = ea; }
        #       /** \brief Get the epsilon (delta) model angle threshold in radians. */
        # inline double getEpsAngle () const { return (eps_angle_); }

    # /** \brief @b SACSegmentationFromNormals represents the PCL nodelet segmentation class for Sample Consensus methods and
    #   * models that require the use of surface normals for estimation.
    #   * \ingroup segmentation
    #   */
    # cdef cppclass SACSegmentationFromNormals[T, N](SACSegmentation[T])
    cdef cppclass SACSegmentationFromNormals[T, N]:
        SACSegmentationFromNormals()
        void setOptimizeCoefficients (bool)
        void setModelType (SacModel)
        void setMethodType (int)
        void setNormalDistanceWeight (float)
        void setMaxIterations (int)
        void setDistanceThreshold (float)
        void setRadiusLimits (float, float)
        void setInputCloud (shared_ptr[PointCloud[T]])
        void setInputNormals (shared_ptr[PointCloud[N]])
        void setEpsAngle (double ea)
        void segment (PointIndices, ModelCoefficients)
        void setMinMaxOpeningAngle(double, double)
        void getMinMaxOpeningAngle(double, double)
        # Add
        # /** \brief Empty constructor. */
        # SACSegmentationFromNormals () : 
        #   normals_ (), 
        #   distance_weight_ (0.1), 
        #   distance_from_origin_ (0), 
        #   min_angle_ (), 
        #   max_angle_ ()
        # {};
        # 
        # /** \brief Provide a pointer to the input dataset that contains the point normals of 
        #   * the XYZ dataset.
        #   * \param[in] normals the const boost shared pointer to a PointCloud message
        #   */
        # inline void setInputNormals (const PointCloudNConstPtr &normals) { normals_ = normals; }
        # 
        # /** \brief Get a pointer to the normals of the input XYZ point cloud dataset. */
        # inline PointCloudNConstPtr getInputNormals () const { return (normals_); }
        # 
        # /** \brief Set the relative weight (between 0 and 1) to give to the angular 
        #   * distance (0 to pi/2) between point normals and the plane normal.
        #   * \param[in] distance_weight the distance/angular weight
        #   */
        # inline void setNormalDistanceWeight (double distance_weight) { distance_weight_ = distance_weight; }
        # 
        # /** \brief Get the relative weight (between 0 and 1) to give to the angular distance (0 to pi/2) between point
        #   * normals and the plane normal. */
        # inline double getNormalDistanceWeight () const { return (distance_weight_); }
        # 
        # /** \brief Set the minimum opning angle for a cone model.
        #   * \param oa the opening angle which we need minumum to validate a cone model.
        #   */
        # inline void setMinMaxOpeningAngle (const double &min_angle, const double &max_angle)
        # 
        # /** \brief Get the opening angle which we need minumum to validate a cone model. */
        # inline void getMinMaxOpeningAngle (double &min_angle, double &max_angle)
        # 
        # /** \brief Set the distance we expect a plane model to be from the origin
        #   * \param[in] d distance from the template plane modl to the origin
        #   */
        # inline void setDistanceFromOrigin (const double d) { distance_from_origin_ = d; }
        # 
        # /** \brief Get the distance of a plane model from the origin. */
        # inline double getDistanceFromOrigin () const { return (distance_from_origin_); }


ctypedef SACSegmentation[PointXYZ] SACSegmentation_t
ctypedef SACSegmentation[PointXYZI] SACSegmentation_PointXYZI_t
ctypedef SACSegmentation[PointXYZRGB] SACSegmentation_PointXYZRGB_t
ctypedef SACSegmentation[PointXYZRGBA] SACSegmentation_PointXYZRGBA_t
ctypedef SACSegmentationFromNormals[PointXYZ,Normal] SACSegmentationNormal_t
ctypedef SACSegmentationFromNormals[PointXYZI,Normal] SACSegmentation_PointXYZI_Normal_t
ctypedef SACSegmentationFromNormals[PointXYZRGB,Normal] SACSegmentation_PointXYZRGB_Normal_t
ctypedef SACSegmentationFromNormals[PointXYZRGBA,Normal] SACSegmentation_PointXYZRGBA_Normal_t
###

# comparator.h
# namespace pcl
# brief Comparator is the base class for comparators that compare two points given some function.
# Currently intended for use with OrganizedConnectedComponentSegmentation
# author Alex Trevor
# template <typename PointT> class Comparator
cdef extern from "pcl/segmentation/comparator.h" namespace "pcl":
    cdef cppclass Comparator[T]:
        Comparator()
        # public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # typedef boost::shared_ptr<Comparator<PointT> > Ptr;
        # typedef boost::shared_ptr<const Comparator<PointT> > ConstPtr;
        # 
        # /** \brief Set the input cloud for the comparator.
        #   * \param[in] cloud the point cloud this comparator will operate on
        #   */
        # virtual void setInputCloud (const PointCloudConstPtr& cloud)
        # 
        # /** \brief Get the input cloud this comparator operates on. */
        # virtual PointCloudConstPtr getInputCloud () const
        # 
        # /** \brief Compares the two points in the input cloud designated by these two indices.
        #   * This is pure virtual and must be implemented by subclasses with some comparison function.
        #   * \param[in] idx1 the index of the first point.
        #   * \param[in] idx2 the index of the second point.
        #   */
        # virtual bool compare (int idx1, int idx2) const = 0;
        # 
        # protected:
        #   PointCloudConstPtr input_;
###

# plane_coefficient_comparator.h
# namespace pcl
# brief PlaneCoefficientComparator is a Comparator that operates on plane coefficients, for use in planar segmentation.
# In conjunction with OrganizedConnectedComponentSegmentation, this allows planes to be segmented from organized data.
# author Alex Trevor
# template<typename PointT, typename PointNT> class PlaneCoefficientComparator: public Comparator<PointT>
cdef extern from "pcl/segmentation/plane_coefficient_comparator.h" namespace "pcl":
    cdef cppclass PlaneCoefficientComparator[T, NT](Comparator[T]):
        PlaneCoefficientComparator()
        # PlaneCoefficientComparator (boost::shared_ptr<std::vector<float> >& plane_coeff_d)
        # public:
        # typedef typename Comparator<PointT>::PointCloud PointCloud;
        # typedef typename Comparator<PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef typename pcl::PointCloud<PointNT> PointCloudN;
        # typedef typename PointCloudN::Ptr PointCloudNPtr;
        # typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
        # typedef boost::shared_ptr<PlaneCoefficientComparator<PointT, PointNT> > Ptr;
        # typedef boost::shared_ptr<const PlaneCoefficientComparator<PointT, PointNT> > ConstPtr;
        # using pcl::Comparator<PointT>::input_;
        # 
        # virtual void setInputCloud (const PointCloudConstPtr& cloud)
        # /** \brief Provide a pointer to the input normals.
        #   * \param[in] normals the input normal cloud
        # inline void setInputNormals (const PointCloudNConstPtr &normals)
        # /** \brief Get the input normals. */
        # inline PointCloudNConstPtr getInputNormals () const
        # /** \brief Provide a pointer to a vector of the d-coefficient of the planes' hessian normal form.  a, b, and c are provided by the normal cloud.
        #   * \param[in] plane_coeff_d a pointer to the plane coefficients.
        # void setPlaneCoeffD (boost::shared_ptr<std::vector<float> >& plane_coeff_d)
        # 
        # /** \brief Provide a pointer to a vector of the d-coefficient of the planes' hessian normal form.  a, b, and c are provided by the normal cloud.
        #   * \param[in] plane_coeff_d a pointer to the plane coefficients.
        # void setPlaneCoeffD (std::vector<float>& plane_coeff_d)
        # /** \brief Get a pointer to the vector of the d-coefficient of the planes' hessian normal form. */
        # const std::vector<float>& getPlaneCoeffD () const
        # /** \brief Set the tolerance in radians for difference in normal direction between neighboring points, to be considered part of the same plane.
        #   * \param[in] angular_threshold the tolerance in radians
        # virtual void setAngularThreshold (float angular_threshold)
        # /** \brief Get the angular threshold in radians for difference in normal direction between neighboring points, to be considered part of the same plane. */
        # inline float getAngularThreshold () const
        # /** \brief Set the tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane.
        #   * \param[in] distance_threshold the tolerance in meters (at 1m)
        #   * \param[in] depth_dependent whether to scale the threshold based on range from the sensor (default: false)
        # void setDistanceThreshold (float distance_threshold, bool depth_dependent = false)
        # /** \brief Get the distance threshold in meters (d component of plane equation) between neighboring points, to be considered part of the same plane. */
        # inline float getDistanceThreshold () const
        # /** \brief Compare points at two indices by their plane equations.  True if the angle between the normals is less than the angular threshold,
        #   * and the difference between the d component of the normals is less than distance threshold, else false
        #   * \param idx1 The first index for the comparison
        #   * \param idx2 The second index for the comparison
        # virtual bool compare (int idx1, int idx2) const
        # 
        # protected:
        # PointCloudNConstPtr normals_;
        # boost::shared_ptr<std::vector<float> > plane_coeff_d_;
        # float angular_threshold_;
        # float distance_threshold_;
        # bool depth_dependent_;
        # Eigen::Vector3f z_axis_;
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
###

### Inheritance class ###

# edge_aware_plane_comparator.h
# namespace pcl
# /** \brief EdgeAwarePlaneComparator is a Comparator that operates on plane coefficients, 
#   * for use in planar segmentation.
#   * In conjunction with OrganizedConnectedComponentSegmentation, this allows planes to be segmented from organized data.
#   * \author Stefan Holzer, Alex Trevor
#   */
# template<typename PointT, typename PointNT>
# class EdgeAwarePlaneComparator: public PlaneCoefficientComparator<PointT, PointNT>
cdef extern from "pcl/segmentation/edge_aware_plane_comparator.h" namespace "pcl":
    cdef cppclass EdgeAwarePlaneComparator[T, NT](PlaneCoefficientComparator[T, NT]):
        EdgeAwarePlaneComparator()
        # EdgeAwarePlaneComparator (const float *distance_map)
        # public:
        # typedef typename Comparator<PointT>::PointCloud PointCloud;
        # typedef typename Comparator<PointT>::PointCloudConstPtr PointCloudConstPtr;
#       # typedef typename pcl::PointCloud<PointNT> PointCloudN;
#       # typedef typename PointCloudN::Ptr PointCloudNPtr;
#       # typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
#       # typedef boost::shared_ptr<EdgeAwarePlaneComparator<PointT, PointNT> > Ptr;
#       # typedef boost::shared_ptr<const EdgeAwarePlaneComparator<PointT, PointNT> > ConstPtr;
#       # using pcl::PlaneCoefficientComparator<PointT, PointNT>::input_;
#       # using pcl::PlaneCoefficientComparator<PointT, PointNT>::normals_;
#       # using pcl::PlaneCoefficientComparator<PointT, PointNT>::plane_coeff_d_;
#       # using pcl::PlaneCoefficientComparator<PointT, PointNT>::angular_threshold_;
#       # using pcl::PlaneCoefficientComparator<PointT, PointNT>::distance_threshold_;
#       # 
#       # /** \brief Set a distance map to use. For an example of a valid distance map see 
#       #   * \ref OrganizedIntegralImageNormalEstimation
#       #   * \param[in] distance_map the distance map to use
#       #   */
#       # inline void setDistanceMap (const float *distance_map)
#       # /** \brief Return the distance map used. */
#       # const float* getDistanceMap () const
#       # 
#       # protected:
#       # /** \brief Compare two neighboring points, by using normal information, curvature, and euclidean distance information.
#       #   * \param[in] idx1 The index of the first point.
#       #   * \param[in] idx2 The index of the second point.
#       # bool compare (int idx1, int idx2) const
#       # protected:
#       # const float* distance_map_;
###

# euclidean_cluster_comparator.h
# namespace pcl
# /** \brief EuclideanClusterComparator is a comparator used for finding clusters supported by planar surfaces.
#   * This needs to be run as a second pass after extracting planar surfaces, using MultiPlaneSegmentation for example.
#   * \author Alex Trevor
# template<typename PointT, typename PointNT, typename PointLT>
# class EuclideanClusterComparator: public Comparator<PointT>
cdef extern from "pcl/segmentation/euclidean_cluster_comparator.h" namespace "pcl":
    cdef cppclass EuclideanClusterComparator[T, NT, LT](Comparator[T]):
        EuclideanClusterComparator()
#       public:
#       typedef typename Comparator<PointT>::PointCloud PointCloud;
#       typedef typename Comparator<PointT>::PointCloudConstPtr PointCloudConstPtr;
#       typedef typename pcl::PointCloud<PointNT> PointCloudN;
#       typedef typename PointCloudN::Ptr PointCloudNPtr;
#       typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
#       typedef typename pcl::PointCloud<PointLT> PointCloudL;
#       typedef typename PointCloudL::Ptr PointCloudLPtr;
#       typedef typename PointCloudL::ConstPtr PointCloudLConstPtr;
#       typedef boost::shared_ptr<EuclideanClusterComparator<PointT, PointNT, PointLT> > Ptr;
#       typedef boost::shared_ptr<const EuclideanClusterComparator<PointT, PointNT, PointLT> > ConstPtr;
#       using pcl::Comparator<PointT>::input_;
#       
#       virtual void setInputCloud (const PointCloudConstPtr& cloud)
#       /** \brief Provide a pointer to the input normals.
#         * \param[in] normals the input normal cloud
#       inline void setInputNormals (const PointCloudNConstPtr &normals)
#       /** \brief Get the input normals. */
#       inline PointCloudNConstPtr getInputNormals () const
#       /** \brief Set the tolerance in radians for difference in normal direction between neighboring points, to be considered part of the same plane.
#         * \param[in] angular_threshold the tolerance in radians
#       virtual inline void setAngularThreshold (float angular_threshold)
#       /** \brief Get the angular threshold in radians for difference in normal direction between neighboring points, to be considered part of the same plane. */
#       inline float getAngularThreshold () const
#       /** \brief Set the tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane.
#         * \param[in] distance_threshold the tolerance in meters
#       inline void setDistanceThreshold (float distance_threshold, bool depth_dependent)
#       /** \brief Get the distance threshold in meters (d component of plane equation) between neighboring points, to be considered part of the same plane. */
#       inline float getDistanceThreshold () const
#       /** \brief Set label cloud
#         * \param[in] labels The label cloud
#       void setLabels (PointCloudLPtr& labels)
#       /** \brief Set labels in the label cloud to exclude.
#         * \param exclude_labels a vector of bools corresponding to whether or not a given label should be considered
#       void setExcludeLabels (std::vector<bool>& exclude_labels)
#       /** \brief Compare points at two indices by their plane equations.  True if the angle between the normals is less than the angular threshold,
#         * and the difference between the d component of the normals is less than distance threshold, else false
#         * \param idx1 The first index for the comparison
#         * \param idx2 The second index for the comparison
#       virtual bool compare (int idx1, int idx2) const
#       
#       protected:
#       PointCloudNConstPtr normals_;
#       PointCloudLPtr labels_;
#       boost::shared_ptr<std::vector<bool> > exclude_labels_;
#       float angular_threshold_;
#       float distance_threshold_;
#       bool depth_dependent_;
#       Eigen::Vector3f z_axis_;
###

# euclidean_plane_coefficient_comparator.h
# namespace pcl
# /** \brief EuclideanPlaneCoefficientComparator is a Comparator that operates on plane coefficients, 
#   * for use in planar segmentation.
#   * In conjunction with OrganizedConnectedComponentSegmentation, this allows planes to be segmented from organized data.
#   * \author Alex Trevor
# template<typename PointT, typename PointNT>
# class EuclideanPlaneCoefficientComparator: public PlaneCoefficientComparator<PointT, PointNT>
cdef extern from "pcl/segmentation/euclidean_plane_coefficient_comparator.h" namespace "pcl":
    cdef cppclass EuclideanPlaneCoefficientComparator[T, NT](PlaneCoefficientComparator[T, NT]):
        EuclideanPlaneCoefficientComparator()
#       public:
#       typedef typename Comparator<PointT>::PointCloud PointCloud;
#       typedef typename Comparator<PointT>::PointCloudConstPtr PointCloudConstPtr;
#       typedef typename pcl::PointCloud<PointNT> PointCloudN;
#       typedef typename PointCloudN::Ptr PointCloudNPtr;
#       typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
#       typedef boost::shared_ptr<EuclideanPlaneCoefficientComparator<PointT, PointNT> > Ptr;
#       typedef boost::shared_ptr<const EuclideanPlaneCoefficientComparator<PointT, PointNT> > ConstPtr;
#       using pcl::Comparator<PointT>::input_;
#       using pcl::PlaneCoefficientComparator<PointT, PointNT>::normals_;
#       using pcl::PlaneCoefficientComparator<PointT, PointNT>::angular_threshold_;
#       using pcl::PlaneCoefficientComparator<PointT, PointNT>::distance_threshold_;
# 
#       /** \brief Compare two neighboring points, by using normal information, and euclidean distance information.
#         * \param[in] idx1 The index of the first point.
#         * \param[in] idx2 The index of the second point.
#         */
#       virtual bool compare (int idx1, int idx2) const
###

# extract_clusters.h
# namespace pcl
# brief Decompose a region of space into clusters based on the Euclidean distance between points
# param cloud the point cloud message
# param tree the spatial locator (e.g., kd-tree) used for nearest neighbors searching
# note the tree has to be created as a spatial locator on \a cloud
# param tolerance the spatial cluster tolerance as a measure in L2 Euclidean space
# param clusters the resultant clusters containing point indices (as a vector of PointIndices)
# param min_pts_per_cluster minimum number of points that a cluster may contain (default: 1)
# param max_pts_per_cluster maximum number of points that a cluster may contain (default: max int)
# ingroup segmentation
# template <typename PointT> void extractEuclideanClusters (
#       const PointCloud<PointT> &cloud, const boost::shared_ptr<search::Search<PointT> > &tree, 
#       float tolerance, std::vector<PointIndices> &clusters, 
#       unsigned int min_pts_per_cluster = 1, unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ());
# 
#   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#   /** \brief Decompose a region of space into clusters based on the Euclidean distance between points
#     * \param cloud the point cloud message
#     * \param indices a list of point indices to use from \a cloud
#     * \param tree the spatial locator (e.g., kd-tree) used for nearest neighbors searching
#     * \note the tree has to be created as a spatial locator on \a cloud and \a indices
#     * \param tolerance the spatial cluster tolerance as a measure in L2 Euclidean space
#     * \param clusters the resultant clusters containing point indices (as a vector of PointIndices)
#     * \param min_pts_per_cluster minimum number of points that a cluster may contain (default: 1)
#     * \param max_pts_per_cluster maximum number of points that a cluster may contain (default: max int)
#     * \ingroup segmentation
#     */
#   template <typename PointT> void 
#   extractEuclideanClusters (
#       const PointCloud<PointT> &cloud, const std::vector<int> &indices, 
#       const boost::shared_ptr<search::Search<PointT> > &tree, float tolerance, std::vector<PointIndices> &clusters, 
#       unsigned int min_pts_per_cluster = 1, unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ());
# 
#   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#   /** \brief Decompose a region of space into clusters based on the euclidean distance between points, and the normal
#     * angular deviation
#     * \param cloud the point cloud message
#     * \param normals the point cloud message containing normal information
#     * \param tree the spatial locator (e.g., kd-tree) used for nearest neighbors searching
#     * \note the tree has to be created as a spatial locator on \a cloud
#     * \param tolerance the spatial cluster tolerance as a measure in the L2 Euclidean space
#     * \param clusters the resultant clusters containing point indices (as a vector of PointIndices)
#     * \param eps_angle the maximum allowed difference between normals in radians for cluster/region growing
#     * \param min_pts_per_cluster minimum number of points that a cluster may contain (default: 1)
#     * \param max_pts_per_cluster maximum number of points that a cluster may contain (default: max int)
#     * \ingroup segmentation
#     */
#   template <typename PointT, typename Normal> void 
#   extractEuclideanClusters (
#       const PointCloud<PointT> &cloud, const PointCloud<Normal> &normals, 
#       float tolerance, const boost::shared_ptr<KdTree<PointT> > &tree, 
#       std::vector<PointIndices> &clusters, double eps_angle, 
#       unsigned int min_pts_per_cluster = 1, 
#       unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ())
#   {
#     if (tree->getInputCloud ()->points.size () != cloud.points.size ())
#     {
#       PCL_ERROR ("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset (%zu) than the input cloud (%zu)!\n", tree->getInputCloud ()->points.size (), cloud.points.size ());
#       return;
#     }
#     if (cloud.points.size () != normals.points.size ())
#     {
#       PCL_ERROR ("[pcl::extractEuclideanClusters] Number of points in the input point cloud (%zu) different than normals (%zu)!\n", cloud.points.size (), normals.points.size ());
#       return;
#     }
# 
#     // Create a bool vector of processed point indices, and initialize it to false
#     std::vector<bool> processed (cloud.points.size (), false);
# 
#     std::vector<int> nn_indices;
#     std::vector<float> nn_distances;
#     // Process all points in the indices vector
#     for (size_t i = 0; i < cloud.points.size (); ++i)
#     {
#       if (processed[i])
#         continue;
# 
#       std::vector<unsigned int> seed_queue;
#       int sq_idx = 0;
#       seed_queue.push_back (i);
# 
#       processed[i] = true;
# 
#       while (sq_idx < static_cast<int> (seed_queue.size ()))
#       {
#         // Search for sq_idx
#         if (!tree->radiusSearch (seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
#         {
#           sq_idx++;
#           continue;
#         }
# 
#         for (size_t j = 1; j < nn_indices.size (); ++j)             // nn_indices[0] should be sq_idx
#         {
#           if (processed[nn_indices[j]])                         // Has this point been processed before ?
#             continue;
# 
#           //processed[nn_indices[j]] = true;
#           // [-1;1]
#           double dot_p = normals.points[i].normal[0] * normals.points[nn_indices[j]].normal[0] +
#                          normals.points[i].normal[1] * normals.points[nn_indices[j]].normal[1] +
#                          normals.points[i].normal[2] * normals.points[nn_indices[j]].normal[2];
#           if ( fabs (acos (dot_p)) < eps_angle )
#           {
#             processed[nn_indices[j]] = true;
#             seed_queue.push_back (nn_indices[j]);
#           }
#         }
# 
#         sq_idx++;
#       }
# 
#       // If this queue is satisfactory, add to the clusters
#       if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
#       {
#         pcl::PointIndices r;
#         r.indices.resize (seed_queue.size ());
#         for (size_t j = 0; j < seed_queue.size (); ++j)
#           r.indices[j] = seed_queue[j];
# 
#         // These two lines should not be needed: (can anyone confirm?) -FF
#         std::sort (r.indices.begin (), r.indices.end ());
#         r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());
# 
#         r.header = cloud.header;
#         clusters.push_back (r);   // We could avoid a copy by working directly in the vector
#       }
#     }
#   }
# 
# 
#   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#   /** \brief Decompose a region of space into clusters based on the euclidean distance between points, and the normal
#     * angular deviation
#     * \param cloud the point cloud message
#     * \param normals the point cloud message containing normal information
#     * \param indices a list of point indices to use from \a cloud
#     * \param tree the spatial locator (e.g., kd-tree) used for nearest neighbors searching
#     * \note the tree has to be created as a spatial locator on \a cloud
#     * \param tolerance the spatial cluster tolerance as a measure in the L2 Euclidean space
#     * \param clusters the resultant clusters containing point indices (as PointIndices)
#     * \param eps_angle the maximum allowed difference between normals in degrees for cluster/region growing
#     * \param min_pts_per_cluster minimum number of points that a cluster may contain (default: 1)
#     * \param max_pts_per_cluster maximum number of points that a cluster may contain (default: max int)
#     * \ingroup segmentation
#     */
#   template <typename PointT, typename Normal> 
#   void extractEuclideanClusters (
#       const PointCloud<PointT> &cloud, const PointCloud<Normal> &normals, 
#       const std::vector<int> &indices, const boost::shared_ptr<KdTree<PointT> > &tree, 
#       float tolerance, std::vector<PointIndices> &clusters, double eps_angle, 
#       unsigned int min_pts_per_cluster = 1, 
#       unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ())
#   {
#     // \note If the tree was created over <cloud, indices>, we guarantee a 1-1 mapping between what the tree returns
#     //and indices[i]
#     if (tree->getInputCloud ()->points.size () != cloud.points.size ())
#     {
#       PCL_ERROR ("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset (%zu) than the input cloud (%zu)!\n", tree->getInputCloud ()->points.size (), cloud.points.size ());
#       return;
#     }
#     if (tree->getIndices ()->size () != indices.size ())
#     {
#       PCL_ERROR ("[pcl::extractEuclideanClusters] Tree built for a different set of indices (%zu) than the input set (%zu)!\n", tree->getIndices ()->size (), indices.size ());
#       return;
#     }
#     if (cloud.points.size () != normals.points.size ())
#     {
#       PCL_ERROR ("[pcl::extractEuclideanClusters] Number of points in the input point cloud (%zu) different than normals (%zu)!\n", cloud.points.size (), normals.points.size ());
#       return;
#     }
#     // Create a bool vector of processed point indices, and initialize it to false
#     std::vector<bool> processed (cloud.points.size (), false);
# 
#     std::vector<int> nn_indices;
#     std::vector<float> nn_distances;
#     // Process all points in the indices vector
#     for (size_t i = 0; i < indices.size (); ++i)
#     {
#       if (processed[indices[i]])
#         continue;
# 
#       std::vector<int> seed_queue;
#       int sq_idx = 0;
#       seed_queue.push_back (indices[i]);
# 
#       processed[indices[i]] = true;
# 
#       while (sq_idx < static_cast<int> (seed_queue.size ()))
#       {
#         // Search for sq_idx
#         if (!tree->radiusSearch (cloud.points[seed_queue[sq_idx]], tolerance, nn_indices, nn_distances))
#         {
#           sq_idx++;
#           continue;
#         }
# 
#         for (size_t j = 1; j < nn_indices.size (); ++j)             // nn_indices[0] should be sq_idx
#         {
#           if (processed[nn_indices[j]])                             // Has this point been processed before ?
#             continue;
# 
#           //processed[nn_indices[j]] = true;
#           // [-1;1]
#           double dot_p =
#             normals.points[indices[i]].normal[0] * normals.points[indices[nn_indices[j]]].normal[0] +
#             normals.points[indices[i]].normal[1] * normals.points[indices[nn_indices[j]]].normal[1] +
#             normals.points[indices[i]].normal[2] * normals.points[indices[nn_indices[j]]].normal[2];
#           if ( fabs (acos (dot_p)) < eps_angle )
#           {
#             processed[nn_indices[j]] = true;
#             seed_queue.push_back (nn_indices[j]);
#           }
#         }
# 
#         sq_idx++;
#       }
# 
#       // If this queue is satisfactory, add to the clusters
#       if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
#       {
#         pcl::PointIndices r;
#         r.indices.resize (seed_queue.size ());
#         for (size_t j = 0; j < seed_queue.size (); ++j)
#           r.indices[j] = seed_queue[j];
# 
#         // These two lines should not be needed: (can anyone confirm?) -FF
#         std::sort (r.indices.begin (), r.indices.end ());
#         r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());
# 
#         r.header = cloud.header;
#         clusters.push_back (r);
#       }
#     }
#   }
# 

# EuclideanClusterExtraction represents a segmentation class for cluster extraction in an Euclidean sense.
# author Radu Bogdan Rusu
# ingroup segmentation
# template <typename PointT>
# class EuclideanClusterExtraction: public PCLBase<PointT>
# cdef extern from "pcl/segmentation/sac_segmentation.h" namespace "pcl":
cdef extern from "pcl/segmentation/extract_clusters.h" namespace "pcl":
    cdef cppclass EuclideanClusterExtraction[T](PCLBase[T]):
        EuclideanClusterExtraction()
        # public:
        # EuclideanClusterExtraction () : tree_ (), 
        #                                 cluster_tolerance_ (0),
        #                                 min_pts_per_cluster_ (1), 
        #                                 max_pts_per_cluster_ (std::numeric_limits<int>::max ())
        
        # brief Provide a pointer to the search object.
        # param[in] tree a pointer to the spatial search object.
        # inline void setSearchMethod (const KdTreePtr &tree) 
        void setSearchMethod (const pclkdt.KdTreePtr_t &tree)
        
        # brief Get a pointer to the search method used. 
        # @todo fix this for a generic search tree
        # inline KdTreePtr getSearchMethod () const 
        pclkdt.KdTreePtr_t getSearchMethod ()
        
        # brief Set the spatial cluster tolerance as a measure in the L2 Euclidean space
        # param[in] tolerance the spatial cluster tolerance as a measure in the L2 Euclidean space
        # inline void setClusterTolerance (double tolerance) 
        void setClusterTolerance (double tolerance) 
        
        # brief Get the spatial cluster tolerance as a measure in the L2 Euclidean space.
        # inline double getClusterTolerance () const 
        double getClusterTolerance () const 
        
        # brief Set the minimum number of points that a cluster needs to contain in order to be considered valid.
        # param[in] min_cluster_size the minimum cluster size
        # inline void setMinClusterSize (int min_cluster_size) 
        void setMinClusterSize (int min_cluster_size) 
        
        # brief Get the minimum number of points that a cluster needs to contain in order to be considered valid.
        # inline int getMinClusterSize () const 
        int getMinClusterSize ()
        
        # brief Set the maximum number of points that a cluster needs to contain in order to be considered valid.
        # param[in] max_cluster_size the maximum cluster size
        # inline void setMaxClusterSize (int max_cluster_size) 
        void setMaxClusterSize (int max_cluster_size) 
        
        # brief Get the maximum number of points that a cluster needs to contain in order to be considered valid.
        # inline int getMaxClusterSize () const 
        int getMaxClusterSize ()
        
        # brief Cluster extraction in a PointCloud given by <setInputCloud (), setIndices ()>
        # param[out] clusters the resultant point clusters
        # void extract (std::vector<PointIndices> &clusters);
        void extract (vector[PointIndices] &clusters)
        
ctypedef EuclideanClusterExtraction[PointXYZ] EuclideanClusterExtraction_t
ctypedef EuclideanClusterExtraction[PointXYZI] EuclideanClusterExtraction_PointXYZI_t
ctypedef EuclideanClusterExtraction[PointXYZRGB] EuclideanClusterExtraction_PointXYZRGB_t
ctypedef EuclideanClusterExtraction[PointXYZRGBA] EuclideanClusterExtraction_PointXYZRGBA_t
###


# extract_labeled_clusters.h
# namespace pcl
# {
#   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#   /** \brief Decompose a region of space into clusters based on the Euclidean distance between points
#     * \param[in] cloud the point cloud message
#     * \param[in] tree the spatial locator (e.g., kd-tree) used for nearest neighbors searching
#     * \note the tree has to be created as a spatial locator on \a cloud
#     * \param[in] tolerance the spatial cluster tolerance as a measure in L2 Euclidean space
#     * \param[out] labeled_clusters the resultant clusters containing point indices (as a vector of PointIndices)
#     * \param[in] min_pts_per_cluster minimum number of points that a cluster may contain (default: 1)
#     * \param[in] max_pts_per_cluster maximum number of points that a cluster may contain (default: max int)
#     * \param[in] max_label
#     * \ingroup segmentation
#     */
#   template <typename PointT> void 
#   extractLabeledEuclideanClusters (
#       const PointCloud<PointT> &cloud, const boost::shared_ptr<search::Search<PointT> > &tree, 
#       float tolerance, std::vector<std::vector<PointIndices> > &labeled_clusters, 
#       unsigned int min_pts_per_cluster = 1, unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) (), 
#       unsigned int max_label = (std::numeric_limits<int>::max));


# brief @b LabeledEuclideanClusterExtraction represents a segmentation class for cluster extraction in an Euclidean sense, with label info.
# author Koen Buys
# ingroup segmentation
# template <typename PointT>
# class LabeledEuclideanClusterExtraction: public PCLBase<PointT>
# {
        # typedef PCLBase<PointT> BasePCLBase;
        # 
        # public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # typedef typename pcl::search::Search<PointT> KdTree;
        # typedef typename pcl::search::Search<PointT>::Ptr KdTreePtr;
        # typedef PointIndices::Ptr PointIndicesPtr;
        # typedef PointIndices::ConstPtr PointIndicesConstPtr;
        # 
        # /** \brief Empty constructor. */
        # LabeledEuclideanClusterExtraction () : 
        #   tree_ (), 
        #   cluster_tolerance_ (0),
        #   min_pts_per_cluster_ (1), 
        #   max_pts_per_cluster_ (std::numeric_limits<int>::max ()),
        #   max_label_ (std::numeric_limits<int>::max ())
        # {};
        # 
        # /** \brief Provide a pointer to the search object.
        #   * \param[in] tree a pointer to the spatial search object.
        #   */
        # inline void setSearchMethod (const KdTreePtr &tree) { tree_ = tree; }
        # 
        # /** \brief Get a pointer to the search method used. */
        # inline KdTreePtr getSearchMethod () const { return (tree_); }
        # 
        # /** \brief Set the spatial cluster tolerance as a measure in the L2 Euclidean space
        #   * \param[in] tolerance the spatial cluster tolerance as a measure in the L2 Euclidean space
        #   */
        # inline void setClusterTolerance (double tolerance) { cluster_tolerance_ = tolerance; }
        # 
        # /** \brief Get the spatial cluster tolerance as a measure in the L2 Euclidean space. */
        # inline double getClusterTolerance () const { return (cluster_tolerance_); }
        # 
        # /** \brief Set the minimum number of points that a cluster needs to contain in order to be considered valid.
        #   * \param[in] min_cluster_size the minimum cluster size
        #   */
        # inline void setMinClusterSize (int min_cluster_size) { min_pts_per_cluster_ = min_cluster_size; }
        # 
        # /** \brief Get the minimum number of points that a cluster needs to contain in order to be considered valid. */
        # inline int getMinClusterSize () const { return (min_pts_per_cluster_); }
        # 
        # /** \brief Set the maximum number of points that a cluster needs to contain in order to be considered valid.
        #   * \param[in] max_cluster_size the maximum cluster size
        #   */
        # inline void setMaxClusterSize (int max_cluster_size) { max_pts_per_cluster_ = max_cluster_size; }
        # 
        # /** \brief Get the maximum number of points that a cluster needs to contain in order to be considered valid. */
        # inline int getMaxClusterSize () const { return (max_pts_per_cluster_); }
        # 
        # /** \brief Set the maximum number of labels in the cloud.
        #   * \param[in] max_label the maximum
        #   */
        # inline void setMaxLabels (unsigned int max_label) { max_label_ = max_label; }
        # 
        # /** \brief Get the maximum number of labels */
        # inline unsigned int getMaxLabels () const { return (max_label_); }
        # 
        # /** \brief Cluster extraction in a PointCloud given by <setInputCloud (), setIndices ()>
        #   * \param[out] labeled_clusters the resultant point clusters
        #   */
        # void extract (std::vector<std::vector<PointIndices> > &labeled_clusters);
        # 
        # protected:
        #     // Members derived from the base class
        #     using BasePCLBase::input_;
        #     using BasePCLBase::indices_;
        #     using BasePCLBase::initCompute;
        #     using BasePCLBase::deinitCompute;
        # 
        # /** \brief A pointer to the spatial search object. */
        # KdTreePtr tree_;
        # /** \brief The spatial cluster tolerance as a measure in the L2 Euclidean space. */
        # double cluster_tolerance_;
        # /** \brief The minimum number of points that a cluster needs to contain in order to be considered valid (default = 1). */
        # int min_pts_per_cluster_;
        # /** \brief The maximum number of points that a cluster needs to contain in order to be considered valid (default = MAXINT). */
        # int max_pts_per_cluster_;
        # /** \brief The maximum number of labels we can find in this pointcloud (default = MAXINT)*/
        # unsigned int max_label_;
        # /** \brief Class getName method. */
        # virtual std::string getClassName () const { return ("LabeledEuclideanClusterExtraction"); }
        # 

        #   brief Sort clusters method (for std::sort). 
        #   ingroup segmentation
        #   inline bool compareLabeledPointClusters (const pcl::PointIndices &a, const pcl::PointIndices &b)
        #   {
        #     return (a.indices.size () < b.indices.size ());
        #   }
###

# extract_polygonal_prism_data.h
# namespace pcl
# {
#   /** \brief General purpose method for checking if a 3D point is inside or
#     * outside a given 2D polygon. 
#     * \note this method accepts any general 3D point that is projected onto the
#     * 2D polygon, but performs an internal XY projection of both the polygon and the point. 
#     * \param point a 3D point projected onto the same plane as the polygon
#     * \param polygon a polygon
#     * \ingroup segmentation
#     */
#   template <typename PointT> bool isPointIn2DPolygon (const PointT &point, const pcl::PointCloud<PointT> &polygon);
# 
#   /** \brief Check if a 2d point (X and Y coordinates considered only!) is
#     * inside or outside a given polygon. This method assumes that both the point
#     * and the polygon are projected onto the XY plane.
#     *
#     * \note (This is highly optimized code taken from http://www.visibone.com/inpoly/)
#     *       Copyright (c) 1995-1996 Galacticomm, Inc.  Freeware source code.
#     * \param point a 3D point projected onto the same plane as the polygon
#     * \param polygon a polygon
#     * \ingroup segmentation
#     */
#   template <typename PointT> bool 
#   isXYPointIn2DXYPolygon (const PointT &point, const pcl::PointCloud<PointT> &polygon);
# 
#   ////////////////////////////////////////////////////////////////////////////////////////////
#   /** \brief @b ExtractPolygonalPrismData uses a set of point indices that
#     * represent a planar model, and together with a given height, generates a 3D
#     * polygonal prism. The polygonal prism is then used to segment all points
#     * lying inside it.
#     *
#     * An example of its usage is to extract the data lying within a set of 3D
#     * boundaries (e.g., objects supported by a plane).
#     *
#     * \author Radu Bogdan Rusu
#     * \ingroup segmentation
#     */

#   template <typename PointT>
#   class ExtractPolygonalPrismData : public PCLBase<PointT>
cdef extern from "pcl/segmentation/extract_polygonal_prism_data.h" namespace "pcl":
    cdef cppclass ExtractPolygonalPrismData[T](PCLBase[T]):
        ExtractPolygonalPrismData()
        # public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # typedef PointIndices::Ptr PointIndicesPtr;
        # typedef PointIndices::ConstPtr PointIndicesConstPtr;
        # 
        # brief Empty constructor.
        # ExtractPolygonalPrismData () : planar_hull_ (), min_pts_hull_ (3), 
        #                                height_limit_min_ (0), height_limit_max_ (FLT_MAX),
        #                                vpx_ (0), vpy_ (0), vpz_ (0)
        # {};
        # 
        # brief Provide a pointer to the input planar hull dataset.
        # param[in] hull the input planar hull dataset
        # inline void setInputPlanarHull (const PointCloudConstPtr &hull) { planar_hull_ = hull; }
        # 
        # brief Get a pointer the input planar hull dataset.
        # inline PointCloudConstPtr getInputPlanarHull () const { return (planar_hull_); }
        # 
        # brief Set the height limits. All points having distances to the
        # model outside this interval will be discarded.
        # param[in] height_min the minimum allowed distance to the plane model value
        # param[in] height_max the maximum allowed distance to the plane model value
        # inline void setHeightLimits (double height_min, double height_max)
        # 
        # brief Get the height limits (min/max) as set by the user. The
        # default values are -FLT_MAX, FLT_MAX. 
        # param[out] height_min the resultant min height limit
        # param[out] height_max the resultant max height limit
        # inline void getHeightLimits (double &height_min, double &height_max) const
        # 
        # brief Set the viewpoint.
        # param[in] vpx the X coordinate of the viewpoint
        # param[in] vpy the Y coordinate of the viewpoint
        # param[in] vpz the Z coordinate of the viewpoint
        # inline void setViewPoint (float vpx, float vpy, float vpz)
        # 
        # brief Get the viewpoint.
        # inline void getViewPoint (float &vpx, float &vpy, float &vpz) const
        # 
        # /** \brief Cluster extraction in a PointCloud given by <setInputCloud (), setIndices ()>
        #   * \param[out] output the resultant point indices that support the model found (inliers)
        # void segment (PointIndices &output);
        # 
        # protected:
        # brief A pointer to the input planar hull dataset.
        # PointCloudConstPtr planar_hull_;
        # 
        # brief The minimum number of points needed on the convex hull.
        # int min_pts_hull_;
        # 
        # brief The minimum allowed height (distance to the model) a point
        # will be considered from. 
        # double height_limit_min_;
        # 
        # brief The maximum allowed height (distance to the model) a point will be considered from. 
        # double height_limit_max_;
        # 
        # brief Values describing the data acquisition viewpoint. Default: 0,0,0.
        # float vpx_, vpy_, vpz_;
        # 
        # brief Class getName method.
        # virtual std::string getClassName () const { return ("ExtractPolygonalPrismData"); }
###

# organized_connected_component_segmentation.h
# namespace pcl
# {
#   /** \brief OrganizedConnectedComponentSegmentation allows connected
#     * components to be found within organized point cloud data, given a
#     * comparison function.  Given an input cloud and a comparator, it will
#     * output a PointCloud of labels, giving each connected component a unique
#     * id, along with a vector of PointIndices corresponding to each component.
#     * See OrganizedMultiPlaneSegmentation for an example application.
#     *
#     * \author Alex Trevor, Suat Gedikli
#     */
#   template <typename PointT, typename PointLT>
#   class OrganizedConnectedComponentSegmentation : public PCLBase<PointT>
#   {
#     using PCLBase<PointT>::input_;
#     using PCLBase<PointT>::indices_;
#     using PCLBase<PointT>::initCompute;
#     using PCLBase<PointT>::deinitCompute;
# 
#     public:
#       typedef typename pcl::PointCloud<PointT> PointCloud;
#       typedef typename PointCloud::Ptr PointCloudPtr;
#       typedef typename PointCloud::ConstPtr PointCloudConstPtr;
#       
#       typedef typename pcl::PointCloud<PointLT> PointCloudL;
#       typedef typename PointCloudL::Ptr PointCloudLPtr;
#       typedef typename PointCloudL::ConstPtr PointCloudLConstPtr;
# 
#       typedef typename pcl::Comparator<PointT> Comparator;
#       typedef typename Comparator::Ptr ComparatorPtr;
#       typedef typename Comparator::ConstPtr ComparatorConstPtr;
#       
#       /** \brief Constructor for OrganizedConnectedComponentSegmentation
#         * \param[in] compare A pointer to the comparator to be used for segmentation.  Must be an instance of pcl::Comparator.
#         */
#       OrganizedConnectedComponentSegmentation (const ComparatorConstPtr& compare)
#         : compare_ (compare)
#       {
#       }
# 
#       /** \brief Destructor for OrganizedConnectedComponentSegmentation. */
#       virtual
#       ~OrganizedConnectedComponentSegmentation ()
#       {
#       }
# 
#       /** \brief Provide a pointer to the comparator to be used for segmentation.
#         * \param[in] compare the comparator
#         */
#       void
#       setComparator (const ComparatorConstPtr& compare)
#       {
#         compare_ = compare;
#       }
#       
#       /** \brief Get the comparator.*/
#       ComparatorConstPtr
#       getComparator () const { return (compare_); }
# 
#       /** \brief Perform the connected component segmentation.
#         * \param[out] labels a PointCloud of labels: each connected component will have a unique id.
#         * \param[out] label_indices a vector of PointIndices corresponding to each label / component id.
#         */
#       void
#       segment (pcl::PointCloud<PointLT>& labels, std::vector<pcl::PointIndices>& label_indices) const;
#       
#       /** \brief Find the boundary points / contour of a connected component
#         * \param[in] start_idx the first (lowest) index of the connected component for which a boundary shoudl be returned
#         * \param[in] labels the Label cloud produced by segmentation
#         * \param[out] boundary_indices the indices of the boundary points for the label corresponding to start_idx
#         */
#       static void
#       findLabeledRegionBoundary (int start_idx, PointCloudLPtr labels, pcl::PointIndices& boundary_indices);      
#       
# 
#     protected:
#       ComparatorConstPtr compare_;
#       
#       inline unsigned
#       findRoot (const std::vector<unsigned>& runs, unsigned index) const
#       {
#         register unsigned idx = index;
#         while (runs[idx] != idx)
#           idx = runs[idx];
# 
#         return (idx);
#       }
###

# organized_multi_plane_segmentation.h
# namespace pcl
# {
#   /** \brief OrganizedMultiPlaneSegmentation finds all planes present in the
#     * input cloud, and outputs a vector of plane equations, as well as a vector
#     * of point clouds corresponding to the inliers of each detected plane.  Only
#     * planes with more than min_inliers points are detected.
#     * Templated on point type, normal type, and label type
#     *
#     * \author Alex Trevor, Suat Gedikli
#     */
#   template<typename PointT, typename PointNT, typename PointLT>
#   class OrganizedMultiPlaneSegmentation : public PCLBase<PointT>
#   {
#     using PCLBase<PointT>::input_;
#     using PCLBase<PointT>::indices_;
#     using PCLBase<PointT>::initCompute;
#     using PCLBase<PointT>::deinitCompute;
# 
#     public:
#       typedef pcl::PointCloud<PointT> PointCloud;
#       typedef typename PointCloud::Ptr PointCloudPtr;
#       typedef typename PointCloud::ConstPtr PointCloudConstPtr;
# 
#       typedef typename pcl::PointCloud<PointNT> PointCloudN;
#       typedef typename PointCloudN::Ptr PointCloudNPtr;
#       typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
# 
#       typedef typename pcl::PointCloud<PointLT> PointCloudL;
#       typedef typename PointCloudL::Ptr PointCloudLPtr;
#       typedef typename PointCloudL::ConstPtr PointCloudLConstPtr;
# 
#       typedef typename pcl::PlaneCoefficientComparator<PointT, PointNT> PlaneComparator;
#       typedef typename PlaneComparator::Ptr PlaneComparatorPtr;
#       typedef typename PlaneComparator::ConstPtr PlaneComparatorConstPtr;
# 
#       typedef typename pcl::PlaneRefinementComparator<PointT, PointNT, PointLT> PlaneRefinementComparator;
#       typedef typename PlaneRefinementComparator::Ptr PlaneRefinementComparatorPtr;
#       typedef typename PlaneRefinementComparator::ConstPtr PlaneRefinementComparatorConstPtr;
# 
#       /** \brief Constructor for OrganizedMultiPlaneSegmentation. */
#       OrganizedMultiPlaneSegmentation () :
#         normals_ (), 
#         min_inliers_ (1000), 
#         angular_threshold_ (pcl::deg2rad (3.0)), 
#         distance_threshold_ (0.02),
#         maximum_curvature_ (0.001),
#         project_points_ (false), 
#         compare_ (new PlaneComparator ()), refinement_compare_ (new PlaneRefinementComparator ())
#       {
#       }
# 
#       /** \brief Destructor for OrganizedMultiPlaneSegmentation. */
#       virtual
#       ~OrganizedMultiPlaneSegmentation ()
#       {
#       }
# 
#       /** \brief Provide a pointer to the input normals.
#         * \param[in] normals the input normal cloud
#         */
#       inline void
#       setInputNormals (const PointCloudNConstPtr &normals) 
#       {
#         normals_ = normals;
#       }
# 
#       /** \brief Get the input normals. */
#       inline PointCloudNConstPtr
#       getInputNormals () const
#       {
#         return (normals_);
#       }
# 
#       /** \brief Set the minimum number of inliers required for a plane
#         * \param[in] min_inliers the minimum number of inliers required per plane
#         */
#       inline void
#       setMinInliers (unsigned min_inliers)
#       {
#         min_inliers_ = min_inliers;
#       }
# 
#       /** \brief Get the minimum number of inliers required per plane. */
#       inline unsigned
#       getMinInliers () const
#       {
#         return (min_inliers_);
#       }
# 
#       /** \brief Set the tolerance in radians for difference in normal direction between neighboring points, to be considered part of the same plane.
#         * \param[in] angular_threshold the tolerance in radians
#         */
#       inline void
#       setAngularThreshold (double angular_threshold)
#       {
#         angular_threshold_ = angular_threshold;
#       }
# 
#       /** \brief Get the angular threshold in radians for difference in normal direction between neighboring points, to be considered part of the same plane. */
#       inline double
#       getAngularThreshold () const
#       {
#         return (angular_threshold_);
#       }
# 
#       /** \brief Set the tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane.
#         * \param[in] distance_threshold the tolerance in meters
#         */
#       inline void
#       setDistanceThreshold (double distance_threshold)
#       {
#         distance_threshold_ = distance_threshold;
#       }
# 
#       /** \brief Get the distance threshold in meters (d component of plane equation) between neighboring points, to be considered part of the same plane. */
#       inline double
#       getDistanceThreshold () const
#       {
#         return (distance_threshold_);
#       }
# 
#       /** \brief Set the maximum curvature allowed for a planar region.
#         * \param[in] maximum_curvature the maximum curvature
#         */
#       inline void
#       setMaximumCurvature (double maximum_curvature)
#       {
#         maximum_curvature_ = maximum_curvature;
#       }
# 
#       /** \brief Get the maximum curvature allowed for a planar region. */
#       inline double
#       getMaximumCurvature () const
#       {
#         return (maximum_curvature_);
#       }
# 
#       /** \brief Provide a pointer to the comparator to be used for segmentation.
#         * \param[in] compare A pointer to the comparator to be used for segmentation.
#         */
#       void
#       setComparator (const PlaneComparatorPtr& compare)
#       {
#         compare_ = compare;
#       }
# 
#       /** \brief Provide a pointer to the comparator to be used for refinement.
#         * \param[in] compare A pointer to the comparator to be used for refinement.
#         */
#       void
#       setRefinementComparator (const PlaneRefinementComparatorPtr& compare)
#       {
#         refinement_compare_ = compare;
#       }
# 
#       /** \brief Set whether or not to project boundary points to the plane, or leave them in the original 3D space.
#         * \param[in] project_points true if points should be projected, false if not.
#         */
#       void
#       setProjectPoints (bool project_points)
#       {
#         project_points_ = project_points;
#       }
# 
#       /** \brief Segmentation of all planes in a point cloud given by setInputCloud(), setIndices()
#         * \param[out] model_coefficients a vector of model_coefficients for each plane found in the input cloud
#         * \param[out] inlier_indices a vector of inliers for each detected plane
#         * \param[out] centroids a vector of centroids for each plane
#         * \param[out] covariances a vector of covariance matricies for the inliers of each plane
#         * \param[out] labels a point cloud for the connected component labels of each pixel
#         * \param[out] label_indices a vector of PointIndices for each labeled component
#         */
#       void
#       segment (std::vector<ModelCoefficients>& model_coefficients, 
#                std::vector<PointIndices>& inlier_indices,
#                std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& centroids,
#                std::vector <Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> >& covariances,
#                pcl::PointCloud<PointLT>& labels, 
#                std::vector<pcl::PointIndices>& label_indices);
# 
#       /** \brief Segmentation of all planes in a point cloud given by setInputCloud(), setIndices()
#         * \param[out] model_coefficients a vector of model_coefficients for each plane found in the input cloud
#         * \param[out] inlier_indices a vector of inliers for each detected plane
#         */
#       void
#       segment (std::vector<ModelCoefficients>& model_coefficients, 
#                std::vector<PointIndices>& inlier_indices);
# 
#       /** \brief Segmentation of all planes in a point cloud given by setInputCloud(), setIndices()
#         * \param[out] regions a list of resultant planar polygonal regions
#         */
#       void
#       segment (std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT> > >& regions);
#       
#       /** \brief Perform a segmentation, as well as an additional refinement step.  This helps with including points whose normals may not match neighboring points well, but may match the planar model well.
#         * \param[out] regions A list of regions generated by segmentation and refinement.
#         */
#       void
#       segmentAndRefine (std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT> > >& regions);
# 
#       /** \brief Perform a segmentation, as well as additional refinement step.  Returns intermediate data structures for use in
#         * subsequent processing.
#         * \param[out] regions A vector of PlanarRegions generated by segmentation
#         * \param[out] model_coefficients A vector of model coefficients for each segmented plane
#         * \param[out] inlier_indices A vector of PointIndices, indicating the inliers to each segmented plane
#         * \param[out] labels A PointCloud<PointLT> corresponding to the resulting segmentation.
#         * \param[out] label_indices A vector of PointIndices for each label
#         * \param[out] boundary_indices A vector of PointIndices corresponding to the outer boundary / contour of each label
#         */
#       void
#       segmentAndRefine (std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT> > >& regions,
#                         std::vector<ModelCoefficients>& model_coefficients,
#                         std::vector<PointIndices>& inlier_indices,
#                         PointCloudLPtr& labels,
#                         std::vector<pcl::PointIndices>& label_indices,
#                         std::vector<pcl::PointIndices>& boundary_indices);
#       
#       /** \brief Perform a refinement of an initial segmentation, by comparing points to adjacent regions detected by the initial segmentation.
#         * \param [in] model_coefficients The list of segmented model coefficients
#         * \param [in] inlier_indices The list of segmented inlier indices, corresponding to each model
#         * \param [in] centroids The list of centroids corresponding to each segmented plane
#         * \param [in] covariances The list of covariances corresponding to each segemented plane
#         * \param [in] labels The labels produced by the initial segmentation
#         * \param [in] label_indices The list of indices corresponding to each label
#         */
#       void
#       refine (std::vector<ModelCoefficients>& model_coefficients, 
#               std::vector<PointIndices>& inlier_indices,
#               std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& centroids,
#               std::vector <Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> >& covariances,
#               PointCloudLPtr& labels,
#               std::vector<pcl::PointIndices>& label_indices);
# 
#     protected:
# 
#       /** \brief A pointer to the input normals */
#       PointCloudNConstPtr normals_;
# 
#       /** \brief The minimum number of inliers required for each plane. */
#       unsigned min_inliers_;
# 
#       /** \brief The tolerance in radians for difference in normal direction between neighboring points, to be considered part of the same plane. */
#       double angular_threshold_;
# 
#       /** \brief The tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane. */
#       double distance_threshold_;
# 
#       /** \brief The tolerance for maximum curvature after fitting a plane.  Used to remove smooth, but non-planar regions. */
#       double maximum_curvature_;
# 
#       /** \brief Whether or not points should be projected to the plane, or left in the original 3D space. */
#       bool project_points_;
# 
#       /** \brief A comparator for comparing neighboring pixels' plane equations. */
#       PlaneComparatorPtr compare_;
# 
#       /** \brief A comparator for use on the refinement step.  Compares points to regions segmented in the first pass. */
#       PlaneRefinementComparatorPtr refinement_compare_;
# 
#       /** \brief Class getName method. */
#       virtual std::string
#       getClassName () const
#       {
#         return ("OrganizedMultiPlaneSegmentation");
#       }
#   };
# 
###

# planar_polygon_fusion.h
# namespace pcl
# {
#   /** \brief PlanarPolygonFusion takes a list of 2D planar polygons and
#     * attempts to reduce them to a minimum set that best represents the scene,
#     * based on various given comparators.
#     */
#   template <typename PointT>
#   class PlanarPolygonFusion
#   {
#     public:
#       /** \brief Constructor */
#       PlanarPolygonFusion () : regions_ () {}
#      
#       /** \brief Destructor */
#       virtual ~PlanarPolygonFusion () {}
# 
#       /** \brief Reset the state (clean the list of planar models). */
#       void 
#       reset ()
#       {
#         regions_.clear ();
#       }
#       
#       /** \brief Set the list of 2D planar polygons to refine.
#         * \param[in] input the list of 2D planar polygons to refine
#         */
#       void
#       addInputPolygons (const std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT> > > &input)
#       {
#         int start = static_cast<int> (regions_.size ());
#         regions_.resize (regions_.size () + input.size ());
#         for(size_t i = 0; i < input.size (); i++)
#           regions_[start+i] = input[i];
#       }
# 
#     protected:
#       /** \brief Internal list of planar states. */
#       std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions_;
#   };
###

# planar_region.h
# namespace pcl
# {
#   /** \brief PlanarRegion represents a set of points that lie in a plane.  Inherits summary statistics about these points from Region3D, and  summary statistics of a 3D collection of points.
#     * \author Alex Trevor
#     */
#   template <typename PointT>
#   class PlanarRegion : public pcl::Region3D<PointT>, public pcl::PlanarPolygon<PointT>
#   {
#     protected:
#       using Region3D<PointT>::centroid_;
#       using Region3D<PointT>::covariance_; 
#       using Region3D<PointT>::count_;
#       using PlanarPolygon<PointT>::contour_;
#       using PlanarPolygon<PointT>::coefficients_;
# 
#     public:
#       /** \brief Empty constructor for PlanarRegion. */
#       PlanarRegion () : contour_labels_ ()
#       {}
# 
#       /** \brief Constructor for Planar region from a Region3D and a PlanarPolygon. 
#         * \param[in] region a Region3D for the input data
#         * \param[in] polygon a PlanarPolygon for the input region
#         */
#       PlanarRegion (const pcl::Region3D<PointT>& region, const pcl::PlanarPolygon<PointT>& polygon) :
#         contour_labels_ ()
#       {
#         centroid_ = region.centroid;
#         covariance_ = region.covariance;
#         count_ = region.count;
#         contour_ = polygon.contour;
#         coefficients_ = polygon.coefficients;
#       }
#       
#       /** \brief Destructor. */
#       virtual ~PlanarRegion () {}
# 
#       /** \brief Constructor for PlanarRegion.
#         * \param[in] centroid the centroid of the region.
#         * \param[in] covariance the covariance of the region.
#         * \param[in] count the number of points in the region.
#         * \param[in] contour the contour / boudnary for the region
#         * \param[in] coefficients the model coefficients (a,b,c,d) for the plane
#         */
#       PlanarRegion (const Eigen::Vector3f& centroid, const Eigen::Matrix3f& covariance, unsigned count,
#                     const typename pcl::PointCloud<PointT>::VectorType& contour,
#                     const Eigen::Vector4f& coefficients) :
#         contour_labels_ ()
#       {
#         centroid_ = centroid;
#         covariance_ = covariance;
#         count_ = count;
#         contour_ = contour;
#         coefficients_ = coefficients;
#       }
#       
#     private:
#       /** \brief The labels (good=true, bad=false) for whether or not this boundary was observed, 
#         * or was due to edge of frame / occlusion boundary. 
#         */
#       std::vector<bool> contour_labels_;
# 
#     public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
###

# plane_refinement_comparator.h
# namespace pcl
# {
#   /** \brief PlaneRefinementComparator is a Comparator that operates on plane coefficients, 
#     * for use in planar segmentation.
#     * In conjunction with OrganizedConnectedComponentSegmentation, this allows planes to be segmented from organized data.
#     *
#     * \author Alex Trevor, Suat Gedikli
#     */
#   template<typename PointT, typename PointNT, typename PointLT>
#   class PlaneRefinementComparator: public PlaneCoefficientComparator<PointT, PointNT>
#   {
#     public:
#       typedef typename Comparator<PointT>::PointCloud PointCloud;
#       typedef typename Comparator<PointT>::PointCloudConstPtr PointCloudConstPtr;
#       
#       typedef typename pcl::PointCloud<PointNT> PointCloudN;
#       typedef typename PointCloudN::Ptr PointCloudNPtr;
#       typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
# 
#       typedef typename pcl::PointCloud<PointLT> PointCloudL;
#       typedef typename PointCloudL::Ptr PointCloudLPtr;
#       typedef typename PointCloudL::ConstPtr PointCloudLConstPtr;
# 
#       typedef boost::shared_ptr<PlaneRefinementComparator<PointT, PointNT, PointLT> > Ptr;
#       typedef boost::shared_ptr<const PlaneRefinementComparator<PointT, PointNT, PointLT> > ConstPtr;
# 
#       using pcl::PlaneCoefficientComparator<PointT, PointNT>::input_;
#       using pcl::PlaneCoefficientComparator<PointT, PointNT>::normals_;
#       using pcl::PlaneCoefficientComparator<PointT, PointNT>::distance_threshold_;
#       using pcl::PlaneCoefficientComparator<PointT, PointNT>::plane_coeff_d_;
# 
# 
#       /** \brief Empty constructor for PlaneCoefficientComparator. */
#      PlaneRefinementComparator ()
#         : models_ ()
#         , labels_ ()
#         , refine_labels_ ()
#         , label_to_model_ ()
#         , depth_dependent_ (false)
#       {
#       }
# 
#       /** \brief Empty constructor for PlaneCoefficientComparator. 
#         * \param[in] models
#         * \param[in] refine_labels
#         */
#       PlaneRefinementComparator (boost::shared_ptr<std::vector<pcl::ModelCoefficients> >& models,
#                                  boost::shared_ptr<std::vector<bool> >& refine_labels)
#         : models_ (models)
#         , labels_ ()
#         , refine_labels_ (refine_labels)
#         , label_to_model_ ()
#         , depth_dependent_ (false)
#       {
#       }
# 
#       /** \brief Destructor for PlaneCoefficientComparator. */
#       virtual
#       ~PlaneRefinementComparator ()
#       {
#       }
# 
#       /** \brief Set the vector of model coefficients to which we will compare.
#         * \param[in] models a vector of model coefficients produced by the initial segmentation step.
#         */
#       void
#       setModelCoefficients (boost::shared_ptr<std::vector<pcl::ModelCoefficients> >& models)
#       {
#         models_ = models;
#       }
# 
#       /** \brief Set the vector of model coefficients to which we will compare.
#         * \param[in] models a vector of model coefficients produced by the initial segmentation step.
#         */
#       void
#       setModelCoefficients (std::vector<pcl::ModelCoefficients>& models)
#       {
#         models_ = boost::make_shared<std::vector<pcl::ModelCoefficients> >(models);
#       }
# 
#       /** \brief Set which labels should be refined.  This is a vector of bools 0-max_label, true if the label should be refined.
#         * \param[in] refine_labels A vector of bools 0-max_label, true if the label should be refined.
#         */
#       void
#       setRefineLabels (boost::shared_ptr<std::vector<bool> >& refine_labels)
#       {
#         refine_labels_ = refine_labels;
#       }
#       
#       /** \brief Set which labels should be refined.  This is a vector of bools 0-max_label, true if the label should be refined.
#         * \param[in] refine_labels A vector of bools 0-max_label, true if the label should be refined.
#         */
#       void
#       setRefineLabels (std::vector<bool>& refine_labels)
#       {
#         refine_labels_ = boost::make_shared<std::vector<bool> >(refine_labels);
#       }
# 
#       /** \brief A mapping from label to index in the vector of models, allowing the model coefficients of a label to be accessed.
#         * \param[in] label_to_model A vector of size max_label, with the index of each corresponding model in models
#         */
#       inline void
#       setLabelToModel (boost::shared_ptr<std::vector<int> >& label_to_model)
#       {
#         label_to_model_ = label_to_model;
#       }
#       
#       /** \brief A mapping from label to index in the vector of models, allowing the model coefficients of a label to be accessed.
#         * \param[in] label_to_model A vector of size max_label, with the index of each corresponding model in models
#         */
#       inline void
#       setLabelToModel (std::vector<int>& label_to_model)
#       {
#         label_to_model_ = boost::make_shared<std::vector<int> >(label_to_model);
#       }
# 
#       /** \brief Get the vector of model coefficients to which we will compare. */
#       inline boost::shared_ptr<std::vector<pcl::ModelCoefficients> >
#       getModelCoefficients () const
#       {
#         return (models_);
#       }
# 
#       /** \brief ...
#         * \param[in] labels
#         */
#       inline void
#       setLabels (PointCloudLPtr& labels)
#       {
#         labels_ = labels;
#       }
# 
#       /** \brief Compare two neighboring points
#         * \param[in] idx1 The index of the first point.
#         * \param[in] idx2 The index of the second point.
#         */
#       virtual bool
#       compare (int idx1, int idx2) const
#       {
#         int current_label = labels_->points[idx1].label;
#         int next_label = labels_->points[idx2].label;
# 
#         if (!((*refine_labels_)[current_label] && !(*refine_labels_)[next_label]))
#           return (false);
#         
#         const pcl::ModelCoefficients& model_coeff = (*models_)[(*label_to_model_)[current_label]];
#         
#         PointT pt = input_->points[idx2];
#         double ptp_dist = fabs (model_coeff.values[0] * pt.x + 
#                                 model_coeff.values[1] * pt.y + 
#                                 model_coeff.values[2] * pt.z +
#                                 model_coeff.values[3]);
#         
#         // depth dependent
#         float threshold = distance_threshold_;
#         if (depth_dependent_)
#         {
#           //Eigen::Vector4f origin = input_->sensor_origin_;
#           Eigen::Vector3f vec = input_->points[idx1].getVector3fMap ();// - origin.head<3> ();
#           
#           float z = vec.dot (z_axis_);
#           threshold *= z * z;
#         }
#         
#         return (ptp_dist < threshold);
#       }
# 
#     protected:
#       boost::shared_ptr<std::vector<pcl::ModelCoefficients> > models_;
#       PointCloudLPtr labels_;
#       boost::shared_ptr<std::vector<bool> > refine_labels_;
#       boost::shared_ptr<std::vector<int> > label_to_model_;
#       bool depth_dependent_;
#       using PlaneCoefficientComparator<PointT, PointNT>::z_axis_;
###

# region_3d.h
# namespace pcl
# {
#   /** \brief Region3D represents summary statistics of a 3D collection of points.
#     * \author Alex Trevor
#     */
#   template <typename PointT>
#   class Region3D
#   {
#     public:
#       /** \brief Empty constructor for Region3D. */
#       Region3D () : centroid_ (Eigen::Vector3f::Zero ()), covariance_ (Eigen::Matrix3f::Identity ()), count_ (0)
#       {
#       }
#       
#       /** \brief Constructor for Region3D. 
#         * \param[in] centroid The centroid of the region.
#         * \param[in] covariance The covariance of the region.
#         * \param[in] count The number of points in the region.
#         */
#       Region3D (Eigen::Vector3f& centroid, Eigen::Matrix3f& covariance, unsigned count) 
#         : centroid_ (centroid), covariance_ (covariance), count_ (count)
#       {
#       }
#      
#       /** \brief Destructor. */
#       virtual ~Region3D () {}
# 
#       /** \brief Get the centroid of the region. */
#       inline Eigen::Vector3f 
#       getCentroid () const
#       {
#         return (centroid_);
#       }
#       
#       /** \brief Get the covariance of the region. */
#       inline Eigen::Matrix3f
#       getCovariance () const
#       {
#         return (covariance_);
#       }
#       
#       /** \brief Get the number of points in the region. */
#       unsigned
#       getCount () const
#       {
#         return (count_);
#       }
# 
#     protected:
#       /** \brief The centroid of the region. */
#       Eigen::Vector3f centroid_;
#       
#       /** \brief The covariance of the region. */
#       Eigen::Matrix3f covariance_;
#       
#       /** \brief The number of points in the region. */
#       unsigned count_;
# 
#     public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
###

# rgb_plane_coefficient_comparator.h
# namespace pcl
# {
#   /** \brief RGBPlaneCoefficientComparator is a Comparator that operates on plane coefficients, 
#     * for use in planar segmentation.  Also takes into account RGB, so we can segmented different colored co-planar regions.
#     * In conjunction with OrganizedConnectedComponentSegmentation, this allows planes to be segmented from organized data.
#     *
#     * \author Alex Trevor
#     */
#   template<typename PointT, typename PointNT>
#   class RGBPlaneCoefficientComparator: public PlaneCoefficientComparator<PointT, PointNT>
#   {
#     public:
#       typedef typename Comparator<PointT>::PointCloud PointCloud;
#       typedef typename Comparator<PointT>::PointCloudConstPtr PointCloudConstPtr;
#       
#       typedef typename pcl::PointCloud<PointNT> PointCloudN;
#       typedef typename PointCloudN::Ptr PointCloudNPtr;
#       typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
#       
#       typedef boost::shared_ptr<RGBPlaneCoefficientComparator<PointT, PointNT> > Ptr;
#       typedef boost::shared_ptr<const RGBPlaneCoefficientComparator<PointT, PointNT> > ConstPtr;
# 
#       using pcl::Comparator<PointT>::input_;
#       using pcl::PlaneCoefficientComparator<PointT, PointNT>::normals_;
#       using pcl::PlaneCoefficientComparator<PointT, PointNT>::angular_threshold_;
#       using pcl::PlaneCoefficientComparator<PointT, PointNT>::distance_threshold_;
# 
#       /** \brief Empty constructor for RGBPlaneCoefficientComparator. */
#       RGBPlaneCoefficientComparator ()
#         : color_threshold_ (50.0f)
#       {
#       }
# 
#       /** \brief Constructor for RGBPlaneCoefficientComparator.
#         * \param[in] plane_coeff_d a reference to a vector of d coefficients of plane equations.  Must be the same size as the input cloud and input normals.  a, b, and c coefficients are in the input normals.
#         */
#       RGBPlaneCoefficientComparator (boost::shared_ptr<std::vector<float> >& plane_coeff_d) 
#         : PlaneCoefficientComparator<PointT, PointNT> (plane_coeff_d), color_threshold_ (50.0f)
#       {
#       }
#       
#       /** \brief Destructor for RGBPlaneCoefficientComparator. */
#       virtual
#       ~RGBPlaneCoefficientComparator ()
#       {
#       }
# 
#       /** \brief Set the tolerance in color space between neighboring points, to be considered part of the same plane.
#         * \param[in] color_threshold The distance in color space
#         */
#       inline void
#       setColorThreshold (float color_threshold)
#       {
#         color_threshold_ = color_threshold * color_threshold;
#       }
# 
#       /** \brief Get the color threshold between neighboring points, to be considered part of the same plane. */
#       inline float
#       getColorThreshold () const
#       {
#         return (color_threshold_);
#       }
# 
#       /** \brief Compare two neighboring points, by using normal information, euclidean distance, and color information.
#         * \param[in] idx1 The index of the first point.
#         * \param[in] idx2 The index of the second point.
#         */
#       bool
#       compare (int idx1, int idx2) const
#       {
#         float dx = input_->points[idx1].x - input_->points[idx2].x;
#         float dy = input_->points[idx1].y - input_->points[idx2].y;
#         float dz = input_->points[idx1].z - input_->points[idx2].z;
#         float dist = sqrtf (dx*dx + dy*dy + dz*dz);
#         int dr = input_->points[idx1].r - input_->points[idx2].r;
#         int dg = input_->points[idx1].g - input_->points[idx2].g;
#         int db = input_->points[idx1].b - input_->points[idx2].b;
#         //Note: This is not the best metric for color comparisons, we should probably use HSV space.
#         float color_dist = static_cast<float> (dr*dr + dg*dg + db*db);
#         return ( (dist < distance_threshold_)
#                  && (normals_->points[idx1].getNormalVector3fMap ().dot (normals_->points[idx2].getNormalVector3fMap () ) > angular_threshold_ )
#                  && (color_dist < color_threshold_));
#       }
#       
#     protected:
#       float color_threshold_;
#   };
# 
###

# segment_differences.h
# namespace pcl
# /** \brief Obtain the difference between two aligned point clouds as another point cloud, given a distance threshold.
#   * \param src the input point cloud source
#   * \param tgt the input point cloud target we need to obtain the difference against
#   * \param threshold the distance threshold (tolerance) for point correspondences. (e.g., check if f a point p1 from 
#   * src has a correspondence > threshold than a point p2 from tgt)
#   * \param tree the spatial locator (e.g., kd-tree) used for nearest neighbors searching built over \a tgt
#   * \param output the resultant output point cloud difference
#   * \ingroup segmentation
#   */
# template <typename PointT> 
# void getPointCloudDifference (
#     const pcl::PointCloud<PointT> &src, const pcl::PointCloud<PointT> &tgt, 
#     double threshold, const boost::shared_ptr<pcl::search::Search<PointT> > &tree,
#     pcl::PointCloud<PointT> &output);

# /** \brief @b SegmentDifferences obtains the difference between two spatially
#   * aligned point clouds and returns the difference between them for a maximum
#   * given distance threshold.
#   * \author Radu Bogdan Rusu
#   * \ingroup segmentation
#   */
# template <typename PointT>
# class SegmentDifferences: public PCLBase<PointT>
#     typedef PCLBase<PointT> BasePCLBase;
# 
#     public:
#       typedef pcl::PointCloud<PointT> PointCloud;
#       typedef typename PointCloud::Ptr PointCloudPtr;
#       typedef typename PointCloud::ConstPtr PointCloudConstPtr;
# 
#       typedef typename pcl::search::Search<PointT> KdTree;
#       typedef typename pcl::search::Search<PointT>::Ptr KdTreePtr;
# 
#       typedef PointIndices::Ptr PointIndicesPtr;
#       typedef PointIndices::ConstPtr PointIndicesConstPtr;
# 
#       /** \brief Empty constructor. */
#       SegmentDifferences ()
# 
#       /** \brief Provide a pointer to the target dataset against which we
#         * compare the input cloud given in setInputCloud
#         * \param cloud the target PointCloud dataset
#       inline void setTargetCloud (const PointCloudConstPtr &cloud)
# 
#       /** \brief Get a pointer to the input target point cloud dataset. */
#       inline PointCloudConstPtr const getTargetCloud ()
#       /** \brief Provide a pointer to the search object.
#         * \param tree a pointer to the spatial search object.
#       inline void setSearchMethod (const KdTreePtr &tree)
#       /** \brief Get a pointer to the search method used. */
#       inline KdTreePtr getSearchMethod ()
#       /** \brief Set the maximum distance tolerance (squared) between corresponding
#         * points in the two input datasets.
#         * \param sqr_threshold the squared distance tolerance as a measure in L2 Euclidean space
#       inline void setDistanceThreshold (double sqr_threshold)
#       /** \brief Get the squared distance tolerance between corresponding points as a
#         * measure in the L2 Euclidean space.
#       inline double getDistanceThreshold ()
# 
#       /** \brief Segment differences between two input point clouds.
#         * \param output the resultant difference between the two point clouds as a PointCloud
#         */
#       void segment (PointCloud &output);
#       protected:
#       // Members derived from the base class
#       using BasePCLBase::input_;
#       using BasePCLBase::indices_;
#       using BasePCLBase::initCompute;
#       using BasePCLBase::deinitCompute;
#       /** \brief A pointer to the spatial search object. */
#       KdTreePtr tree_;
#       /** \brief The input target point cloud dataset. */
#       PointCloudConstPtr target_;
#       /** \brief The distance tolerance (squared) as a measure in the L2
#         * Euclidean space between corresponding points. 
#         */
#       double distance_threshold_;
#       /** \brief Class getName method. */
#       virtual std::string getClassName () const { return ("SegmentDifferences"); }
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

###############################################################################
# Activation
###############################################################################


### pcl 1.7.2 ###
# approximate_progressive_morphological_filter.h
# boost.h
# comparator.h
# conditional_euclidean_clustering.h
# crf_normal_segmentation.h
# edge_aware_plane_comparator.h
# euclidean_cluster_comparator.h
# euclidean_plane_coefficient_comparator.h
# extract_clusters.h
# extract_labeled_clusters.h
# extract_polygonal_prism_data.h
# grabcut_segmentation.h
# ground_plane_comparator.h

# min_cut_segmentation.h
# namespace pcl
# template <typename PointT>
# class PCL_EXPORTS MinCutSegmentation : public pcl::PCLBase<PointT>
cdef extern from "pcl/segmentation/min_cut_segmentation.h" namespace "pcl":
    cdef cppclass MinCutSegmentation[T](cpp.PCLBase[T]):
        MinCutSegmentation()
        # public:
        # typedef pcl::search::Search <PointT> KdTree;
        # typedef typename KdTree::Ptr KdTreePtr;
        # typedef pcl::PointCloud< PointT > PointCloud;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # using PCLBase <PointT>::input_;
        # using PCLBase <PointT>::indices_;
        # using PCLBase <PointT>::initCompute;
        # using PCLBase <PointT>::deinitCompute;
        # public:
        # typedef boost::adjacency_list_traits< boost::vecS, boost::vecS, boost::directedS > Traits;
        # typedef boost::adjacency_list< boost::vecS, boost::vecS, boost::directedS,
        #                              boost::property< boost::vertex_name_t, std::string,
        #                                boost::property< boost::vertex_index_t, long,
        #                                  boost::property< boost::vertex_color_t, boost::default_color_type,
        #                                    boost::property< boost::vertex_distance_t, long,
        #                                      boost::property< boost::vertex_predecessor_t, Traits::edge_descriptor > > > > >,
        #                              boost::property< boost::edge_capacity_t, double,
        #                                boost::property< boost::edge_residual_capacity_t, double,
        #                                  boost::property< boost::edge_reverse_t, Traits::edge_descriptor > > > > mGraph;
        # typedef boost::property_map< mGraph, boost::edge_capacity_t >::type CapacityMap;
        # typedef boost::property_map< mGraph, boost::edge_reverse_t>::type ReverseEdgeMap;
        # typedef Traits::vertex_descriptor VertexDescriptor;
        # typedef boost::graph_traits< mGraph >::edge_descriptor EdgeDescriptor;
        # typedef boost::graph_traits< mGraph >::out_edge_iterator OutEdgeIterator;
        # typedef boost::graph_traits< mGraph >::vertex_iterator VertexIterator;
        # typedef boost::property_map< mGraph, boost::edge_residual_capacity_t >::type ResidualCapacityMap;
        # typedef boost::property_map< mGraph, boost::vertex_index_t >::type IndexMap;
        # typedef boost::graph_traits< mGraph >::in_edge_iterator InEdgeIterator;
        # public:
        # /** \brief This method simply sets the input point cloud.
        #   * \param[in] cloud the const boost shared pointer to a PointCloud
        # virtual void setInputCloud (const PointCloudConstPtr &cloud);
        # /** \brief Returns normalization value for binary potentials. For more information see the article. */
        double getSigma ()
        # /** \brief Allows to set the normalization value for the binary potentials as described in the article.
        #   * \param[in] sigma new normalization value
        void setSigma (double sigma)
        # /** \brief Returns radius to the background. */
        double getRadius ()
        # /** \brief Allows to set the radius to the background.
        #   * \param[in] radius new radius to the background
        void setRadius (double radius)
        # /** \brief Returns weight that every edge from the source point has. */
        double getSourceWeight ()
        # /** \brief Allows to set weight for source edges. Every edge that comes from the source point will have that weight.
        #   * \param[in] weight new weight
        void setSourceWeight (double weight)
        # /** \brief Returns search method that is used for finding KNN.
        #   * The graph is build such way that it contains the edges that connect point and its KNN.
        # KdTreePtr getSearchMethod () const;
        # /** \brief Allows to set search method for finding KNN.
        #   * The graph is build such way that it contains the edges that connect point and its KNN.
        #   * \param[in] search search method that will be used for finding KNN.
        # void setSearchMethod (const KdTreePtr& tree);
        # /** \brief Returns the number of neighbours to find. */
        unsigned int getNumberOfNeighbours ()
        # /** \brief Allows to set the number of neighbours to find.
        #   * \param[in] number_of_neighbours new number of neighbours
        void setNumberOfNeighbours (unsigned int neighbour_number)
        # /** \brief Returns the points that must belong to foreground. */
        # std::vector<PointT, Eigen::aligned_allocator<PointT> > getForegroundPoints () const;
        # /** \brief Allows to specify points which are known to be the points of the object.
        #   * \param[in] foreground_points point cloud that contains foreground points. At least one point must be specified.
        # void setForegroundPoints (typename pcl::PointCloud<PointT>::Ptr foreground_points);
        # /** \brief Returns the points that must belong to background. */
        # std::vector<PointT, Eigen::aligned_allocator<PointT> > getBackgroundPoints () const;
        # /** \brief Allows to specify points which are known to be the points of the background.
        #   * \param[in] background_points point cloud that contains background points.
        # void setBackgroundPoints (typename pcl::PointCloud<PointT>::Ptr background_points);
        # /** \brief This method launches the segmentation algorithm and returns the clusters that were
        #   * obtained during the segmentation. The indices of points that belong to the object will be stored
        #   * in the cluster with index 1, other indices will be stored in the cluster with index 0.
        #   * \param[out] clusters clusters that were obtained. Each cluster is an array of point indices.
        # void extract (vector <pcl::PointIndices>& clusters);
        # /** \brief Returns that flow value that was calculated during the segmentation. */
        double getMaxFlow ()
        # /** \brief Returns the graph that was build for finding the minimum cut. */
        # typename boost::shared_ptr<typename pcl::MinCutSegmentation<PointT>::mGraph> getGraph () const;
        # /** \brief Returns the colored cloud. Points that belong to the object have the same color. */
        # pcl::PointCloud<pcl::PointXYZRGB>::Ptr getColoredCloud ();
        # protected:
        # /** \brief This method simply builds the graph that will be used during the segmentation. */
        bool buildGraph ()
        # /** \brief Returns unary potential(data cost) for the given point index.
        #   * In other words it calculates weights for (source, point) and (point, sink) edges.
        #   * \param[in] point index of the point for which weights will be calculated
        #   * \param[out] source_weight calculated weight for the (source, point) edge
        #   * \param[out] sink_weight calculated weight for the (point, sink) edge
        void calculateUnaryPotential (int point, double& source_weight, double& sink_weight)
        # /** \brief This method simply adds the edge from the source point to the target point with a given weight.
        #   * \param[in] source index of the source point of the edge
        #   * \param[in] target index of the target point of the edge
        #   * \param[in] weight weight that will be assigned to the (source, target) edge
        bool addEdge (int source, int target, double weight)
        # /** \brief Returns the binary potential(smooth cost) for the given indices of points.
        #   * In other words it returns weight that must be assigned to the edge from source to target point.
        #   * \param[in] source index of the source point of the edge
        #   * \param[in] target index of the target point of the edge
        double calculateBinaryPotential (int source, int target)
        # brief This method recalculates unary potentials(data cost) if some changes were made, instead of creating new graph. */
        bool recalculateUnaryPotentials ()
        # brief This method recalculates binary potentials(smooth cost) if some changes were made, instead of creating new graph. */
        bool recalculateBinaryPotentials ()
        # /** \brief This method analyzes the residual network and assigns a label to every point in the cloud.
        #   * \param[in] residual_capacity residual network that was obtained during the segmentation
        # void assembleLabels (ResidualCapacityMap& residual_capacity);
        # protected:
        # /** \brief Stores the sigma coefficient. It is used for finding smooth costs. More information can be found in the article. */
        # double inverse_sigma_;
        # /** \brief Signalizes if the binary potentials are valid. */
        # bool binary_potentials_are_valid_;
        # /** \brief Used for comparison of the floating point numbers. */
        # double epsilon_;
        # /** \brief Stores the distance to the background. */
        # double radius_;
        # /** \brief Signalizes if the unary potentials are valid. */
        # bool unary_potentials_are_valid_;
        # /** \brief Stores the weight for every edge that comes from source point. */
        # double source_weight_;
        # /** \brief Stores the search method that will be used for finding K nearest neighbors. Neighbours are used for building the graph. */
        # KdTreePtr search_;
        # /** \brief Stores the number of neighbors to find. */
        # unsigned int number_of_neighbours_;
        # /** \brief Signalizes if the graph is valid. */
        # bool graph_is_valid_;
        # /** \brief Stores the points that are known to be in the foreground. */
        # std::vector<PointT, Eigen::aligned_allocator<PointT> > foreground_points_;
        # /** \brief Stores the points that are known to be in the background. */
        # std::vector<PointT, Eigen::aligned_allocator<PointT> > background_points_;
        # /** \brief After the segmentation it will contain the segments. */
        # std::vector <pcl::PointIndices> clusters_;
        # /** \brief Stores the graph for finding the maximum flow. */
        # boost::shared_ptr<mGraph> graph_;
        # /** \brief Stores the capacity of every edge in the graph. */
        # boost::shared_ptr<CapacityMap> capacity_;
        # /** \brief Stores reverse edges for every edge in the graph. */
        # boost::shared_ptr<ReverseEdgeMap> reverse_edges_;
        # /** \brief Stores the vertices of the graph. */
        # std::vector< VertexDescriptor > vertices_;
        # /** \brief Stores the information about the edges that were added to the graph. It is used to avoid the duplicate edges. */
        # std::vector< std::set<int> > edge_marker_;
        # /** \brief Stores the vertex that serves as source. */
        # VertexDescriptor source_;
        # /** \brief Stores the vertex that serves as sink. */
        # VertexDescriptor sink_;
        # /** \brief Stores the maximum flow value that was calculated during the segmentation. */
        # double max_flow_;
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
###


# organized_connected_component_segmentation.h
# namespace pcl
# /** \brief OrganizedConnectedComponentSegmentation allows connected
#   * components to be found within organized point cloud data, given a
#   * comparison function.  Given an input cloud and a comparator, it will
#   * output a PointCloud of labels, giving each connected component a unique
#   * id, along with a vector of PointIndices corresponding to each component.
#   * See OrganizedMultiPlaneSegmentation for an example application.
#   * \author Alex Trevor, Suat Gedikli
#   */
# template <typename PointT, typename PointLT>
# class OrganizedConnectedComponentSegmentation : public PCLBase<PointT>
# {
cdef extern from "pcl/segmentation/organized_connected_component_segmentation.h" namespace "pcl":
    cdef cppclass OrganizedConnectedComponentSegmentation[T, LT](cpp.PCLBase[T]):
        OrganizedConnectedComponentSegmentation()
		using PCLBase<PointT>::input_;
		using PCLBase<PointT>::indices_;
		using PCLBase<PointT>::initCompute;
		using PCLBase<PointT>::deinitCompute;
		
    	public:
      	typedef typename pcl::PointCloud<PointT> PointCloud;
      	typedef typename PointCloud::Ptr PointCloudPtr;
      	typedef typename PointCloud::ConstPtr PointCloudConstPtr;
      	
      	typedef typename pcl::PointCloud<PointLT> PointCloudL;
      	typedef typename PointCloudL::Ptr PointCloudLPtr;
      	typedef typename PointCloudL::ConstPtr PointCloudLConstPtr;
		typedef typename pcl::Comparator<PointT> Comparator;
      	typedef typename Comparator::Ptr ComparatorPtr;
      	typedef typename Comparator::ConstPtr ComparatorConstPtr;
      	
      	/** \brief Constructor for OrganizedConnectedComponentSegmentation
         * \param[in] compare A pointer to the comparator to be used for segmentation.  Must be an instance of pcl::Comparator.
        */
      	OrganizedConnectedComponentSegmentation (const ComparatorConstPtr& compare) : compare_ (compare)
      	/** \brief Destructor for OrganizedConnectedComponentSegmentation. */
      	virtual ~OrganizedConnectedComponentSegmentation ()
		
      	/** \brief Provide a pointer to the comparator to be used for segmentation.
         * \param[in] compare the comparator
        */
      	void setComparator (const ComparatorConstPtr& compare)
      	
      	/** \brief Get the comparator.*/
      	ComparatorConstPtr getComparator () const { return (compare_); }
		
      	/** \brief Perform the connected component segmentation.
         * \param[out] labels a PointCloud of labels: each connected component will have a unique id.
         * \param[out] label_indices a vector of PointIndices corresponding to each label / component id.
        */
      	void segment (pcl::PointCloud<PointLT>& labels, std::vector<pcl::PointIndices>& label_indices) const;
      	
      	/** \brief Find the boundary points / contour of a connected component
         * \param[in] start_idx the first (lowest) index of the connected component for which a boundary shoudl be returned
         * \param[in] labels the Label cloud produced by segmentation
         * \param[out] boundary_indices the indices of the boundary points for the label corresponding to start_idx
        */
      	static void findLabeledRegionBoundary (int start_idx, PointCloudLPtr labels, pcl::PointIndices& boundary_indices);      


###

# organized_multi_plane_segmentation.h
namespace pcl
{
  /** \brief OrganizedMultiPlaneSegmentation finds all planes present in the
    * input cloud, and outputs a vector of plane equations, as well as a vector
    * of point clouds corresponding to the inliers of each detected plane.  Only
    * planes with more than min_inliers points are detected.
    * Templated on point type, normal type, and label type
    *
    * \author Alex Trevor, Suat Gedikli
    */
  template<typename PointT, typename PointNT, typename PointLT>
  class OrganizedMultiPlaneSegmentation : public PCLBase<PointT>
  {
    using PCLBase<PointT>::input_;
    using PCLBase<PointT>::indices_;
    using PCLBase<PointT>::initCompute;
    using PCLBase<PointT>::deinitCompute;

    public:
      typedef pcl::PointCloud<PointT> PointCloud;
      typedef typename PointCloud::Ptr PointCloudPtr;
      typedef typename PointCloud::ConstPtr PointCloudConstPtr;

      typedef typename pcl::PointCloud<PointNT> PointCloudN;
      typedef typename PointCloudN::Ptr PointCloudNPtr;
      typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;

      typedef typename pcl::PointCloud<PointLT> PointCloudL;
      typedef typename PointCloudL::Ptr PointCloudLPtr;
      typedef typename PointCloudL::ConstPtr PointCloudLConstPtr;

      typedef typename pcl::PlaneCoefficientComparator<PointT, PointNT> PlaneComparator;
      typedef typename PlaneComparator::Ptr PlaneComparatorPtr;
      typedef typename PlaneComparator::ConstPtr PlaneComparatorConstPtr;

      typedef typename pcl::PlaneRefinementComparator<PointT, PointNT, PointLT> PlaneRefinementComparator;
      typedef typename PlaneRefinementComparator::Ptr PlaneRefinementComparatorPtr;
      typedef typename PlaneRefinementComparator::ConstPtr PlaneRefinementComparatorConstPtr;

      /** \brief Constructor for OrganizedMultiPlaneSegmentation. */
      OrganizedMultiPlaneSegmentation () :
        normals_ (), 
        min_inliers_ (1000), 
        angular_threshold_ (pcl::deg2rad (3.0)), 
        distance_threshold_ (0.02),
        maximum_curvature_ (0.001),
        project_points_ (false), 
        compare_ (new PlaneComparator ()), refinement_compare_ (new PlaneRefinementComparator ())
      {
      }

      /** \brief Destructor for OrganizedMultiPlaneSegmentation. */
      virtual
      ~OrganizedMultiPlaneSegmentation ()
      {
      }

      /** \brief Provide a pointer to the input normals.
        * \param[in] normals the input normal cloud
        */
      inline void
      setInputNormals (const PointCloudNConstPtr &normals) 
      {
        normals_ = normals;
      }

      /** \brief Get the input normals. */
      inline PointCloudNConstPtr
      getInputNormals () const
      {
        return (normals_);
      }

      /** \brief Set the minimum number of inliers required for a plane
        * \param[in] min_inliers the minimum number of inliers required per plane
        */
      inline void
      setMinInliers (unsigned min_inliers)
      {
        min_inliers_ = min_inliers;
      }

      /** \brief Get the minimum number of inliers required per plane. */
      inline unsigned
      getMinInliers () const
      {
        return (min_inliers_);
      }

      /** \brief Set the tolerance in radians for difference in normal direction between neighboring points, to be considered part of the same plane.
        * \param[in] angular_threshold the tolerance in radians
        */
      inline void
      setAngularThreshold (double angular_threshold)
      {
        angular_threshold_ = angular_threshold;
      }

      /** \brief Get the angular threshold in radians for difference in normal direction between neighboring points, to be considered part of the same plane. */
      inline double
      getAngularThreshold () const
      {
        return (angular_threshold_);
      }

      /** \brief Set the tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane.
        * \param[in] distance_threshold the tolerance in meters
        */
      inline void
      setDistanceThreshold (double distance_threshold)
      {
        distance_threshold_ = distance_threshold;
      }

      /** \brief Get the distance threshold in meters (d component of plane equation) between neighboring points, to be considered part of the same plane. */
      inline double
      getDistanceThreshold () const
      {
        return (distance_threshold_);
      }

      /** \brief Set the maximum curvature allowed for a planar region.
        * \param[in] maximum_curvature the maximum curvature
        */
      inline void
      setMaximumCurvature (double maximum_curvature)
      {
        maximum_curvature_ = maximum_curvature;
      }

      /** \brief Get the maximum curvature allowed for a planar region. */
      inline double
      getMaximumCurvature () const
      {
        return (maximum_curvature_);
      }

      /** \brief Provide a pointer to the comparator to be used for segmentation.
        * \param[in] compare A pointer to the comparator to be used for segmentation.
        */
      void
      setComparator (const PlaneComparatorPtr& compare)
      {
        compare_ = compare;
      }

      /** \brief Provide a pointer to the comparator to be used for refinement.
        * \param[in] compare A pointer to the comparator to be used for refinement.
        */
      void
      setRefinementComparator (const PlaneRefinementComparatorPtr& compare)
      {
        refinement_compare_ = compare;
      }

      /** \brief Set whether or not to project boundary points to the plane, or leave them in the original 3D space.
        * \param[in] project_points true if points should be projected, false if not.
        */
      void
      setProjectPoints (bool project_points)
      {
        project_points_ = project_points;
      }

      /** \brief Segmentation of all planes in a point cloud given by setInputCloud(), setIndices()
        * \param[out] model_coefficients a vector of model_coefficients for each plane found in the input cloud
        * \param[out] inlier_indices a vector of inliers for each detected plane
        * \param[out] centroids a vector of centroids for each plane
        * \param[out] covariances a vector of covariance matricies for the inliers of each plane
        * \param[out] labels a point cloud for the connected component labels of each pixel
        * \param[out] label_indices a vector of PointIndices for each labeled component
        */
      void
      segment (std::vector<ModelCoefficients>& model_coefficients, 
               std::vector<PointIndices>& inlier_indices,
               std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& centroids,
               std::vector <Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> >& covariances,
               pcl::PointCloud<PointLT>& labels, 
               std::vector<pcl::PointIndices>& label_indices);

      /** \brief Segmentation of all planes in a point cloud given by setInputCloud(), setIndices()
        * \param[out] model_coefficients a vector of model_coefficients for each plane found in the input cloud
        * \param[out] inlier_indices a vector of inliers for each detected plane
        */
      void
      segment (std::vector<ModelCoefficients>& model_coefficients, 
               std::vector<PointIndices>& inlier_indices);

      /** \brief Segmentation of all planes in a point cloud given by setInputCloud(), setIndices()
        * \param[out] regions a list of resultant planar polygonal regions
        */
      void
      segment (std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT> > >& regions);
      
      /** \brief Perform a segmentation, as well as an additional refinement step.  This helps with including points whose normals may not match neighboring points well, but may match the planar model well.
        * \param[out] regions A list of regions generated by segmentation and refinement.
        */
      void
      segmentAndRefine (std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT> > >& regions);

      /** \brief Perform a segmentation, as well as additional refinement step.  Returns intermediate data structures for use in
        * subsequent processing.
        * \param[out] regions A vector of PlanarRegions generated by segmentation
        * \param[out] model_coefficients A vector of model coefficients for each segmented plane
        * \param[out] inlier_indices A vector of PointIndices, indicating the inliers to each segmented plane
        * \param[out] labels A PointCloud<PointLT> corresponding to the resulting segmentation.
        * \param[out] label_indices A vector of PointIndices for each label
        * \param[out] boundary_indices A vector of PointIndices corresponding to the outer boundary / contour of each label
        */
      void
      segmentAndRefine (std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT> > >& regions,
                        std::vector<ModelCoefficients>& model_coefficients,
                        std::vector<PointIndices>& inlier_indices,
                        PointCloudLPtr& labels,
                        std::vector<pcl::PointIndices>& label_indices,
                        std::vector<pcl::PointIndices>& boundary_indices);
      
      /** \brief Perform a refinement of an initial segmentation, by comparing points to adjacent regions detected by the initial segmentation.
        * \param [in] model_coefficients The list of segmented model coefficients
        * \param [in] inlier_indices The list of segmented inlier indices, corresponding to each model
        * \param [in] centroids The list of centroids corresponding to each segmented plane
        * \param [in] covariances The list of covariances corresponding to each segemented plane
        * \param [in] labels The labels produced by the initial segmentation
        * \param [in] label_indices The list of indices corresponding to each label
        */
      void
      refine (std::vector<ModelCoefficients>& model_coefficients, 
              std::vector<PointIndices>& inlier_indices,
              std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& centroids,
              std::vector <Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> >& covariances,
              PointCloudLPtr& labels,
              std::vector<pcl::PointIndices>& label_indices);

    protected:

      /** \brief A pointer to the input normals */
      PointCloudNConstPtr normals_;

      /** \brief The minimum number of inliers required for each plane. */
      unsigned min_inliers_;

      /** \brief The tolerance in radians for difference in normal direction between neighboring points, to be considered part of the same plane. */
      double angular_threshold_;

      /** \brief The tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane. */
      double distance_threshold_;

      /** \brief The tolerance for maximum curvature after fitting a plane.  Used to remove smooth, but non-planar regions. */
      double maximum_curvature_;

      /** \brief Whether or not points should be projected to the plane, or left in the original 3D space. */
      bool project_points_;

      /** \brief A comparator for comparing neighboring pixels' plane equations. */
      PlaneComparatorPtr compare_;

      /** \brief A comparator for use on the refinement step.  Compares points to regions segmented in the first pass. */
      PlaneRefinementComparatorPtr refinement_compare_;

      /** \brief Class getName method. */
      virtual std::string
      getClassName () const
      {
        return ("OrganizedMultiPlaneSegmentation");
      }
  };

}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/segmentation/impl/organized_multi_plane_segmentation.hpp>
#endif

#endif //#ifndef PCL_SEGMENTATION_ORGANIZED_MULTI_PLANE_SEGMENTATION_H_
###

# planar_polygon_fusion.h
namespace pcl
{
  /** \brief PlanarPolygonFusion takes a list of 2D planar polygons and
    * attempts to reduce them to a minimum set that best represents the scene,
    * based on various given comparators.
    */
  template <typename PointT>
  class PlanarPolygonFusion
  {
    public:
      /** \brief Constructor */
      PlanarPolygonFusion () : regions_ () {}
     
      /** \brief Destructor */
      virtual ~PlanarPolygonFusion () {}

      /** \brief Reset the state (clean the list of planar models). */
      void 
      reset ()
      {
        regions_.clear ();
      }
      
      /** \brief Set the list of 2D planar polygons to refine.
        * \param[in] input the list of 2D planar polygons to refine
        */
      void
      addInputPolygons (const std::vector<PlanarRegion<PointT>, Eigen::aligned_allocator<PlanarRegion<PointT> > > &input)
      {
        int start = static_cast<int> (regions_.size ());
        regions_.resize (regions_.size () + input.size ());
        for(size_t i = 0; i < input.size (); i++)
          regions_[start+i] = input[i];
      }

    protected:
      /** \brief Internal list of planar states. */
      std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions_;
  };
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/segmentation/impl/planar_polygon_fusion.hpp>
#endif

#endif // PCL_SEGMENTATION_PLANAR_POLYGON_FUSION_H_
###

# planar_region.h
namespace pcl
{
  /** \brief PlanarRegion represents a set of points that lie in a plane.  Inherits summary statistics about these points from Region3D, and  summary statistics of a 3D collection of points.
    * \author Alex Trevor
    */
  template <typename PointT>
  class PlanarRegion : public pcl::Region3D<PointT>, public pcl::PlanarPolygon<PointT>
  {
    protected:
      using Region3D<PointT>::centroid_;
      using Region3D<PointT>::covariance_; 
      using Region3D<PointT>::count_;
      using PlanarPolygon<PointT>::contour_;
      using PlanarPolygon<PointT>::coefficients_;

    public:
      /** \brief Empty constructor for PlanarRegion. */
      PlanarRegion () : contour_labels_ ()
      {}

      /** \brief Constructor for Planar region from a Region3D and a PlanarPolygon. 
        * \param[in] region a Region3D for the input data
        * \param[in] polygon a PlanarPolygon for the input region
        */
      PlanarRegion (const pcl::Region3D<PointT>& region, const pcl::PlanarPolygon<PointT>& polygon) :
        contour_labels_ ()
      {
        centroid_ = region.centroid;
        covariance_ = region.covariance;
        count_ = region.count;
        contour_ = polygon.contour;
        coefficients_ = polygon.coefficients;
      }
      
      /** \brief Destructor. */
      virtual ~PlanarRegion () {}

      /** \brief Constructor for PlanarRegion.
        * \param[in] centroid the centroid of the region.
        * \param[in] covariance the covariance of the region.
        * \param[in] count the number of points in the region.
        * \param[in] contour the contour / boudnary for the region
        * \param[in] coefficients the model coefficients (a,b,c,d) for the plane
        */
      PlanarRegion (const Eigen::Vector3f& centroid, const Eigen::Matrix3f& covariance, unsigned count,
                    const typename pcl::PointCloud<PointT>::VectorType& contour,
                    const Eigen::Vector4f& coefficients) :
        contour_labels_ ()
      {
        centroid_ = centroid;
        covariance_ = covariance;
        count_ = count;
        contour_ = contour;
        coefficients_ = coefficients;
      }
      
    private:
      /** \brief The labels (good=true, bad=false) for whether or not this boundary was observed, 
        * or was due to edge of frame / occlusion boundary. 
        */
      std::vector<bool> contour_labels_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}

#endif //PCL_SEGMENTATION_PLANAR_REGION_H_
###


# plane_coefficient_comparator.h
namespace pcl
{
  /** \brief PlaneCoefficientComparator is a Comparator that operates on plane coefficients, for use in planar segmentation.
    * In conjunction with OrganizedConnectedComponentSegmentation, this allows planes to be segmented from organized data.
    *
    * \author Alex Trevor
    */
  template<typename PointT, typename PointNT>
  class PlaneCoefficientComparator: public Comparator<PointT>
  {
    public:
      typedef typename Comparator<PointT>::PointCloud PointCloud;
      typedef typename Comparator<PointT>::PointCloudConstPtr PointCloudConstPtr;
      
      typedef typename pcl::PointCloud<PointNT> PointCloudN;
      typedef typename PointCloudN::Ptr PointCloudNPtr;
      typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
      
      typedef boost::shared_ptr<PlaneCoefficientComparator<PointT, PointNT> > Ptr;
      typedef boost::shared_ptr<const PlaneCoefficientComparator<PointT, PointNT> > ConstPtr;

      using pcl::Comparator<PointT>::input_;
      
      /** \brief Empty constructor for PlaneCoefficientComparator. */
      PlaneCoefficientComparator ()
        : normals_ ()
        , plane_coeff_d_ ()
        , angular_threshold_ (pcl::deg2rad (2.0f))
        , distance_threshold_ (0.02f)
        , depth_dependent_ (true)
        , z_axis_ (Eigen::Vector3f (0.0, 0.0, 1.0) )
      {
      }

      /** \brief Constructor for PlaneCoefficientComparator.
        * \param[in] plane_coeff_d a reference to a vector of d coefficients of plane equations.  Must be the same size as the input cloud and input normals.  a, b, and c coefficients are in the input normals.
        */
      PlaneCoefficientComparator (boost::shared_ptr<std::vector<float> >& plane_coeff_d) 
        : normals_ ()
        , plane_coeff_d_ (plane_coeff_d)
        , angular_threshold_ (pcl::deg2rad (2.0f))
        , distance_threshold_ (0.02f)
        , depth_dependent_ (true)
        , z_axis_ (Eigen::Vector3f (0.0f, 0.0f, 1.0f) )
      {
      }
      
      /** \brief Destructor for PlaneCoefficientComparator. */
      virtual
      ~PlaneCoefficientComparator ()
      {
      }

      virtual void 
      setInputCloud (const PointCloudConstPtr& cloud)
      {
        input_ = cloud;
      }
      
      /** \brief Provide a pointer to the input normals.
        * \param[in] normals the input normal cloud
        */
      inline void
      setInputNormals (const PointCloudNConstPtr &normals)
      {
        normals_ = normals;
      }

      /** \brief Get the input normals. */
      inline PointCloudNConstPtr
      getInputNormals () const
      {
        return (normals_);
      }

      /** \brief Provide a pointer to a vector of the d-coefficient of the planes' hessian normal form.  a, b, and c are provided by the normal cloud.
        * \param[in] plane_coeff_d a pointer to the plane coefficients.
        */
      void
      setPlaneCoeffD (boost::shared_ptr<std::vector<float> >& plane_coeff_d)
      {
        plane_coeff_d_ = plane_coeff_d;
      }

      /** \brief Provide a pointer to a vector of the d-coefficient of the planes' hessian normal form.  a, b, and c are provided by the normal cloud.
        * \param[in] plane_coeff_d a pointer to the plane coefficients.
        */
      void
      setPlaneCoeffD (std::vector<float>& plane_coeff_d)
      {
        plane_coeff_d_ = boost::make_shared<std::vector<float> >(plane_coeff_d);
      }
      
      /** \brief Get a pointer to the vector of the d-coefficient of the planes' hessian normal form. */
      const std::vector<float>&
      getPlaneCoeffD () const
      {
        return (plane_coeff_d_);
      }

      /** \brief Set the tolerance in radians for difference in normal direction between neighboring points, to be considered part of the same plane.
        * \param[in] angular_threshold the tolerance in radians
        */
      virtual void
      setAngularThreshold (float angular_threshold)
      {
        angular_threshold_ = cosf (angular_threshold);
      }
      
      /** \brief Get the angular threshold in radians for difference in normal direction between neighboring points, to be considered part of the same plane. */
      inline float
      getAngularThreshold () const
      {
        return (acosf (angular_threshold_) );
      }

      /** \brief Set the tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane.
        * \param[in] distance_threshold the tolerance in meters (at 1m)
        * \param[in] depth_dependent whether to scale the threshold based on range from the sensor (default: false)
        */
      void
      setDistanceThreshold (float distance_threshold, 
                            bool depth_dependent = false)
      {
        distance_threshold_ = distance_threshold;
        depth_dependent_ = depth_dependent;
      }

      /** \brief Get the distance threshold in meters (d component of plane equation) between neighboring points, to be considered part of the same plane. */
      inline float
      getDistanceThreshold () const
      {
        return (distance_threshold_);
      }
      
      /** \brief Compare points at two indices by their plane equations.  True if the angle between the normals is less than the angular threshold,
        * and the difference between the d component of the normals is less than distance threshold, else false
        * \param idx1 The first index for the comparison
        * \param idx2 The second index for the comparison
        */
      virtual bool
      compare (int idx1, int idx2) const
      {
        float threshold = distance_threshold_;
        if (depth_dependent_)
        {
          Eigen::Vector3f vec = input_->points[idx1].getVector3fMap ();
          
          float z = vec.dot (z_axis_);
          threshold *= z * z;
        }
        return ( (fabs ((*plane_coeff_d_)[idx1] - (*plane_coeff_d_)[idx2]) < threshold)
                 && (normals_->points[idx1].getNormalVector3fMap ().dot (normals_->points[idx2].getNormalVector3fMap () ) > angular_threshold_ ) );
      }
      
    protected:
      PointCloudNConstPtr normals_;
      boost::shared_ptr<std::vector<float> > plane_coeff_d_;
      float angular_threshold_;
      float distance_threshold_;
      bool depth_dependent_;
      Eigen::Vector3f z_axis_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}

#endif // PCL_SEGMENTATION_PLANE_COEFFICIENT_COMPARATOR_H_
###


# plane_refinement_comparator.h
namespace pcl
{
  /** \brief PlaneRefinementComparator is a Comparator that operates on plane coefficients, 
    * for use in planar segmentation.
    * In conjunction with OrganizedConnectedComponentSegmentation, this allows planes to be segmented from organized data.
    *
    * \author Alex Trevor, Suat Gedikli
    */
  template<typename PointT, typename PointNT, typename PointLT>
  class PlaneRefinementComparator: public PlaneCoefficientComparator<PointT, PointNT>
  {
    public:
      typedef typename Comparator<PointT>::PointCloud PointCloud;
      typedef typename Comparator<PointT>::PointCloudConstPtr PointCloudConstPtr;
      
      typedef typename pcl::PointCloud<PointNT> PointCloudN;
      typedef typename PointCloudN::Ptr PointCloudNPtr;
      typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;

      typedef typename pcl::PointCloud<PointLT> PointCloudL;
      typedef typename PointCloudL::Ptr PointCloudLPtr;
      typedef typename PointCloudL::ConstPtr PointCloudLConstPtr;

      typedef boost::shared_ptr<PlaneRefinementComparator<PointT, PointNT, PointLT> > Ptr;
      typedef boost::shared_ptr<const PlaneRefinementComparator<PointT, PointNT, PointLT> > ConstPtr;

      using pcl::PlaneCoefficientComparator<PointT, PointNT>::input_;
      using pcl::PlaneCoefficientComparator<PointT, PointNT>::normals_;
      using pcl::PlaneCoefficientComparator<PointT, PointNT>::distance_threshold_;
      using pcl::PlaneCoefficientComparator<PointT, PointNT>::plane_coeff_d_;


      /** \brief Empty constructor for PlaneCoefficientComparator. */
     PlaneRefinementComparator ()
        : models_ ()
        , labels_ ()
        , refine_labels_ ()
        , label_to_model_ ()
        , depth_dependent_ (false)
      {
      }

      /** \brief Empty constructor for PlaneCoefficientComparator. 
        * \param[in] models
        * \param[in] refine_labels
        */
      PlaneRefinementComparator (boost::shared_ptr<std::vector<pcl::ModelCoefficients> >& models,
                                 boost::shared_ptr<std::vector<bool> >& refine_labels)
        : models_ (models)
        , labels_ ()
        , refine_labels_ (refine_labels)
        , label_to_model_ ()
        , depth_dependent_ (false)
      {
      }

      /** \brief Destructor for PlaneCoefficientComparator. */
      virtual
      ~PlaneRefinementComparator ()
      {
      }

      /** \brief Set the vector of model coefficients to which we will compare.
        * \param[in] models a vector of model coefficients produced by the initial segmentation step.
        */
      void
      setModelCoefficients (boost::shared_ptr<std::vector<pcl::ModelCoefficients> >& models)
      {
        models_ = models;
      }

      /** \brief Set the vector of model coefficients to which we will compare.
        * \param[in] models a vector of model coefficients produced by the initial segmentation step.
        */
      void
      setModelCoefficients (std::vector<pcl::ModelCoefficients>& models)
      {
        models_ = boost::make_shared<std::vector<pcl::ModelCoefficients> >(models);
      }

      /** \brief Set which labels should be refined.  This is a vector of bools 0-max_label, true if the label should be refined.
        * \param[in] refine_labels A vector of bools 0-max_label, true if the label should be refined.
        */
      void
      setRefineLabels (boost::shared_ptr<std::vector<bool> >& refine_labels)
      {
        refine_labels_ = refine_labels;
      }
      
      /** \brief Set which labels should be refined.  This is a vector of bools 0-max_label, true if the label should be refined.
        * \param[in] refine_labels A vector of bools 0-max_label, true if the label should be refined.
        */
      void
      setRefineLabels (std::vector<bool>& refine_labels)
      {
        refine_labels_ = boost::make_shared<std::vector<bool> >(refine_labels);
      }

      /** \brief A mapping from label to index in the vector of models, allowing the model coefficients of a label to be accessed.
        * \param[in] label_to_model A vector of size max_label, with the index of each corresponding model in models
        */
      inline void
      setLabelToModel (boost::shared_ptr<std::vector<int> >& label_to_model)
      {
        label_to_model_ = label_to_model;
      }
      
      /** \brief A mapping from label to index in the vector of models, allowing the model coefficients of a label to be accessed.
        * \param[in] label_to_model A vector of size max_label, with the index of each corresponding model in models
        */
      inline void
      setLabelToModel (std::vector<int>& label_to_model)
      {
        label_to_model_ = boost::make_shared<std::vector<int> >(label_to_model);
      }

      /** \brief Get the vector of model coefficients to which we will compare. */
      inline boost::shared_ptr<std::vector<pcl::ModelCoefficients> >
      getModelCoefficients () const
      {
        return (models_);
      }

      /** \brief ...
        * \param[in] labels
        */
      inline void
      setLabels (PointCloudLPtr& labels)
      {
        labels_ = labels;
      }

      /** \brief Compare two neighboring points
        * \param[in] idx1 The index of the first point.
        * \param[in] idx2 The index of the second point.
        */
      virtual bool
      compare (int idx1, int idx2) const
      {
        int current_label = labels_->points[idx1].label;
        int next_label = labels_->points[idx2].label;

        if (!((*refine_labels_)[current_label] && !(*refine_labels_)[next_label]))
          return (false);
        
        const pcl::ModelCoefficients& model_coeff = (*models_)[(*label_to_model_)[current_label]];
        
        PointT pt = input_->points[idx2];
        double ptp_dist = fabs (model_coeff.values[0] * pt.x + 
                                model_coeff.values[1] * pt.y + 
                                model_coeff.values[2] * pt.z +
                                model_coeff.values[3]);
        
        // depth dependent
        float threshold = distance_threshold_;
        if (depth_dependent_)
        {
          //Eigen::Vector4f origin = input_->sensor_origin_;
          Eigen::Vector3f vec = input_->points[idx1].getVector3fMap ();// - origin.head<3> ();
          
          float z = vec.dot (z_axis_);
          threshold *= z * z;
        }
        
        return (ptp_dist < threshold);
      }

    protected:
      boost::shared_ptr<std::vector<pcl::ModelCoefficients> > models_;
      PointCloudLPtr labels_;
      boost::shared_ptr<std::vector<bool> > refine_labels_;
      boost::shared_ptr<std::vector<int> > label_to_model_;
      bool depth_dependent_;
      using PlaneCoefficientComparator<PointT, PointNT>::z_axis_;
  };
}

#endif // PCL_SEGMENTATION_PLANE_COEFFICIENT_COMPARATOR_H_
###


# progressive_morphological_filter.h
namespace pcl
{
  /** \brief
    * Implements the Progressive Morphological Filter for segmentation of ground points.
    * Description can be found in the article
    * "A Progressive Morphological Filter for Removing Nonground Measurements from
    * Airborne LIDAR Data"
    * by K. Zhang, S. Chen, D. Whitman, M. Shyu, J. Yan, and C. Zhang.
    */
  template <typename PointT>
  class PCL_EXPORTS ProgressiveMorphologicalFilter : public pcl::PCLBase<PointT>
  {
    public:

      typedef pcl::PointCloud <PointT> PointCloud;

      using PCLBase <PointT>::input_;
      using PCLBase <PointT>::indices_;
      using PCLBase <PointT>::initCompute;
      using PCLBase <PointT>::deinitCompute;

    public:

      /** \brief Constructor that sets default values for member variables. */
      ProgressiveMorphologicalFilter ();

      virtual
      ~ProgressiveMorphologicalFilter ();

      /** \brief Get the maximum window size to be used in filtering ground returns. */
      inline int
      getMaxWindowSize () const { return (max_window_size_); }

      /** \brief Set the maximum window size to be used in filtering ground returns. */
      inline void
      setMaxWindowSize (int max_window_size) { max_window_size_ = max_window_size; }

      /** \brief Get the slope value to be used in computing the height threshold. */
      inline float
      getSlope () const { return (slope_); }

      /** \brief Set the slope value to be used in computing the height threshold. */
      inline void
      setSlope (float slope) { slope_ = slope; }

      /** \brief Get the maximum height above the parameterized ground surface to be considered a ground return. */
      inline float
      getMaxDistance () const { return (max_distance_); }
      
      /** \brief Set the maximum height above the parameterized ground surface to be considered a ground return. */
      inline void
      setMaxDistance (float max_distance) { max_distance_ = max_distance; }

      /** \brief Get the initial height above the parameterized ground surface to be considered a ground return. */
      inline float
      getInitialDistance () const { return (initial_distance_); }

      /** \brief Set the initial height above the parameterized ground surface to be considered a ground return. */
      inline void
      setInitialDistance (float initial_distance) { initial_distance_ = initial_distance; }

      /** \brief Get the cell size. */
      inline float
      getCellSize () const { return (cell_size_); }
      
      /** \brief Set the cell size. */
      inline void
      setCellSize (float cell_size) { cell_size_ = cell_size; }

      /** \brief Get the base to be used in computing progressive window sizes. */
      inline float
      getBase () const { return (base_); }

      /** \brief Set the base to be used in computing progressive window sizes. */
      inline void
      setBase (float base) { base_ = base; }

      /** \brief Get flag indicating whether or not to exponentially grow window sizes? */
      inline bool
      getExponential () const { return (exponential_); }

      /** \brief Set flag indicating whether or not to exponentially grow window sizes? */
      inline void
      setExponential (bool exponential) { exponential_ = exponential; }

      /** \brief This method launches the segmentation algorithm and returns indices of
        * points determined to be ground returns.
        * \param[out] ground indices of points determined to be ground returns.
        */
      virtual void
      extract (std::vector<int>& ground);

    protected:

      /** \brief Maximum window size to be used in filtering ground returns. */
      int max_window_size_;

      /** \brief Slope value to be used in computing the height threshold. */
      float slope_;

      /** \brief Maximum height above the parameterized ground surface to be considered a ground return. */
      float max_distance_;

      /** \brief Initial height above the parameterized ground surface to be considered a ground return. */
      float initial_distance_;

      /** \brief Cell size. */
      float cell_size_;

      /** \brief Base to be used in computing progressive window sizes. */
      float base_;

      /** \brief Exponentially grow window sizes? */
      bool exponential_;
  };
}

###

# region_3d.h
namespace pcl
{
  /** \brief Region3D represents summary statistics of a 3D collection of points.
    * \author Alex Trevor
    */
  template <typename PointT>
  class Region3D
  {
    public:
      /** \brief Empty constructor for Region3D. */
      Region3D () : centroid_ (Eigen::Vector3f::Zero ()), covariance_ (Eigen::Matrix3f::Identity ()), count_ (0)
      {
      }
      
      /** \brief Constructor for Region3D. 
        * \param[in] centroid The centroid of the region.
        * \param[in] covariance The covariance of the region.
        * \param[in] count The number of points in the region.
        */
      Region3D (Eigen::Vector3f& centroid, Eigen::Matrix3f& covariance, unsigned count) 
        : centroid_ (centroid), covariance_ (covariance), count_ (count)
      {
      }
     
      /** \brief Destructor. */
      virtual ~Region3D () {}

      /** \brief Get the centroid of the region. */
      inline Eigen::Vector3f 
      getCentroid () const
      {
        return (centroid_);
      }
      
      /** \brief Get the covariance of the region. */
      inline Eigen::Matrix3f
      getCovariance () const
      {
        return (covariance_);
      }
      
      /** \brief Get the number of points in the region. */
      unsigned
      getCount () const
      {
        return (count_);
      }

      /** \brief Get the curvature of the region. */
      float
      getCurvature () const
      {
        return (curvature_);
      }

      /** \brief Set the curvature of the region. */
      void
      setCurvature (float curvature)
      {
        curvature_ = curvature;
      }

    protected:
      /** \brief The centroid of the region. */
      Eigen::Vector3f centroid_;
      
      /** \brief The covariance of the region. */
      Eigen::Matrix3f covariance_;
      
      /** \brief The number of points in the region. */
      unsigned count_;

      /** \brief The mean curvature of the region. */
      float curvature_;
      
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}

#endif //#ifndef  PCL_SEGMENTATION_REGION_3D_H_
###


# region_growing.h
namespace pcl
{
  /** \brief
    * Implements the well known Region Growing algorithm used for segmentation.
    * Description can be found in the article
    * "Segmentation of point clouds using smoothness constraint"
    * by T. Rabbania, F. A. van den Heuvelb, G. Vosselmanc.
    * In addition to residual test, the possibility to test curvature is added.
    */
  template <typename PointT, typename NormalT>
  class PCL_EXPORTS RegionGrowing : public pcl::PCLBase<PointT>
  {
    public:

      typedef pcl::search::Search <PointT> KdTree;
      typedef typename KdTree::Ptr KdTreePtr;
      typedef pcl::PointCloud <NormalT> Normal;
      typedef typename Normal::Ptr NormalPtr;
      typedef pcl::PointCloud <PointT> PointCloud;

      using PCLBase <PointT>::input_;
      using PCLBase <PointT>::indices_;
      using PCLBase <PointT>::initCompute;
      using PCLBase <PointT>::deinitCompute;

    public:

      /** \brief Constructor that sets default values for member variables. */
      RegionGrowing ();

      /** \brief This destructor destroys the cloud, normals and search method used for
        * finding KNN. In other words it frees memory.
        */
      virtual
      ~RegionGrowing ();

      /** \brief Get the minimum number of points that a cluster needs to contain in order to be considered valid. */
      int
      getMinClusterSize ();

      /** \brief Set the minimum number of points that a cluster needs to contain in order to be considered valid. */
      void
      setMinClusterSize (int min_cluster_size);

      /** \brief Get the maximum number of points that a cluster needs to contain in order to be considered valid. */
      int
      getMaxClusterSize ();

      /** \brief Set the maximum number of points that a cluster needs to contain in order to be considered valid. */
      void
      setMaxClusterSize (int max_cluster_size);

      /** \brief Returns the flag value. This flag signalizes which mode of algorithm will be used.
        * If it is set to true than it will work as said in the article. This means that
        * it will be testing the angle between normal of the current point and it's neighbours normal.
        * Otherwise, it will be testing the angle between normal of the current point
        * and normal of the initial point that was chosen for growing new segment.
        */
      bool
      getSmoothModeFlag () const;

      /** \brief This function allows to turn on/off the smoothness constraint.
        * \param[in] value new mode value, if set to true then the smooth version will be used.
        */
      void
      setSmoothModeFlag (bool value);

      /** \brief Returns the flag that signalize if the curvature test is turned on/off. */
      bool
      getCurvatureTestFlag () const;

      /** \brief Allows to turn on/off the curvature test. Note that at least one test
        * (residual or curvature) must be turned on. If you are turning curvature test off
        * then residual test will be turned on automatically.
        * \param[in] value new value for curvature test. If set to true then the test will be turned on
        */
      virtual void
      setCurvatureTestFlag (bool value);

      /** \brief Returns the flag that signalize if the residual test is turned on/off. */
      bool
      getResidualTestFlag () const;

      /** \brief
        * Allows to turn on/off the residual test. Note that at least one test
        * (residual or curvature) must be turned on. If you are turning residual test off
        * then curvature test will be turned on automatically.
        * \param[in] value new value for residual test. If set to true then the test will be turned on
        */
      virtual void
      setResidualTestFlag (bool value);

      /** \brief Returns smoothness threshold. */
      float
      getSmoothnessThreshold () const;

      /** \brief Allows to set smoothness threshold used for testing the points.
        * \param[in] theta new threshold value for the angle between normals
        */
      void
      setSmoothnessThreshold (float theta);

      /** \brief Returns residual threshold. */
      float
      getResidualThreshold () const;

      /** \brief Allows to set residual threshold used for testing the points.
        * \param[in] residual new threshold value for residual testing
        */
      void
      setResidualThreshold (float residual);

      /** \brief Returns curvature threshold. */
      float
      getCurvatureThreshold () const;

      /** \brief Allows to set curvature threshold used for testing the points.
        * \param[in] curvature new threshold value for curvature testing
        */
      void
      setCurvatureThreshold (float curvature);

      /** \brief Returns the number of nearest neighbours used for KNN. */
      unsigned int
      getNumberOfNeighbours () const;

      /** \brief Allows to set the number of neighbours. For more information check the article.
        * \param[in] neighbour_number number of neighbours to use
        */
      void
      setNumberOfNeighbours (unsigned int neighbour_number);

      /** \brief Returns the pointer to the search method that is used for KNN. */
      KdTreePtr
      getSearchMethod () const;

      /** \brief Allows to set search method that will be used for finding KNN.
        * \param[in] tree pointer to a KdTree
        */
      void
      setSearchMethod (const KdTreePtr& tree);

      /** \brief Returns normals. */
      NormalPtr
      getInputNormals () const;

      /** \brief This method sets the normals. They are needed for the algorithm, so if
        * no normals will be set, the algorithm would not be able to segment the points.
        * \param[in] norm normals that will be used in the algorithm
        */
      void
      setInputNormals (const NormalPtr& norm);

      /** \brief This method launches the segmentation algorithm and returns the clusters that were
        * obtained during the segmentation.
        * \param[out] clusters clusters that were obtained. Each cluster is an array of point indices.
        */
      virtual void
      extract (std::vector <pcl::PointIndices>& clusters);

      /** \brief For a given point this function builds a segment to which it belongs and returns this segment.
        * \param[in] index index of the initial point which will be the seed for growing a segment.
        * \param[out] cluster cluster to which the point belongs.
        */
      virtual void
      getSegmentFromPoint (int index, pcl::PointIndices& cluster);

      /** \brief If the cloud was successfully segmented, then function
        * returns colored cloud. Otherwise it returns an empty pointer.
        * Points that belong to the same segment have the same color.
        * But this function doesn't guarantee that different segments will have different
        * color(it all depends on RNG). Points that were not listed in the indices array will have red color.
        */
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr
      getColoredCloud ();

      /** \brief If the cloud was successfully segmented, then function
        * returns colored cloud. Otherwise it returns an empty pointer.
        * Points that belong to the same segment have the same color.
        * But this function doesn't guarantee that different segments will have different
        * color(it all depends on RNG). Points that were not listed in the indices array will have red color.
        */
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
      getColoredCloudRGBA ();

    protected:

      /** \brief This method simply checks if it is possible to execute the segmentation algorithm with
        * the current settings. If it is possible then it returns true.
        */
      virtual bool
      prepareForSegmentation ();

      /** \brief This method finds KNN for each point and saves them to the array
        * because the algorithm needs to find KNN a few times.
        */
      virtual void
      findPointNeighbours ();

      /** \brief This function implements the algorithm described in the article
        * "Segmentation of point clouds using smoothness constraint"
        * by T. Rabbania, F. A. van den Heuvelb, G. Vosselmanc.
        */
      void
      applySmoothRegionGrowingAlgorithm ();

      /** \brief This method grows a segment for the given seed point. And returns the number of its points.
        * \param[in] initial_seed index of the point that will serve as the seed point
        * \param[in] segment_number indicates which number this segment will have
        */
      int
      growRegion (int initial_seed, int segment_number);

      /** \brief This function is checking if the point with index 'nghbr' belongs to the segment.
        * If so, then it returns true. It also checks if this point can serve as the seed.
        * \param[in] initial_seed index of the initial point that was passed to the growRegion() function
        * \param[in] point index of the current seed point
        * \param[in] nghbr index of the point that is neighbour of the current seed
        * \param[out] is_a_seed this value is set to true if the point with index 'nghbr' can serve as the seed
        */
      virtual bool
      validatePoint (int initial_seed, int point, int nghbr, bool& is_a_seed) const;

      /** \brief This function simply assembles the regions from list of point labels.
        * Each cluster is an array of point indices.
        */
      void
      assembleRegions ();

    protected:

      /** \brief Stores the minimum number of points that a cluster needs to contain in order to be considered valid. */
      int min_pts_per_cluster_;

      /** \brief Stores the maximum number of points that a cluster needs to contain in order to be considered valid. */
      int max_pts_per_cluster_;

      /** \brief Flag that signalizes if the smoothness constraint will be used. */
      bool smooth_mode_flag_;

      /** \brief If set to true then curvature test will be done during segmentation. */
      bool curvature_flag_;

      /** \brief If set to true then residual test will be done during segmentation. */
      bool residual_flag_;

      /** \brief Thershold used for testing the smoothness between points. */
      float theta_threshold_;

      /** \brief Thershold used in residual test. */
      float residual_threshold_;

      /** \brief Thershold used in curvature test. */
      float curvature_threshold_;

      /** \brief Number of neighbours to find. */
      unsigned int neighbour_number_;

      /** \brief Serch method that will be used for KNN. */
      KdTreePtr search_;

      /** \brief Contains normals of the points that will be segmented. */
      NormalPtr normals_;

      /** \brief Contains neighbours of each point. */
      std::vector<std::vector<int> > point_neighbours_;

      /** \brief Point labels that tells to which segment each point belongs. */
      std::vector<int> point_labels_;

      /** \brief If set to true then normal/smoothness test will be done during segmentation.
        * It is always set to true for the usual region growing algorithm. It is used for turning on/off the test
        * for smoothness in the child class RegionGrowingRGB.*/
      bool normal_flag_;

      /** \brief Tells how much points each segment contains. Used for reserving memory. */
      std::vector<int> num_pts_in_segment_;

      /** \brief After the segmentation it will contain the segments. */
      std::vector <pcl::PointIndices> clusters_;

      /** \brief Stores the number of segments. */
      int number_of_segments_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  /** \brief This function is used as a comparator for sorting. */
  inline bool
  comparePair (std::pair<float, int> i, std::pair<float, int> j)
  {
    return (i.first < j.first);
  }
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/segmentation/impl/region_growing.hpp>
#endif

#endif
###


# region_growing_rgb.h
namespace pcl
{
  /** \brief
    * Implements the well known Region Growing algorithm used for segmentation.
    * Description can be found in the article
    * "Segmentation of point clouds using smoothness constraint"
    * by T. Rabbania, F. A. van den Heuvelb, G. Vosselmanc.
    * In addition to residual test, the possibility to test curvature is added.
    */
  template <typename PointT, typename NormalT>
  class PCL_EXPORTS RegionGrowing : public pcl::PCLBase<PointT>
  {
    public:

      typedef pcl::search::Search <PointT> KdTree;
      typedef typename KdTree::Ptr KdTreePtr;
      typedef pcl::PointCloud <NormalT> Normal;
      typedef typename Normal::Ptr NormalPtr;
      typedef pcl::PointCloud <PointT> PointCloud;

      using PCLBase <PointT>::input_;
      using PCLBase <PointT>::indices_;
      using PCLBase <PointT>::initCompute;
      using PCLBase <PointT>::deinitCompute;

    public:

      /** \brief Constructor that sets default values for member variables. */
      RegionGrowing ();

      /** \brief This destructor destroys the cloud, normals and search method used for
        * finding KNN. In other words it frees memory.
        */
      virtual
      ~RegionGrowing ();

      /** \brief Get the minimum number of points that a cluster needs to contain in order to be considered valid. */
      int
      getMinClusterSize ();

      /** \brief Set the minimum number of points that a cluster needs to contain in order to be considered valid. */
      void
      setMinClusterSize (int min_cluster_size);

      /** \brief Get the maximum number of points that a cluster needs to contain in order to be considered valid. */
      int
      getMaxClusterSize ();

      /** \brief Set the maximum number of points that a cluster needs to contain in order to be considered valid. */
      void
      setMaxClusterSize (int max_cluster_size);

      /** \brief Returns the flag value. This flag signalizes which mode of algorithm will be used.
        * If it is set to true than it will work as said in the article. This means that
        * it will be testing the angle between normal of the current point and it's neighbours normal.
        * Otherwise, it will be testing the angle between normal of the current point
        * and normal of the initial point that was chosen for growing new segment.
        */
      bool
      getSmoothModeFlag () const;

      /** \brief This function allows to turn on/off the smoothness constraint.
        * \param[in] value new mode value, if set to true then the smooth version will be used.
        */
      void
      setSmoothModeFlag (bool value);

      /** \brief Returns the flag that signalize if the curvature test is turned on/off. */
      bool
      getCurvatureTestFlag () const;

      /** \brief Allows to turn on/off the curvature test. Note that at least one test
        * (residual or curvature) must be turned on. If you are turning curvature test off
        * then residual test will be turned on automatically.
        * \param[in] value new value for curvature test. If set to true then the test will be turned on
        */
      virtual void
      setCurvatureTestFlag (bool value);

      /** \brief Returns the flag that signalize if the residual test is turned on/off. */
      bool
      getResidualTestFlag () const;

      /** \brief
        * Allows to turn on/off the residual test. Note that at least one test
        * (residual or curvature) must be turned on. If you are turning residual test off
        * then curvature test will be turned on automatically.
        * \param[in] value new value for residual test. If set to true then the test will be turned on
        */
      virtual void
      setResidualTestFlag (bool value);

      /** \brief Returns smoothness threshold. */
      float
      getSmoothnessThreshold () const;

      /** \brief Allows to set smoothness threshold used for testing the points.
        * \param[in] theta new threshold value for the angle between normals
        */
      void
      setSmoothnessThreshold (float theta);

      /** \brief Returns residual threshold. */
      float
      getResidualThreshold () const;

      /** \brief Allows to set residual threshold used for testing the points.
        * \param[in] residual new threshold value for residual testing
        */
      void
      setResidualThreshold (float residual);

      /** \brief Returns curvature threshold. */
      float
      getCurvatureThreshold () const;

      /** \brief Allows to set curvature threshold used for testing the points.
        * \param[in] curvature new threshold value for curvature testing
        */
      void
      setCurvatureThreshold (float curvature);

      /** \brief Returns the number of nearest neighbours used for KNN. */
      unsigned int
      getNumberOfNeighbours () const;

      /** \brief Allows to set the number of neighbours. For more information check the article.
        * \param[in] neighbour_number number of neighbours to use
        */
      void
      setNumberOfNeighbours (unsigned int neighbour_number);

      /** \brief Returns the pointer to the search method that is used for KNN. */
      KdTreePtr
      getSearchMethod () const;

      /** \brief Allows to set search method that will be used for finding KNN.
        * \param[in] tree pointer to a KdTree
        */
      void
      setSearchMethod (const KdTreePtr& tree);

      /** \brief Returns normals. */
      NormalPtr
      getInputNormals () const;

      /** \brief This method sets the normals. They are needed for the algorithm, so if
        * no normals will be set, the algorithm would not be able to segment the points.
        * \param[in] norm normals that will be used in the algorithm
        */
      void
      setInputNormals (const NormalPtr& norm);

      /** \brief This method launches the segmentation algorithm and returns the clusters that were
        * obtained during the segmentation.
        * \param[out] clusters clusters that were obtained. Each cluster is an array of point indices.
        */
      virtual void
      extract (std::vector <pcl::PointIndices>& clusters);

      /** \brief For a given point this function builds a segment to which it belongs and returns this segment.
        * \param[in] index index of the initial point which will be the seed for growing a segment.
        * \param[out] cluster cluster to which the point belongs.
        */
      virtual void
      getSegmentFromPoint (int index, pcl::PointIndices& cluster);

      /** \brief If the cloud was successfully segmented, then function
        * returns colored cloud. Otherwise it returns an empty pointer.
        * Points that belong to the same segment have the same color.
        * But this function doesn't guarantee that different segments will have different
        * color(it all depends on RNG). Points that were not listed in the indices array will have red color.
        */
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr
      getColoredCloud ();

      /** \brief If the cloud was successfully segmented, then function
        * returns colored cloud. Otherwise it returns an empty pointer.
        * Points that belong to the same segment have the same color.
        * But this function doesn't guarantee that different segments will have different
        * color(it all depends on RNG). Points that were not listed in the indices array will have red color.
        */
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
      getColoredCloudRGBA ();

    protected:

      /** \brief This method simply checks if it is possible to execute the segmentation algorithm with
        * the current settings. If it is possible then it returns true.
        */
      virtual bool
      prepareForSegmentation ();

      /** \brief This method finds KNN for each point and saves them to the array
        * because the algorithm needs to find KNN a few times.
        */
      virtual void
      findPointNeighbours ();

      /** \brief This function implements the algorithm described in the article
        * "Segmentation of point clouds using smoothness constraint"
        * by T. Rabbania, F. A. van den Heuvelb, G. Vosselmanc.
        */
      void
      applySmoothRegionGrowingAlgorithm ();

      /** \brief This method grows a segment for the given seed point. And returns the number of its points.
        * \param[in] initial_seed index of the point that will serve as the seed point
        * \param[in] segment_number indicates which number this segment will have
        */
      int
      growRegion (int initial_seed, int segment_number);

      /** \brief This function is checking if the point with index 'nghbr' belongs to the segment.
        * If so, then it returns true. It also checks if this point can serve as the seed.
        * \param[in] initial_seed index of the initial point that was passed to the growRegion() function
        * \param[in] point index of the current seed point
        * \param[in] nghbr index of the point that is neighbour of the current seed
        * \param[out] is_a_seed this value is set to true if the point with index 'nghbr' can serve as the seed
        */
      virtual bool
      validatePoint (int initial_seed, int point, int nghbr, bool& is_a_seed) const;

      /** \brief This function simply assembles the regions from list of point labels.
        * Each cluster is an array of point indices.
        */
      void
      assembleRegions ();

    protected:

      /** \brief Stores the minimum number of points that a cluster needs to contain in order to be considered valid. */
      int min_pts_per_cluster_;

      /** \brief Stores the maximum number of points that a cluster needs to contain in order to be considered valid. */
      int max_pts_per_cluster_;

      /** \brief Flag that signalizes if the smoothness constraint will be used. */
      bool smooth_mode_flag_;

      /** \brief If set to true then curvature test will be done during segmentation. */
      bool curvature_flag_;

      /** \brief If set to true then residual test will be done during segmentation. */
      bool residual_flag_;

      /** \brief Thershold used for testing the smoothness between points. */
      float theta_threshold_;

      /** \brief Thershold used in residual test. */
      float residual_threshold_;

      /** \brief Thershold used in curvature test. */
      float curvature_threshold_;

      /** \brief Number of neighbours to find. */
      unsigned int neighbour_number_;

      /** \brief Serch method that will be used for KNN. */
      KdTreePtr search_;

      /** \brief Contains normals of the points that will be segmented. */
      NormalPtr normals_;

      /** \brief Contains neighbours of each point. */
      std::vector<std::vector<int> > point_neighbours_;

      /** \brief Point labels that tells to which segment each point belongs. */
      std::vector<int> point_labels_;

      /** \brief If set to true then normal/smoothness test will be done during segmentation.
        * It is always set to true for the usual region growing algorithm. It is used for turning on/off the test
        * for smoothness in the child class RegionGrowingRGB.*/
      bool normal_flag_;

      /** \brief Tells how much points each segment contains. Used for reserving memory. */
      std::vector<int> num_pts_in_segment_;

      /** \brief After the segmentation it will contain the segments. */
      std::vector <pcl::PointIndices> clusters_;

      /** \brief Stores the number of segments. */
      int number_of_segments_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  /** \brief This function is used as a comparator for sorting. */
  inline bool
  comparePair (std::pair<float, int> i, std::pair<float, int> j)
  {
    return (i.first < j.first);
  }
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/segmentation/impl/region_growing.hpp>
#endif

#endif
###


# rgb_plane_coefficient_comparator.h
namespace pcl
{
  /** \brief RGBPlaneCoefficientComparator is a Comparator that operates on plane coefficients, 
    * for use in planar segmentation.  Also takes into account RGB, so we can segmented different colored co-planar regions.
    * In conjunction with OrganizedConnectedComponentSegmentation, this allows planes to be segmented from organized data.
    *
    * \author Alex Trevor
    */
  template<typename PointT, typename PointNT>
  class RGBPlaneCoefficientComparator: public PlaneCoefficientComparator<PointT, PointNT>
  {
    public:
      typedef typename Comparator<PointT>::PointCloud PointCloud;
      typedef typename Comparator<PointT>::PointCloudConstPtr PointCloudConstPtr;
      
      typedef typename pcl::PointCloud<PointNT> PointCloudN;
      typedef typename PointCloudN::Ptr PointCloudNPtr;
      typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
      
      typedef boost::shared_ptr<RGBPlaneCoefficientComparator<PointT, PointNT> > Ptr;
      typedef boost::shared_ptr<const RGBPlaneCoefficientComparator<PointT, PointNT> > ConstPtr;

      using pcl::Comparator<PointT>::input_;
      using pcl::PlaneCoefficientComparator<PointT, PointNT>::normals_;
      using pcl::PlaneCoefficientComparator<PointT, PointNT>::angular_threshold_;
      using pcl::PlaneCoefficientComparator<PointT, PointNT>::distance_threshold_;

      /** \brief Empty constructor for RGBPlaneCoefficientComparator. */
      RGBPlaneCoefficientComparator ()
        : color_threshold_ (50.0f)
      {
      }

      /** \brief Constructor for RGBPlaneCoefficientComparator.
        * \param[in] plane_coeff_d a reference to a vector of d coefficients of plane equations.  Must be the same size as the input cloud and input normals.  a, b, and c coefficients are in the input normals.
        */
      RGBPlaneCoefficientComparator (boost::shared_ptr<std::vector<float> >& plane_coeff_d) 
        : PlaneCoefficientComparator<PointT, PointNT> (plane_coeff_d), color_threshold_ (50.0f)
      {
      }
      
      /** \brief Destructor for RGBPlaneCoefficientComparator. */
      virtual
      ~RGBPlaneCoefficientComparator ()
      {
      }

      /** \brief Set the tolerance in color space between neighboring points, to be considered part of the same plane.
        * \param[in] color_threshold The distance in color space
        */
      inline void
      setColorThreshold (float color_threshold)
      {
        color_threshold_ = color_threshold * color_threshold;
      }

      /** \brief Get the color threshold between neighboring points, to be considered part of the same plane. */
      inline float
      getColorThreshold () const
      {
        return (color_threshold_);
      }

      /** \brief Compare two neighboring points, by using normal information, euclidean distance, and color information.
        * \param[in] idx1 The index of the first point.
        * \param[in] idx2 The index of the second point.
        */
      bool
      compare (int idx1, int idx2) const
      {
        float dx = input_->points[idx1].x - input_->points[idx2].x;
        float dy = input_->points[idx1].y - input_->points[idx2].y;
        float dz = input_->points[idx1].z - input_->points[idx2].z;
        float dist = sqrtf (dx*dx + dy*dy + dz*dz);
        int dr = input_->points[idx1].r - input_->points[idx2].r;
        int dg = input_->points[idx1].g - input_->points[idx2].g;
        int db = input_->points[idx1].b - input_->points[idx2].b;
        //Note: This is not the best metric for color comparisons, we should probably use HSV space.
        float color_dist = static_cast<float> (dr*dr + dg*dg + db*db);
        return ( (dist < distance_threshold_)
                 && (normals_->points[idx1].getNormalVector3fMap ().dot (normals_->points[idx2].getNormalVector3fMap () ) > angular_threshold_ )
                 && (color_dist < color_threshold_));
      }
      
    protected:
      float color_threshold_;
  };
}

#endif // PCL_SEGMENTATION_PLANE_COEFFICIENT_COMPARATOR_H_
###


# sac_segmentation.h
namespace pcl
{
  /** \brief @b SACSegmentation represents the Nodelet segmentation class for
    * Sample Consensus methods and models, in the sense that it just creates a
    * Nodelet wrapper for generic-purpose SAC-based segmentation.
    * \author Radu Bogdan Rusu
    * \ingroup segmentation
    */
  template <typename PointT>
  class SACSegmentation : public PCLBase<PointT>
  {
    using PCLBase<PointT>::initCompute;
    using PCLBase<PointT>::deinitCompute;

     public:
      using PCLBase<PointT>::input_;
      using PCLBase<PointT>::indices_;

      typedef pcl::PointCloud<PointT> PointCloud;
      typedef typename PointCloud::Ptr PointCloudPtr;
      typedef typename PointCloud::ConstPtr PointCloudConstPtr;
      typedef typename pcl::search::Search<PointT>::Ptr SearchPtr;

      typedef typename SampleConsensus<PointT>::Ptr SampleConsensusPtr;
      typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;

      /** \brief Empty constructor. 
        * \param[in] random if true set the random seed to the current time, else set to 12345 (default: false)
        */
      SACSegmentation (bool random = false) 
        : model_ ()
        , sac_ ()
        , model_type_ (-1)
        , method_type_ (0)
        , threshold_ (0)
        , optimize_coefficients_ (true)
        , radius_min_ (-std::numeric_limits<double>::max ())
        , radius_max_ (std::numeric_limits<double>::max ())
        , samples_radius_ (0.0)
        , samples_radius_search_ ()
        , eps_angle_ (0.0)
        , axis_ (Eigen::Vector3f::Zero ())
        , max_iterations_ (50)
        , probability_ (0.99)
        , random_ (random)
      {
      }

      /** \brief Empty destructor. */
      virtual ~SACSegmentation () { /*srv_.reset ();*/ };

      /** \brief The type of model to use (user given parameter).
        * \param[in] model the model type (check \a model_types.h)
        */
      inline void 
      setModelType (int model) { model_type_ = model; }

      /** \brief Get the type of SAC model used. */
      inline int 
      getModelType () const { return (model_type_); }

      /** \brief Get a pointer to the SAC method used. */
      inline SampleConsensusPtr 
      getMethod () const { return (sac_); }

      /** \brief Get a pointer to the SAC model used. */
      inline SampleConsensusModelPtr 
      getModel () const { return (model_); }

      /** \brief The type of sample consensus method to use (user given parameter).
        * \param[in] method the method type (check \a method_types.h)
        */
      inline void 
      setMethodType (int method) { method_type_ = method; }

      /** \brief Get the type of sample consensus method used. */
      inline int 
      getMethodType () const { return (method_type_); }

      /** \brief Distance to the model threshold (user given parameter).
        * \param[in] threshold the distance threshold to use
        */
      inline void 
      setDistanceThreshold (double threshold) { threshold_ = threshold; }

      /** \brief Get the distance to the model threshold. */
      inline double 
      getDistanceThreshold () const { return (threshold_); }

      /** \brief Set the maximum number of iterations before giving up.
        * \param[in] max_iterations the maximum number of iterations the sample consensus method will run
        */
      inline void 
      setMaxIterations (int max_iterations) { max_iterations_ = max_iterations; }

      /** \brief Get maximum number of iterations before giving up. */
      inline int 
      getMaxIterations () const { return (max_iterations_); }

      /** \brief Set the probability of choosing at least one sample free from outliers.
        * \param[in] probability the model fitting probability
        */
      inline void 
      setProbability (double probability) { probability_ = probability; }

      /** \brief Get the probability of choosing at least one sample free from outliers. */
      inline double 
      getProbability () const { return (probability_); }

      /** \brief Set to true if a coefficient refinement is required.
        * \param[in] optimize true for enabling model coefficient refinement, false otherwise
        */
      inline void 
      setOptimizeCoefficients (bool optimize) { optimize_coefficients_ = optimize; }

      /** \brief Get the coefficient refinement internal flag. */
      inline bool 
      getOptimizeCoefficients () const { return (optimize_coefficients_); }

      /** \brief Set the minimum and maximum allowable radius limits for the model (applicable to models that estimate
        * a radius)
        * \param[in] min_radius the minimum radius model
        * \param[in] max_radius the maximum radius model
        */
      inline void
      setRadiusLimits (const double &min_radius, const double &max_radius)
      {
        radius_min_ = min_radius;
        radius_max_ = max_radius;
      }

      /** \brief Get the minimum and maximum allowable radius limits for the model as set by the user.
        * \param[out] min_radius the resultant minimum radius model
        * \param[out] max_radius the resultant maximum radius model
        */
      inline void
      getRadiusLimits (double &min_radius, double &max_radius)
      {
        min_radius = radius_min_;
        max_radius = radius_max_;
      }

      /** \brief Set the maximum distance allowed when drawing random samples
        * \param[in] radius the maximum distance (L2 norm)
        * \param search
        */
      inline void
      setSamplesMaxDist (const double &radius, SearchPtr search)
      {
        samples_radius_ = radius;
        samples_radius_search_ = search;
      }

      /** \brief Get maximum distance allowed when drawing random samples
        *
        * \param[out] radius the maximum distance (L2 norm)
        */
      inline void
      getSamplesMaxDist (double &radius)
      {
        radius = samples_radius_;
      }

      /** \brief Set the axis along which we need to search for a model perpendicular to.
        * \param[in] ax the axis along which we need to search for a model perpendicular to
        */
      inline void 
      setAxis (const Eigen::Vector3f &ax) { axis_ = ax; }

      /** \brief Get the axis along which we need to search for a model perpendicular to. */
      inline Eigen::Vector3f 
      getAxis () const { return (axis_); }

      /** \brief Set the angle epsilon (delta) threshold.
        * \param[in] ea the maximum allowed difference between the model normal and the given axis in radians.
        */
      inline void 
      setEpsAngle (double ea) { eps_angle_ = ea; }

      /** \brief Get the epsilon (delta) model angle threshold in radians. */
      inline double 
      getEpsAngle () const { return (eps_angle_); }

      /** \brief Base method for segmentation of a model in a PointCloud given by <setInputCloud (), setIndices ()>
        * \param[in] inliers the resultant point indices that support the model found (inliers)
        * \param[out] model_coefficients the resultant model coefficients
        */
      virtual void 
      segment (PointIndices &inliers, ModelCoefficients &model_coefficients);

    protected:
      /** \brief Initialize the Sample Consensus model and set its parameters.
        * \param[in] model_type the type of SAC model that is to be used
        */
      virtual bool 
      initSACModel (const int model_type);

      /** \brief Initialize the Sample Consensus method and set its parameters.
        * \param[in] method_type the type of SAC method to be used
        */
      virtual void 
      initSAC (const int method_type);

      /** \brief The model that needs to be segmented. */
      SampleConsensusModelPtr model_;

      /** \brief The sample consensus segmentation method. */
      SampleConsensusPtr sac_;

      /** \brief The type of model to use (user given parameter). */
      int model_type_;

      /** \brief The type of sample consensus method to use (user given parameter). */
      int method_type_;

      /** \brief Distance to the model threshold (user given parameter). */
      double threshold_;

      /** \brief Set to true if a coefficient refinement is required. */
      bool optimize_coefficients_;

      /** \brief The minimum and maximum radius limits for the model. Applicable to all models that estimate a radius. */
      double radius_min_, radius_max_;

      /** \brief The maximum distance of subsequent samples from the first (radius search) */
      double samples_radius_;

      /** \brief The search object for picking subsequent samples using radius search */
      SearchPtr samples_radius_search_;

      /** \brief The maximum allowed difference between the model normal and the given axis. */
      double eps_angle_;

      /** \brief The axis along which we need to search for a model perpendicular to. */
      Eigen::Vector3f axis_;

      /** \brief Maximum number of iterations before giving up (user given parameter). */
      int max_iterations_;

      /** \brief Desired probability of choosing at least one sample free from outliers (user given parameter). */
      double probability_;

      /** \brief Set to true if we need a random seed. */
      bool random_;

      /** \brief Class get name method. */
      virtual std::string 
      getClassName () const { return ("SACSegmentation"); }
  };

  /** \brief @b SACSegmentationFromNormals represents the PCL nodelet segmentation class for Sample Consensus methods and
    * models that require the use of surface normals for estimation.
    * \ingroup segmentation
    */
  template <typename PointT, typename PointNT>
  class SACSegmentationFromNormals: public SACSegmentation<PointT>
  {
    using SACSegmentation<PointT>::model_;
    using SACSegmentation<PointT>::model_type_;
    using SACSegmentation<PointT>::radius_min_;
    using SACSegmentation<PointT>::radius_max_;
    using SACSegmentation<PointT>::eps_angle_;
    using SACSegmentation<PointT>::axis_;
    using SACSegmentation<PointT>::random_;

    public:
      using PCLBase<PointT>::input_;
      using PCLBase<PointT>::indices_;

      typedef typename SACSegmentation<PointT>::PointCloud PointCloud;
      typedef typename PointCloud::Ptr PointCloudPtr;
      typedef typename PointCloud::ConstPtr PointCloudConstPtr;

      typedef typename pcl::PointCloud<PointNT> PointCloudN;
      typedef typename PointCloudN::Ptr PointCloudNPtr;
      typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;

      typedef typename SampleConsensus<PointT>::Ptr SampleConsensusPtr;
      typedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
      typedef typename SampleConsensusModelFromNormals<PointT, PointNT>::Ptr SampleConsensusModelFromNormalsPtr;

      /** \brief Empty constructor.
        * \param[in] random if true set the random seed to the current time, else set to 12345 (default: false)
        */
      SACSegmentationFromNormals (bool random = false) 
        : SACSegmentation<PointT> (random)
        , normals_ ()
        , distance_weight_ (0.1)
        , distance_from_origin_ (0)
        , min_angle_ ()
        , max_angle_ ()
      {};

      /** \brief Provide a pointer to the input dataset that contains the point normals of 
        * the XYZ dataset.
        * \param[in] normals the const boost shared pointer to a PointCloud message
        */
      inline void 
      setInputNormals (const PointCloudNConstPtr &normals) { normals_ = normals; }

      /** \brief Get a pointer to the normals of the input XYZ point cloud dataset. */
      inline PointCloudNConstPtr 
      getInputNormals () const { return (normals_); }

      /** \brief Set the relative weight (between 0 and 1) to give to the angular 
        * distance (0 to pi/2) between point normals and the plane normal.
        * \param[in] distance_weight the distance/angular weight
        */
      inline void 
      setNormalDistanceWeight (double distance_weight) { distance_weight_ = distance_weight; }

      /** \brief Get the relative weight (between 0 and 1) to give to the angular distance (0 to pi/2) between point
        * normals and the plane normal. */
      inline double 
      getNormalDistanceWeight () const { return (distance_weight_); }

      /** \brief Set the minimum opning angle for a cone model.
        * \param min_angle the opening angle which we need minumum to validate a cone model.
        * \param max_angle the opening angle which we need maximum to validate a cone model.
        */
      inline void
      setMinMaxOpeningAngle (const double &min_angle, const double &max_angle)
      {
        min_angle_ = min_angle;
        max_angle_ = max_angle;
      }
 
      /** \brief Get the opening angle which we need minumum to validate a cone model. */
      inline void
      getMinMaxOpeningAngle (double &min_angle, double &max_angle)
      {
        min_angle = min_angle_;
        max_angle = max_angle_;
      }

      /** \brief Set the distance we expect a plane model to be from the origin
        * \param[in] d distance from the template plane modl to the origin
        */
      inline void
      setDistanceFromOrigin (const double d) { distance_from_origin_ = d; }

      /** \brief Get the distance of a plane model from the origin. */
      inline double
      getDistanceFromOrigin () const { return (distance_from_origin_); }

    protected:
      /** \brief A pointer to the input dataset that contains the point normals of the XYZ dataset. */
      PointCloudNConstPtr normals_;

      /** \brief The relative weight (between 0 and 1) to give to the angular
        * distance (0 to pi/2) between point normals and the plane normal. 
        */
      double distance_weight_;

      /** \brief The distance from the template plane to the origin. */
      double distance_from_origin_;

      /** \brief The minimum and maximum allowed opening angle of valid cone model. */
      double min_angle_;
      double max_angle_;

      /** \brief Initialize the Sample Consensus model and set its parameters.
        * \param[in] model_type the type of SAC model that is to be used
        */
      virtual bool 
      initSACModel (const int model_type);

      /** \brief Class get name method. */
      virtual std::string 
      getClassName () const { return ("SACSegmentationFromNormals"); }
  };
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/segmentation/impl/sac_segmentation.hpp>
#endif

#endif  //#ifndef PCL_SEGMENTATION_SAC_SEGMENTATION_H_
###


# seeded_hue_segmentation.h
namespace pcl
{
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /** \brief Decompose a region of space into clusters based on the Euclidean distance between points
    * \param[in] cloud the point cloud message
    * \param[in] tree the spatial locator (e.g., kd-tree) used for nearest neighbors searching
    * \note the tree has to be created as a spatial locator on \a cloud
    * \param[in] tolerance the spatial cluster tolerance as a measure in L2 Euclidean space
    * \param[in] indices_in the cluster containing the seed point indices (as a vector of PointIndices)
    * \param[out] indices_out 
    * \param[in] delta_hue
    * \todo look how to make this templated!
    * \ingroup segmentation
    */
  void 
  seededHueSegmentation (const PointCloud<PointXYZRGB>                           &cloud, 
                         const boost::shared_ptr<search::Search<PointXYZRGB> >   &tree, 
                         float                                                   tolerance, 
                         PointIndices                                            &indices_in, 
                         PointIndices                                            &indices_out, 
                         float                                                   delta_hue = 0.0);

  /** \brief Decompose a region of space into clusters based on the Euclidean distance between points
    * \param[in] cloud the point cloud message
    * \param[in] tree the spatial locator (e.g., kd-tree) used for nearest neighbors searching
    * \note the tree has to be created as a spatial locator on \a cloud
    * \param[in] tolerance the spatial cluster tolerance as a measure in L2 Euclidean space
    * \param[in] indices_in the cluster containing the seed point indices (as a vector of PointIndices)
    * \param[out] indices_out 
    * \param[in] delta_hue
    * \todo look how to make this templated!
    * \ingroup segmentation
    */
  void 
  seededHueSegmentation (const PointCloud<PointXYZRGB>                           &cloud, 
                         const boost::shared_ptr<search::Search<PointXYZRGBL> >  &tree, 
                         float                                                   tolerance, 
                         PointIndices                                            &indices_in, 
                         PointIndices                                            &indices_out, 
                         float                                                   delta_hue = 0.0);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /** \brief SeededHueSegmentation 
    * \author Koen Buys
    * \ingroup segmentation
    */
  class SeededHueSegmentation: public PCLBase<PointXYZRGB>
  {
    typedef PCLBase<PointXYZRGB> BasePCLBase;

    public:
      typedef pcl::PointCloud<PointXYZRGB> PointCloud;
      typedef PointCloud::Ptr PointCloudPtr;
      typedef PointCloud::ConstPtr PointCloudConstPtr;

      typedef pcl::search::Search<PointXYZRGB> KdTree;
      typedef pcl::search::Search<PointXYZRGB>::Ptr KdTreePtr;

      typedef PointIndices::Ptr PointIndicesPtr;
      typedef PointIndices::ConstPtr PointIndicesConstPtr;

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Empty constructor. */
      SeededHueSegmentation () : tree_ (), cluster_tolerance_ (0), delta_hue_ (0.0)
      {};

      /** \brief Provide a pointer to the search object.
        * \param[in] tree a pointer to the spatial search object.
        */
      inline void 
      setSearchMethod (const KdTreePtr &tree) { tree_ = tree; }

      /** \brief Get a pointer to the search method used. */
      inline KdTreePtr 
      getSearchMethod () const { return (tree_); }

      /** \brief Set the spatial cluster tolerance as a measure in the L2 Euclidean space
        * \param[in] tolerance the spatial cluster tolerance as a measure in the L2 Euclidean space
        */
      inline void 
      setClusterTolerance (double tolerance) { cluster_tolerance_ = tolerance; }

      /** \brief Get the spatial cluster tolerance as a measure in the L2 Euclidean space. */
      inline double 
      getClusterTolerance () const { return (cluster_tolerance_); }

      /** \brief Set the tollerance on the hue
        * \param[in] delta_hue the new delta hue
        */
      inline void 
      setDeltaHue (float delta_hue) { delta_hue_ = delta_hue; }

      /** \brief Get the tolerance on the hue */
      inline float 
      getDeltaHue () const { return (delta_hue_); }

      /** \brief Cluster extraction in a PointCloud given by <setInputCloud (), setIndices ()>
        * \param[in] indices_in
        * \param[out] indices_out
        */
      void 
      segment (PointIndices &indices_in, PointIndices &indices_out);

    protected:
      // Members derived from the base class
      using BasePCLBase::input_;
      using BasePCLBase::indices_;
      using BasePCLBase::initCompute;
      using BasePCLBase::deinitCompute;

      /** \brief A pointer to the spatial search object. */
      KdTreePtr tree_;

      /** \brief The spatial cluster tolerance as a measure in the L2 Euclidean space. */
      double cluster_tolerance_;

      /** \brief The allowed difference on the hue*/
      float delta_hue_;

      /** \brief Class getName method. */
      virtual std::string getClassName () const { return ("seededHueSegmentation"); }
  };
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/segmentation/impl/seeded_hue_segmentation.hpp>
#endif

#endif  //#ifndef PCL_SEEDED_HUE_SEGMENTATION_H_
###


# segment_differences.h
namespace pcl
{
  ////////////////////////////////////////////////////////////////////////////////////////////
  /** \brief Obtain the difference between two aligned point clouds as another point cloud, given a distance threshold.
    * \param src the input point cloud source
    * \param tgt the input point cloud target we need to obtain the difference against
    * \param threshold the distance threshold (tolerance) for point correspondences. (e.g., check if f a point p1 from 
    * src has a correspondence > threshold than a point p2 from tgt)
    * \param tree the spatial locator (e.g., kd-tree) used for nearest neighbors searching built over \a tgt
    * \param output the resultant output point cloud difference
    * \ingroup segmentation
    */
  template <typename PointT> 
  void getPointCloudDifference (
      const pcl::PointCloud<PointT> &src, const pcl::PointCloud<PointT> &tgt, 
      double threshold, const boost::shared_ptr<pcl::search::Search<PointT> > &tree,
      pcl::PointCloud<PointT> &output);

  ////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////
  /** \brief @b SegmentDifferences obtains the difference between two spatially
    * aligned point clouds and returns the difference between them for a maximum
    * given distance threshold.
    * \author Radu Bogdan Rusu
    * \ingroup segmentation
    */
  template <typename PointT>
  class SegmentDifferences: public PCLBase<PointT>
  {
    typedef PCLBase<PointT> BasePCLBase;

    public:
      typedef pcl::PointCloud<PointT> PointCloud;
      typedef typename PointCloud::Ptr PointCloudPtr;
      typedef typename PointCloud::ConstPtr PointCloudConstPtr;

      typedef typename pcl::search::Search<PointT> KdTree;
      typedef typename pcl::search::Search<PointT>::Ptr KdTreePtr;

      typedef PointIndices::Ptr PointIndicesPtr;
      typedef PointIndices::ConstPtr PointIndicesConstPtr;

      /** \brief Empty constructor. */
      SegmentDifferences () : 
        tree_ (), target_ (), distance_threshold_ (0)
      {};

      /** \brief Provide a pointer to the target dataset against which we
        * compare the input cloud given in setInputCloud
        *
        * \param cloud the target PointCloud dataset
        */
      inline void 
      setTargetCloud (const PointCloudConstPtr &cloud) { target_ = cloud; }

      /** \brief Get a pointer to the input target point cloud dataset. */
      inline PointCloudConstPtr const 
      getTargetCloud () { return (target_); }

      /** \brief Provide a pointer to the search object.
        * \param tree a pointer to the spatial search object.
        */
      inline void 
      setSearchMethod (const KdTreePtr &tree) { tree_ = tree; }

      /** \brief Get a pointer to the search method used. */
      inline KdTreePtr 
      getSearchMethod () { return (tree_); }

      /** \brief Set the maximum distance tolerance (squared) between corresponding
        * points in the two input datasets.
        *
        * \param sqr_threshold the squared distance tolerance as a measure in L2 Euclidean space
        */
      inline void 
      setDistanceThreshold (double sqr_threshold) { distance_threshold_ = sqr_threshold; }

      /** \brief Get the squared distance tolerance between corresponding points as a
        * measure in the L2 Euclidean space.
        */
      inline double 
      getDistanceThreshold () { return (distance_threshold_); }

      /** \brief Segment differences between two input point clouds.
        * \param output the resultant difference between the two point clouds as a PointCloud
        */
      void 
      segment (PointCloud &output);

    protected:
      // Members derived from the base class
      using BasePCLBase::input_;
      using BasePCLBase::indices_;
      using BasePCLBase::initCompute;
      using BasePCLBase::deinitCompute;

      /** \brief A pointer to the spatial search object. */
      KdTreePtr tree_;

      /** \brief The input target point cloud dataset. */
      PointCloudConstPtr target_;

      /** \brief The distance tolerance (squared) as a measure in the L2
        * Euclidean space between corresponding points. 
        */
      double distance_threshold_;

      /** \brief Class getName method. */
      virtual std::string 
      getClassName () const { return ("SegmentDifferences"); }
  };
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/segmentation/impl/segment_differences.hpp>
#endif

#endif  //#ifndef PCL_SEGMENT_DIFFERENCES_H_
###


# supervoxel_clustering.h
namespace pcl
{
  /** \brief Supervoxel container class - stores a cluster extracted using supervoxel clustering 
   */
  template <typename PointT>
  class Supervoxel
  {
    public:
      Supervoxel () :
        voxels_ (new pcl::PointCloud<PointT> ()),
        normals_ (new pcl::PointCloud<Normal> ())
        {  } 
      
      typedef boost::shared_ptr<Supervoxel<PointT> > Ptr;
      typedef boost::shared_ptr<const Supervoxel<PointT> > ConstPtr;

      /** \brief Gets the centroid of the supervoxel
       *  \param[out] centroid_arg centroid of the supervoxel
       */ 
      void
      getCentroidPoint (PointXYZRGBA &centroid_arg)
      {
        centroid_arg = centroid_;
      }
      
      /** \brief Gets the point normal for the supervoxel 
       * \param[out] normal_arg Point normal of the supervoxel
       * \note This isn't an average, it is a normal computed using all of the voxels in the supervoxel as support
       */ 
      void
      getCentroidPointNormal (PointNormal &normal_arg)
      {
        normal_arg.x = centroid_.x;
        normal_arg.y = centroid_.y;
        normal_arg.z = centroid_.z;
        normal_arg.normal_x = normal_.normal_x;
        normal_arg.normal_y = normal_.normal_y;
        normal_arg.normal_z = normal_.normal_z;
        normal_arg.curvature = normal_.curvature;
      }
      
      /** \brief The normal calculated for the voxels contained in the supervoxel */
      pcl::Normal normal_;
      /** \brief The centroid of the supervoxel - average voxel */
      pcl::PointXYZRGBA centroid_;
      /** \brief A Pointcloud of the voxels in the supervoxel */
      typename pcl::PointCloud<PointT>::Ptr voxels_;
      /** \brief A Pointcloud of the normals for the points in the supervoxel */
      typename pcl::PointCloud<Normal>::Ptr normals_;
                
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
  };
  
  /** \brief Implements a supervoxel algorithm based on voxel structure, normals, and rgb values
   *   \note Supervoxels are oversegmented volumetric patches (usually surfaces) 
   *   \note Usually, color isn't needed (and can be detrimental)- spatial structure is mainly used
    * - J. Papon, A. Abramov, M. Schoeler, F. Woergoetter
    *   Voxel Cloud Connectivity Segmentation - Supervoxels from PointClouds
    *   In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2013 
    *  \author Jeremie Papon (jpapon@gmail.com)
    */
  template <typename PointT>
  class PCL_EXPORTS SupervoxelClustering : public pcl::PCLBase<PointT>
  {
    //Forward declaration of friended helper class
    class SupervoxelHelper;
    friend class SupervoxelHelper;
    public:
      /** \brief VoxelData is a structure used for storing data within a pcl::octree::OctreePointCloudAdjacencyContainer
       *  \note It stores xyz, rgb, normal, distance, an index, and an owner.
       */
      class VoxelData
      {
        public:
          VoxelData ():
            xyz_ (0.0f, 0.0f, 0.0f),
            rgb_ (0.0f, 0.0f, 0.0f),
            normal_ (0.0f, 0.0f, 0.0f, 0.0f),
            curvature_ (0.0f),
            owner_ (0)
            {}
            
          /** \brief Gets the data of in the form of a point
           *  \param[out] point_arg Will contain the point value of the voxeldata
           */  
          void
          getPoint (PointT &point_arg) const;
          
          /** \brief Gets the data of in the form of a normal
           *  \param[out] normal_arg Will contain the normal value of the voxeldata
           */            
          void
          getNormal (Normal &normal_arg) const;
          
          Eigen::Vector3f xyz_;
          Eigen::Vector3f rgb_;
          Eigen::Vector4f normal_;
          float curvature_;
          float distance_;
          int idx_;
          SupervoxelHelper* owner_;
          
        public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      };
      
      typedef pcl::octree::OctreePointCloudAdjacencyContainer<PointT, VoxelData> LeafContainerT;
      typedef std::vector <LeafContainerT*> LeafVectorT;
      
      typedef typename pcl::PointCloud<PointT> PointCloudT;
      typedef typename pcl::PointCloud<Normal> NormalCloudT;
      typedef typename pcl::octree::OctreePointCloudAdjacency<PointT, LeafContainerT> OctreeAdjacencyT;
      typedef typename pcl::octree::OctreePointCloudSearch <PointT> OctreeSearchT;
      typedef typename pcl::search::KdTree<PointT> KdTreeT;
      typedef boost::shared_ptr<std::vector<int> > IndicesPtr;
           
      using PCLBase <PointT>::initCompute;
      using PCLBase <PointT>::deinitCompute;
      using PCLBase <PointT>::input_;
      
      typedef boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS, uint32_t, float> VoxelAdjacencyList;
      typedef VoxelAdjacencyList::vertex_descriptor VoxelID;
      typedef VoxelAdjacencyList::edge_descriptor EdgeID;
      
      
    public:

      /** \brief Constructor that sets default values for member variables. 
       *  \param[in] voxel_resolution The resolution (in meters) of voxels used
       *  \param[in] seed_resolution The average size (in meters) of resulting supervoxels
       *  \param[in] use_single_camera_transform Set to true if point density in cloud falls off with distance from origin (such as with a cloud coming from one stationary camera), set false if input cloud is from multiple captures from multiple locations.
       */
      SupervoxelClustering (float voxel_resolution, float seed_resolution, bool use_single_camera_transform = true);

      /** \brief This destructor destroys the cloud, normals and search method used for
        * finding neighbors. In other words it frees memory.
        */
      virtual
      ~SupervoxelClustering ();

      /** \brief Set the resolution of the octree voxels */
      void
      setVoxelResolution (float resolution);
      
      /** \brief Get the resolution of the octree voxels */
      float 
      getVoxelResolution () const;
      
      /** \brief Set the resolution of the octree seed voxels */
      void
      setSeedResolution (float seed_resolution);
      
      /** \brief Get the resolution of the octree seed voxels */
      float 
      getSeedResolution () const;
        
      /** \brief Set the importance of color for supervoxels */
      void
      setColorImportance (float val);
      
      /** \brief Set the importance of spatial distance for supervoxels */
      void
      setSpatialImportance (float val);
            
      /** \brief Set the importance of scalar normal product for supervoxels */
      void
      setNormalImportance (float val);
      
      /** \brief This method launches the segmentation algorithm and returns the supervoxels that were
       * obtained during the segmentation.
       * \param[out] supervoxel_clusters A map of labels to pointers to supervoxel structures
       */
      virtual void
      extract (std::map<uint32_t,typename Supervoxel<PointT>::Ptr > &supervoxel_clusters);

      /** \brief This method sets the cloud to be supervoxelized
       * \param[in] cloud The cloud to be supervoxelize
       */
      virtual void
      setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr& cloud);
      
      /** \brief This method sets the normals to be used for supervoxels (should be same size as input cloud)
      * \param[in] normal_cloud The input normals                         
      */
      virtual void
      setNormalCloud (typename NormalCloudT::ConstPtr normal_cloud);
      
      /** \brief This method refines the calculated supervoxels - may only be called after extract
       * \param[in] num_itr The number of iterations of refinement to be done (2 or 3 is usually sufficient)
       * \param[out] supervoxel_clusters The resulting refined supervoxels
       */
      virtual void
      refineSupervoxels (int num_itr, std::map<uint32_t,typename Supervoxel<PointT>::Ptr > &supervoxel_clusters);
      
      ////////////////////////////////////////////////////////////
      /** \brief Returns an RGB colorized cloud showing superpixels
        * Otherwise it returns an empty pointer.
        * Points that belong to the same supervoxel have the same color.
        * But this function doesn't guarantee that different segments will have different
        * color(it's random). Points that are unlabeled will be black
        * \note This will expand the label_colors_ vector so that it can accomodate all labels
        */
      typename pcl::PointCloud<PointXYZRGBA>::Ptr
      getColoredCloud () const;
      
      /** \brief Returns a deep copy of the voxel centroid cloud */
      typename pcl::PointCloud<PointT>::Ptr
      getVoxelCentroidCloud () const;
      
      /** \brief Returns labeled cloud
        * Points that belong to the same supervoxel have the same label.
        * Labels for segments start from 1, unlabled points have label 0
        */
      typename pcl::PointCloud<PointXYZL>::Ptr
      getLabeledCloud () const;
      
      /** \brief Returns an RGB colorized voxelized cloud showing superpixels
       * Otherwise it returns an empty pointer.
       * Points that belong to the same supervoxel have the same color.
       * But this function doesn't guarantee that different segments will have different
       * color(it's random). Points that are unlabeled will be black
       * \note This will expand the label_colors_ vector so that it can accomodate all labels
       */
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
      getColoredVoxelCloud () const;
      
      /** \brief Returns labeled voxelized cloud
       * Points that belong to the same supervoxel have the same label.
       * Labels for segments start from 1, unlabled points have label 0
       */      
      pcl::PointCloud<pcl::PointXYZL>::Ptr
      getLabeledVoxelCloud () const;

      /** \brief Gets the adjacency list (Boost Graph library) which gives connections between supervoxels
       *  \param[out] adjacency_list_arg BGL graph where supervoxel labels are vertices, edges are touching relationships
       */
      void
      getSupervoxelAdjacencyList (VoxelAdjacencyList &adjacency_list_arg) const;
      
      /** \brief Get a multimap which gives supervoxel adjacency
       *  \param[out] label_adjacency Multi-Map which maps a supervoxel label to all adjacent supervoxel labels
       */
      void 
      getSupervoxelAdjacency (std::multimap<uint32_t, uint32_t> &label_adjacency) const;
            
      /** \brief Static helper function which returns a pointcloud of normals for the input supervoxels 
       *  \param[in] supervoxel_clusters Supervoxel cluster map coming from this class
       *  \returns Cloud of PointNormals of the supervoxels
       * 
       */
      static pcl::PointCloud<pcl::PointNormal>::Ptr
      makeSupervoxelNormalCloud (std::map<uint32_t,typename Supervoxel<PointT>::Ptr > &supervoxel_clusters);
      
      /** \brief Returns the current maximum (highest) label */
      int
      getMaxLabel () const;
      
    private:
      
      /** \brief This method initializes the label_colors_ vector (assigns random colors to labels)
       * \note Checks to see if it is already big enough - if so, does not reinitialize it
       */
      void
      initializeLabelColors ();
      
      /** \brief This method simply checks if it is possible to execute the segmentation algorithm with
        * the current settings. If it is possible then it returns true.
        */
      virtual bool
      prepareForSegmentation ();

      /** \brief This selects points to use as initial supervoxel centroids
       *  \param[out] seed_points The selected points
       */
      void
      selectInitialSupervoxelSeeds (std::vector<PointT, Eigen::aligned_allocator<PointT> > &seed_points);
      
      /** \brief This method creates the internal supervoxel helpers based on the provided seed points
       *  \param[in] seed_points The selected points
       */
      void
      createSupervoxelHelpers (std::vector<PointT, Eigen::aligned_allocator<PointT> > &seed_points);
      
      /** \brief This performs the superpixel evolution */
      void
      expandSupervoxels (int depth);

      /** \brief This sets the data of the voxels in the tree */
      void 
      computeVoxelData ();
     
      /** \brief Reseeds the supervoxels by finding the voxel closest to current centroid */
      void
      reseedSupervoxels ();
      
      /** \brief Constructs the map of supervoxel clusters from the internal supervoxel helpers */
      void
      makeSupervoxels (std::map<uint32_t,typename Supervoxel<PointT>::Ptr > &supervoxel_clusters);
      
      /** \brief Stores the resolution used in the octree */
      float resolution_;
    
      /** \brief Stores the resolution used to seed the superpixels */
      float seed_resolution_;
      
      /** \brief Distance function used for comparing voxelDatas */
      float
      voxelDataDistance (const VoxelData &v1, const VoxelData &v2) const;
      
      /** \brief Transform function used to normalize voxel density versus distance from camera */
      void
      transformFunction (PointT &p);
      
      /** \brief Contains a KDtree for the voxelized cloud */
      typename pcl::search::KdTree<PointT>::Ptr voxel_kdtree_;
      
      /** \brief Octree Adjacency structure with leaves at voxel resolution */
      typename OctreeAdjacencyT::Ptr adjacency_octree_;
      
      /** \brief Contains the Voxelized centroid Cloud */
      typename PointCloudT::Ptr voxel_centroid_cloud_;
      
      /** \brief Contains the Voxelized centroid Cloud */
      typename NormalCloudT::ConstPtr input_normals_;
      
      /** \brief Importance of color in clustering */
      float color_importance_;
      /** \brief Importance of distance from seed center in clustering */
      float spatial_importance_;
      /** \brief Importance of similarity in normals for clustering */
      float normal_importance_;
      
      /** \brief Stores the colors used for the superpixel labels*/
      std::vector<uint32_t> label_colors_;
      
      /** \brief Internal storage class for supervoxels 
       * \note Stores pointers to leaves of clustering internal octree, 
       * \note so should not be used outside of clustering class 
       */
      class SupervoxelHelper
      {
        public:
          SupervoxelHelper (uint32_t label, SupervoxelClustering* parent_arg):
            label_ (label),
            parent_ (parent_arg)
          { }
          
          void
          addLeaf (LeafContainerT* leaf_arg);
        
          void
          removeLeaf (LeafContainerT* leaf_arg);
        
          void
          removeAllLeaves ();
          
          void 
          expand ();
          
          void 
          refineNormals ();
          
          void 
          updateCentroid ();
          
          void 
          getVoxels (typename pcl::PointCloud<PointT>::Ptr &voxels) const;
          
          void 
          getNormals (typename pcl::PointCloud<Normal>::Ptr &normals) const;
          
          typedef float (SupervoxelClustering::*DistFuncPtr)(const VoxelData &v1, const VoxelData &v2);
          
          uint32_t
          getLabel () const 
          { return label_; }
          
          Eigen::Vector4f 
          getNormal () const 
          { return centroid_.normal_; }
          
          Eigen::Vector3f 
          getRGB () const 
          { return centroid_.rgb_; }
          
          Eigen::Vector3f
          getXYZ () const 
          { return centroid_.xyz_;}
          
          void
          getXYZ (float &x, float &y, float &z) const
          { x=centroid_.xyz_[0]; y=centroid_.xyz_[1]; z=centroid_.xyz_[2]; }
          
          void
          getRGB (uint32_t &rgba) const
          { 
            rgba = static_cast<uint32_t>(centroid_.rgb_[0]) << 16 | 
                   static_cast<uint32_t>(centroid_.rgb_[1]) << 8 | 
                   static_cast<uint32_t>(centroid_.rgb_[2]); 
          }
          
          void 
          getNormal (pcl::Normal &normal_arg) const 
          { 
            normal_arg.normal_x = centroid_.normal_[0];
            normal_arg.normal_y = centroid_.normal_[1];
            normal_arg.normal_z = centroid_.normal_[2];
            normal_arg.curvature = centroid_.curvature_;
          }
          
          void
          getNeighborLabels (std::set<uint32_t> &neighbor_labels) const;
          
          VoxelData
          getCentroid () const
          { return centroid_; }
            
          
          size_t
          size () const { return leaves_.size (); }
        private:
          //Stores leaves
          std::set<LeafContainerT*> leaves_;
          uint32_t label_;
          VoxelData centroid_;
          SupervoxelClustering* parent_;
        public:
          //Type VoxelData may have fixed-size Eigen objects inside
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      };
      
      //Make boost::ptr_list can access the private class SupervoxelHelper
      friend void boost::checked_delete<> (const typename pcl::SupervoxelClustering<PointT>::SupervoxelHelper *);
      
      typedef boost::ptr_list<SupervoxelHelper> HelperListT;
      HelperListT supervoxel_helpers_;
      
      //TODO DEBUG REMOVE
      StopWatch timer_;
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      
     

  };

}

###


