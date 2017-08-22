# -*- coding: utf-8 -*-

from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp
cimport pcl_features as pclftr

# boost
from boost_shared_ptr cimport shared_ptr

###############################################################################
# Types
###############################################################################

### base class ###

# keypoint.h
# template <typename PointInT, typename PointOutT>
# class Keypoint : public PCLBase<PointInT>
cdef extern from "pcl/keypoints/keypoint.h" namespace "pcl":
    cdef cppclass Keypoint[In, Out](cpp.PCLBase[In]):
        Keypoint ()
        # public:
        # brief Provide a pointer to the input dataset that we need to estimate features at every point for.
        # param cloud the const boost shared pointer to a PointCloud message
        # void setSearchSurface (const PointCloudInConstPtr &cloud)
        # void setSearchSurface (const PointCloud[In] &cloud)
        
        # brief Get a pointer to the surface point cloud dataset.
        # PointCloudInConstPtr getSearchSurface ()
        # PointCloud[In] getSearchSurface ()
        
        # brief Provide a pointer to the search object.
        # param tree a pointer to the spatial search object.
        # void setSearchMethod (const KdTreePtr &tree)
        # void setSearchMethod (-.KdTree &tree)
        
        # brief Get a pointer to the search method used.
        # KdTreePtr getSearchMethod ()
        # -.KdTree getSearchMethod ()
        
        # brief Get the internal search parameter.
        double getSearchParameter ()
        
        # brief Set the number of k nearest neighbors to use for the feature estimation.
        # param k the number of k-nearest neighbors
        void setKSearch (int k)
        
        # brief get the number of k nearest neighbors used for the feature estimation. */
        int getKSearch ()
        
        # brief Set the sphere radius that is to be used for determining the nearest neighbors used for the key point detection
        # param radius the sphere radius used as the maximum distance to consider a point a neighbor
        void setRadiusSearch (double radius)
        
        # brief Get the sphere radius used for determining the neighbors. */
        double getRadiusSearch ()
        
        # brief Base method for key point detection for all points given in <setInputCloud (), setIndices ()> using
        # the surface in setSearchSurface () and the spatial locator in setSearchMethod ()
        # param output the resultant point cloud model dataset containing the estimated features
        # inline void compute (PointCloudOut &output);
        void compute (cpp.PointCloud[Out] &output)
        
        # brief Search for k-nearest neighbors using the spatial locator from \a setSearchmethod, and the given surface
        # from \a setSearchSurface.
        # param index the index of the query point
        # param parameter the search parameter (either k or radius)
        # param indices the resultant vector of indices representing the k-nearest neighbors
        # param distances the resultant vector of distances representing the distances from the query point to the
        # k-nearest neighbors
        # inline int searchForNeighbors (int index, double parameter, vector[int] &indices, vector[float] &distances)
        int searchForNeighbors (int index, double parameter, vector[int] &indices, vector[float] &distances)


###

# harris_keypoint3D.h (1.6.0)
# harris_3d.h (1.7.2)
# template <typename PointInT, typename PointOutT, typename NormalT = pcl::Normal>
# class HarrisKeypoint3D : public Keypoint<PointInT, PointOutT>
cdef extern from "pcl/keypoints/harris_keypoint3D.h" namespace "pcl":
    cdef cppclass HarrisKeypoint3D[In, Out, NormalT](Keypoint[In, Out]):
        HarrisKeypoint3D ()
        # HarrisKeypoint3D (ResponseMethod method = HARRIS, float radius = 0.01f, float threshold = 0.0f)
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Keypoint<PointInT, PointOutT>::KdTree KdTree;
        # typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # typedef typename pcl::PointCloud<NormalT> PointCloudN;
        # typedef typename PointCloudN::Ptr PointCloudNPtr;
        # typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
        
        # typedef enum {HARRIS = 1, NOBLE, LOWE, TOMASI, CURVATURE} ResponseMethod;
        
        # brief Set the method of the response to be calculated.
        # param[in] type
        # void setMethod (ResponseMethod type)
        # void setMethod (ResponseMethod2 type)
        void setMethod (int type)
        
        # * \brief Set the radius for normal estimation and non maxima supression.
        # * \param[in] radius
        # void setRadius (float radius)
        void setRadius (float radius)
        
        # * \brief Set the threshold value for detecting corners. This is only evaluated if non maxima suppression is turned on.
        # * \brief note non maxima suppression needs to be activated in order to use this feature.
        # * \param[in] threshold
        void setThreshold (float threshold)
        
        # * \brief Whether non maxima suppression should be applied or the response for each point should be returned
        # * \note this value needs to be turned on in order to apply thresholding and refinement
        # * \param[in] nonmax default is false
        # void setNonMaxSupression (bool = false)
        void setNonMaxSupression (bool param)
        
        # * \brief Whether the detected key points should be refined or not. If turned of, the key points are a subset of the original point cloud. Otherwise the key points may be arbitrary.
        # * \brief note non maxima supression needs to be on in order to use this feature.
        # * \param[in] do_refine
        void setRefine (bool do_refine)
        
        # * \brief Set normals if precalculated normals are available.
        # * \param normals
        # void setNormals (const PointCloudNPtr &normals)
        # void setNormals (const cpp.PointCloud[NormalT] &normals)
        
        # * \brief Provide a pointer to a dataset to add additional information
        # * to estimate the features for every point in the input dataset.  This
        # * is optional, if this is not set, it will only use the data in the
        # * input cloud to estimate the features.  This is useful when you only
        # * need to compute the features for a downsampled cloud.
        # * \param[in] cloud a pointer to a PointCloud message
        # virtual void setSearchSurface (const PointCloudInConstPtr &cloud)
        # void setSearchSurface (const PointCloudInConstPtr &cloud)
        
        # * \brief Initialize the scheduler and set the number of threads to use.
        # * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
        # inline void setNumberOfThreads (int nr_threads)
        void setNumberOfThreads (int nr_threads)


ctypedef HarrisKeypoint3D[cpp.PointXYZ, cpp.PointXYZI, cpp.Normal] HarrisKeypoint3D_t
ctypedef HarrisKeypoint3D[cpp.PointXYZI, cpp.PointXYZI, cpp.Normal] HarrisKeypoint3D_PointXYZI_t
ctypedef HarrisKeypoint3D[cpp.PointXYZRGB, cpp.PointXYZI, cpp.Normal] HarrisKeypoint3D_PointXYZRGB_t
ctypedef HarrisKeypoint3D[cpp.PointXYZRGBA, cpp.PointXYZI, cpp.Normal] HarrisKeypoint3D_PointXYZRGBA_t
ctypedef shared_ptr[HarrisKeypoint3D[cpp.PointXYZ, cpp.PointXYZI, cpp.Normal]] HarrisKeypoint3DPtr_t
ctypedef shared_ptr[HarrisKeypoint3D[cpp.PointXYZI, cpp.PointXYZI, cpp.Normal]] HarrisKeypoint3D_PointXYZI_Ptr_t
ctypedef shared_ptr[HarrisKeypoint3D[cpp.PointXYZRGB, cpp.PointXYZI, cpp.Normal]] HarrisKeypoint3D_PointXYZRGB_Ptr_t
ctypedef shared_ptr[HarrisKeypoint3D[cpp.PointXYZRGBA, cpp.PointXYZI, cpp.Normal]] HarrisKeypoint3D_PointXYZRGBA_Ptr_t
###

# narf_keypoint.h
# class PCL_EXPORTS NarfKeypoint : public Keypoint<PointWithRange, int>
cdef extern from "pcl/keypoints/narf_keypoint.h" namespace "pcl":
    cdef cppclass NarfKeypoint(Keypoint[cpp.PointWithRange, int]):
        NarfKeypoint ()
        NarfKeypoint (pclftr.RangeImageBorderExtractor range_image_border_extractor, float support_size)
        # NarfKeypoint (RangeImageBorderExtractor* range_image_border_extractor=NULL, float support_size=-1.0f);
        # public:
        # // =====TYPEDEFS=====
        # typedef Keypoint<PointWithRange, int> BaseClass;
        # typedef Keypoint<PointWithRange, int>::PointCloudOut PointCloudOut;
        # // =====PUBLIC STRUCTS=====
        # //! Parameters used in this class
        # cdef struct Parameters
        # {
        #     Parameters() : support_size(-1.0f), max_no_of_interest_points(-1), min_distance_between_interest_points(0.25f),
        #            optimal_distance_to_high_surface_change(0.25), min_interest_value(0.45f),
        #            min_surface_change_score(0.2f), optimal_range_image_patch_size(10),
        #            distance_for_additional_points(0.0f), add_points_on_straight_edges(false),
        #            do_non_maximum_suppression(true), no_of_polynomial_approximations_per_point(0),
        #            max_no_of_threads(1), use_recursive_scale_reduction(false),
        #            calculate_sparse_interest_image(true) {}
        # 
        #     float support_size;  //!< This defines the area 'covered' by an interest point (in meters)
        #     int max_no_of_interest_points;  //!< The maximum number of interest points that will be returned
        #     float min_distance_between_interest_points;  /**< Minimum distance between maximas
        #                                            *  (this is a factor for support_size, i.e. the distance is
        #                                            *  min_distance_between_interest_points*support_size) */
        #     float optimal_distance_to_high_surface_change;  /**< The distance we want keep between keypoints and areas
        #                                               *  of high surface change
        #                                               *  (this is a factor for support_size, i.e., the distance is
        #                                               *  optimal_distance_to_high_surface_change*support_size) */
        #     float min_interest_value;  //!< The minimum value to consider a point as an interest point
        #     float min_surface_change_score;  //!< The minimum value  of the surface change score to consider a point
        #     int optimal_range_image_patch_size;  /**< The size (in pixels) of the image patches from which the interest value
        #                                    *  should be computed. This influences, which range image is selected from
        #                                    *  the scale space to compute the interest value of a pixel at a certain
        #                                    *  distance. */
        #     // TODO:
        #     float distance_for_additional_points;  /**< All points in this distance to a found maximum, that
        #                                      *  are above min_interest_value are also added as interest points
        #                                      *  (this is a factor for support_size, i.e. the distance is
        #                                      *  distance_for_additional_points*support_size) */
        #     bool add_points_on_straight_edges;  /**< If this is set to true, there will also be interest points on
        #                                   *   straight edges, e.g., just indicating an area of high surface change */
        #     bool do_non_maximum_suppression;  /**< If this is set to false there will be much more points
        #                                 *  (can be used to spread points over the whole scene
        #                                 *  (combined with a low min_interest_value)) */
        #     bool no_of_polynomial_approximations_per_point; /**< If this is >0, the exact position of the interest point is
        #                                                  determined using bivariate polynomial approximations of the
        #                                                  interest values of the area. */
        #     int max_no_of_threads;  //!< The maximum number of threads this code is allowed to use with OPNEMP
        #     bool use_recursive_scale_reduction;  /**< Try to decrease runtime by extracting interest points at lower reolution
        #                                    *  in areas that contain enough points, i.e., have lower range. */
        #     bool calculate_sparse_interest_image;  /**< Use some heuristics to decide which areas of the interest image
        #                                         can be left out to improve the runtime. */
        # };
        # 
        # =====PUBLIC METHODS=====
        # Erase all data calculated for the current range image
        void clearData ()
        
        # //! Set the RangeImageBorderExtractor member (required)
        # void setRangeImageBorderExtractor (RangeImageBorderExtractor* range_image_border_extractor);
        void setRangeImageBorderExtractor (pclftr.RangeImageBorderExtractor range_image_border_extractor)
        
        # //! Get the RangeImageBorderExtractor member
        # RangeImageBorderExtractor* getRangeImageBorderExtractor ()
        pclftr.RangeImageBorderExtractor getRangeImageBorderExtractor ()
        
        # //! Set the RangeImage member of the RangeImageBorderExtractor
        # void setRangeImage (const RangeImage* range_image)
        # void setRangeImage (const RangeImage_Ptr range_image)
        
        # /** Extract interest value per image point */
        # float* getInterestImage () { calculateInterestImage(); return interest_image_;}
        # float[] getInterestImage ()
        
        # //! Extract maxima from an interest image
        # const ::pcl::PointCloud<InterestPoint>& getInterestPoints () { calculateInterestPoints(); return *interest_points_;}
        
        # //! Set all points in the image that are interest points to true, the rest to false
        # const std::vector<bool>& getIsInterestPointImage ()
        
        # //! Getter for the parameter struct
        # Parameters& getParameters ()
        
        # //! Getter for the range image of range_image_border_extractor_
        # const RangeImage& getRangeImage ();
        
        # //! Overwrite the compute function of the base class
        # void compute (PointCloudOut& output);

# ingroup keypoints
# operator
# inline std::ostream& operator << (std::ostream& os, const NarfKeypoint::Parameters& p)

ctypedef NarfKeypoint NarfKeypoint_t
ctypedef shared_ptr[NarfKeypoint] NarfKeypointPtr_t
###

# sift_keypoint.h
# template <typename PointInT, typename PointOutT>
# class SIFTKeypoint : public Keypoint<PointInT, PointOutT>
cdef extern from "pcl/keypoints/sift_keypoint.h" namespace "pcl":
    cdef cppclass SIFTKeypoint[In, Out](Keypoint[In, Out]):
        SIFTKeypoint ()
        # public:
        # /** \brief Specify the range of scales over which to search for keypoints
        # * \param min_scale the standard deviation of the smallest scale in the scale space
        # * \param nr_octaves the number of octaves (i.e. doublings of scale) to compute 
        # * \param nr_scales_per_octave the number of scales to compute within each octave
        void setScales (float min_scale, int nr_octaves, int nr_scales_per_octave)
        
        # /** \brief Provide a threshold to limit detection of keypoints without sufficient contrast
        # * \param min_contrast the minimum contrast required for detection
        void setMinimumContrast (float min_contrast)


# pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
ctypedef SIFTKeypoint[cpp.PointNormal, cpp.PointWithScale] SIFTKeypoint_t
ctypedef shared_ptr[SIFTKeypoint[cpp.PointNormal, cpp.PointWithScale]] SIFTKeypointPtr_t
###

# smoothed_surfaces_keypoint.h
# template <typename PointT, typename PointNT>
# class SmoothedSurfacesKeypoint : public Keypoint <PointT, PointT>
cdef extern from "pcl/keypoints/smoothed_surfaces_keypoint.h" namespace "pcl":
    cdef cppclass SmoothedSurfacesKeypoint[In, Out](Keypoint[In, Out]):
        SmoothedSurfacesKeypoint ()
        # public:
        # void addSmoothedPointCloud (const PointCloudTConstPtr &cloud, const PointCloudNTConstPtr &normals, KdTreePtr &kdtree, float &scale);
        
        void resetClouds ()
        
        # inline void setNeighborhoodConstant (float neighborhood_constant)
        
        # inline float getNeighborhoodConstant ()
        
        # inline void setInputNormals (const PointCloudNTConstPtr &normals)
        
        # inline void setInputScale (float input_scale)
        
        # void detectKeypoints (PointCloudT &output);


###

# uniform_sampling.h
# template <typename PointInT>
# class UniformSampling: public Keypoint<PointInT, int>
cdef extern from "pcl/keypoints/uniform_sampling.h" namespace "pcl":
    cdef cppclass UniformSampling[In](Keypoint[In, int]):
        UniformSampling ()
        # public:
        # brief Set the 3D grid leaf size.
        # param radius the 3D grid leaf size
        void setRadiusSearch (double radius)


ctypedef UniformSampling[cpp.PointXYZ] UniformSampling_t
ctypedef UniformSampling[cpp.PointXYZI] UniformSampling_PointXYZI_t
ctypedef UniformSampling[cpp.PointXYZRGB] UniformSampling_PointXYZRGB_t
ctypedef UniformSampling[cpp.PointXYZRGBA] UniformSampling_PointXYZRGBA_t
ctypedef shared_ptr[UniformSampling[cpp.PointXYZ]] UniformSamplingPtr_t
ctypedef shared_ptr[UniformSampling[cpp.PointXYZI]] UniformSampling_PointXYZI_Ptr_t
ctypedef shared_ptr[UniformSampling[cpp.PointXYZRGB]] UniformSampling_PointXYZRGB_Ptr_t
ctypedef shared_ptr[UniformSampling[cpp.PointXYZRGBA]] UniformSampling_PointXYZRGBA_Ptr_t
###

###############################################################################
# Enum
###############################################################################

# 1.6.0
# NG : use Template parameters Class Internal
# typedef enum {HARRIS = 1, NOBLE, LOWE, TOMASI, CURVATURE} ResponseMethod;

# 1.7.2
# NG : use Template parameters Class Internal
# RESPONSEMETHOD_HARRIS "pcl::HarrisKeypoint3D::HARRIS", 
# RESPONSEMETHOD_NOBLE "pcl::HarrisKeypoint3D::NOBLE", 
# RESPONSEMETHOD_LOWE "pcl::HarrisKeypoint3D::LOWE", 
# RESPONSEMETHOD_TOMASI "pcl::HarrisKeypoint3D::TOMASI", 
# RESPONSEMETHOD_CURVATURE "pcl::HarrisKeypoint3D::CURVATURE"


