# -*- coding: utf-8 -*-

from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# pcl
cimport pcl_defs as cpp
cimport pcl_features_172 as pclftr
cimport pcl_kdtree_172 as pclkdt

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
cdef extern from "pcl/keypoints/harris_3d.h" namespace "pcl":
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


############################
# 1.7.2 Add

# agast_2d.h
# namespace pcl
# namespace keypoints
# namespace agast
# /** \brief Abstract detector class for AGAST corner point detectors.
#  * Adapted from the C++ implementation of Elmar Mair 
#  * (http://www6.in.tum.de/Main/ResearchAgast).
#  * \author Stefan Holzer
#  * \ingroup keypoints
#  */
# class PCL_EXPORTS AbstractAgastDetector
        # AbstractAgastDetector (const size_t width, 
        #                        const size_t height, 
        #                        const double threshold,
        #                        const double bmax) 
        # public:
        # typedef boost::shared_ptr<AbstractAgastDetector> Ptr;
        # typedef boost::shared_ptr<const AbstractAgastDetector> ConstPtr;
        # /** \brief Constructor. 
        # * \param[in] width the width of the image to process
        # * \param[in] height the height of the image to process
        # * \param[in] threshold the corner detection threshold
        # * \param[in] bmax the max image value (default: 255)
        # */
        # /** \brief Detects corner points. 
        # * \param intensity_data
        # * \param output
        # */
        # void 
        # detectKeypoints (const std::vector<unsigned char> &intensity_data, 
        #            pcl::PointCloud<pcl::PointUV> &output);
        # /** \brief Detects corner points. 
        # * \param intensity_data
        # * \param output
        # */
        # void 
        # detectKeypoints (const std::vector<float> &intensity_data, 
        #            pcl::PointCloud<pcl::PointUV> &output);
        # /** \brief Applies non-max-suppression. 
        # * \param[in] intensity_data the image data
        # * \param[in] input the keypoint positions
        # * \param[out] output the resultant keypoints after non-max-supression
        # */
        # void
        # applyNonMaxSuppression (const std::vector<unsigned char>& intensity_data, 
        #                   const pcl::PointCloud<pcl::PointUV> &input, 
        #                   pcl::PointCloud<pcl::PointUV> &output);
        # /** \brief Applies non-max-suppression. 
        # * \param[in] intensity_data the image data
        # * \param[in] input the keypoint positions
        # * \param[out] output the resultant keypoints after non-max-supression
        # */
        # void
        # applyNonMaxSuppression (const std::vector<float>& intensity_data, 
        #                   const pcl::PointCloud<pcl::PointUV> &input, 
        #                   pcl::PointCloud<pcl::PointUV> &output);
        # /** \brief Computes corner score. 
        # * \param[in] im the pixels to compute the score at
        # */
        # virtual int 
        # computeCornerScore (const unsigned char* im) const = 0;
        # /** \brief Computes corner score. 
        # * \param[in] im the pixels to compute the score at
        # */
        # virtual int 
        # computeCornerScore (const float* im) const = 0;
        # /** \brief Sets the threshold for corner detection.
        # * \param[in] threshold the threshold used for corner detection.
        # */
        # inline void
        # setThreshold (const double threshold)
        # /** \brief Get the threshold for corner detection, as set by the user. */
        # inline double
        # getThreshold ()
        # /** \brief Sets the maximum number of keypoints to return. The
        # * estimated keypoints are sorted by their internal score.
        # * \param[in] nr_max_keypoints set the maximum number of keypoints to return
        # */
        # inline void
        # setMaxKeypoints (const unsigned int nr_max_keypoints)
        # /** \brief Get the maximum nuber of keypoints to return, as set by the user. */
        # inline unsigned int 
        # getMaxKeypoints ()
        # /** \brief Detects points of interest (i.e., keypoints) in the given image
        # * \param[in] im the image to detect keypoints in 
        # * \param[out] corners_all the resultant set of keypoints detected
        # */
        # virtual void 
        # detect (const unsigned char* im, 
        #   std::vector<pcl::PointUV, Eigen::aligned_allocator<pcl::PointUV> > &corners_all) const = 0;
        # /** \brief Detects points of interest (i.e., keypoints) in the given image
        # * \param[in] im the image to detect keypoints in 
        # */
        # virtual void 
        # detect (const float* im, 
        #   std::vector<pcl::PointUV, Eigen::aligned_allocator<pcl::PointUV> > &) const = 0;
        # protected:
        # /** \brief Structure holding an index and the associated keypoint score. */
        # struct ScoreIndex
        # {
        # int idx;
        # int score;
        # };
        # /** \brief Score index comparator. */
        # struct CompareScoreIndex
        # {
        # /** \brief Comparator
        # * \param[in] i1 the first score index
        # * \param[in] i2 the second score index
        # */
        # inline bool
        # operator() (const ScoreIndex &i1, const ScoreIndex &i2)
        # {
        # return (i1.score > i2.score);
        # }
        # };
        # /** \brief Initializes the sample pattern. */
        # virtual void
        # initPattern () = 0;
        # /** \brief Non-max-suppression helper method.
        # * \param[in] input the keypoint positions
        # * \param[in] scores the keypoint scores computed on the image data
        # * \param[out] output the resultant keypoints after non-max-supression
        # */
        # void
        # applyNonMaxSuppression (const pcl::PointCloud<pcl::PointUV> &input, 
        #                   const std::vector<ScoreIndex>& scores, 
        #                   pcl::PointCloud<pcl::PointUV> &output);
        # /** \brief Computes corner scores for the specified points. 
        # * \param im
        # * \param corners_all
        # * \param scores
        # */
        # void 
        # computeCornerScores (const unsigned char* im, 
        #                const std::vector<pcl::PointUV, Eigen::aligned_allocator<pcl::PointUV> > & corners_all, 
        #                std::vector<ScoreIndex> & scores);
        # /** \brief Computes corner scores for the specified points. 
        # * \param im
        # * \param corners_all
        # * \param scores
        # */
        # void 
        # computeCornerScores (const float* im, 
        #                const std::vector<pcl::PointUV, Eigen::aligned_allocator<pcl::PointUV> > & corners_all, 
        #                std::vector<ScoreIndex> & scores);
        # /** \brief Width of the image to process. */
        # size_t width_;
        # /** \brief Height of the image to process. */
        # size_t height_;
        # /** \brief Threshold for corner detection. */
        # double threshold_;
        # /** \brief The maximum number of keypoints to return. */
        # unsigned int nr_max_keypoints_;
        # /** \brief Max image value. */
        # double bmax_;

        # namespace pcl
        # namespace keypoints
        # namespace agast
        # /** \brief Detector class for AGAST corner point detector (7_12s). 
        # *        
        # * Adapted from the C++ implementation of Elmar Mair 
        # * (http://www6.in.tum.de/Main/ResearchAgast).
        # *
        # * \author Stefan Holzer
        # * \ingroup keypoints
        # */
        # class PCL_EXPORTS AgastDetector7_12s : public AbstractAgastDetector
        # AgastDetector7_12s (const size_t width, 
        #           const size_t height, 
        #           const double threshold,
        #           const double bmax = 255) 
        # public:
        # typedef boost::shared_ptr<AgastDetector7_12s> Ptr;
        # typedef boost::shared_ptr<const AgastDetector7_12s> ConstPtr;
        # /** \brief Computes corner score. 
        # * \param im 
        # */
        # int 
        # computeCornerScore (const unsigned char* im) const;
        # /** \brief Computes corner score. 
        # * \param im 
        # */
        # int 
        # computeCornerScore (const float* im) const;
        # /** \brief Detects points of interest (i.e., keypoints) in the given image
        # * \param[in] im the image to detect keypoints in 
        # * \param[out] corners_all the resultant set of keypoints detected
        # */
        # void 
        # detect (const unsigned char* im, std::vector<pcl::PointUV, Eigen::aligned_allocator<pcl::PointUV> > &corners_all) const;
        # /** \brief Detects points of interest (i.e., keypoints) in the given image
        # * \param[in] im the image to detect keypoints in 
        # * \param[out] corners_all the resultant set of keypoints detected
        # */
        # void 
        # detect (const float* im, std::vector<pcl::PointUV, Eigen::aligned_allocator<pcl::PointUV> > &corners_all) const;
        # protected:
        # /** \brief Initializes the sample pattern. */
        # void 
        # initPattern ();
###

        # namespace pcl
        # namespace keypoints
        # namespace agast
        # /** \brief Detector class for AGAST corner point detector (5_8). 
        # *        
        # * Adapted from the C++ implementation of Elmar Mair 
        # * (http://www6.in.tum.de/Main/ResearchAgast).
        # *
        # * \author Stefan Holzer
        # * \ingroup keypoints
        # */
        # class PCL_EXPORTS AgastDetector5_8 : public AbstractAgastDetector
        # public:
            # typedef boost::shared_ptr<AgastDetector5_8> Ptr;
            # typedef boost::shared_ptr<const AgastDetector5_8> ConstPtr;
            # /** \brief Constructor. 
            # * \param[in] width the width of the image to process
            # * \param[in] height the height of the image to process
            # * \param[in] threshold the corner detection threshold
            # * \param[in] bmax the max image value (default: 255)
            # */
            # AgastDetector5_8 (const size_t width, 
            #               const size_t height, 
            #               const double threshold,
            #               const double bmax = 255) 
            # /** \brief Computes corner score. 
            # * \param im 
            # */
            # int computeCornerScore (const unsigned char* im) const;
            # /** \brief Computes corner score. 
            # * \param im 
            # */
            # int computeCornerScore (const float* im) const;
            # /** \brief Detects points of interest (i.e., keypoints) in the given image
            # * \param[in] im the image to detect keypoints in 
            # * \param[out] corners_all the resultant set of keypoints detected
            # */
            # void detect (const unsigned char* im, std::vector<pcl::PointUV, Eigen::aligned_allocator<pcl::PointUV> > &corners_all) const;
            # /** \brief Detects points of interest (i.e., keypoints) in the given image
            # * \param[in] im the image to detect keypoints in 
            # * \param[out] corners_all the resultant set of keypoints detected
            # */
            # void detect (const float* im, std::vector<pcl::PointUV, Eigen::aligned_allocator<pcl::PointUV> > &corners_all) const;
            # protected:
            # /** \brief Initializes the sample pattern. */
            # void initPattern ();
###

        # namespace pcl
        # namespace keypoints
        # namespace agast
        # /** \brief Detector class for AGAST corner point detector (OAST 9_16). 
        #   *        
        #   * Adapted from the C++ implementation of Elmar Mair 
        #   * (http://www6.in.tum.de/Main/ResearchAgast).
        #   *
        #   * \author Stefan Holzer
        #   * \ingroup keypoints
        #   */
        # class PCL_EXPORTS OastDetector9_16 : public AbstractAgastDetector
            # public:
            # typedef boost::shared_ptr<OastDetector9_16> Ptr;
            # typedef boost::shared_ptr<const OastDetector9_16> ConstPtr;
            # /** \brief Constructor. 
            #   * \param[in] width the width of the image to process
            #   * \param[in] height the height of the image to process
            #   * \param[in] threshold the corner detection threshold
            #   * \param[in] bmax the max image value (default: 255)
            #   */
            # OastDetector9_16 (const size_t width, 
            #                   const size_t height, 
            #                   const double threshold,
            #                   const double bmax = 255) 
            # 
            # /** \brief Computes corner score. 
            # * \param im 
            # */
            # int computeCornerScore (const unsigned char* im) const;
            # /** \brief Computes corner score. 
            # * \param im 
            # */
            # int computeCornerScore (const float* im) const;
            # /** \brief Detects points of interest (i.e., keypoints) in the given image
            # * \param[in] im the image to detect keypoints in 
            # * \param[out] corners_all the resultant set of keypoints detected
            # */
            # void detect (const unsigned char* im, std::vector<pcl::PointUV, Eigen::aligned_allocator<pcl::PointUV> > &corners_all) const;
            # /** \brief Detects points of interest (i.e., keypoints) in the given image
            # * \param[in] im the image to detect keypoints in 
            # * \param[out] corners_all the resultant set of keypoints detected
            # */
            # void detect (const float* im, std::vector<pcl::PointUV, Eigen::aligned_allocator<pcl::PointUV> > &corners_all) const;
            # protected:
            # /** \brief Initializes the sample pattern. */
            # void initPattern ();
###

        # namespace pcl
        # namespace keypoints
        # namespace internal
        # /////////////////////////////////////////////////////////////////////////////////////
        # template <typename Out> 
        # struct AgastApplyNonMaxSuppresion
        # {
        # AgastApplyNonMaxSuppresion (
        #     const std::vector<unsigned char> &image_data, 
        #     const pcl::PointCloud<pcl::PointUV> &tmp_cloud,
        #     const pcl::keypoints::agast::AbstractAgastDetector::Ptr &detector,
        #     pcl::PointCloud<Out> &output)
        # {
        #   pcl::PointCloud<pcl::PointUV> output_temp;
        #   detector->applyNonMaxSuppression (image_data, tmp_cloud, output_temp);
        #   pcl::copyPointCloud<pcl::PointUV, Out> (output_temp, output);
        # }
        
        # /////////////////////////////////////////////////////////////////////////////////////
        # template <>
        # struct AgastApplyNonMaxSuppresion<pcl::PointUV>
        # {
        #   AgastApplyNonMaxSuppresion (
        #       const std::vector<unsigned char> &image_data, 
        #       const pcl::PointCloud<pcl::PointUV> &tmp_cloud,
        #       const pcl::keypoints::agast::AbstractAgastDetector::Ptr &detector,
        #       pcl::PointCloud<pcl::PointUV> &output)
        #   {
        #     detector->applyNonMaxSuppression (image_data, tmp_cloud, output);
        #   }
        # };
        # /////////////////////////////////////////////////////////////////////////////////////
        # template <typename Out> 
        # struct AgastDetector
        # {
        #   AgastDetector (
        #       const std::vector<unsigned char> &image_data, 
        #       const pcl::keypoints::agast::AbstractAgastDetector::Ptr &detector,
        #       pcl::PointCloud<Out> &output)
        #   {
        #     pcl::PointCloud<pcl::PointUV> output_temp;
        #     detector->detectKeypoints (image_data, output_temp);
        #     pcl::copyPointCloud<pcl::PointUV, Out> (output_temp, output);
        #   }
        # };
        # /////////////////////////////////////////////////////////////////////////////////////
        # template <>
        # struct AgastDetector<pcl::PointUV>
        # {
        #   AgastDetector (
        #       const std::vector<unsigned char> &image_data, 
        #       const pcl::keypoints::agast::AbstractAgastDetector::Ptr &detector,
        #       pcl::PointCloud<pcl::PointUV> &output)
        #   {
        #     detector->detectKeypoints (image_data, output);
        #   }
        # };

# namespace pcl
# /** \brief Detects 2D AGAST corner points. Based on the original work and
# * paper reference by
# *
# * \par
# * Elmar Mair, Gregory D. Hager, Darius Burschka, Michael Suppa, and Gerhard Hirzinger. 
# * Adaptive and generic corner detection based on the accelerated segment test. 
# * In Proceedings of the European Conference on Computer Vision (ECCV'10), September 2010.
# *
# * \note This is an abstract base class. All children must implement a detectKeypoints method, based on the type of AGAST keypoint to be used.
# *
# * \author Stefan Holzer, Radu B. Rusu
# * \ingroup keypoints
# */
# template <typename PointInT, typename PointOutT, typename IntensityT = pcl::common::IntensityFieldAccessor<PointInT> >
# class AgastKeypoint2DBase : public Keypoint<PointInT, PointOutT>
        # AgastKeypoint2DBase ()
        # public:
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Keypoint<PointInT, PointOutT>::KdTree KdTree;
        # typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # typedef pcl::keypoints::agast::AbstractAgastDetector::Ptr AgastDetectorPtr;
        # using Keypoint<PointInT, PointOutT>::name_;
        # using Keypoint<PointInT, PointOutT>::input_;
        # using Keypoint<PointInT, PointOutT>::indices_;
        # using Keypoint<PointInT, PointOutT>::k_;

        # 
        # /** \brief Sets the threshold for corner detection.
        # * \param[in] threshold the threshold used for corner detection.
        # */
        # inline void setThreshold (const double threshold)
        # /** \brief Get the threshold for corner detection, as set by the user. */
        # inline double getThreshold ()
        # /** \brief Sets the maximum number of keypoints to return. The
        # * estimated keypoints are sorted by their internal score.
        # * \param[in] nr_max_keypoints set the maximum number of keypoints to return
        # */
        # inline void setMaxKeypoints (const unsigned int nr_max_keypoints)
        # /** \brief Get the maximum nuber of keypoints to return, as set by the user. */
        # inline unsigned int getMaxKeypoints ()
        # /** \brief Sets the max image data value (affects how many iterations AGAST does)
        # * \param[in] bmax the max image data value
        # */
        # inline void setMaxDataValue (const double bmax)
        # /** \brief Get the bmax image value, as set by the user. */
        # inline double getMaxDataValue ()
        # /** \brief Sets whether non-max-suppression is applied or not.
        # * \param[in] enabled determines whether non-max-suppression is enabled.
        # */
        # inline void setNonMaxSuppression (const bool enabled)
        # /** \brief Returns whether non-max-suppression is applied or not. */
        # inline bool getNonMaxSuppression ()
        # inline void setAgastDetector (const AgastDetectorPtr &detector)
        # inline AgastDetectorPtr getAgastDetector ()
        # protected:
        # /** \brief Initializes everything and checks whether input data is fine. */
        # bool initCompute ();
        # /** \brief Detects the keypoints.
        # * \param[out] output the resultant keypoints
        # */
        # virtual void detectKeypoints (PointCloudOut &output) = 0;
        # /** \brief Intensity field accessor. */
        # IntensityT intensity_;
        # /** \brief Threshold for corner detection. */
        # double threshold_;
        # /** \brief Determines whether non-max-suppression is activated. */
        # bool apply_non_max_suppression_;
        # /** \brief Max image value. */
        # double bmax_;
        # /** \brief The Agast detector to use. */
        # AgastDetectorPtr detector_;
        # /** \brief The maximum number of keypoints to return. */
        # unsigned int nr_max_keypoints_;
### 

# /** \brief Detects 2D AGAST corner points. Based on the original work and
# * paper reference by
# * \par
# * Elmar Mair, Gregory D. Hager, Darius Burschka, Michael Suppa, and Gerhard Hirzinger. 
# * Adaptive and generic corner detection based on the accelerated segment test. 
# * In Proceedings of the European Conference on Computer Vision (ECCV'10), September 2010.
# * Code example:
# * \code
# * pcl::PointCloud<pcl::PointXYZRGBA> cloud;
# * pcl::AgastKeypoint2D<pcl::PointXYZRGBA> agast;
# * agast.setThreshold (30);
# * agast.setInputCloud (cloud);
# * PointCloud<pcl::PointUV> keypoints;
# * agast.compute (keypoints);
# * \endcode
# * \note The AGAST keypoint type used is 7_12s.
# * \author Stefan Holzer, Radu B. Rusu
# * \ingroup keypoints
# */
# template <typename PointInT, typename PointOutT = pcl::PointUV>
# class AgastKeypoint2D : public AgastKeypoint2DBase<PointInT, PointOutT, pcl::common::IntensityFieldAccessor<PointInT> >
        # AgastKeypoint2D()
        # public:
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # using Keypoint<PointInT, PointOutT>::name_;
        # using Keypoint<PointInT, PointOutT>::input_;
        # using Keypoint<PointInT, PointOutT>::indices_;
        # using Keypoint<PointInT, PointOutT>::k_;
        # using AgastKeypoint2DBase<PointInT, PointOutT, pcl::common::IntensityFieldAccessor<PointInT> >::intensity_;
        # using AgastKeypoint2DBase<PointInT, PointOutT, pcl::common::IntensityFieldAccessor<PointInT> >::threshold_;
        # using AgastKeypoint2DBase<PointInT, PointOutT, pcl::common::IntensityFieldAccessor<PointInT> >::bmax_;
        # using AgastKeypoint2DBase<PointInT, PointOutT, pcl::common::IntensityFieldAccessor<PointInT> >::apply_non_max_suppression_;
        # using AgastKeypoint2DBase<PointInT, PointOutT, pcl::common::IntensityFieldAccessor<PointInT> >::detector_;
        # using AgastKeypoint2DBase<PointInT, PointOutT, pcl::common::IntensityFieldAccessor<PointInT> >::nr_max_keypoints_;
        # protected:
        # /** \brief Detects the keypoints.
        #  * \param[out] output the resultant keypoints
        #  */
        # virtual void detectKeypoints (PointCloudOut &output);

# /** \brief Detects 2D AGAST corner points. Based on the original work and
# * paper reference by
# *
# * \par
# * Elmar Mair, Gregory D. Hager, Darius Burschka, Michael Suppa, and Gerhard Hirzinger. 
# * Adaptive and generic corner detection based on the accelerated segment test. 
# * In Proceedings of the European Conference on Computer Vision (ECCV'10), September 2010.
# *
# * Code example:
# *
# * \code
# * pcl::PointCloud<pcl::PointXYZRGBA> cloud;
# * pcl::AgastKeypoint2D<pcl::PointXYZRGBA> agast;
# * agast.setThreshold (30);
# * agast.setInputCloud (cloud);
# *
# * PointCloud<pcl::PointUV> keypoints;
# * agast.compute (keypoints);
# * \endcode
# *
# * \note This is a specialized version for PointXYZ clouds, and operates on depth (z) as float. The output keypoints are of the PointXY type.
# * \note The AGAST keypoint type used is 7_12s.
# *
# * \author Stefan Holzer, Radu B. Rusu
# * \ingroup keypoints
# */
# template <>
# class AgastKeypoint2D<pcl::PointXYZ, pcl::PointUV>
# : public AgastKeypoint2DBase<pcl::PointXYZ, pcl::PointUV, pcl::common::IntensityFieldAccessor<pcl::PointXYZ> > 
#   public:
#   AgastKeypoint2D ()
#   protected:
#   /** \brief Detects the keypoints.
#   * \param[out] output the resultant keypoints
#   */
#   virtual void detectKeypoints (pcl::PointCloud<pcl::PointUV> &output);
# 
###

# harris_3d.h
# namespace pcl
# /** \brief HarrisKeypoint3D uses the idea of 2D Harris keypoints, but instead of using image gradients, it uses
# * surface normals.
# * \author Suat Gedikli
# * \ingroup keypoints
# */
# template <typename PointInT, typename PointOutT, typename NormalT = pcl::Normal>
# class HarrisKeypoint3D : public Keypoint<PointInT, PointOutT>
        # /** \brief Constructor
        #  * \param[in] method the method to be used to determine the corner responses
        #  * \param[in] radius the radius for normal estimation as well as for non maxima suppression
        #  * \param[in] threshold the threshold to filter out weak corners
        #  */
        # HarrisKeypoint3D (ResponseMethod method = HARRIS, float radius = 0.01f, float threshold = 0.0f)
        # HarrisKeypoint3D ()
        # public:
        # typedef boost::shared_ptr<HarrisKeypoint3D<PointInT, PointOutT, NormalT> > Ptr;
        # typedef boost::shared_ptr<const HarrisKeypoint3D<PointInT, PointOutT, NormalT> > ConstPtr;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Keypoint<PointInT, PointOutT>::KdTree KdTree;
        # typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # typedef typename pcl::PointCloud<NormalT> PointCloudN;
        # typedef typename PointCloudN::Ptr PointCloudNPtr;
        # typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
        # using Keypoint<PointInT, PointOutT>::name_;
        # using Keypoint<PointInT, PointOutT>::input_;
        # using Keypoint<PointInT, PointOutT>::indices_;
        # using Keypoint<PointInT, PointOutT>::surface_;
        # using Keypoint<PointInT, PointOutT>::tree_;
        # using Keypoint<PointInT, PointOutT>::k_;
        # using Keypoint<PointInT, PointOutT>::search_radius_;
        # using Keypoint<PointInT, PointOutT>::search_parameter_;
        # using Keypoint<PointInT, PointOutT>::keypoints_indices_;
        # using Keypoint<PointInT, PointOutT>::initCompute;
        # using PCLBase<PointInT>::setInputCloud;
        # typedef enum {HARRIS = 1, NOBLE, LOWE, TOMASI, CURVATURE} ResponseMethod;
        # /** \brief Provide a pointer to the input dataset
        # * \param[in] cloud the const boost shared pointer to a PointCloud message
        # */
        # virtual void setInputCloud (const PointCloudInConstPtr &cloud);
        # /** \brief Set the method of the response to be calculated.
        # * \param[in] type
        # */
        # void 
        # setMethod (ResponseMethod type);
        # /** \brief Set the radius for normal estimation and non maxima supression.
        # * \param[in] radius
        # */
        # void 
        # setRadius (float radius);
        # /** \brief Set the threshold value for detecting corners. This is only evaluated if non maxima suppression is turned on.
        # * \brief note non maxima suppression needs to be activated in order to use this feature.
        # * \param[in] threshold
        # */
        # void 
        # setThreshold (float threshold);
        # /** \brief Whether non maxima suppression should be applied or the response for each point should be returned
        # * \note this value needs to be turned on in order to apply thresholding and refinement
        # * \param[in] nonmax default is false
        # */
        # void 
        # setNonMaxSupression (bool = false);
        # /** \brief Whether the detected key points should be refined or not. If turned of, the key points are a subset of the original point cloud. Otherwise the key points may be arbitrary.
        # * \brief note non maxima supression needs to be on in order to use this feature.
        # * \param[in] do_refine
        # */
        # void 
        # setRefine (bool do_refine);
        # /** \brief Set normals if precalculated normals are available.
        # * \param normals
        # */
        # void 
        # setNormals (const PointCloudNConstPtr &normals);
        # /** \brief Provide a pointer to a dataset to add additional information
        # * to estimate the features for every point in the input dataset.  This
        # * is optional, if this is not set, it will only use the data in the
        # * input cloud to estimate the features.  This is useful when you only
        # * need to compute the features for a downsampled cloud.
        # * \param[in] cloud a pointer to a PointCloud message
        # */
        # virtual void setSearchSurface (const PointCloudInConstPtr &cloud) { surface_ = cloud; normals_.reset(); }
        # /** \brief Initialize the scheduler and set the number of threads to use.
        # * \param nr_threads the number of hardware threads to use (0 sets the value back to automatic)
        # */
        # inline void setNumberOfThreads (unsigned int nr_threads = 0)
        # protected:
        # bool
        # initCompute ();
        # void detectKeypoints (PointCloudOut &output);
        # /** \brief gets the corner response for valid input points*/
        # void responseHarris (PointCloudOut &output) const;
        # void responseNoble (PointCloudOut &output) const;
        # void responseLowe (PointCloudOut &output) const;
        # void responseTomasi (PointCloudOut &output) const;
        # void responseCurvature (PointCloudOut &output) const;
        # void refineCorners (PointCloudOut &corners) const;
        # /** \brief calculates the upper triangular part of unnormalized covariance matrix over the normals given by the indices.*/
        # void calculateNormalCovar (const std::vector<int>& neighbors, float* coefficients) const;
###

# harris_6d.h
# namespace pcl
# /** \brief Keypoint detector for detecting corners in 3D (XYZ), 2D (intensity) AND mixed versions of these.
# * \author Suat Gedikli
# * \ingroup keypoints
# */
# template <typename PointInT, typename PointOutT, typename NormalT = pcl::Normal>
# class HarrisKeypoint6D : public Keypoint<PointInT, PointOutT>
        # /**
        #  * @brief Constructor
        #  * @param radius the radius for normal estimation as well as for non maxima suppression
        #  * @param threshold the threshold to filter out weak corners
        #  */
        # HarrisKeypoint6D (float radius = 0.01, float threshold = 0.0)
        # HarrisKeypoint6D ()
        # public:
        # typedef boost::shared_ptr<HarrisKeypoint6D<PointInT, PointOutT, NormalT> > Ptr;
        # typedef boost::shared_ptr<const HarrisKeypoint6D<PointInT, PointOutT, NormalT> > ConstPtr;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Keypoint<PointInT, PointOutT>::KdTree KdTree;
        # typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # using Keypoint<PointInT, PointOutT>::name_;
        # using Keypoint<PointInT, PointOutT>::input_;
        # using Keypoint<PointInT, PointOutT>::indices_;
        # using Keypoint<PointInT, PointOutT>::surface_;
        # using Keypoint<PointInT, PointOutT>::tree_;
        # using Keypoint<PointInT, PointOutT>::k_;
        # using Keypoint<PointInT, PointOutT>::search_radius_;
        # using Keypoint<PointInT, PointOutT>::search_parameter_;
        # using Keypoint<PointInT, PointOutT>::keypoints_indices_;
        # 
        # /**
        # * @brief set the radius for normal estimation and non maxima supression.
        # * @param radius
        # */
        # void setRadius (float radius);
        # /**
        # * @brief set the threshold value for detecting corners. This is only evaluated if non maxima suppression is turned on.
        # * @brief note non maxima suppression needs to be activated in order to use this feature.
        # * @param threshold
        # */
        # void setThreshold (float threshold);
        # /**
        # * @brief whether non maxima suppression should be applied or the response for each point should be returned
        # * @note this value needs to be turned on in order to apply thresholding and refinement
        # * @param nonmax default is false
        # */
        # void setNonMaxSupression (bool = false);
        # /**
        # * @brief whether the detected key points should be refined or not. If turned of, the key points are a subset of the original point cloud. Otherwise the key points may be arbitrary.
        # * @brief note non maxima supression needs to be on in order to use this feature.
        # * @param do_refine
        # */
        # void setRefine (bool do_refine);
        # virtual void
        # setSearchSurface (const PointCloudInConstPtr &cloud) { surface_ = cloud; normals_->clear (); intensity_gradients_->clear ();}
        # /** \brief Initialize the scheduler and set the number of threads to use.
        # * \param nr_threads the number of hardware threads to use (0 sets the value back to automatic)
        # */
        # inline void
        # setNumberOfThreads (unsigned int nr_threads = 0) { threads_ = nr_threads; }
        # protected:
        # void detectKeypoints (PointCloudOut &output);
        # void responseTomasi (PointCloudOut &output) const;
        # void refineCorners (PointCloudOut &corners) const;
        # void calculateCombinedCovar (const std::vector<int>& neighbors, float* coefficients) const;
###

# iss_3d.h
# namespace pcl
# /** \brief ISSKeypoint3D detects the Intrinsic Shape Signatures keypoints for a given
# * point cloud. This class is based on a particular implementation made by Federico
# * Tombari and Samuele Salti and it has been explicitly adapted to PCL.
# * For more information about the original ISS detector, see:
# *\par
# * Yu Zhong, “Intrinsic shape signatures: A shape descriptor for 3D object recognition,”
# * Computer Vision Workshops (ICCV Workshops), 2009 IEEE 12th International Conference on ,
# * vol., no., pp.689-696, Sept. 27 2009-Oct. 4 2009
# * Code example:
# * \code
# * pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model (new pcl::PointCloud<pcl::PointXYZRGBA> ());;
# * pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model_keypoints (new pcl::PointCloud<pcl::PointXYZRGBA> ());
# * pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA> ());
# * // Fill in the model cloud
# * double model_resolution;
# * // Compute model_resolution
# * pcl::ISSKeypoint3D<pcl::PointXYZRGBA, pcl::PointXYZRGBA> iss_detector;
# * iss_detector.setSearchMethod (tree);
# * iss_detector.setSalientRadius (6 * model_resolution);
# * iss_detector.setNonMaxRadius (4 * model_resolution);
# * iss_detector.setThreshold21 (0.975);
# * iss_detector.setThreshold32 (0.975);
# * iss_detector.setMinNeighbors (5);
# * iss_detector.setNumberOfThreads (4);
# * iss_detector.setInputCloud (model);
# * iss_detector.compute (*model_keypoints);
# * \endcode
# * \author Gioia Ballin
# * \ingroup keypoints
# */
# template <typename PointInT, typename PointOutT, typename NormalT = pcl::Normal>
# class ISSKeypoint3D : public Keypoint<PointInT, PointOutT>
        # /** \brief Constructor.
        # * \param[in] salient_radius the radius of the spherical neighborhood used to compute the scatter matrix.
        # */
        # ISSKeypoint3D (double salient_radius = 0.0001)
        # ISSKeypoint3D ()
        # public:
        # typedef boost::shared_ptr<ISSKeypoint3D<PointInT, PointOutT, NormalT> > Ptr;
        # typedef boost::shared_ptr<const ISSKeypoint3D<PointInT, PointOutT, NormalT> > ConstPtr;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename pcl::PointCloud<NormalT> PointCloudN;
        # typedef typename PointCloudN::Ptr PointCloudNPtr;
        # typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
        # typedef typename pcl::octree::OctreePointCloudSearch<PointInT> OctreeSearchIn;
        # typedef typename OctreeSearchIn::Ptr OctreeSearchInPtr;
        # using Keypoint<PointInT, PointOutT>::name_;
        # using Keypoint<PointInT, PointOutT>::input_;
        # using Keypoint<PointInT, PointOutT>::surface_;
        # using Keypoint<PointInT, PointOutT>::tree_;
        # using Keypoint<PointInT, PointOutT>::search_radius_;
        # using Keypoint<PointInT, PointOutT>::search_parameter_;
        # using Keypoint<PointInT, PointOutT>::keypoints_indices_;
        # 
        #   /** \brief Set the radius of the spherical neighborhood used to compute the scatter matrix.
        # * \param[in] salient_radius the radius of the spherical neighborhood
        # */
        # void
        # setSalientRadius (double salient_radius);
        # /** \brief Set the radius for the application of the non maxima supression algorithm.
        # * \param[in] non_max_radius the non maxima suppression radius
        # */
        # void
        # setNonMaxRadius (double non_max_radius);
        # /** \brief Set the radius used for the estimation of the surface normals of the input cloud. If the radius is
        # * too large, the temporal performances of the detector may degrade significantly.
        # * \param[in] normal_radius the radius used to estimate surface normals
        # */
        # void
        # setNormalRadius (double normal_radius);
        # /** \brief Set the radius used for the estimation of the boundary points. If the radius is too large,
        # * the temporal performances of the detector may degrade significantly.
        # * \param[in] border_radius the radius used to compute the boundary points
        # */
        # void
        # setBorderRadius (double border_radius);
        # /** \brief Set the upper bound on the ratio between the second and the first eigenvalue.
        # * \param[in] gamma_21 the upper bound on the ratio between the second and the first eigenvalue
        # */
        # void
        # setThreshold21 (double gamma_21);
        # /** \brief Set the upper bound on the ratio between the third and the second eigenvalue.
        # * \param[in] gamma_32 the upper bound on the ratio between the third and the second eigenvalue
        # */
        # void
        # setThreshold32 (double gamma_32);
        # /** \brief Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
        # * \param[in] min_neighbors the minimum number of neighbors required
        # */
        # void
        # setMinNeighbors (int min_neighbors);
        # /** \brief Set the normals if pre-calculated normals are available.
        # * \param[in] normals the given cloud of normals
        # */
        # void
        # setNormals (const PointCloudNConstPtr &normals);
        # /** \brief Set the decision boundary (angle threshold) that marks points as boundary or regular.
        # * (default \f$\pi / 2.0\f$)
        # * \param[in] angle the angle threshold
        # */
        # inline void setAngleThreshold (float angle)
        # /** \brief Initialize the scheduler and set the number of threads to use.
        # * \param[in] nr_threads the number of hardware threads to use (0 sets the value back to automatic)
        # */
        # inline void setNumberOfThreads (unsigned int nr_threads = 0)
        # protected:
        # /** \brief Compute the boundary points for the given input cloud.
        # * \param[in] input the input cloud
        # * \param[in] border_radius the radius used to compute the boundary points
        # * \param[in] angle_threshold the decision boundary that marks the points as boundary
        # * \return the vector of boolean values in which the information about the boundary points is stored
        # */
        # bool* getBoundaryPoints (PointCloudIn &input, double border_radius, float angle_threshold);
        # /** \brief Compute the scatter matrix for a point index.
        # * \param[in] current_index the index of the point
        # * \param[out] cov_m the point scatter matrix
        # */
        # void getScatterMatrix (const int &current_index, Eigen::Matrix3d &cov_m);
        # /** \brief Perform the initial checks before computing the keypoints.
        # *  \return true if all the checks are passed, false otherwise
        # */
        # bool initCompute ();
        # /** \brief Detect the keypoints by performing the EVD of the scatter matrix.
        # * \param[out] output the resultant cloud of keypoints
        # */
        # void detectKeypoints (PointCloudOut &output);
        # /** \brief The radius of the spherical neighborhood used to compute the scatter matrix.*/
        # double salient_radius_;
        # /** \brief The non maxima suppression radius. */
        # double non_max_radius_;
        # /** \brief The radius used to compute the normals of the input cloud. */
        # double normal_radius_;
        # /** \brief The radius used to compute the boundary points of the input cloud. */
        # double border_radius_;
        # /** \brief The upper bound on the ratio between the second and the first eigenvalue returned by the EVD. */
        # double gamma_21_;
        # /** \brief The upper bound on the ratio between the third and the second eigenvalue returned by the EVD. */
        # double gamma_32_;
        # /** \brief Store the third eigen value associated to each point in the input cloud. */
        # double *third_eigen_value_;
        # /** \brief Store the information about the boundary points of the input cloud. */
        # bool *edge_points_;
        # /** \brief Minimum number of neighbors that has to be found while applying the non maxima suppression algorithm. */
        # int min_neighbors_;
        # /** \brief The cloud of normals related to the input surface. */
        # PointCloudNConstPtr normals_;
        # /** \brief The decision boundary (angle threshold) that marks points as boundary or regular. (default \f$\pi / 2.0\f$) */
        # float angle_threshold_;
        # /** \brief The number of threads that has to be used by the scheduler. */
        # unsigned int threads_;
#### 

# # susan.h
# namespace pcl
# /** \brief SUSANKeypoint implements a RGB-D extension of the SUSAN detector inluding normal 
#  * directions variation in top of intensity variation. 
#  * It is different from Harris in that it exploits normals directly so it is faster.  
#  * Original paper "SUSAN 窶A New Approach to Low Level Image Processing", Smith,
#  * Stephen M. and Brady, J. Michael 
#  *
#  * \author Nizar Sallem 
#  * \ingroup keypoints
#  */
# template <typename PointInT, typename PointOutT, typename NormalT = pcl::Normal, typename IntensityT= pcl::common::IntensityFieldAccessor<PointInT> >
# class SUSANKeypoint : public Keypoint<PointInT, PointOutT>
        # /** \brief Constructor
        #   * \param[in] radius the radius for normal estimation as well as for non maxima suppression
        #   * \param[in] distance_threshold to test if the nucleus is far enough from the centroid
        #   * \param[in] angular_threshold to test if normals are parallel
        #   * \param[in] intensity_threshold to test if points are of same color
        #   */
        # SUSANKeypoint (float radius = 0.01f, 
        #                float distance_threshold = 0.001f, 
        #                float angular_threshold = 0.0001f, 
        #                float intensity_threshold = 7.0f)
        # SUSANKeypoint()
        # public:
        # typedef boost::shared_ptr<SUSANKeypoint<PointInT, PointOutT, NormalT, IntensityT> > Ptr;
        # typedef boost::shared_ptr<const SUSANKeypoint<PointInT, PointOutT, NormalT, Intensity> > ConstPtr;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Keypoint<PointInT, PointOutT>::KdTree KdTree;
        # typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
        # typedef typename pcl::PointCloud<NormalT> PointCloudN;
        # typedef typename PointCloudN::Ptr PointCloudNPtr;
        # typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
        # using Keypoint<PointInT, PointOutT>::name_;
        # using Keypoint<PointInT, PointOutT>::input_;
        # using Keypoint<PointInT, PointOutT>::indices_;
        # using Keypoint<PointInT, PointOutT>::surface_;
        # using Keypoint<PointInT, PointOutT>::tree_;
        # using Keypoint<PointInT, PointOutT>::k_;
        # using Keypoint<PointInT, PointOutT>::search_radius_;
        # using Keypoint<PointInT, PointOutT>::search_parameter_;
        # using Keypoint<PointInT, PointOutT>::keypoints_indices_;
        # using Keypoint<PointInT, PointOutT>::initCompute;
        # /** \brief set the radius for normal estimation and non maxima supression.
        # * \param[in] radius
        # */
        # void setRadius (float radius);
        # void setDistanceThreshold (float distance_threshold);
        # /** \brief set the angular_threshold value for detecting corners. Normals are considered as 
        # * parallel if 1 - angular_threshold <= (Ni.Nj) <= 1
        # * \param[in] angular_threshold
        # */
        # void setAngularThreshold (float angular_threshold);
        # /** \brief set the intensity_threshold value for detecting corners. 
        # * \param[in] intensity_threshold
        # */
        # void setIntensityThreshold (float intensity_threshold);
        # /**
        # * \brief set normals if precalculated normals are available.
        # * \param normals
        # */
        # void setNormals (const PointCloudNConstPtr &normals);
        # virtual void setSearchSurface (const PointCloudInConstPtr &cloud);
        # /** \brief Initialize the scheduler and set the number of threads to use.
        # * \param nr_threads the number of hardware threads to use (0 sets the value back to automatic)
        # */
        # void setNumberOfThreads (unsigned int nr_threads);
        # /** \brief Apply non maxima suppression to the responses to keep strongest corners.
        # * \note in SUSAN points with less response or stronger corners
        # */
        # void setNonMaxSupression (bool nonmax);
        # /** \brief Filetr false positive using geometric criteria. 
        # * The nucleus and the centroid should at least distance_threshold_ from each other AND all the 
        # * points belonging to the USAN must be within the segment [nucleus centroid].
        # * \param[in] validate 
        # */
        # void setGeometricValidation (bool validate);
        # protected:
        # bool initCompute ();
        # void detectKeypoints (PointCloudOut &output);
        # /** \brief return true if a point lies within the line between the nucleus and the centroid
        # * \param[in] nucleus coordinate of the nucleus
        # * \param[in] centroid of the SUSAN
        # * \param[in] nc to centroid vector (used to speed up since it is constant for a given
        # * neighborhood)
        # * \param[in] point the query point to test against
        # * \return true if the point lies within [nucleus centroid]
        # */
        # bool isWithinNucleusCentroid (const Eigen::Vector3f& nucleus,
        #                        const Eigen::Vector3f& centroid,
        #                        const Eigen::Vector3f& nc,
        #                        const PointInT& point) const;
###


# harris_3d.h
###

# harris_6d.h
###

# iss_3d.h
###


