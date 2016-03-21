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

# harris_keypoint3D.h (1.6.0)
# template <typename PointInT, typename PointOutT, typename NormalT = pcl::Normal>
class HarrisKeypoint3D : public Keypoint<PointInT, PointOutT>
cdef extern from "pcl/keypoints/harris_keypoint3D.h" namespace "pcl":
    cdef cppclass HarrisKeypoint3D[In, Out, NormalT](Keypoint[In, Out]):
        HarrisKeypoint3D ()
        # HarrisKeypoint3D (ResponseMethod method = HARRIS, float radius = 0.01f, float threshold = 0.0f)
        # typedef enum {HARRIS = 1, NOBLE, LOWE, TOMASI, CURVATURE} ResponseMethod;
        # * \brief Set the method of the response to be calculated.
        # * \param[in] type
        # void setMethod (ResponseMethod type)
        # * \brief Set the radius for normal estimation and non maxima supression.
        # * \param[in] radius
        void setRadius (float radius)
        # * \brief Set the threshold value for detecting corners. This is only evaluated if non maxima suppression is turned on.
        # * \brief note non maxima suppression needs to be activated in order to use this feature.
        # * \param[in] threshold
        void setThreshold (float threshold)
        # * \brief Whether non maxima suppression should be applied or the response for each point should be returned
        # * \note this value needs to be turned on in order to apply thresholding and refinement
        # * \param[in] nonmax default is false
        void setNonMaxSupression (bool = false)
        # * \brief Whether the detected key points should be refined or not. If turned of, the key points are a subset of the original point cloud. Otherwise the key points may be arbitrary.
        # * \brief note non maxima supression needs to be on in order to use this feature.
        # * \param[in] do_refine
        void setRefine (bool do_refine)
        # * \brief Set normals if precalculated normals are available.
        # * \param normals
        void setNormals (PointCloud_Notmal_Ptr_t normals)
        # * \brief Provide a pointer to a dataset to add additional information
        # * to estimate the features for every point in the input dataset.  This
        # * is optional, if this is not set, it will only use the data in the
        # * input cloud to estimate the features.  This is useful when you only
        # * need to compute the features for a downsampled cloud.
        # * \param[in] cloud a pointer to a PointCloud message
        void setSearchSurface (const PointCloudInConstPtr &cloud)
        # * \brief Initialize the scheduler and set the number of threads to use.
        # * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
        inline void setNumberOfThreads (int nr_threads)
        # protected:
        # bool initCompute ();
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

# sift_keypoint.h
# template <typename PointInT, typename PointOutT>
# class SIFTKeypoint : public Keypoint<PointInT, PointOutT>
cdef extern from "pcl/keypoints/sift_keypoint.h" namespace "pcl":
    cdef cppclass SIFTKeypoint[In, Out](Keypoint[In, Out]):
        SIFTKeypoint ()
        # public:
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # typedef typename Keypoint<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Keypoint<PointInT, PointOutT>::KdTree KdTree;
        # using Keypoint<PointInT, PointOutT>::name_;
        # using Keypoint<PointInT, PointOutT>::input_;
        # using Keypoint<PointInT, PointOutT>::indices_;
        # using Keypoint<PointInT, PointOutT>::surface_;
        # using Keypoint<PointInT, PointOutT>::tree_;
        # using Keypoint<PointInT, PointOutT>::initCompute; 
        # 
        # /** \brief Specify the range of scales over which to search for keypoints
        # * \param min_scale the standard deviation of the smallest scale in the scale space
        # * \param nr_octaves the number of octaves (i.e. doublings of scale) to compute 
        # * \param nr_scales_per_octave the number of scales to compute within each octave
        void setScales (float min_scale, int nr_octaves, int nr_scales_per_octave)
        # /** \brief Provide a threshold to limit detection of keypoints without sufficient contrast
        # * \param min_contrast the minimum contrast required for detection
        void setMinimumContrast (float min_contrast)
        # protected:
        # bool initCompute ();
        # /** \brief Detect the SIFT keypoints for a set of points given in setInputCloud () using the spatial locator in 
        # * setSearchMethod ().
        # * \param output the resultant cloud of keypoints
        # void detectKeypoints (PointCloudOut &output);
###

# smoothed_surfaces_keypoint.h
# template <typename PointT, typename PointNT>
# class SmoothedSurfacesKeypoint : public Keypoint <PointT, PointT>
cdef extern from "pcl/keypoints/smoothed_surfaces_keypoint.h" namespace "pcl":
    cdef cppclass SmoothedSurfacesKeypoint[In, Out](Keypoint[In, Out]):
        SmoothedSurfacesKeypoint ()
        # public:
        # using PCLBase<PointT>::input_;
        # using Keypoint<PointT, PointT>::name_;
        # using Keypoint<PointT, PointT>::tree_;
        # using Keypoint<PointT, PointT>::initCompute;
        # typedef pcl::PointCloud<PointT> PointCloudT;
        # typedef typename PointCloudT::ConstPtr PointCloudTConstPtr;
        # typedef pcl::PointCloud<PointNT> PointCloudNT;
        # typedef typename PointCloudNT::ConstPtr PointCloudNTConstPtr;
        # typedef typename PointCloudT::Ptr PointCloudTPtr;
        # typedef typename Keypoint<PointT, PointT>::KdTreePtr KdTreePtr;
        # void addSmoothedPointCloud (const PointCloudTConstPtr &cloud,
        #                      const PointCloudNTConstPtr &normals,
        #                      KdTreePtr &kdtree,
        #                      float &scale);
        void resetClouds ()
        # inline void setNeighborhoodConstant (float neighborhood_constant)
        # inline float getNeighborhoodConstant ()
        # inline void setInputNormals (const PointCloudNTConstPtr &normals)
        # inline void setInputScale (float input_scale)
        # void detectKeypoints (PointCloudT &output);
        # protected:
        # bool initCompute ();
###

# uniform_sampling.h
# template <typename PointInT>
# class UniformSampling: public Keypoint<PointInT, int>
cdef extern from "pcl/keypoints/uniform_sampling.h" namespace "pcl":
    cdef cppclass UniformSampling[In](Keypoint[In, int]):
        UniformSampling ()
        # typedef typename Keypoint<PointInT, int>::PointCloudIn PointCloudIn;
        # typedef typename Keypoint<PointInT, int>::PointCloudOut PointCloudOut;
        # using Keypoint<PointInT, int>::name_;
        # using Keypoint<PointInT, int>::input_;
        # using Keypoint<PointInT, int>::indices_;
        # using Keypoint<PointInT, int>::search_radius_;
        # using Keypoint<PointInT, int>::getClassName;
        # public:
        # /** \brief Set the 3D grid leaf size.
        # * \param radius the 3D grid leaf size
        void setRadiusSearch (double radius)
        # protected:
        # brief Simple structure to hold an nD centroid and the number of points in a leaf.
        # struct Leaf
        # {
        #   Leaf () : idx (-1) { }
        #   int idx;
        # };
        # /** \brief The 3D grid leaves. */
        # boost::unordered_map<size_t, Leaf> leaves_;
        # /** \brief The size of a leaf. */
        # Eigen::Vector4f leaf_size_;
        # /** \brief Internal leaf sizes stored as 1/leaf_size_ for efficiency reasons. */ 
        # Eigen::Array4f inverse_leaf_size_;
        # /** \brief The minimum and maximum bin coordinates, the number of divisions, and the division multiplier. */
        # Eigen::Vector4i min_b_, max_b_, div_b_, divb_mul_;
        # /** \brief Downsample a Point Cloud using a voxelized grid approach
        # * \param output the resultant point cloud message
        # */
        # void detectKeypoints (PointCloudOut &output);

ctypedef UniformSampling[cpp.PointXYZ] UniformSampling_t
ctypedef UniformSampling[cpp.PointXYZI] UniformSampling_PointXYZI_t
ctypedef UniformSampling[cpp.PointXYZRGB] UniformSampling_PointXYZRGB_t
ctypedef UniformSampling[cpp.PointXYZRGBA] UniformSampling_PointXYZRGBA_t
ctypedef shared_ptr[UniformSampling[cpp.PointXYZ]] UniformSamplingPtr_t
ctypedef shared_ptr[UniformSampling[cpp.PointXYZI]] UniformSampling_PointXYZI_Ptr_t
ctypedef shared_ptr[UniformSampling[cpp.PointXYZRGB]] UniformSampling_PointXYZRGB_Ptr_t
ctypedef shared_ptr[UniformSampling[cpp.PointXYZRGBA]] UniformSampling_PointXYZRGBA_Ptr_t
###

