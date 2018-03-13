# -*- coding: utf-8 -*-

from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from boost_shared_ptr cimport shared_ptr

cimport eigen as eigen3

# main
cimport pcl_defs as cpp
cimport pcl_kdtree_172 as pclkdt
cimport pcl_range_image_172 as pcl_r_img

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

# > 1.7.2
# board.h
# namespace pcl
# /** \brief BOARDLocalReferenceFrameEstimation implements the BOrder Aware Repeatable Directions algorithm
# * for local reference frame estimation as described here:
# *  - A. Petrelli, L. Di Stefano,
# *    "On the repeatability of the local reference frame for partial shape matching",
# *    13th International Conference on Computer Vision (ICCV), 2011
# *
# * \author Alioscia Petrelli (original), Tommaso Cavallari (PCL port)
# * \ingroup features
# */
# template<typename PointInT, typename PointNT, typename PointOutT = ReferenceFrame>
# class BOARDLocalReferenceFrameEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/board.h" namespace "pcl":
    cdef cppclass BOARDLocalReferenceFrameEstimation[In, NT, Out](FeatureFromNormals[In, NT, Out]):
        BOARDLocalReferenceFrameEstimation()
        # public:
        # typedef boost::shared_ptr<BOARDLocalReferenceFrameEstimation<PointInT, PointNT, PointOutT> > Ptr;
        # typedef boost::shared_ptr<const BOARDLocalReferenceFrameEstimation<PointInT, PointNT, PointOutT> > ConstPtr;
        # 
        # /** \brief Constructor. */
        # BOARDLocalReferenceFrameEstimation () :
        #   tangent_radius_ (0.0f),
        #   find_holes_ (false),
        #   margin_thresh_ (0.85f),
        #   check_margin_array_size_ (24),
        #   hole_size_prob_thresh_ (0.2f),
        #   steep_thresh_ (0.1f),
        #   check_margin_array_ (),
        #   margin_array_min_angle_ (),
        #   margin_array_max_angle_ (),
        #   margin_array_min_angle_normal_ (),
        #   margin_array_max_angle_normal_ ()
        # {
        #   feature_name_ = "BOARDLocalReferenceFrameEstimation";
        #   setCheckMarginArraySize (check_margin_array_size_);
        # }
        # 
        # /** \brief Empty destructor */
        # virtual ~BOARDLocalReferenceFrameEstimation () {}
        # 
        # //Getters/Setters
        # /** \brief Set the maximum distance of the points used to estimate the x_axis and y_axis of the BOARD Reference Frame for a given point.
        #   *
        #   * \param[in] radius The search radius for x and y axes. If not set or set to 0 the parameter given with setRadiusSearch is used.
        #   */
        # inline void setTangentRadius (float radius)
        # 
        # /** \brief Get the maximum distance of the points used to estimate the x_axis and y_axis of the BOARD Reference Frame for a given point.
        #   *
        #   * \return The search radius for x and y axes. If set to 0 the parameter given with setRadiusSearch is used.
        #   */
        # inline float getTangentRadius () const
        # 
        # /** \brief Sets whether holes in the margin of the support, for each point, are searched and accounted for in the estimation of the 
        # *          Reference Frame or not.
        # *
        # * \param[in] find_holes Enable/Disable the search for holes in the support.
        # */
        # inline void setFindHoles (bool find_holes)
        # 
        # /** \brief Gets whether holes in the margin of the support, for each point, are searched and accounted for in the estimation of the 
        # *          Reference Frame or not.
        # *
        # * \return The search for holes in the support is enabled/disabled.
        # */
        # inline bool getFindHoles () const
        # 
        # /** \brief Sets the percentage of the search radius (or tangent radius if set) after which a point is considered part of the support margin.
        # *
        # * \param[in] margin_thresh the percentage of the search radius after which a point is considered a margin point.
        # */
        # inline void setMarginThresh (float margin_thresh)
        # 
        # /** \brief Gets the percentage of the search radius (or tangent radius if set) after which a point is considered part of the support margin.
        # *
        # * \return The percentage of the search radius after which a point is considered a margin point.
        # */
        # inline float getMarginThresh () const
        # 
        # /** \brief Sets the number of slices in which is divided the margin for the search of missing regions.
        # *
        # * \param[in] size the number of margin slices.
        # */
        # void setCheckMarginArraySize (int size)
        # 
        # /** \brief Gets the number of slices in which is divided the margin for the search of missing regions.
        # *
        # * \return the number of margin slices.
        # */
        # inline int getCheckMarginArraySize () const
        # 
        # /** \brief Given the angle width of a hole in the support margin, sets the minimum percentage of a circumference this angle 
        # *         must cover to be considered a missing region in the support and hence used for the estimation of the Reference Frame.
        # * \param[in] prob_thresh the minimum percentage of a circumference after which a hole is considered in the calculation
        # */
        # inline void setHoleSizeProbThresh (float prob_thresh)
        # 
        # /** \brief Given the angle width of a hole in the support margin, gets the minimum percentage of a circumference this angle 
        # *         must cover to be considered a missing region in the support and hence used for the estimation of the Reference Frame.
        # * \return the minimum percentage of a circumference after which a hole is considered in the calculation
        # */
        # inline float getHoleSizeProbThresh () const
        # 
        # /** \brief Sets the minimum steepness that the normals of the points situated on the borders of the hole, with reference
        # *         to the normal of the best point found by the algorithm, must have in order to be considered in the calculation of the Reference Frame.
        # * \param[in] steep_thresh threshold that defines if a missing region contains a point with the most different normal.
        # */
        # inline void setSteepThresh (float steep_thresh)
        # 
        # /** \brief Gets the minimum steepness that the normals of the points situated on the borders of the hole, with reference
        # *         to the normal of the best point found by the algorithm, must have in order to be considered in the calculation of the Reference Frame.
        # * \return threshold that defines if a missing region contains a point with the most different normal.
        # */
        # inline float getSteepThresh () const


###

# cppf.h
# namespace pcl
#   /** \brief
#     * \param[in] p1 
#     * \param[in] n1
#     * \param[in] p2 
#     * \param[in] n2
#     * \param[in] c1
#     * \param[in] c2
#     * \param[out] f1
#     * \param[out] f2
#     * \param[out] f3
#     * \param[out] f4
#     * \param[out] f5
#     * \param[out] f6
#     * \param[out] f7
#     * \param[out] f8
#     * \param[out] f9
#     * \param[out] f10
#     */
#   computeCPPFPairFeature (const Eigen::Vector4f &p1, const Eigen::Vector4f &n1, const Eigen::Vector4i &c1,
#                             const Eigen::Vector4f &p2, const Eigen::Vector4f &n2, const Eigen::Vector4i &c2,
#                             float &f1, float &f2, float &f3, float &f4, float &f5, float &f6, float &f7, float &f8, float &f9, float &f10);
# 
##
# cppf.h
# namespace pcl
# /** \brief Class that calculates the "surflet" features for each pair in the given
# * pointcloud. Please refer to the following publication for more details:
# *    C. Choi, Henrik Christensen
# *    3D Pose Estimation of Daily Objects Using an RGB-D Camera
# *    Proceedings of IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
# *    2012
# *
# * PointOutT is meant to be pcl::CPPFSignature - contains the 10 values of the Surflet
# * feature and in addition, alpha_m for the respective pair - optimization proposed by
# * the authors (see above)
# *
# * \author Martin Szarski, Alexandru-Eugen Ichim
# */
# template <typename PointInT, typename PointNT, typename PointOutT>
# class CPPFEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/cppf.h" namespace "pcl":
    cdef cppclass CPPFEstimation[In, NT, Out](FeatureFromNormals[In, NT, Out]):
        CPPFEstimation()
        # public:
        # typedef boost::shared_ptr<CPPFEstimation<PointInT, PointNT, PointOutT> > Ptr;
        # typedef boost::shared_ptr<const CPPFEstimation<PointInT, PointNT, PointOutT> > ConstPtr;
        # using PCLBase<PointInT>::indices_;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # typedef pcl::PointCloud<PointOutT> PointCloudOut;
        # 
        # /** \brief Empty Constructor. */
        # CPPFEstimation ();


###

# crh.h
# namespace pcl
# /** \brief CRHEstimation estimates the Camera Roll Histogram (CRH) descriptor for a given
# * point cloud dataset containing XYZ data and normals, as presented in:
# *   - CAD-Model Recognition and 6 DOF Pose Estimation
# *     A. Aldoma, N. Blodow, D. Gossow, S. Gedikli, R.B. Rusu, M. Vincze and G. Bradski
# *     ICCV 2011, 3D Representation and Recognition (3dRR11) workshop
# *     Barcelona, Spain, (2011)
# *
# * The suggested PointOutT is pcl::Histogram<90>. //dc (real) + 44 complex numbers (real, imaginary) + nyquist (real)
# *
# * \author Aitor Aldoma
# * \ingroup features
# */
# template<typename PointInT, typename PointNT, typename PointOutT = pcl::Histogram<90> >
# class CRHEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/crh.h" namespace "pcl":
    cdef cppclass CRHEstimation[In, NT, Out](FeatureFromNormals[In, NT, Out]):
        CRHEstimation()
        # public:
        # typedef boost::shared_ptr<CRHEstimation<PointInT, PointNT, PointOutT> > Ptr;
        # typedef boost::shared_ptr<const CRHEstimation<PointInT, PointNT, PointOutT> > ConstPtr;
        # 
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # 
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # 
        # /** \brief Constructor. */
        # CRHEstimation () : vpx_ (0), vpy_ (0), vpz_ (0), nbins_ (90)
        # 
        # /** \brief Set the viewpoint.
        # * \param[in] vpx the X coordinate of the viewpoint
        # * \param[in] vpy the Y coordinate of the viewpoint
        # * \param[in] vpz the Z coordinate of the viewpoint
        # */
        # inline void setViewPoint (float vpx, float vpy, float vpz)
        # 
        # /** \brief Get the viewpoint. 
        # * \param[out] vpx the X coordinate of the viewpoint
        # * \param[out] vpy the Y coordinate of the viewpoint
        # * \param[out] vpz the Z coordinate of the viewpoint
        # */
        # inline void getViewPoint (float &vpx, float &vpy, float &vpz)
        # inline void setCentroid (Eigen::Vector4f & centroid)


###

# don.h
# namespace pcl
# /** \brief A Difference of Normals (DoN) scale filter implementation for point cloud data.
# * For each point in the point cloud two normals estimated with a differing search radius (sigma_s, sigma_l)
# * are subtracted, the difference of these normals provides a scale-based feature which
# * can be further used to filter the point cloud, somewhat like the Difference of Guassians
# * in image processing, but instead on surfaces. Best results are had when the two search
# * radii are related as sigma_l=10*sigma_s, the octaves between the two search radii
# * can be though of as a filter bandwidth. For appropriate values and thresholds it
# * can be used for surface edge extraction.
# * \attention The input normals given by setInputNormalsSmall and setInputNormalsLarge have
# * to match the input point cloud given by setInputCloud. This behavior is different than
# * feature estimation methods that extend FeatureFromNormals, which match the normals
# * with the search surface.
# * \note For more information please see
# *    <b>Yani Ioannou. Automatic Urban Modelling using Mobile Urban LIDAR Data.
# *    Thesis (Master, Computing), Queen's University, March, 2010.</b>
# * \author Yani Ioannou.
# * \ingroup features
# */
# template <typename PointInT, typename PointNT, typename PointOutT>
# class DifferenceOfNormalsEstimation : public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/don.h" namespace "pcl":
    cdef cppclass DifferenceOfNormalsEstimation[In, NT, Out](Feature[In, Out]):
        DifferenceOfNormalsEstimation()
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using PCLBase<PointInT>::input_;
        # typedef typename pcl::PointCloud<PointNT> PointCloudN;
        # typedef typename PointCloudN::Ptr PointCloudNPtr;
        # typedef typename PointCloudN::ConstPtr PointCloudNConstPtr;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # public:
        # typedef boost::shared_ptr<DifferenceOfNormalsEstimation<PointInT, PointNT, PointOutT> > Ptr;
        # typedef boost::shared_ptr<const DifferenceOfNormalsEstimation<PointInT, PointNT, PointOutT> > ConstPtr;
        # 
        # /**
        # * Creates a new Difference of Normals filter.
        # */
        # DifferenceOfNormalsEstimation ()
        # virtual ~DifferenceOfNormalsEstimation ()
        # 
        # /**
        # * Set the normals calculated using a smaller search radius (scale) for the DoN operator.
        # * @param normals the smaller radius (scale) of the DoN filter.
        # */
        # inline void setNormalScaleSmall (const PointCloudNConstPtr &normals)
        # 
        # /**
        # * Set the normals calculated using a larger search radius (scale) for the DoN operator.
        # * @param normals the larger radius (scale) of the DoN filter.
        # */
        # inline void setNormalScaleLarge (const PointCloudNConstPtr &normals)
        # 
        # /**
        # * Computes the DoN vector for each point in the input point cloud and outputs the vector cloud to the given output.
        # * @param output the cloud to output the DoN vector cloud to.
        # */
        # virtual void computeFeature (PointCloudOut &output);
        # 
        # /**
        # * Initialize for computation of features.
        # * @return true if parameters (input normals, input) are sufficient to perform computation.
        # */
        # virtual bool initCompute ();


###

# gfpfh.h
# namespace pcl
# /** \brief @b GFPFHEstimation estimates the Global Fast Point Feature Histogram (GFPFH) descriptor for a given point
#   * cloud dataset containing points and labels.
#   * @note If you use this code in any academic work, please cite:
#   * <ul>
#   * <li> R.B. Rusu, A. Holzbach, M. Beetz.
#   *      Detecting and Segmenting Objects for Mobile Manipulation.
#   *      In the S3DV Workshop of the 12th International Conference on Computer Vision (ICCV),
#   *      2009.
#   * </li>
#   * </ul>
#   * \author Radu B. Rusu
#   * \ingroup features
#   */
# template <typename PointInT, typename PointLT, typename PointOutT>
# class GFPFHEstimation : public FeatureFromLabels<PointInT, PointLT, PointOutT>
cdef extern from "pcl/features/gfpfh.h" namespace "pcl":
    cdef cppclass GFPFHEstimation[In, LT, Out](FeatureFromLabels[In, LT, Out]):
        DifferenceOfNormalsEstimation()
        # public:
        # typedef boost::shared_ptr<GFPFHEstimation<PointInT, PointLT, PointOutT> > Ptr;
        # typedef boost::shared_ptr<const GFPFHEstimation<PointInT, PointLT, PointOutT> > ConstPtr;
        # using FeatureFromLabels<PointInT, PointLT, PointOutT>::feature_name_;
        # using FeatureFromLabels<PointInT, PointLT, PointOutT>::getClassName;
        # using FeatureFromLabels<PointInT, PointLT, PointOutT>::indices_;
        # using FeatureFromLabels<PointInT, PointLT, PointOutT>::k_;
        # using FeatureFromLabels<PointInT, PointLT, PointOutT>::search_parameter_;
        # using FeatureFromLabels<PointInT, PointLT, PointOutT>::surface_;
        # 
        # using FeatureFromLabels<PointInT, PointLT, PointOutT>::input_;
        # using FeatureFromLabels<PointInT, PointLT, PointOutT>::labels_;
        # 
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudIn  PointCloudIn;
        # 
        # /** \brief Empty constructor. */
        # GFPFHEstimation () : octree_leaf_size_ (0.01), number_of_classes_ (16), descriptor_size_ (PointOutT::descriptorSize ())
        # 
        # /** \brief Set the size of the octree leaves.
        #   */
        # inline void setOctreeLeafSize (double size)
        # 
        # /** \brief Get the sphere radius used for determining the neighbors. */
        # inline double getOctreeLeafSize ()
        # 
        # /** \brief Return the empty label value. */
        # inline uint32_t emptyLabel () const
        # 
        # /** \brief Return the number of different classes. */
        # inline uint32_t getNumberOfClasses () const
        # 
        # /** \brief Set the number of different classes.
        #  * \param n number of different classes.
        #  */
        # inline void setNumberOfClasses (uint32_t n)
        # 
        # /** \brief Return the size of the descriptor. */
        # inline int descriptorSize () const
        # 
        # /** \brief Overloaded computed method from pcl::Feature.
        #   * \param[out] output the resultant point cloud model dataset containing the estimated features
        #   */
        # void compute (PointCloudOut &output);


###

# linear_least_squares_normal.h
# namespace pcl
# /** \brief Surface normal estimation on dense data using a least-squares estimation based on a first-order Taylor approximation.
# * \author Stefan Holzer, Cedric Cagniart
# */
# template <typename PointInT, typename PointOutT>
# class LinearLeastSquaresNormalEstimation : public Feature<PointInT, PointOutT>
cdef extern from "pcl/features/linear_least_squares_normal.h" namespace "pcl":
    cdef cppclass LinearLeastSquaresNormalEstimation[In, Out](Feature[In, Out]):
        LinearLeastSquaresNormalEstimation()
        # public:
        # typedef boost::shared_ptr<LinearLeastSquaresNormalEstimation<PointInT, PointOutT> > Ptr;
        # typedef boost::shared_ptr<const LinearLeastSquaresNormalEstimation<PointInT, PointOutT> > ConstPtr;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudIn  PointCloudIn;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # using Feature<PointInT, PointOutT>::input_;
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::tree_;
        # using Feature<PointInT, PointOutT>::k_;
        # 
        # /** \brief Constructor */
        # LinearLeastSquaresNormalEstimation () :
        #   use_depth_dependent_smoothing_(false),
        #   max_depth_change_factor_(1.0f),
        #   normal_smoothing_size_(9.0f)
        # 
        # /** \brief Destructor */
        # virtual ~LinearLeastSquaresNormalEstimation ();
        # 
        # /** \brief Computes the normal at the specified position. 
        # * \param[in] pos_x x position (pixel)
        # * \param[in] pos_y y position (pixel)
        # * \param[out] normal the output estimated normal 
        # */
        # void computePointNormal (const int pos_x, const int pos_y, PointOutT &normal)
        # 
        # /** \brief Set the normal smoothing size
        # * \param[in] normal_smoothing_size factor which influences the size of the area used to smooth normals 
        # * (depth dependent if useDepthDependentSmoothing is true)
        # */
        # void setNormalSmoothingSize (float normal_smoothing_size)
        # 
        # /** \brief Set whether to use depth depending smoothing or not
        #  * \param[in] use_depth_dependent_smoothing decides whether the smoothing is depth dependent
        #  */
        # void setDepthDependentSmoothing (bool use_depth_dependent_smoothing)
        # 
        # /** \brief The depth change threshold for computing object borders
        #  * \param[in] max_depth_change_factor the depth change threshold for computing object borders based on 
        #  * depth changes
        #  */
        # void setMaxDepthChangeFactor (float max_depth_change_factor)
        # 
        # /** \brief Provide a pointer to the input dataset (overwrites the PCLBase::setInputCloud method)
        # * \param[in] cloud the const boost shared pointer to a PointCloud message
        # */
        # virtual inline void setInputCloud (const typename PointCloudIn::ConstPtr &cloud) 


###

# pcl 1.7 package base ng(linux)
# (source code build ok?)
# moment_of_inertia_estimation.h
# namespace pcl
# /** 
#   * Implements the method for extracting features based on moment of inertia.
#   * It also calculates AABB, OBB and eccentricity of the projected cloud.
#   */
# template <typename PointT>
# class PCL_EXPORTS MomentOfInertiaEstimation : public pcl::PCLBase <PointT>
cdef extern from "pcl/features/moment_of_inertia_estimation.h" namespace "pcl":
    cdef cppclass MomentOfInertiaEstimation[PointT](cpp.PCLBase[PointT]):
        MomentOfInertiaEstimation()
        # /** \brief Constructor that sets default values for member variables. */
        # MomentOfInertiaEstimation ();
        # public:
        # typedef typename pcl::PCLBase <PointT>::PointCloudConstPtr PointCloudConstPtr;
        # typedef typename pcl::PCLBase <PointT>::PointIndicesConstPtr PointIndicesConstPtr;
        # public:
        # /** \brief Provide a pointer to the input dataset
        # * \param[in] cloud the const boost shared pointer to a PointCloud message
        # */
        # virtual void setInputCloud (const PointCloudConstPtr& cloud)
        void setInputCloud (const cpp.PCLBase[PointT]& cloud)
        
        # \brief Provide a pointer to the vector of indices that represents the input data.
        # \param[in] indices a pointer to the vector of indices that represents the input data.
        # virtual void setIndices (const IndicesPtr& indices);
        # void setIndices (const IndicesPtr& indices)
        
        # /** \brief Provide a pointer to the vector of indices that represents the input data.
        # * \param[in] indices a pointer to the vector of indices that represents the input data.
        # */
        # void setIndices (const IndicesConstPtr& indices)
        
        # /** \brief Provide a pointer to the vector of indices that represents the input data.
        # * \param[in] indices a pointer to the vector of indices that represents the input data.
        # */
        # virtual void setIndices (const PointIndicesConstPtr& indices);
        # void setIndices (const PointIndicesConstPtr& indices)
        
        # /** \brief Set the indices for the points laying within an interest region of 
        #   * the point cloud.
        #   * \note you shouldn't call this method on unorganized point clouds!
        #   * \param[in] row_start the offset on rows
        #   * \param[in] col_start the offset on columns
        #   * \param[in] nb_rows the number of rows to be considered row_start included
        #   * \param[in] nb_cols the number of columns to be considered col_start included
        #   */
        # virtual void setIndices (size_t row_start, size_t col_start, size_t nb_rows, size_t nb_cols);
        void setIndices (size_t row_start, size_t col_start, size_t nb_rows, size_t nb_cols)
        
        # /** \brief This method allows to set the angle step. It is used for the rotation
        # * of the axis which is used for moment of inertia/eccentricity calculation.
        # * \param[in] step angle step
        # */
        # void setAngleStep (const float step);
        void setAngleStep (const float step)
        
        # /** \brief Returns the angle step. */
        # float getAngleStep () const;
        float getAngleStep ()
        
        # /** \brief This method allows to set the normalize_ flag. If set to false, then
        # * point_mass_ will be used to scale the moment of inertia values. Otherwise,
        # * point_mass_ will be set to 1 / number_of_points. Default value is true.
        # * \param[in] need_to_normalize desired value
        # */
        # void setNormalizePointMassFlag (bool need_to_normalize);
        void setNormalizePointMassFlag (bool need_to_normalize)
        
        # /** \brief Returns the normalize_ flag. */
        # bool getNormalizePointMassFlag () const;
        bool getNormalizePointMassFlag ()
        
        # /** \brief This method allows to set point mass that will be used for
        # * moment of inertia calculation. It is needed to scale moment of inertia values.
        # * default value is 0.0001.
        # * \param[in] point_mass point mass
        # */
        # void setPointMass (const float point_mass);
        void setPointMass (const float point_mass)
        
        # /** \brief Returns the mass of point. */
        # float getPointMass () const;
        float getPointMass ()
        
        # /** \brief This method launches the computation of all features. After execution
        # * it sets is_valid_ flag to true and each feature can be accessed with the
        # * corresponding get method.
        # */
        # void compute ();
        void compute ()
        
        # 
        # /** \brief This method gives access to the computed axis aligned bounding box. It returns true
        # * if the current values (eccentricity, moment of inertia etc) are valid and false otherwise.
        # * \param[out] min_point min point of the AABB
        # * \param[out] max_point max point of the AABB
        # */
        # bool getAABB (PointT& min_point, PointT& max_point) const;
        bool getAABB (PointT& min_point, PointT& max_point)
        
        # /** \brief This method gives access to the computed oriented bounding box. It returns true
        # * if the current values (eccentricity, moment of inertia etc) are valid and false otherwise.
        # * Note that in order to get the OBB, each vertex of the given AABB (specified with min_point and max_point)
        # * must be rotated with the given rotational matrix (rotation transform) and then positioned.
        # * Also pay attention to the fact that this is not the minimal possible bounding box. This is the bounding box
        # * which is oriented in accordance with the eigen vectors.
        # * \param[out] min_point min point of the OBB
        # * \param[out] max_point max point of the OBB
        # * \param[out] position position of the OBB
        # * \param[out] rotational_matrix this matrix represents the rotation transform
        # */
        # bool getOBB (PointT& min_point, PointT& max_point, PointT& position, Eigen::Matrix3f& rotational_matrix) const;
        bool getOBB (PointT& min_point, PointT& max_point, PointT& position, eigen3.Matrix3f& rotational_matrix)
        
        # /** \brief This method gives access to the computed eigen values. It returns true
        # * if the current values (eccentricity, moment of inertia etc) are valid and false otherwise.
        # * \param[out] major major eigen value
        # * \param[out] middle middle eigen value
        # * \param[out] minor minor eigen value
        # */
        # bool getEigenValues (float& major, float& middle, float& minor) const;
        bool getEigenValues (float& major, float& middle, float& minor)
        
        # /** \brief This method gives access to the computed eigen vectors. It returns true
        # * if the current values (eccentricity, moment of inertia etc) are valid and false otherwise.
        # * \param[out] major axis which corresponds to the eigen vector with the major eigen value
        # * \param[out] middle axis which corresponds to the eigen vector with the middle eigen value
        # * \param[out] minor axis which corresponds to the eigen vector with the minor eigen value
        # */
        # bool getEigenVectors (Eigen::Vector3f& major, Eigen::Vector3f& middle, Eigen::Vector3f& minor) const;
        bool getEigenVectors (eigen3.Vector3f& major, eigen3.Vector3f& middle, eigen3.Vector3f& minor)
        
        # /** \brief This method gives access to the computed moments of inertia. It returns true
        # * if the current values (eccentricity, moment of inertia etc) are valid and false otherwise.
        # * \param[out] moment_of_inertia computed moments of inertia
        # */
        # bool getMomentOfInertia (std::vector <float>& moment_of_inertia) const;
        bool getMomentOfInertia (vector [float]& moment_of_inertia)
        
        # /** \brief This method gives access to the computed ecentricities. It returns true
        # * if the current values (eccentricity, moment of inertia etc) are valid and false otherwise.
        # * \param[out] eccentricity computed eccentricities
        # */
        # bool getEccentricity (std::vector <float>& eccentricity) const;
        bool getEccentricity (vector [float]& eccentricity)
        
        # /** \brief This method gives access to the computed mass center. It returns true
        # * if the current values (eccentricity, moment of inertia etc) are valid and false otherwise.
        # * Note that when mass center of a cloud is computed, mass point is always considered equal 1.
        # * \param[out] mass_center computed mass center
        # */
        # bool getMassCenter (Eigen::Vector3f& mass_center) const;
        bool getMassCenter (eigen3.Vector3f& mass_center)


ctypedef MomentOfInertiaEstimation[cpp.PointXYZ] MomentOfInertiaEstimation_t
ctypedef MomentOfInertiaEstimation[cpp.PointXYZI] MomentOfInertiaEstimation_PointXYZI_t
ctypedef MomentOfInertiaEstimation[cpp.PointXYZRGB] MomentOfInertiaEstimation_PointXYZRGB_t
ctypedef MomentOfInertiaEstimation[cpp.PointXYZRGBA] MomentOfInertiaEstimation_PointXYZRGBA_t
ctypedef shared_ptr[MomentOfInertiaEstimation[cpp.PointXYZ]] MomentOfInertiaEstimationPtr_t
ctypedef shared_ptr[MomentOfInertiaEstimation[cpp.PointXYZI]] MomentOfInertiaEstimation_PointXYZI_Ptr_t
ctypedef shared_ptr[MomentOfInertiaEstimation[cpp.PointXYZRGB]] MomentOfInertiaEstimation_PointXYZRGB_Ptr_t
ctypedef shared_ptr[MomentOfInertiaEstimation[cpp.PointXYZRGBA]] MomentOfInertiaEstimation_PointXYZRGBA_Ptr_t
###

# our_cvfh.h
# namespace pcl
# /** \brief OURCVFHEstimation estimates the Oriented, Unique and Repetable Clustered Viewpoint Feature Histogram (CVFH) descriptor for a given
#  * point cloud dataset given XYZ data and normals, as presented in:
#  *     - OUR-CVFH Oriented, Unique and Repeatable Clustered Viewpoint Feature Histogram for Object Recognition and 6DOF Pose Estimation
#  *     A. Aldoma, F. Tombari, R.B. Rusu and M. Vincze
#  *     DAGM-OAGM 2012
#  *     Graz, Austria
#  * The suggested PointOutT is pcl::VFHSignature308.
#  * \author Aitor Aldoma
#  * \ingroup features
#  */
# template<typename PointInT, typename PointNT, typename PointOutT = pcl::VFHSignature308>
# class OURCVFHEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/our_cvfh.h" namespace "pcl":
    cdef cppclass OURCVFHEstimation[In, NT, Out](FeatureFromNormals[In, NT, Out]):
        OURCVFHEstimation()
        # public:
        # typedef boost::shared_ptr<OURCVFHEstimation<PointInT, PointNT, PointOutT> > Ptr;
        # typedef boost::shared_ptr<const OURCVFHEstimation<PointInT, PointNT, PointOutT> > ConstPtr;
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::k_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # 
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename pcl::search::Search<PointNormal>::Ptr KdTreePtr;
        # typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
        # /** \brief Empty constructor. */
        # OURCVFHEstimation () :
        # vpx_ (0), vpy_ (0), vpz_ (0), leaf_size_ (0.005f), normalize_bins_ (false), curv_threshold_ (0.03f), cluster_tolerance_ (leaf_size_ * 3),
        #     eps_angle_threshold_ (0.125f), min_points_ (50), radius_normals_ (leaf_size_ * 3), centroids_dominant_orientations_ (),
        #     dominant_normals_ ()
        # 
        # /** \brief Creates an affine transformation from the RF axes
        # * \param[in] evx the x-axis
        # * \param[in] evy the y-axis
        # * \param[in] evz the z-axis
        # * \param[out] transformPC the resulting transformation
        # * \param[in] center_mat 4x4 matrix concatenated to the resulting transformation
        # */
        # inline Eigen::Matrix4f createTransFromAxes (Eigen::Vector3f & evx, Eigen::Vector3f & evy, Eigen::Vector3f & evz, Eigen::Affine3f & transformPC, Eigen::Matrix4f & center_mat)
        # 
        # /** \brief Computes SGURF and the shape distribution based on the selected SGURF
        # * \param[in] processed the input cloud
        # * \param[out] output the resulting signature
        # * \param[in] cluster_indices the indices of the stable cluster
        # */
        # void computeRFAndShapeDistribution (PointInTPtr & processed, PointCloudOut &output, std::vector<pcl::PointIndices> & cluster_indices);
        # 
        # /** \brief Computes SGURF
        # * \param[in] centroid the centroid of the cluster
        # * \param[in] normal_centroid the average of the normals
        # * \param[in] processed the input cloud
        # * \param[out] transformations the transformations aligning the cloud to the SGURF axes
        # * \param[out] grid the cloud transformed internally
        # * \param[in] indices the indices of the stable cluster
        # */
        # bool sgurf (Eigen::Vector3f & centroid, Eigen::Vector3f & normal_centroid, PointInTPtr & processed, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & transformations, PointInTPtr & grid, pcl::PointIndices & indices);
        # 
        # /** \brief Removes normals with high curvature caused by real edges or noisy data
        # * \param[in] cloud pointcloud to be filtered
        # * \param[in] indices_to_use
        # * \param[out] indices_out the indices of the points with higher curvature than threshold
        # * \param[out] indices_in the indices of the remaining points after filtering
        # * \param[in] threshold threshold value for curvature
        # */
        # void filterNormalsWithHighCurvature (const pcl::PointCloud<PointNT> & cloud, std::vector<int> & indices_to_use, std::vector<int> &indices_out, std::vector<int> &indices_in, float threshold);
        # 
        # /** \brief Set the viewpoint.
        # * \param[in] vpx the X coordinate of the viewpoint
        # * \param[in] vpy the Y coordinate of the viewpoint
        # * \param[in] vpz the Z coordinate of the viewpoint
        # */
        # inline void setViewPoint (float vpx, float vpy, float vpz)
        # 
        # /** \brief Set the radius used to compute normals
        # * \param[in] radius_normals the radius
        # */
        # inline void setRadiusNormals (float radius_normals)
        # 
        # /** \brief Get the viewpoint. 
        # * \param[out] vpx the X coordinate of the viewpoint
        # * \param[out] vpy the Y coordinate of the viewpoint
        # * \param[out] vpz the Z coordinate of the viewpoint
        # */
        # inline void getViewPoint (float &vpx, float &vpy, float &vpz)
        # 
        # /** \brief Get the centroids used to compute different CVFH descriptors
        # * \param[out] centroids vector to hold the centroids
        # */
        # inline void getCentroidClusters (std::vector<Eigen::Vector3f> & centroids)
        # 
        # /** \brief Get the normal centroids used to compute different CVFH descriptors
        # * \param[out] centroids vector to hold the normal centroids
        # */
        # inline void getCentroidNormalClusters (std::vector<Eigen::Vector3f> & centroids)
        # 
        # /** \brief Sets max. Euclidean distance between points to be added to the cluster 
        # * \param[in] d the maximum Euclidean distance
        # */
        # inline void setClusterTolerance (float d)
        # 
        # /** \brief Sets max. deviation of the normals between two points so they can be clustered together
        # * \param[in] d the maximum deviation
        # */
        # inline void setEPSAngleThreshold (float d)
        # 
        # /** \brief Sets curvature threshold for removing normals
        # * \param[in] d the curvature threshold
        # */
        # inline void setCurvatureThreshold (float d)
        # 
        # /** \brief Set minimum amount of points for a cluster to be considered
        # * \param[in] min the minimum amount of points to be set
        # */
        # inline void setMinPoints (size_t min)
        # 
        # /** \brief Sets wether if the signatures should be normalized or not
        # * \param[in] normalize true if normalization is required, false otherwise
        # */
        # inline void setNormalizeBins (bool normalize)
        # 
        # /** \brief Gets the indices of the original point cloud used to compute the signatures
        # * \param[out] indices vector of point indices
        # */
        # inline void getClusterIndices (std::vector<pcl::PointIndices> & indices)
        # 
        # /** \brief Gets the number of non-disambiguable axes that correspond to each centroid
        # * \param[out] cluster_axes vector mapping each centroid to the number of signatures
        # */
        # inline void getClusterAxes (std::vector<short> & cluster_axes)
        # 
        # /** \brief Sets the refinement factor for the clusters
        # * \param[in] rc the factor used to decide if a point is used to estimate a stable cluster
        # */
        # void setRefineClusters (float rc)
        # 
        # /** \brief Returns the transformations aligning the point cloud to the corresponding SGURF
        # * \param[out] trans vector of transformations
        # */
        # void getTransforms (std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & trans)
        # 
        # /** \brief Returns a boolean vector indicating of the transformation obtained by getTransforms() represents
        # * a valid SGURF
        # * \param[out] valid vector of booleans
        # */
        # void getValidTransformsVec (std::vector<bool> & valid)
        # 
        # /** \brief Sets the min axis ratio between the SGURF axes to decide if disambiguition is feasible
        # * \param[in] f the ratio between axes
        # */
        # void setAxisRatio (float f)
        # 
        # /** \brief Sets the min disambiguition axis value to generate several SGURFs for the cluster when disambiguition is difficult
        # * \param[in] f the min axis value
        # */
        # void setMinAxisValue (float f)
        # 
        # /** \brief Overloaded computed method from pcl::Feature.
        # * \param[out] output the resultant point cloud model dataset containing the estimated features
        # */
        # void compute (PointCloudOut &output);


####

# pfh_tools.h
# namespace pcl
#   /** \brief Compute the 4-tuple representation containing the three angles and one distance between two points
#     * represented by Cartesian coordinates and normals.
#     * \note For explanations about the features, please see the literature mentioned above (the order of the
#     * features might be different).
#     * \param[in] p1 the first XYZ point
#     * \param[in] n1 the first surface normal
#     * \param[in] p2 the second XYZ point
#     * \param[in] n2 the second surface normal
#     * \param[out] f1 the first angular feature (angle between the projection of nq_idx and u)
#     * \param[out] f2 the second angular feature (angle between nq_idx and v)
#     * \param[out] f3 the third angular feature (angle between np_idx and |p_idx - q_idx|)
#     * \param[out] f4 the distance feature (p_idx - q_idx)
#     *
#     * \note For efficiency reasons, we assume that the point data passed to the method is finite.
#     * \ingroup features
#     */
#   PCL_EXPORTS bool 
#   computePairFeatures (const Eigen::Vector4f &p1, const Eigen::Vector4f &n1, 
#                        const Eigen::Vector4f &p2, const Eigen::Vector4f &n2, 
#                        float &f1, float &f2, float &f3, float &f4);
# 
#   PCL_EXPORTS bool
#   computeRGBPairFeatures (const Eigen::Vector4f &p1, const Eigen::Vector4f &n1, const Eigen::Vector4i &colors1,
#                           const Eigen::Vector4f &p2, const Eigen::Vector4f &n2, const Eigen::Vector4i &colors2,
#                           float &f1, float &f2, float &f3, float &f4, float &f5, float &f6, float &f7);
# 
###

# rops_estimation.h
# namespace pcl
# /** \brief
# * This class implements the method for extracting RoPS features presented in the article
# * "Rotational Projection Statistics for 3D Local Surface Description and Object Recognition" by
# * Yulan Guo, Ferdous Sohel, Mohammed Bennamoun, Min Lu and Jianwei Wan.
# */
# template <typename PointInT, typename PointOutT>
# class PCL_EXPORTS ROPSEstimation : public pcl::Feature <PointInT, PointOutT>
cdef extern from "pcl/features/rops_estimation.h" namespace "pcl":
    cdef cppclass ROPSEstimation[In, Out](Feature[In, Out]):
        ROPSEstimation()
        # public:
        # using Feature <PointInT, PointOutT>::input_;
        # using Feature <PointInT, PointOutT>::indices_;
        # using Feature <PointInT, PointOutT>::surface_;
        # using Feature <PointInT, PointOutT>::tree_;
        # typedef typename pcl::Feature <PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename pcl::Feature <PointInT, PointOutT>::PointCloudIn PointCloudIn;
        # public:
        # /** \brief Simple constructor. */
        # ROPSEstimation ();
        # 
        # /** \brief Virtual destructor. */
        # virtual ~ROPSEstimation ();
        # 
        # /** \brief Allows to set the number of partition bins that is used for distribution matrix calculation.
        # * \param[in] number_of_bins number of partition bins
        # */
        # void setNumberOfPartitionBins (unsigned int number_of_bins);
        # 
        # /** \brief Returns the nmber of partition bins. */
        # unsigned int getNumberOfPartitionBins () const;
        # 
        # /** \brief This method sets the number of rotations.
        # * \param[in] number_of_rotations number of rotations
        # */
        # void setNumberOfRotations (unsigned int number_of_rotations);
        # 
        # /** \brief returns the number of rotations. */
        # unsigned int getNumberOfRotations () const;
        # 
        # /** \brief Allows to set the support radius that is used to crop the local surface of the point.
        # * \param[in] support_radius support radius
        # */
        # void setSupportRadius (float support_radius);
        # 
        # /** \brief Returns the support radius. */
        # float getSupportRadius () const;
        # 
        # /** \brief This method sets the triangles of the mesh.
        # * \param[in] triangles list of triangles of the mesh
        # */
        # void setTriangles (const std::vector <pcl::Vertices>& triangles);
        # 
        # /** \brief Returns the triangles of the mesh.
        # * \param[out] triangles triangles of tthe mesh
        # */
        # void getTriangles (std::vector <pcl::Vertices>& triangles) const;


###

# rsd.h
# namespace pcl
#   /** \brief Transform a list of 2D matrices into a point cloud containing the values in a vector (Histogram<N>).
#     * Can be used to transform the 2D histograms obtained in \ref RSDEstimation into a point cloud.
#     * @note The template paramter N should be (greater or) equal to the product of the number of rows and columns.
#     * \param[in] histograms2D the list of neighborhood 2D histograms
#     * \param[out] histogramsPC the dataset containing the linearized matrices
#     * \ingroup features
#     */
#   template <int N> void getFeaturePointCloud (const std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > &histograms2D, PointCloud<Histogram<N> > &histogramsPC)
# 
#   /** \brief Estimate the Radius-based Surface Descriptor (RSD) for a given point based on its spatial neighborhood of 3D points with normals
#     * \param[in] surface the dataset containing the XYZ points
#     * \param[in] normals the dataset containing the surface normals at each point in the dataset
#     * \param[in] indices the neighborhood point indices in the dataset (first point is used as the reference)
#     * \param[in] max_dist the upper bound for the considered distance interval
#     * \param[in] nr_subdiv the number of subdivisions for the considered distance interval
#     * \param[in] plane_radius maximum radius, above which everything can be considered planar
#     * \param[in] radii the output point of a type that should have r_min and r_max fields
#     * \param[in] compute_histogram if not false, the full neighborhood histogram is provided, usable as a point signature
#     * \ingroup features
#     */
#   template <typename PointInT, typename PointNT, typename PointOutT> Eigen::MatrixXf
#   computeRSD (boost::shared_ptr<const pcl::PointCloud<PointInT> > &surface, boost::shared_ptr<const pcl::PointCloud<PointNT> > &normals,
#              const std::vector<int> &indices, double max_dist,
#              int nr_subdiv, double plane_radius, PointOutT &radii, bool compute_histogram = false);
# 
#   /** \brief Estimate the Radius-based Surface Descriptor (RSD) for a given point based on its spatial neighborhood of 3D points with normals
#     * \param[in] normals the dataset containing the surface normals at each point in the dataset
#     * \param[in] indices the neighborhood point indices in the dataset (first point is used as the reference)
#     * \param[in] sqr_dists the squared distances from the first to all points in the neighborhood
#     * \param[in] max_dist the upper bound for the considered distance interval
#     * \param[in] nr_subdiv the number of subdivisions for the considered distance interval
#     * \param[in] plane_radius maximum radius, above which everything can be considered planar
#     * \param[in] radii the output point of a type that should have r_min and r_max fields
#     * \param[in] compute_histogram if not false, the full neighborhood histogram is provided, usable as a point signature
#     * \ingroup features
#     */
#   template <typename PointNT, typename PointOutT> Eigen::MatrixXf
#   computeRSD (boost::shared_ptr<const pcl::PointCloud<PointNT> > &normals,
#              const std::vector<int> &indices, const std::vector<float> &sqr_dists, double max_dist,
#              int nr_subdiv, double plane_radius, PointOutT &radii, bool compute_histogram = false);
# 
##
# rsd.h
# namespace pcl
# /** \brief @b RSDEstimation estimates the Radius-based Surface Descriptor (minimal and maximal radius of the local surface's curves)
# * for a given point cloud dataset containing points and normals.
# *
# * @note If you use this code in any academic work, please cite:
# *
# * <ul>
# * <li> Z.C. Marton , D. Pangercic , N. Blodow , J. Kleinehellefort, M. Beetz
# *      General 3D Modelling of Novel Objects from a Single View
# *      In Proceedings of the 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
# *      Taipei, Taiwan, October 18-22, 2010
# * </li>
# * <li> Z.C. Marton, D. Pangercic, N. Blodow, Michael Beetz.
# *      Combined 2D-3D Categorization and Classification for Multimodal Perception Systems.
# *      In The International Journal of Robotics Research, Sage Publications
# *      pages 1378--1402, Volume 30, Number 11, September 2011.
# * </li>
# * </ul>
# *
# * @note The code is stateful as we do not expect this class to be multicore parallelized.
# * \author Zoltan-Csaba Marton
# * \ingroup features
# */
# template <typename PointInT, typename PointNT, typename PointOutT>
# class RSDEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
# Note : Travis CI error (not found rsd.h)
# cdef extern from "pcl/features/rsd.h" namespace "pcl":
#     cdef cppclass RSDEstimation[In, NT, Out](FeatureFromNormals[In, NT, Out]):
#         RSDEstimation()
        # public:
        # using Feature<PointInT, PointOutT>::feature_name_;
        # using Feature<PointInT, PointOutT>::getClassName;
        # using Feature<PointInT, PointOutT>::indices_;
        # using Feature<PointInT, PointOutT>::search_radius_;
        # using Feature<PointInT, PointOutT>::search_parameter_;
        # using Feature<PointInT, PointOutT>::surface_;
        # using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        # typedef typename Feature<PointInT, PointOutT>::PointCloudIn  PointCloudIn;
        # typedef typename boost::shared_ptr<RSDEstimation<PointInT, PointNT, PointOutT> > Ptr;
        # typedef typename boost::shared_ptr<const RSDEstimation<PointInT, PointNT, PointOutT> > ConstPtr;
        # 
        # /** \brief Empty constructor. */
        # RSDEstimation () : nr_subdiv_ (5), plane_radius_ (0.2), save_histograms_ (false)
        # 
        # /** \brief Set the number of subdivisions for the considered distance interval.
        #  * \param[in] nr_subdiv the number of subdivisions
        #  */
        # inline void setNrSubdivisions (int nr_subdiv)
        # 
        # /** \brief Get the number of subdivisions for the considered distance interval. */
        # inline int getNrSubdivisions () const
        # 
        # /** \brief Set the maximum radius, above which everything can be considered planar.
        # * \note the order of magnitude should be around 10-20 times the search radius (0.2 works well for typical datasets).
        # * \note on accurate 3D data (e.g. openni sernsors) a search radius as low as 0.01 still gives good results.
        # * \param[in] plane_radius the new plane radius
        # */
        # inline void setPlaneRadius (double plane_radius)
        # 
        # /** \brief Get the maximum radius, above which everything can be considered planar. */
        # inline double getPlaneRadius () const
        # 
        # /** \brief Disables the setting of the number of k nearest neighbors to use for the feature estimation. */
        # inline void setKSearch (int) 
        # 
        # /** \brief Set whether the full distance-angle histograms should be saved.
        # * @note Obtain the list of histograms by getHistograms ()
        # * \param[in] save set to true if histograms should be saved
        # */
        # inline void setSaveHistograms (bool save)
        # 
        # /** \brief Returns whether the full distance-angle histograms are being saved. */
        # inline bool getSaveHistograms () const
        # 
        # /** \brief Returns a pointer to the list of full distance-angle histograms for all points. */
        # inline boost::shared_ptr<std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > > getHistograms () const { return (histograms_); }


###

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
#         # void compute (PointCloudOut &);


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
        # bool 
        # computePairFeatures (const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals, 
        #                    int p_idx, int q_idx, float &f1, float &f2, float &f3, float &f4);
        # * \brief Estimate the SPFH (Simple Point Feature Histograms) individual signatures of the three angular
        # * (f1, f2, f3) features for a given point based on its spatial neighborhood of 3D points with normals
        # * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
        # * \param[in] normals the dataset containing the surface normals at each point in \a cloud
        # * \param[in] p_idx the index of the query point (source)
        # * \param[in] row the index row in feature histogramms
        # * \param[in] indices the k-neighborhood point indices in the dataset
        # * \param[out] hist_f1 the resultant SPFH histogram for feature f1
        # * \param[out] hist_f2 the resultant SPFH histogram for feature f2
        # * \param[out] hist_f3 the resultant SPFH histogram for feature f3
        # void 
        # computePointSPFHSignature (const pcl::PointCloud<PointInT> &cloud, 
        #                          const pcl::PointCloud<PointNT> &normals, int p_idx, int row, 
        #                          const std::vector<int> &indices, 
        #                          Eigen::MatrixXf &hist_f1, Eigen::MatrixXf &hist_f2, Eigen::MatrixXf &hist_f3);
        # * \brief Weight the SPFH (Simple Point Feature Histograms) individual histograms to create the final FPFH
        # * (Fast Point Feature Histogram) for a given point based on its 3D spatial neighborhood
        # * \param[in] hist_f1 the histogram feature vector of \a f1 values over the given patch
        # * \param[in] hist_f2 the histogram feature vector of \a f2 values over the given patch
        # * \param[in] hist_f3 the histogram feature vector of \a f3 values over the given patch
        # * \param[in] indices the point indices of p_idx's k-neighborhood in the point cloud
        # * \param[in] dists the distances from p_idx to all its k-neighbors
        # * \param[out] fpfh_histogram the resultant FPFH histogram representing the feature at the query point
        # void 
        # weightPointSPFHSignature (const Eigen::MatrixXf &hist_f1, 
        #                         const Eigen::MatrixXf &hist_f2, 
        #                         const Eigen::MatrixXf &hist_f3, 
        #                         const std::vector<int> &indices, 
        #                         const std::vector<float> &dists, 
        #                         Eigen::VectorXf &fpfh_histogram);
        # * \brief Set the number of subdivisions for each angular feature interval.
        # * \param[in] nr_bins_f1 number of subdivisions for the first angular feature
        # * \param[in] nr_bins_f2 number of subdivisions for the second angular feature
        # * \param[in] nr_bins_f3 number of subdivisions for the third angular feature
        inline void setNrSubdivisions (int , int , int )
         # * \brief Get the number of subdivisions for each angular feature interval. 
        # * \param[out] nr_bins_f1 number of subdivisions for the first angular feature
        # * \param[out] nr_bins_f2 number of subdivisions for the second angular feature
        # * \param[out] nr_bins_f3 number of subdivisions for the third angular feature
        inline void getNrSubdivisions (int &, int &, int &)
###

# template <typename PointInT, typename PointNT>
# class FPFHEstimation<PointInT, PointNT, Eigen::MatrixXf> : public FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>
# cdef extern from "pcl/features/feature.h" namespace "pcl":
#     cdef cppclass FPFHEstimation[T, NT]:
#         FPFHEstimation()
#         # public:
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::k_;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::nr_bins_f1_;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::nr_bins_f2_;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::nr_bins_f3_;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::hist_f1_;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::hist_f2_;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::hist_f3_;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::indices_;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::search_parameter_;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::input_;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::compute;
#         # using FPFHEstimation<PointInT, PointNT, pcl::FPFHSignature33>::fpfh_histogram_;

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

