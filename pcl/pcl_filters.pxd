from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

from boost_shared_ptr cimport shared_ptr

# ###
# # approximate_voxel_grid.h
# 
# # template <typename PointT>
# # struct xNdCopyEigenPointFunctor
# cdef extern from "pcl/filters/approximate_voxel_grid.h" namespace "pcl":
#     cdef struct xNdCopyEigenPointFunctor[T]:
#         # ctypedef typename traits::POD<PointT>::type Pod;
#         # xNdCopyEigenPointFunctor (const Eigen::VectorXf &p1, PointT &p2)
#         # template<typename Key> inline void operator() ()
# 
# # template <typename PointT>
# # struct xNdCopyPointEigenFunctor
# cdef extern from "pcl/filters/approximate_voxel_grid.h" namespace "pcl":
#     cdef struct xNdCopyPointEigenFunctor[T]:
#         # ctypedef typename traits::POD<PointT>::type Pod;
#         # xNdCopyPointEigenFunctor (const PointT &p1, Eigen::VectorXf &p2)
#         # template<typename Key> inline void operator() ()
# 
# # template <typename PointT>
# # class ApproximateVoxelGrid: public Filter<PointT>
# cdef extern from "pcl/filters/approximate_voxel_grid.h" namespace "pcl":
#     cdef cppclass ApproximateVoxelGrid[T]:
#         ApproximateVoxelGrid()
#         # ApproximateVoxelGrid (const ApproximateVoxelGrid &src) : 
#         # inline ApproximateVoxelGrid& operator = (const ApproximateVoxelGrid &src)
#         # using Filter<PointT>::filter_name_;
#         # using Filter<PointT>::getClassName;
#         # using Filter<PointT>::input_;
#         # using Filter<PointT>::indices_;
#         # ctypedef typename Filter<PointT>::PointCloud PointCloud;
#         # ctypedef typename PointCloud::Ptr PointCloudPtr;
#         # ctypedef typename PointCloud::ConstPtr PointCloudConstPtr;
# 
#         # public:
#         # * \brief Set the voxel grid leaf size.
#         # * \param[in] leaf_size the voxel grid leaf size
#         inline void setLeafSize (const Eigen::Vector3f &leaf_size) 
# 
#         # * \brief Set the voxel grid leaf size.
#         # * \param[in] lx the leaf size for X
#         # * \param[in] ly the leaf size for Y
#         # * \param[in] lz the leaf size for Z
#         # */
#         inline void setLeafSize (float lx, float ly, float lz)
# 
#         # /** \brief Get the voxel grid leaf size. */
#         # inline Eigen::Vector3f 
#         # getLeafSize () const { return (leaf_size_); }
# 
#         # * \brief Set to true if all fields need to be downsampled, or false if just XYZ.
#         # * \param downsample the new value (true/false)
#         # */
#         inline void setDownsampleAllData (bool downsample)
# 
#         # * \brief Get the state of the internal downsampling parameter (true if
#         #   * all fields need to be downsampled, false if just XYZ). 
#         #   */
#         inline bool getDownsampleAllData () const
# 
# ###
# 
# # bilateral.h
# # template<typename PointT>
# # class BilateralFilter : public Filter<PointT>
# cdef extern from "pcl/filters/bilateral.h" namespace "pcl":
#     cdef cppclass BilateralFilter[T]:
#         BilateralFilter()
# 
#         # using Filter<PointT>::input_;
#         # using Filter<PointT>::indices_;
#         # ctypedef typename Filter<PointT>::PointCloud PointCloud;
#         # ctypedef typename pcl::search::Search<PointT>::Ptr KdTreePtr;
# 
#         # public:
#         # * \brief Filter the input data and store the results into output
#         #   * \param[out] output the resultant point cloud message
#         #   */
#         void applyFilter (PointCloud &output);
# 
#         # * \brief Compute the intensity average for a single point
#         # * \param[in] pid the point index to compute the weight for
#         # * \param[in] indices the set of nearest neighor indices 
#         # * \param[in] distances the set of nearest neighbor distances
#         # * \return the intensity average at a given point index
#         double  computePointWeight (const int pid, const std::vector<int> &indices, const std::vector<float> &distances);
# 
#         # * \brief Set the half size of the Gaussian bilateral filter window.
#         # * \param[in] sigma_s the half size of the Gaussian bilateral filter window to use
#         # */
#         inline void setHalfSize (const double sigma_s)
# 
#         # * \brief Get the half size of the Gaussian bilateral filter window as set by the user. */
#         double getHalfSize ()
# 
#         #  \brief Set the standard deviation parameter
#         #   * \param[in] sigma_r the new standard deviation parameter
#         #   */
#         void setStdDev (const double sigma_r)
# 
#         # * \brief Get the value of the current standard deviation parameter of the bilateral filter. */
#         double getStdDev ()
# 
#         # * \brief Provide a pointer to the search object.
#         # * \param[in] tree a pointer to the spatial search object.
#         # */
#         void setSearchMethod (const KdTreePtr &tree)
# 
# ###
# # clipper3D.h
# # Override class
# 
# ###
# # conditional_removal.h
# 
# # cdef extern from "pcl/filters/bilateral.h" namespace "pcl::ComparisonOps":
# #   typedef enum
# #   {
# #           GT, GE, LT, LE, EQ
# #   } CompareOp;
# 
# cdef extern from "pcl/filters/conditional_removal.h" namespace "pcl":
#     cdef cppclass PointDataAtOffset[T]:
#         PointDataAtOffset (uint8_t datatype, uint32_t offset)
#         int compare (const PointT& p, const double& val);
# 
#         # template<typename PointT>
#         # class ComparisonBase
#         cdef cppclass ComparisonBase[T]:
#             ComparisonBase()
#             # public:
#             # ctypedef boost::shared_ptr<ComparisonBase<PointT> > Ptr;
#             # ctypedef boost::shared_ptr<const ComparisonBase<PointT> > ConstPtr;
# 
#             # /** \brief Return if the comparison is capable. */
#             inline bool isCapable () const
# 
#             # /** \brief Evaluate function. */
#             # virtual bool evaluate (const PointT &point) const = 0;
# 
# #  template<typename PointT>
# #  class FieldComparison : public ComparisonBase<PointT>
# cdef extern from "pcl/filters/conditional_removal.h" namespace "pcl":
#     cdef cppclass FieldComparison[T]:
#         # FieldComparison (string field_name, CompareOp op, double compare_val)
#         # FieldComparison (const FieldComparison &src) :
#         # inline FieldComparison& operator = (const FieldComparison &src)
#         # using ComparisonBase<PointT>::field_name_;
#         # using ComparisonBase<PointT>::op_;
#         # using ComparisonBase<PointT>::capable_;
# 
#         # public:
#         #   typedef boost::shared_ptr<FieldComparison<PointT> > Ptr;
#         #   typedef boost::shared_ptr<const FieldComparison<PointT> > ConstPtr;
# 
# # template<typename PointT>
# # class PackedRGBComparison : public ComparisonBase<PointT>
# cdef extern from "pcl/filters/conditional_removal.h" namespace "pcl":
#     cdef cppclass PackedRGBComparison[T]:
#         # PackedRGBComparison (string component_name, CompareOp op, double compare_val)
#         # using ComparisonBase<PointT>::capable_;
#         # using ComparisonBase<PointT>::op_;
# 
#       # virtual boolevaluate (const PointT &point) const;
# 
# # template<typename PointT>
# # class PackedHSIComparison : public ComparisonBase<PointT>
# cdef extern from "pcl/filters/conditional_removal.h" namespace "pcl":
#     cdef cppclass PackedHSIComparison[T]:
#         PackedHSIComparison (string component_name, CompareOp op, double compare_val)
#         # using ComparisonBase<PointT>::capable_;
#         # using ComparisonBase<PointT>::op_;
# 
#         # public:
#         # * \brief Construct a PackedHSIComparison 
#         # * \param component_name either "h", "s" or "i"
#         # * \param op the operator to use when making the comparison
#         # * \param compare_val the constant value to compare the component value too
# 
#         # typedef enum
#         # {
#         #   H, // -128 to 127 corresponds to -pi to pi
#         #   S, // 0 to 255
#         #   I  // 0 to 255
#         # } ComponentId;
# 
# # template<typename PointT>
# # class TfQuadraticXYZComparison : public pcl::ComparisonBase<PointT>
# cdef extern from "pcl/filters/conditional_removal.h" namespace "pcl":
#     cdef cppclass TfQuadraticXYZComparison[T]:
#         TfQuadraticXYZComparison ()
#         # * \param op the operator "[OP]" of the comparison "p'Ap + 2v'p + c [OP] 0".
#         # * \param comparison_matrix the matrix "A" of the comparison "p'Ap + 2v'p + c [OP] 0".
#         # * \param comparison_vector the vector "v" of the comparison "p'Ap + 2v'p + c [OP] 0".
#         # * \param comparison_scalar the scalar "c" of the comparison "p'Ap + 2v'p + c [OP] 0".
#         # * \param comparison_transform the transformation of the comparison.
#         # TfQuadraticXYZComparison (const pcl::ComparisonOps::CompareOp op, const Eigen::Matrix3f &comparison_matrix,
#         #                         const Eigen::Vector3f &comparison_vector, const float &comparison_scalar,
#         #                         const Eigen::Affine3f &comparison_transform = Eigen::Affine3f::Identity ());
#         # public:
#         # EIGEN_MAKE_ALIGNED_OPERATOR_NEW     //needed whenever there is a fixed size Eigen:: vector or matrix in a class
#         # ctypedef boost::shared_ptr<TfQuadraticXYZComparison<PointT> > Ptr;
#         # typedef boost::shared_ptr<const TfQuadraticXYZComparison<PointT> > ConstPtr;
# 
#         inline void setComparisonOperator (const pcl::ComparisonOps::CompareOp op)
# 
#         # * \brief set the matrix "A" of the comparison "p'Ap + 2v'p + c [OP] 0".
#         #  */
#         inline void setComparisonMatrix (const Eigen::Matrix3f &matrix)
# 
#         # * \brief set the matrix "A" of the comparison "p'Ap + 2v'p + c [OP] 0".
#         inline void setComparisonMatrix (const Eigen::Matrix4f &homogeneousMatrix)
# 
#         # * \brief set the vector "v" of the comparison "p'Ap + 2v'p + c [OP] 0".
#         inline void setComparisonVector (const Eigen::Vector3f &vector)
# 
#         # * \brief set the vector "v" of the comparison "p'Ap + 2v'p + c [OP] 0".
#         inline void setComparisonVector (const Eigen::Vector4f &homogeneousVector)
# 
#         # * \brief set the scalar "c" of the comparison "p'Ap + 2v'p + c [OP] 0".
#         inline void setComparisonScalar (const float &scalar)
# 
#         # * \brief transform the coordinate system of the comparison. If you think of
#         # * the transformation to be a translation and rotation of the comparison in the
#         # * same coordinate system, you have to provide the inverse transformation.
#         # * This function does not change the original definition of the comparison. Thus,
#         # * each call of this function will assume the original definition of the comparison
#         # * as starting point for the transformation.
#         # *
#         # * @param transform the transformation (rotation and translation) as an affine matrix.
#         # inline void transformComparison (const Eigen::Matrix4f &transform)
# 
#         # * \brief transform the coordinate system of the comparison. If you think of
#         # * the transformation to be a translation and rotation of the comparison in the
#         # * same coordinate system, you have to provide the inverse transformation.
#         # * This function does not change the original definition of the comparison. Thus,
#         # * each call of this function will assume the original definition of the comparison
#         # * as starting point for the transformation.
#         # *
#         # * @param transform the transformation (rotation and translation) as an affine matrix.
#         # inline void transformComparison (const Eigen::Affine3f &transform)
# 
#         # * \brief Determine the result of this comparison.
#         #  \param point the point to evaluate
#         #  \return the result of this comparison.
#         #
#         # virtual bool evaluate (const PointT &point) const;
# 
# 
# cdef extern from "pcl/filters/conditional_removal.h" namespace "pcl":
#     cdef cppclass ConditionBase[T]:
#         ConditionBase ()
#         # public:
#         # ctypedef typename pcl::ComparisonBase<PointT> ComparisonBase;
#         # ctypedef typename ComparisonBase::Ptr ComparisonBasePtr;
#         # ctypedef typename ComparisonBase::ConstPtr ComparisonBaseConstPtr;
#         # ctypedef boost::shared_ptr<ConditionBase<PointT> > Ptr;
#         # ctypedef boost::shared_ptr<const ConditionBase<PointT> > ConstPtr;
# 
#         void addComparison (ComparisonBaseConstPtr comparison);
#         void addCondition (Ptr condition);
#         inline bool isCapable () const
# 
# # template<typename PointT>
# # class ConditionAnd : public ConditionBase<PointT>
# cdef extern from "pcl/filters/conditional_removal.h" namespace "pcl":
#     cdef cppclass ConditionAnd[T]:
#         # using ConditionBase<PointT>::conditions_;
#         # using ConditionBase<PointT>::comparisons_;
# 
#         # public:
#         # ctypedef boost::shared_ptr<ConditionAnd<PointT> > Ptr;
#         # ctypedef boost::shared_ptr<const ConditionAnd<PointT> > ConstPtr;
# 
# # template<typename PointT>
# # class ConditionOr : public ConditionBase<PointT>
# cdef extern from "pcl/filters/conditional_removal.h" namespace "pcl":
#     cdef cppclass ConditionOr[T]:
#         # using ConditionBase<PointT>::conditions_;
#         # using ConditionBase<PointT>::comparisons_;
# 
#         # public:
#         # ctypedef boost::shared_ptr<ConditionOr<PointT> > Ptr;
#         # ctypedef boost::shared_ptr<const ConditionOr<PointT> > ConstPtr;
# 
# # template<typename PointT>
# # class ConditionalRemoval : public Filter<PointT>
# cdef extern from "pcl/filters/conditional_removal.h" namespace "pcl":
#     cdef cppclass ConditionalRemoval[T]:
#         ConditionalRemoval()
#         # ConditionalRemoval(int)
#         # ConditionalRemoval (ConditionBasePtr condition, bool extract_removed_indices = false) :
#         # using Filter<PointT>::input_;
#         # using Filter<PointT>::filter_name_;
#         # using Filter<PointT>::getClassName;
#         # using Filter<PointT>::removed_indices_;
#         # using Filter<PointT>::extract_removed_indices_;
#         # ctypedef typename Filter<PointT>::PointCloud PointCloud;
#         # ctypedef typename PointCloud::Ptr PointCloudPtr;
#         # ctypedef typename PointCloud::ConstPtr PointCloudConstPtr;
#         # public:
#         # ctypedef typename pcl::ConditionBase<PointT> ConditionBase;
#         # ctypedef typename ConditionBase::Ptr ConditionBasePtr;
#         # ctypedef typename ConditionBase::ConstPtr ConditionBaseConstPtr;
#         inline void setKeepOrganized (bool val)
#         inline bool getKeepOrganized () const
#         inline void setUserFilterValue (float val)
#         void setCondition (ConditionBasePtr condition);
# 
# ###
# # crop_box.h
# 
# # template<typename PointT>
# # class CropBox : public FilterIndices<PointT>
# cdef extern from "pcl/filters/crop_box.h" namespace "pcl":
#     cdef cppclass CropBox[T]:
#         CropBox()
#     
#         # using Filter<PointT>::filter_name_;
#         # using Filter<PointT>::getClassName;
#         # using Filter<PointT>::indices_;
#         # using Filter<PointT>::input_;
# 
#         # ctypedef typename Filter<PointT>::PointCloud PointCloud;
#         # ctypedef typename PointCloud::Ptr PointCloudPtr;
#         # ctypedef typename PointCloud::ConstPtr PointCloudConstPtr;
# 
#         # public:
# 
#         # * \brief Set the minimum point of the box
#         # * \param[in] min_pt the minimum point of the box
#         # */
#         inline void setMin (const Eigen::Vector4f &min_pt)
#       
#         # * \brief Get the value of the minimum point of the box, as set by the user
#         # * * \return the value of the internal \a min_pt parameter.
#         # * */
#         inline Eigen::Vector4f getMin () const
# 
#         # * \brief Set the maximum point of the box
#         # * \param[in] max_pt the maximum point of the box
#         inline void setMax (const Eigen::Vector4f &max_pt)
# 
#         # \brief Get the value of the maxiomum point of the box, as set by the user
#         # \return the value of the internal \a max_pt parameter.
#         inline Eigen::Vector4f getMax () const
# 
#         # \brief Set a translation value for the box
#         # \param[in] translation the (tx,ty,tz) values that the box should be translated by
#         inline void setTranslation (const Eigen::Vector3f &translation)
# 
#         # \brief Get the value of the box translation parameter as set by the user. */
#         Eigen::Vector3f getTranslation () const
# 
#         # \brief Set a rotation value for the box
#         # \param[in] rotation the (rx,ry,rz) values that the box should be rotated by
#         inline void setRotation (const Eigen::Vector3f &rotation)
# 
#         # \brief Get the value of the box rotatation parameter, as set by the user. */
#         inline Eigen::Vector3f getRotation () const
# 
#         # \brief Set a transformation that should be applied to the cloud before filtering
#         # \param[in] transform an affine transformation that needs to be applied to the cloud before filtering
#         inline void setTransform (const Eigen::Affine3f &transform)
# 
#         # \brief Get the value of the transformation parameter, as set by the user. */
#         inline Eigen::Affine3f getTransform () const
# 
# #  template<>
# #  class PCL_EXPORTS CropBox<sensor_msgs::PointCloud2> : public FilterIndices<sensor_msgs::PointCloud2>
# #  {
# #    using Filter<sensor_msgs::PointCloud2>::filter_name_;
# #    using Filter<sensor_msgs::PointCloud2>::getClassName;
# #
# #    typedef sensor_msgs::PointCloud2 PointCloud2;
# #    typedef PointCloud2::Ptr PointCloud2Ptr;
# #    typedef PointCloud2::ConstPtr PointCloud2ConstPtr;
# #
# #    public:
# #    /** \brief Empty constructor. */
# #      CropBox () :
# #        min_pt_(Eigen::Vector4f (-1, -1, -1, 1)),
# #        max_pt_(Eigen::Vector4f (1, 1, 1, 1)),
# #        translation_ (Eigen::Vector3f::Zero ()),
# #        rotation_ (Eigen::Vector3f::Zero ()),
# #        transform_(Eigen::Affine3f::Identity ())
# #      {
# #        filter_name_ = "CropBox";
# #      }
# #
# #      /** \brief Set the minimum point of the box
# #        * \param[in] min_pt the minimum point of the box
# #        */
# #      inline void
# #      setMin (const Eigen::Vector4f& min_pt)
# #      {
# #        min_pt_ = min_pt;
# #      }
# #
# #      /** \brief Get the value of the minimum point of the box, as set by the user
# #        * \return the value of the internal \a min_pt parameter.
# #        */
# #      inline Eigen::Vector4f
# #      getMin () const
# #      {
# #        return (min_pt_);
# #      }
# #
# #      /** \brief Set the maximum point of the box
# #        * \param[in] max_pt the maximum point of the box
# #        */
# #      inline void
# #      setMax (const Eigen::Vector4f &max_pt)
# #      {
# #        max_pt_ = max_pt;
# #      }
# #
# #      /** \brief Get the value of the maxiomum point of the box, as set by the user
# #        * \return the value of the internal \a max_pt parameter.
# #        */
# #      inline Eigen::Vector4f
# #      getMax () const
# #      {
# #        return (max_pt_);
# #      }
# #
# #      /** \brief Set a translation value for the box
# #        * \param[in] translation the (tx,ty,tz) values that the box should be translated by
# #        */
# #      inline void
# #      setTranslation (const Eigen::Vector3f &translation)
# #      {
# #        translation_ = translation;
# #      }
# #
# #      /** \brief Get the value of the box translation parameter as set by the user. */
# #      inline Eigen::Vector3f
# #      getTranslation () const
# #      {
# #        return (translation_);
# #      }
# #
# #      /** \brief Set a rotation value for the box
# #        * \param[in] rotation the (rx,ry,rz) values that the box should be rotated by
# #        */
# #      inline void
# #      setRotation (const Eigen::Vector3f &rotation)
# #      {
# #        rotation_ = rotation;
# #      }
# #
# #      /** \brief Get the value of the box rotatation parameter, as set by the user. */
# #      inline Eigen::Vector3f
# #      getRotation () const
# #      {
# #        return (rotation_);
# #      }
# #
# #      /** \brief Set a transformation that should be applied to the cloud before filtering
# #        * \param[in] transform an affine transformation that needs to be applied to the cloud before filtering
# #        */
# #      inline void
# #      setTransform (const Eigen::Affine3f &transform)
# #      {
# #        transform_ = transform;
# #      }
# #
# #      /** \brief Get the value of the transformation parameter, as set by the user. */
# #      inline Eigen::Affine3f
# #      getTransform () const
# #      {
# #        return (transform_);
# #      }
# #
# 
# ###
# # crop_hull.h
# 
# #  template<typename PointT>
# #  class CropHull: public FilterIndices<PointT>
# cdef extern from "pcl/filters/crop_hull.h" namespace "pcl":
#     cdef cppclass CropHull[T]:
#         CropHull()
# 
#         # using Filter<PointT>::filter_name_;
#         # using Filter<PointT>::indices_;
#         # using Filter<PointT>::input_;
# 
#         # ctypedef typename Filter<PointT>::PointCloud PointCloud;
#         # ctypedef typename PointCloud::Ptr PointCloudPtr;
#         # ctypedef typename PointCloud::ConstPtr PointCloudConstPtr;
# 
#         # /** \brief Set the vertices of the hull used to filter points.
#         #  * \param[in] polygons Vector of polygons (Vertices structures) forming
#         #  * the hull used for filtering points.
#         #  */
#         inline void setHullIndices (const std::vector<Vertices>& polygons)
# 
#         # \brief Get the vertices of the hull used to filter points.
#         vector[Vertices] getHullIndices () const
#       
#         #   /** \brief Set the point cloud that the hull indices refer to
#         #  * \param[in] points the point cloud that the hull indices refer to
#         #  */
#         inline void setHullCloud (PointCloudPtr points)
# 
#         #/** \brief Get the point cloud that the hull indices refer to. */
#         PointCloudPtr getHullCloud () const
#     
#         #/** \brief Set the dimensionality of the hull to be used.
#         #  * This should be set to correspond to the dimensionality of the
#         #  * convex/concave hull produced by the pcl::ConvexHull and
#         #  * pcl::ConcaveHull classes.
#         #  * \param[in] dim Dimensionailty of the hull used to filter points.
#         #  */
#         inline void setDim (int dim)
# 
#         # /** \brief Remove points outside the hull (default), or those inside the hull.
#         # * \param[in] crop_outside If true, the filter will remove points
#         # * outside the hull. If false, those inside will be removed.
#         # */
#         inline void setCropOutside(bool crop_outside)
# 
# ###
# # extract_indices.h
# 
# # template<typename PointT>
# # class ExtractIndices : public FilterIndices<PointT>
# cdef extern from "pcl/filters/extract_indices.h" namespace "pcl":
#     cdef cppclass ExtractIndices[T]:
#         ExtractIndices()
#         # ctypedef typename FilterIndices<PointT>::PointCloud PointCloud;
#         # ctypedef typename PointCloud::Ptr PointCloudPtr;
#         # ctypedef typename PointCloud::ConstPtr PointCloudConstPtr;
#         # ctypedef typename pcl::traits::fieldList<PointT>::type FieldList;
# 
#         # * \brief Apply the filter and store the results directly in the input cloud.
#         # * \details This method will save the time and memory copy of an output cloud but can not alter the original size of the input cloud:
#         # * It operates as though setKeepOrganized() is true and will overwrite the filtered points instead of remove them.
#         # * All fields of filtered points are replaced with the value set by setUserFilterValue() (default = NaN).
#         # * This method also automatically alters the input cloud set via setInputCloud().
#         # * It does not alter the value of the internal keep organized boolean as set by setKeepOrganized().
#         # * \param[in/out] cloud The point cloud used for input and output.
#         void filterDirectly (PointCloudPtr &cloud);
# 
# #   template<>
# #   class PCL_EXPORTS ExtractIndices<sensor_msgs::PointCloud2> : public FilterIndices<sensor_msgs::PointCloud2>
# #   {
# #     public:
# #       typedef sensor_msgs::PointCloud2 PointCloud2;
# #       typedef PointCloud2::Ptr PointCloud2Ptr;
# #       typedef PointCloud2::ConstPtr PointCloud2ConstPtr;
# # 
# #       /** \brief Empty constructor. */
# #       ExtractIndices ()
# #       {
# #         use_indices_ = true;
# #         filter_name_ = "ExtractIndices";
# #       }
# # 
# #     protected:
# #       using PCLBase<PointCloud2>::input_;
# #       using PCLBase<PointCloud2>::indices_;
# #       using PCLBase<PointCloud2>::use_indices_;
# #       using Filter<PointCloud2>::filter_name_;
# #       using Filter<PointCloud2>::getClassName;
# #       using FilterIndices<PointCloud2>::negative_;
# #       using FilterIndices<PointCloud2>::keep_organized_;
# #       using FilterIndices<PointCloud2>::user_filter_value_;
# # 
# #       /** \brief Extract point indices into a separate PointCloud
# #         * \param[out] output the resultant point cloud
# #         */
# #       void
# #       applyFilter (PointCloud2 &output);
# # 
# #       /** \brief Extract point indices
# #         * \param indices the resultant indices
# #         */
# #       void
# #       applyFilter (std::vector<int> &indices);
# #
# 
# ###
# # filter.h
# 
# # template<typename PointT>
# # class Filter : public PCLBase<PointT>
# cdef extern from "pcl/filters/filter.h" namespace "pcl":
#     cdef cppclass Filter[T]:
#         Filter()
# 
#         # public:
#         # using PCLBase<PointT>::indices_;
#         # using PCLBase<PointT>::input_;
#         # ctypedef boost::shared_ptr< Filter<PointT> > Ptr;
#         # ctypedef boost::shared_ptr< const Filter<PointT> > ConstPtr;
#         # ctypedef pcl::PointCloud<PointT> PointCloud;
#         # ctypedef typename PointCloud::Ptr PointCloudPtr;
#         # ctypedef typename PointCloud::ConstPtr PointCloudConstPtr;
# 
#         # /** \brief Get the point indices being removed */
#         inline IndicesConstPtr const getRemovedIndices ()
# 
#         # /** \brief Calls the filtering method and returns the filtered dataset in output.
#         #   * \param[out] output the resultant filtered point cloud dataset
#         #   */
#         inline void filter (PointCloud &output)
# 
# #   template<>
# #   class PCL_EXPORTS Filter<sensor_msgs::PointCloud2> : public PCLBase<sensor_msgs::PointCloud2>
# #   {
# #     public:
# #       typedef sensor_msgs::PointCloud2 PointCloud2;
# #       typedef PointCloud2::Ptr PointCloud2Ptr;
# #       typedef PointCloud2::ConstPtr PointCloud2ConstPtr;
# # 
# #       /** \brief Empty constructor. 
# #         * \param[in] extract_removed_indices set to true if the filtered data indices should be saved in a 
# #         * separate list. Default: false.
# #         */
# #       Filter (bool extract_removed_indices = false) : 
# #         removed_indices_ (new std::vector<int>),
# #         extract_removed_indices_ (extract_removed_indices),
# #         filter_name_ ()
# #       {
# #       }
# # 
# #       /** \brief Get the point indices being removed */
# #       inline IndicesConstPtr const
# #       getRemovedIndices ()
# #       {
# #         return (removed_indices_);
# #       }
# # 
# #       /** \brief Calls the filtering method and returns the filtered dataset in output.
# #         * \param[out] output the resultant filtered point cloud dataset
# #         */
# #       void
# #       filter (PointCloud2 &output);
# # 
# 
# ###
# # filter_indices.h
# 
# # template<typename PointT>
# # class FilterIndices : public Filter<PointT>
# cdef extern from "pcl/filters/filter_indices.h" namespace "pcl":
#     cdef cppclass FilterIndices[T]:
#         FilterIndices()
#         # public:
#         # ctypedef pcl::PointCloud<PointT> PointCloud;
# 
#         inline void filter (PointCloud &output)
#         # /** \brief Calls the filtering method and returns the filtered point cloud indices.
#         # * \param[out] indices the resultant filtered point cloud indices
#         # */
#         inline void filter (vector[int] &indices)
# 
#         # /** \brief Set whether the regular conditions for points filtering should apply, or the inverted conditions.
#         #   * \param[in] negative false = normal filter behavior (default), true = inverted behavior.
#         #   */
#         inline void setNegative (bool negative)
# 
#         # /** \brief Get whether the regular conditions for points filtering should apply, or the inverted conditions.
#         #   * \return The value of the internal \a negative_ parameter; false = normal filter behavior (default), true = inverted behavior.
#         #   */
#       inline bool getNegative ()
# 
#         # /** \brief Set whether the filtered points should be kept and set to the value given through \a setUserFilterValue (default: NaN),
#         #   * or removed from the PointCloud, thus potentially breaking its organized structure.
#         #   * \param[in] keep_organized false = remove points (default), true = redefine points, keep structure.
#         #   */
#         inline void setKeepOrganized (bool keep_organized)
# 
#         # /** \brief Get whether the filtered points should be kept and set to the value given through \a setUserFilterValue (default = NaN),
#         #   * or removed from the PointCloud, thus potentially breaking its organized structure.
#         #   * \return The value of the internal \a keep_organized_ parameter; false = remove points (default), true = redefine points, keep structure.
#         #   */
#         inline bool getKeepOrganized ()
# 
#       # /** \brief Provide a value that the filtered points should be set to instead of removing them.
#       #   * Used in conjunction with \a setKeepOrganized ().
#       #   * \param[in] value the user given value that the filtered point dimensions should be set to (default = NaN).
#       #   */
#       inline void setUserFilterValue (float value)
# 
#       # /** \brief Get the point indices being removed
#       #   * \return The value of the internal \a negative_ parameter; false = normal filter behavior (default), true = inverted behavior.
#       #   */
#       inline IndicesConstPtr const getRemovedIndices ()
# 
# #   template<>
# #   class PCL_EXPORTS FilterIndices<sensor_msgs::PointCloud2> : public Filter<sensor_msgs::PointCloud2>
# #   {
# #     public:
# #       typedef sensor_msgs::PointCloud2 PointCloud2;
# # 
# #       /** \brief Constructor.
# #         * \param[in] extract_removed_indices Set to true if you want to extract the indices of points being removed (default = false).
# #         */
# #       FilterIndices (bool extract_removed_indices = false) :
# #           negative_ (false), 
# #           keep_organized_ (false), 
# #           extract_removed_indices_ (extract_removed_indices), 
# #           user_filter_value_ (std::numeric_limits<float>::quiet_NaN ()),
# #           removed_indices_ (new std::vector<int>)
# #       {
# #       }
# # 
# #       /** \brief Empty virtual destructor. */
# #       virtual
# #       ~FilterIndices ()
# #       {
# #       }
# # 
# #       virtual void
# #       filter (PointCloud2 &output)
# #       {
# #         pcl::Filter<PointCloud2>::filter (output);
# #       }
# # 
# #       /** \brief Calls the filtering method and returns the filtered point cloud indices.
# #         * \param[out] indices the resultant filtered point cloud indices
# #         */
# #       void
# #       filter (std::vector<int> &indices);
# # 
# #       /** \brief Set whether the regular conditions for points filtering should apply, or the inverted conditions.
# #         * \param[in] negative false = normal filter behavior (default), true = inverted behavior.
# #         */
# #       inline void
# #       setNegative (bool negative)
# #       {
# #         negative_ = negative;
# #       }
# # 
# #       /** \brief Get whether the regular conditions for points filtering should apply, or the inverted conditions.
# #         * \return The value of the internal \a negative_ parameter; false = normal filter behavior (default), true = inverted behavior.
# #         */
# #       inline bool
# #       getNegative ()
# #       {
# #         return (negative_);
# #       }
# # 
# #       /** \brief Set whether the filtered points should be kept and set to the value given through \a setUserFilterValue (default: NaN),
# #         * or removed from the PointCloud, thus potentially breaking its organized structure.
# #         * \param[in] keep_organized false = remove points (default), true = redefine points, keep structure.
# #         */
# #       inline void
# #       setKeepOrganized (bool keep_organized)
# #       {
# #         keep_organized_ = keep_organized;
# #       }
# # 
# #       /** \brief Get whether the filtered points should be kept and set to the value given through \a setUserFilterValue (default = NaN),
# #         * or removed from the PointCloud, thus potentially breaking its organized structure.
# #         * \return The value of the internal \a keep_organized_ parameter; false = remove points (default), true = redefine points, keep structure.
# #         */
# #       inline bool
# #       getKeepOrganized ()
# #       {
# #         return (keep_organized_);
# #       }
# # 
# #       /** \brief Provide a value that the filtered points should be set to instead of removing them.
# #         * Used in conjunction with \a setKeepOrganized ().
# #         * \param[in] value the user given value that the filtered point dimensions should be set to (default = NaN).
# #         */
# #       inline void
# #       setUserFilterValue (float value)
# #       {
# #         user_filter_value_ = value;
# #       }
# # 
# #       /** \brief Get the point indices being removed
# #         * \return The value of the internal \a negative_ parameter; false = normal filter behavior (default), true = inverted behavior.
# #         */
# #       inline IndicesConstPtr const
# #       getRemovedIndices ()
# #       {
# #         return (removed_indices_);
# #       }
# # 
# 
# ###
# # normal_space.h
# 
# # template<typename PointT, typename NormalT>
# # class NormalSpaceSampling : public FilterIndices<PointT>
# cdef extern from "pcl/filters/normal_space.h" namespace "pcl":
#     cdef cppclass NormalSpaceSampling[T]:
#         NormalSpaceSampling()
#         # using FilterIndices<PointT>::filter_name_;
#         # using FilterIndices<PointT>::getClassName;
#         # using FilterIndices<PointT>::indices_;
#         # using FilterIndices<PointT>::input_;
#         # ctypedef typename FilterIndices<PointT>::PointCloud PointCloud;
#         # ctypedef typename PointCloud::Ptr PointCloudPtr;
#         # ctypedef typename PointCloud::ConstPtr PointCloudConstPtr;
#         # ctypedef typename pcl::PointCloud<NormalT>::Ptr NormalsPtr;
# 
#         # /** \brief Set number of indices to be sampled.
#         #   * \param[in] sample the number of sample indices
#         #   */
#         inline void setSample (unsigned int sample)
# 
#         # /** \brief Get the value of the internal \a sample parameter. */
#         inline unsigned int getSample () const
# 
#         #  \brief Set seed of random function.
#         #   * \param[in] seed the input seed
#         #   */
#         inline void setSeed (unsigned int seed)
# 
#         # /** \brief Get the value of the internal \a seed parameter. */
#         inline unsigned int getSeed () const
# 
#         # /** \brief Set the number of bins in x, y and z direction
#         #   * \param[in] binsx number of bins in x direction
#         #   * \param[in] binsy number of bins in y direction
#         #   * \param[in] binsz number of bins in z direction
#         #   */
#         inline void setBins (unsigned int binsx, unsigned int binsy, unsigned int binsz)
# 
#         # /** \brief Get the number of bins in x, y and z direction
#         #   * \param[out] binsx number of bins in x direction
#         #   * \param[out] binsy number of bins in y direction
#         #   * \param[out] binsz number of bins in z direction
#         #   */
#         inline void getBins (unsigned int& binsx, unsigned int& binsy, unsigned int& binsz) const
# 
#         # * \brief Set the normals computed on the input point cloud
#         #   * \param[in] normals the normals computed for the input cloud
#         #   */
#         inline void setNormals (const NormalsPtr &normals)
# 
#         # * \brief Get the normals computed on the input point cloud */
#         inline NormalsPtr getNormals () const

###
# passthrough.h
cdef extern from "pcl/filters/passthrough.h" namespace "pcl":
    cdef cppclass PassThrough[T]:
        PassThrough()
        void setFilterFieldName (string field_name)
        void setFilterLimits (float, float)
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        void filter(cpp.PointCloud[T] c)

ctypedef PassThrough[cpp.PointXYZ] PassThrough_t
ctypedef PassThrough[cpp.PointXYZRGBA] PassThrough2_t

#   template<>
#   class PCL_EXPORTS PassThrough<sensor_msgs::PointCloud2> : public Filter<sensor_msgs::PointCloud2>
#   {
#     typedef sensor_msgs::PointCloud2 PointCloud2;
#     typedef PointCloud2::Ptr PointCloud2Ptr;
#     typedef PointCloud2::ConstPtr PointCloud2ConstPtr;
# 
#     using Filter<sensor_msgs::PointCloud2>::removed_indices_;
#     using Filter<sensor_msgs::PointCloud2>::extract_removed_indices_;
# 
#     public:
#       /** \brief Constructor. */
#       PassThrough (bool extract_removed_indices = false) :
#         Filter<sensor_msgs::PointCloud2>::Filter (extract_removed_indices), keep_organized_ (false),
#         user_filter_value_ (std::numeric_limits<float>::quiet_NaN ()),
#         filter_field_name_ (""), filter_limit_min_ (-FLT_MAX), filter_limit_max_ (FLT_MAX),
#         filter_limit_negative_ (false)
#       {
#         filter_name_ = "PassThrough";
#       }
# 
#       /** \brief Set whether the filtered points should be kept and set to the
#         * value given through \a setUserFilterValue (default: NaN), or removed
#         * from the PointCloud, thus potentially breaking its organized
#         * structure. By default, points are removed.
#         *
#         * \param[in] val set to true whether the filtered points should be kept and
#         * set to a given user value (default: NaN)
#         */
#       inline void
#       setKeepOrganized (bool val)
#       {
#         keep_organized_ = val;
#       }
# 
#       /** \brief Obtain the value of the internal \a keep_organized_ parameter. */
#       inline bool
#       getKeepOrganized ()
#       {
#         return (keep_organized_);
#       }
# 
#       /** \brief Provide a value that the filtered points should be set to
#         * instead of removing them.  Used in conjunction with \a
#         * setKeepOrganized ().
#         * \param[in] val the user given value that the filtered point dimensions should be set to
#         */
#       inline void
#       setUserFilterValue (float val)
#       {
#         user_filter_value_ = val;
#       }
# 
#       /** \brief Provide the name of the field to be used for filtering data. In conjunction with  \a setFilterLimits,
#         * points having values outside this interval will be discarded.
#         * \param[in] field_name the name of the field that contains values used for filtering
#         */
#       inline void
#       setFilterFieldName (const std::string &field_name)
#       {
#         filter_field_name_ = field_name;
#       }
# 
#       /** \brief Get the name of the field used for filtering. */
#       inline std::string const
#       getFilterFieldName ()
#       {
#         return (filter_field_name_);
#       }
# 
#       /** \brief Set the field filter limits. All points having field values outside this interval will be discarded.
#         * \param[in] limit_min the minimum allowed field value
#         * \param[in] limit_max the maximum allowed field value
#         */
#       inline void
#       setFilterLimits (const double &limit_min, const double &limit_max)
#       {
#         filter_limit_min_ = limit_min;
#         filter_limit_max_ = limit_max;
#       }
# 
#       /** \brief Get the field filter limits (min/max) set by the user. The default values are -FLT_MAX, FLT_MAX. 
#         * \param[out] limit_min the minimum allowed field value
#         * \param[out] limit_max the maximum allowed field value
#         */
#       inline void
#       getFilterLimits (double &limit_min, double &limit_max)
#       {
#         limit_min = filter_limit_min_;
#         limit_max = filter_limit_max_;
#       }
# 
#       /** \brief Set to true if we want to return the data outside the interval specified by setFilterLimits (min, max).
#         * Default: false.
#         * \param[in] limit_negative return data inside the interval (false) or outside (true)
#         */
#       inline void
#       setFilterLimitsNegative (const bool limit_negative)
#       {
#         filter_limit_negative_ = limit_negative;
#       }
# 
#       /** \brief Get whether the data outside the interval (min/max) is to be returned (true) or inside (false). 
#         * \param[out] limit_negative true if data \b outside the interval [min; max] is to be returned, false otherwise
#         */
#       inline void
#       getFilterLimitsNegative (bool &limit_negative)
#       {
#         limit_negative = filter_limit_negative_;
#       }
# 
#       /** \brief Get whether the data outside the interval (min/max) is to be returned (true) or inside (false). 
#         * \return true if data \b outside the interval [min; max] is to be returned, false otherwise
#         */
#       inline bool
#       getFilterLimitsNegative ()
#       {
#         return (filter_limit_negative_);
#       }
# 

# ###
# # plane_clipper3D.h
# 
# # template<typename PointT>
# # class PlaneClipper3D : public Clipper3D<PointT>
# cdef extern from "pcl/filters/plane_clipper3D.h" namespace "pcl":
#     cdef cppclass PlaneClipper3D[T]:
#         # PlaneClipper3D (const Eigen::Vector4f& plane_params);
#         # public:
#         # /**
#         #  * @author Suat Gedikli <gedikli@willowgarage.com>
#         #  * @brief Constructor taking the homogeneous representation of the plane as a Eigen::Vector4f
#         #  * @param[in] plane_params plane parameters, need not necessarily be normalized
#         #  */
# 
#         # /**
#         #   * \brief Set new plane parameters
#         #   * \param plane_params
#         #   */
#         # void setPlaneParameters (const Eigen::Vector4f& plane_params);
# 
#         # /**
#         #   * \brief return the current plane parameters
#         #   * \return the current plane parameters
#         #   */
#         # const Eigen::Vector4f& getPlaneParameters () const;
# 
#         # virtual bool clipPoint3D (const PointT& point) const;
# 
#         # virtual bool clipLineSegment3D (PointT& from, PointT& to) const;
# 
#         # virtual void clipPlanarPolygon3D (const std::vector<PointT>& polygon, std::vector<PointT>& clipped_polygon) const;
# 
#         # virtual void
#         # clipPointCloud3D (const pcl::PointCloud<PointT> &cloud_in, std::vector<int>& clipped, const std::vector<int>& indices = std::vector<int> ()) const;
# 
#         # virtual Clipper3D<PointT>*
#         # clone () const;
# 

###
# project_inliers.h

# # template<typename PointT>
# # class ProjectInliers : public Filter<PointT>
# cdef extern from "pcl/filters/project_inliers.h" namespace "pcl":
#     cdef cppclass ProjectInliers[T]:
#         ProjectInliers ()
# 
#         # using Filter<PointT>::input_;
#         # using Filter<PointT>::indices_;
#         # using Filter<PointT>::filter_name_;
#         # using Filter<PointT>::getClassName;
#         # ctypedef typename Filter<PointT>::PointCloud PointCloud;
#         # ctypedef typename PointCloud::Ptr PointCloudPtr;
#         # ctypedef typename PointCloud::ConstPtr PointCloudConstPtr;
#         # ctypedef typename SampleConsensusModel<PointT>::Ptr SampleConsensusModelPtr;
# 
#         # public:
#         # /** \brief The type of model to use (user given parameter).
#         #   * \param model the model type (check \a model_types.h)
#         #   */
#         inline void setModelType (int model)
# 
#         # /** \brief Get the type of SAC model used. */
#         inline int getModelType ()
# 
#         # /** \brief Provide a pointer to the model coefficients.
#         #   * \param model a pointer to the model coefficients
#         #   */
#         inline void setModelCoefficients (const ModelCoefficientsConstPtr &model)
# 
#         # /** \brief Get a pointer to the model coefficients. */
#         inline ModelCoefficientsConstPtr getModelCoefficients ()
# 
#         # /** \brief Set whether all data will be returned, or only the projected inliers.
#         #   * \param val true if all data should be returned, false if only the projected inliers
#         #   */
#         inline void setCopyAllData (bool val)
# 
#       	# /** \brief Get whether all data is being copied (true), or only the projected inliers (false). */
#       	# inline bool
#       	# getCopyAllData ()


#   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#   /** \brief @b ProjectInliers uses a model and a set of inlier indices from a PointCloud to project them into a
#     * separate PointCloud.
#     * \note setFilterFieldName (), setFilterLimits (), and setFilterLimitNegative () are ignored.
#     * \author Radu Bogdan Rusu
#     * \ingroup filters
#     */
#   template<>
#   class PCL_EXPORTS ProjectInliers<sensor_msgs::PointCloud2> : public Filter<sensor_msgs::PointCloud2>
#   {
#     using Filter<sensor_msgs::PointCloud2>::filter_name_;
#     using Filter<sensor_msgs::PointCloud2>::getClassName;
# 
#     typedef sensor_msgs::PointCloud2 PointCloud2;
#     typedef PointCloud2::Ptr PointCloud2Ptr;
#     typedef PointCloud2::ConstPtr PointCloud2ConstPtr;
# 
#     typedef SampleConsensusModel<PointXYZ>::Ptr SampleConsensusModelPtr;
# 
#     public:
#       /** \brief Empty constructor. */
#       ProjectInliers () : model_type_ (), copy_all_data_ (false), copy_all_fields_ (true), model_ (), sacmodel_ ()
#       {
#         filter_name_ = "ProjectInliers";
#       }
# 
#       /** \brief The type of model to use (user given parameter).
#         * \param[in] model the model type (check \a model_types.h)
#         */
#       inline void
#       setModelType (int model)
#       {
#         model_type_ = model;
#       }
# 
#       /** \brief Get the type of SAC model used. */
#       inline int
#       getModelType () const
#       {
#         return (model_type_);
#       }
# 
#       /** \brief Provide a pointer to the model coefficients.
#         * \param[in] model a pointer to the model coefficients
#         */
#       inline void
#       setModelCoefficients (const ModelCoefficientsConstPtr &model)
#       {
#         model_ = model;
#       }
# 
#       /** \brief Get a pointer to the model coefficients. */
#       inline ModelCoefficientsConstPtr
#       getModelCoefficients () const
#       {
#         return (model_);
#       }
# 
#       /** \brief Set whether all fields should be copied, or only the XYZ.
#         * \param[in] val true if all fields will be returned, false if only XYZ
#         */
#       inline void
#       setCopyAllFields (bool val)
#       {
#         copy_all_fields_ = val;
#       }
# 
#       /** \brief Get whether all fields are being copied (true), or only XYZ (false). */
#       inline bool
#       getCopyAllFields () const
#       {
#         return (copy_all_fields_);
#       }
# 
#       /** \brief Set whether all data will be returned, or only the projected inliers.
#         * \param[in] val true if all data should be returned, false if only the projected inliers
#         */
#       inline void
#       setCopyAllData (bool val)
#       {
#         copy_all_data_ = val;
#       }
# 
#       /** \brief Get whether all data is being copied (true), or only the projected inliers (false). */
#       inline bool
#       getCopyAllData () const
#       {
#         return (copy_all_data_);
#       }

###
# radius_outlier_removal.h

# template<typename PointT>
# class RadiusOutlierRemoval : public FilterIndices<PointT>
cdef extern from "pcl/filters/radius_outlier_removal.h" namespace "pcl":
    cdef cppclass RadiusOutlierRemoval[T]:
        RadiusOutlierRemoval ()
        # protected:
        # ctypedef typename FilterIndices<PointT>::PointCloud PointCloud;
        # ctypedef typename PointCloud::Ptr PointCloudPtr;
        # ctypedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # ctypedef typename pcl::search::Search<PointT>::Ptr SearcherPtr;

        # /** \brief Set the radius of the sphere that will determine which points are neighbors.
        #   * \details The number of points within this distance from the query point will need to be equal or greater
        #   * than setMinNeighborsInRadius() in order to be classified as an inlier point (i.e. will not be filtered).
        #   * \param[in] radius The radius of the sphere for nearest neighbor searching.
        #   */
        inline void setRadiusSearch (double radius)

        # /** \brief Get the radius of the sphere that will determine which points are neighbors.
        #   * \details The number of points within this distance from the query point will need to be equal or greater
        #   * than setMinNeighborsInRadius() in order to be classified as an inlier point (i.e. will not be filtered).
        #   * \return The radius of the sphere for nearest neighbor searching.
        #   */
        inline double getRadiusSearch ()

        # * \brief Set the number of neighbors that need to be present in order to be classified as an inlier.
        # * \details The number of points within setRadiusSearch() from the query point will need to be equal or greater
        # * than this number in order to be classified as an inlier point (i.e. will not be filtered).
        # * \param min_pts The minimum number of neighbors (default = 1).
        inline void setMinNeighborsInRadius (int min_pts)

        # /** \brief Get the number of neighbors that need to be present in order to be classified as an inlier.
        #   * \details The number of points within setRadiusSearch() from the query point will need to be equal or greater
        #   * than this number in order to be classified as an inlier point (i.e. will not be filtered).
        #   * \param min_pts The minimum number of neighbors (default = 1).
        #   */
        inline int getMinNeighborsInRadius ()

#   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#   /** \brief @b RadiusOutlierRemoval is a simple filter that removes outliers if the number of neighbors in a certain
#     * search radius is smaller than a given K.
#     * \note setFilterFieldName (), setFilterLimits (), and setFilterLimitNegative () are ignored.
#     * \author Radu Bogdan Rusu
#     * \ingroup filters
#     */
#   template<>
#   class PCL_EXPORTS RadiusOutlierRemoval<sensor_msgs::PointCloud2> : public Filter<sensor_msgs::PointCloud2>
#   {
#     using Filter<sensor_msgs::PointCloud2>::filter_name_;
#     using Filter<sensor_msgs::PointCloud2>::getClassName;
# 
#     using Filter<sensor_msgs::PointCloud2>::removed_indices_;
#     using Filter<sensor_msgs::PointCloud2>::extract_removed_indices_;
# 
#     typedef pcl::search::Search<pcl::PointXYZ> KdTree;
#     typedef pcl::search::Search<pcl::PointXYZ>::Ptr KdTreePtr;
# 
#     typedef sensor_msgs::PointCloud2 PointCloud2;
#     typedef PointCloud2::Ptr PointCloud2Ptr;
#     typedef PointCloud2::ConstPtr PointCloud2ConstPtr;
# 
#     public:
#       /** \brief Empty constructor. */
#       RadiusOutlierRemoval (bool extract_removed_indices = false) :
#         Filter<sensor_msgs::PointCloud2>::Filter (extract_removed_indices), 
#         search_radius_ (0.0), min_pts_radius_ (1), tree_ ()
#       {
#         filter_name_ = "RadiusOutlierRemoval";
#       }
# 
#       /** \brief Set the sphere radius that is to be used for determining the k-nearest neighbors for filtering.
#         * \param radius the sphere radius that is to contain all k-nearest neighbors
#         */
#       inline void
#       setRadiusSearch (double radius)
#       {
#         search_radius_ = radius;
#       }
# 
#       /** \brief Get the sphere radius used for determining the k-nearest neighbors. */
#       inline double
#       getRadiusSearch ()
#       {
#         return (search_radius_);
#       }
# 
#       /** \brief Set the minimum number of neighbors that a point needs to have in the given search radius in order to
#         * be considered an inlier (i.e., valid).
#         * \param min_pts the minimum number of neighbors
#         */
#       inline void
#       setMinNeighborsInRadius (int min_pts)
#       {
#         min_pts_radius_ = min_pts;
#       }
# 
#       /** \brief Get the minimum number of neighbors that a point needs to have in the given search radius to be
#         * considered an inlier and avoid being filtered. 
#         */
#       inline double
#       getMinNeighborsInRadius ()
#       {
#         return (min_pts_radius_);
#       }

###
# random_sample.h

#  template<typename PointT>
#  class RandomSample : public FilterIndices<PointT>

cdef extern from "pcl/filters/random_sample.h" namespace "pcl":
    cdef cppclass RandomSample[T]:
        RandomSample ()
        # using FilterIndices<PointT>::filter_name_;
        # using FilterIndices<PointT>::getClassName;
        # using FilterIndices<PointT>::indices_;
        # using FilterIndices<PointT>::input_;
        # ctypedef typename FilterIndices<PointT>::PointCloud PointCloud;
        # ctypedef typename PointCloud::Ptr PointCloudPtr;
        # ctypedef typename PointCloud::ConstPtr PointCloudConstPtr;

        # public:

        # /** \brief Set number of indices to be sampled.
        #   * \param sample
        #   */
        inline void setSample (unsigned int sample)

        # /** \brief Get the value of the internal \a sample parameter.
        #   */
        inline unsigned int getSample ()

        # /** \brief Set seed of random function.
        #   * \param seed
        #   */
        inline void setSeed (unsigned int seed)

        # /** \brief Get the value of the internal \a seed parameter.
        #   */
        inline unsigned int getSeed ()

#   /** \brief @b RandomSample applies a random sampling with uniform probability.
#     * \author Justin Rosen
#     * \ingroup filters
#     */
#   template<>
#   class PCL_EXPORTS RandomSample<sensor_msgs::PointCloud2> : public FilterIndices<sensor_msgs::PointCloud2>
#   {
#     using FilterIndices<sensor_msgs::PointCloud2>::filter_name_;
#     using FilterIndices<sensor_msgs::PointCloud2>::getClassName;
# 
#     typedef sensor_msgs::PointCloud2 PointCloud2;
#     typedef PointCloud2::Ptr PointCloud2Ptr;
#     typedef PointCloud2::ConstPtr PointCloud2ConstPtr;
# 
#     public:
#       /** \brief Empty constructor. */
#       RandomSample () : sample_ (UINT_MAX), seed_ (static_cast<unsigned int> (time (NULL)))
#       {
#         filter_name_ = "RandomSample";
#       }
# 
#       /** \brief Set number of indices to be sampled.
#         * \param sample
#         */
#       inline void
#       setSample (unsigned int sample)
#       {
#         sample_ = sample;
#       }
# 
#       /** \brief Get the value of the internal \a sample parameter.
#         */
#       inline unsigned int
#       getSample ()
#       {
#         return (sample_);
#       }
# 
#       /** \brief Set seed of random function.
#         * \param seed
#         */
#       inline void
#       setSeed (unsigned int seed)
#       {
#         seed_ = seed;
#       }
# 
#       /** \brief Get the value of the internal \a seed parameter.
#         */
#       inline unsigned int
#       getSeed ()
#       {
#         return (seed_);
#       }
# 

###
# statistical_outlier_removal.h

cdef extern from "pcl/filters/statistical_outlier_removal.h" namespace "pcl":
    cdef cppclass StatisticalOutlierRemoval[T]:
        StatisticalOutlierRemoval()
        int getMeanK()
        void setMeanK (int nr_k)
        double getStddevMulThresh()
        void setStddevMulThresh (double std_mul)
        bool getNegative()
        void setNegative (bool negative)
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        void filter(cpp.PointCloud[T] &c)

ctypedef StatisticalOutlierRemoval[cpp.PointXYZ] StatisticalOutlierRemoval_t
ctypedef StatisticalOutlierRemoval[cpp.PointXYZRGBA] StatisticalOutlierRemoval2_t

# template<>
# class PCL_EXPORTS StatisticalOutlierRemoval<sensor_msgs::PointCloud2> : public Filter<sensor_msgs::PointCloud2>
#   {
#     using Filter<sensor_msgs::PointCloud2>::filter_name_;
#     using Filter<sensor_msgs::PointCloud2>::getClassName;
# 
#     using Filter<sensor_msgs::PointCloud2>::removed_indices_;
#     using Filter<sensor_msgs::PointCloud2>::extract_removed_indices_;
# 
#     typedef pcl::search::Search<pcl::PointXYZ> KdTree;
#     typedef pcl::search::Search<pcl::PointXYZ>::Ptr KdTreePtr;
# 
#     typedef sensor_msgs::PointCloud2 PointCloud2;
#     typedef PointCloud2::Ptr PointCloud2Ptr;
#     typedef PointCloud2::ConstPtr PointCloud2ConstPtr;
# 
#     public:
#       /** \brief Empty constructor. */
#       StatisticalOutlierRemoval (bool extract_removed_indices = false) :
#         Filter<sensor_msgs::PointCloud2>::Filter (extract_removed_indices), mean_k_ (2), 
#         std_mul_ (0.0), tree_ (), negative_ (false)
#       {
#         filter_name_ = "StatisticalOutlierRemoval";
#       }
# 
#       /** \brief Set the number of points (k) to use for mean distance estimation
#         * \param nr_k the number of points to use for mean distance estimation
#         */
#       inline void
#       setMeanK (int nr_k)
#       {
#         mean_k_ = nr_k;
#       }
# 
#       /** \brief Get the number of points to use for mean distance estimation. */
#       inline int
#       getMeanK ()
#       {
#         return (mean_k_);
#       }
# 
#       /** \brief Set the standard deviation multiplier threshold. All points outside the
#         * \f[ \mu \pm \sigma \cdot std\_mul \f]
#         * will be considered outliers, where \f$ \mu \f$ is the estimated mean,
#         * and \f$ \sigma \f$ is the standard deviation.
#         * \param std_mul the standard deviation multiplier threshold
#         */
#       inline void
#       setStddevMulThresh (double std_mul)
#       {
#         std_mul_ = std_mul;
#       }
# 
#       /** \brief Get the standard deviation multiplier threshold as set by the user. */
#       inline double
#       getStddevMulThresh ()
#       {
#         return (std_mul_);
#       }
# 
#       /** \brief Set whether the indices should be returned, or all points \e except the indices.
#         * \param negative true if all points \e except the input indices will be returned, false otherwise
#         */
#       inline void
#       setNegative (bool negative)
#       {
#         negative_ = negative;
#       }
# 
#       /** \brief Get the value of the internal #negative_ parameter. If
#         * true, all points \e except the input indices will be returned.
#         * \return The value of the "negative" flag
#         */
#       inline bool
#       getNegative ()
#       {
#         return (negative_);
#       }
# 
#       void applyFilter (PointCloud2 &output);


###
# voxel_grid.h

cdef extern from "pcl/filters/voxel_grid.h" namespace "pcl":
    cdef cppclass VoxelGrid[T]:
        VoxelGrid()
        void setLeafSize (float, float, float)
        void setInputCloud (shared_ptr[cpp.PointCloud[T]])
        void filter(cpp.PointCloud[T] c)

        # /** \brief Set to true if all fields need to be downsampled, or false if just XYZ.
        #   * \param[in] downsample the new value (true/false)
        #   */
        # inline void setDownsampleAllData (bool downsample)
        # /** \brief Get the state of the internal downsampling parameter (true if
        #   * all fields need to be downsampled, false if just XYZ). 
        #   */
        # inline bool getDownsampleAllData ()
        # /** \brief Set to true if leaf layout information needs to be saved for later access.
        #   * \param[in] save_leaf_layout the new value (true/false)
        #   */
        # inline void setSaveLeafLayout (bool save_leaf_layout)
        # /** \brief Returns true if leaf layout information will to be saved for later access. */
        # inline bool 
        # getSaveLeafLayout () { return (save_leaf_layout_); }
        # /** \brief Get the minimum coordinates of the bounding box (after
        #   * filtering is performed). 
        #   */
        # inline Eigen::Vector3i 
        # getMinBoxCoordinates () { return (min_b_.head<3> ()); }
        # /** \brief Get the minimum coordinates of the bounding box (after
        #   * filtering is performed). 
        #   */
        # inline Eigen::Vector3i 
        # getMaxBoxCoordinates () { return (max_b_.head<3> ()); }
        # /** \brief Get the number of divisions along all 3 axes (after filtering
        #   * is performed). 
        #   */
        # inline Eigen::Vector3i 
        # getNrDivisions () { return (div_b_.head<3> ()); }

        # /** \brief Get the multipliers to be applied to the grid coordinates in
        #   * order to find the centroid index (after filtering is performed). 
        #   */
        # inline Eigen::Vector3i getDivisionMultiplier () { return (divb_mul_.head<3> ()); }
        # /** \brief Returns the index in the resulting downsampled cloud of the specified point.
        #   *
        #   * \note for efficiency, user must make sure that the saving of the leaf layout is enabled and filtering 
        #   * performed, and that the point is inside the grid, to avoid invalid access (or use
        #   * getGridCoordinates+getCentroidIndexAt)
        #   *
        #   * \param[in] p the point to get the index at
        #   */
        # inline int getCentroidIndex (const PointT &p)
        # /** \brief Returns the indices in the resulting downsampled cloud of the points at the specified grid coordinates,
        #   * relative to the grid coordinates of the specified point (or -1 if the cell was empty/out of bounds).
        #   * \param[in] reference_point the coordinates of the reference point (corresponding cell is allowed to be empty/out of bounds)
        #   * \param[in] relative_coordinates matrix with the columns being the coordinates of the requested cells, relative to the reference point's cell
        #   * \note for efficiency, user must make sure that the saving of the leaf layout is enabled and filtering performed
        #   */
        # inline std::vector<int> getNeighborCentroidIndices (const PointT &reference_point, const Eigen::MatrixXi &relative_coordinates)
        # /** \brief Returns the layout of the leafs for fast access to cells relative to current position.
        #   * \note position at (i-min_x) + (j-min_y)*div_x + (k-min_z)*div_x*div_y holds the index of the element at coordinates (i,j,k) in the grid (-1 if empty)
        #   */
        # inline vector[int] getLeafLayout ()
        # /** \brief Returns the corresponding (i,j,k) coordinates in the grid of point (x,y,z). 
        #   * \param[in] x the X point coordinate to get the (i, j, k) index at
        #   * \param[in] y the Y point coordinate to get the (i, j, k) index at
        #   * \param[in] z the Z point coordinate to get the (i, j, k) index at
        #   */
        # inline Eigen::Vector3i getGridCoordinates (float x, float y, float z) 
        # /** \brief Returns the index in the downsampled cloud corresponding to a given set of coordinates.
        #   * \param[in] ijk the coordinates (i,j,k) in the grid (-1 if empty)
        #   */
        # inline int getCentroidIndexAt (const Eigen::Vector3i &ijk)
        # /** \brief Provide the name of the field to be used for filtering data. In conjunction with  \a setFilterLimits,
        #   * points having values outside this interval will be discarded.
        #   * \param[in] field_name the name of the field that contains values used for filtering
        #   */
        # inline void
        # setFilterFieldName (const std::string &field_name)
        # /** \brief Get the name of the field used for filtering. */
        # inline std::string const
        # getFilterFieldName ()
        # /** \brief Set the field filter limits. All points having field values outside this interval will be discarded.
        #   * \param[in] limit_min the minimum allowed field value
        #   * \param[in] limit_max the maximum allowed field value
        #   */
        # inline void setFilterLimits (const double &limit_min, const double &limit_max)
        # /** \brief Get the field filter limits (min/max) set by the user. The default values are -FLT_MAX, FLT_MAX. 
        #   * \param[out] limit_min the minimum allowed field value
        #   * \param[out] limit_max the maximum allowed field value
        #   */
        # inline void getFilterLimits (double &limit_min, double &limit_max)
        # /** \brief Set to true if we want to return the data outside the interval specified by setFilterLimits (min, max).
        #   * Default: false.
        #   * \param[in] limit_negative return data inside the interval (false) or outside (true)
        #   */
        # inline void
        # setFilterLimitsNegative (const bool limit_negative)
        # /** \brief Get whether the data outside the interval (min/max) is to be returned (true) or inside (false). 
        #   * \param[out] limit_negative true if data \b outside the interval [min; max] is to be returned, false otherwise
        #   */
        # inline void
        # getFilterLimitsNegative (bool &limit_negative)
        # /** \brief Get whether the data outside the interval (min/max) is to be returned (true) or inside (false). 
        #   * \return true if data \b outside the interval [min; max] is to be returned, false otherwise
        #   */
        # inline bool
        # getFilterLimitsNegative ()

ctypedef VoxelGrid[cpp.PointXYZ] VoxelGrid_t
ctypedef VoxelGrid[cpp.PointXYZRGBA] VoxelGrid2_t

###
