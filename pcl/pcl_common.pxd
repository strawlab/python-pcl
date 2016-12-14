# -*- coding: utf-8 -*-

# common/angles.h
# namespace pcl
cdef extern from "pcl/common/angles.h" namespace "pcl":
    # brief Convert an angle from radians to degrees
    # param alpha the input angle (in radians)
    # ingroup common
    # inline float rad2deg (float alpha);
    cdef float rad2deg (float alpha)
    
    # brief Convert an angle from degrees to radians
    # param alpha the input angle (in degrees)
    # ingroup common
    # inline float deg2rad (float alpha);
    cdef float deg2rad (float alpha)
    
    # brief Convert an angle from radians to degrees
    # param alpha the input angle (in radians)
    # ingroup common
    # inline double rad2deg (double alpha);
    cdef double deg2rad (double alpha)
    
    # brief Convert an angle from degrees to radians
    # param alpha the input angle (in degrees)
    # ingroup common
    # inline double deg2rad (double alpha);
    cdef double deg2rad (double alpha)
    
    # brief Normalize an angle to (-PI, PI]
    # param alpha the input angle (in radians)
    # ingroup common
    # inline float normAngle (float alpha);
    cdef float normAngle (float alpha)
###

# bivariate_polynomial.h
# namespace pcl 
# {
# cdef extern from "pcl/common/bivariate_polynomial.h" namespace "pcl":
#   # /** \brief This represents a bivariate polynomial and provides some functionality for it
#   #   * \author Bastian Steder 
#   #   * \ingroup common
#   #   */
#   template<typename real> class BivariatePolynomialT 
#   {
#     public:
#       //-----CONSTRUCTOR&DESTRUCTOR-----
#       /** Constructor */
#       BivariatePolynomialT (int new_degree=0);
#       /** Copy constructor */
#       BivariatePolynomialT (const BivariatePolynomialT& other);
#       /** Destructor */
#       ~BivariatePolynomialT ();
# 
#       //-----OPERATORS-----
#       /** = operator */
#       BivariatePolynomialT&
#       operator= (const BivariatePolynomialT& other) { deepCopy (other); return *this;}
# 
#       //-----METHODS-----
#       /** Initialize members to default values */
#       void setDegree (int new_degree);
# 
#       /** How many parametes has a bivariate polynomial with this degree */
#       unsigned int getNoOfParameters () const { return getNoOfParametersFromDegree (degree);}
# 
#       /** Calculate the value of the polynomial at the given point */
#       real getValue (real x, real y) const;  
# 
#       /** Calculate the gradient of this polynomial
#        *  If forceRecalc is false, it will do nothing when the gradient already exists */
#       void calculateGradient (bool forceRecalc=false);
# 
#       /** Calculate the value of the gradient at the given point */
#       void getValueOfGradient (real x, real y, real& gradX, real& gradY);
# 
#       /** Returns critical points of the polynomial. type can be 0=maximum, 1=minimum, or 2=saddle point
#        *  !!Currently only implemented for degree 2!! */
#       void findCriticalPoints (std::vector<real>& x_values, std::vector<real>& y_values, std::vector<int>& types) const;
#       
#       /** write as binary to a stream */
#       void writeBinary (std::ostream& os) const;
# 
#       /** write as binary into a file */
#       void writeBinary (const char* filename) const;
# 
#       /** read binary from a stream */
#       void readBinary (std::istream& os);
# 
#       /** read binary from a file */
#       void readBinary (const char* filename);
#       
#       /** How many parametes has a bivariate polynomial of the given degree */
#       static unsigned int getNoOfParametersFromDegree (int n) { return ((n+2)* (n+1))/2;}
#
#     protected:
#       //-----METHODS-----
#       /** Delete all members */
#       void memoryCleanUp ();
# 
#       /** Create a deep copy of the given polynomial */
#       void deepCopy (const BivariatePolynomialT<real>& other);
# 
#   template<typename real>
#   std::ostream&
#     operator<< (std::ostream& os, const BivariatePolynomialT<real>& p);
# 
#   typedef BivariatePolynomialT<double> BivariatePolynomiald;
#   typedef BivariatePolynomialT<float>  BivariatePolynomial;
# }  // end namespace
# #include <pcl/common/impl/bivariate_polynomial.hpp>
# #endif
###

# # boost.h
# // Marking all Boost headers as system headers to remove warnings
# #include <boost/fusion/sequence/intrinsic/at_key.hpp>
# #include <boost/shared_ptr.hpp>
# #include <boost/make_shared.hpp>
# #include <boost/mpl/size.hpp>
# #include <boost/date_time/posix_time/posix_time.hpp>
# #include <boost/function.hpp>
# //#include <boost/timer.hpp>
# #include <boost/thread.hpp>
# #include <boost/thread/condition.hpp>
# #include <boost/signals2.hpp>
# #include <boost/signals2/slot.hpp>
# #include <boost/algorithm/string.hpp>
# 
#endif    // PCL_COMMON_BOOST_H_
###

# centroid.h
# namespace pcl
# {
# cdef extern from "pcl/common/centroid.h" namespace "pcl":
  # /** \brief Compute the 3D (X-Y-Z) centroid of a set of points and return it as a 3D vector.
  #   * \param[in] cloud_iterator an iterator over the input point cloud
  #   * \param[out] centroid the output centroid
  #   * \return number of valid point used to determine the centroid. In case of dense point clouds, this is the same as the size of input cloud.
  #   * \note if return value is 0, the centroid is not changed, thus not valid.
  #   * The last compononent of the vector is set to 1, this allow to transform the centroid vector with 4x4 matrices.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # compute3DCentroid (ConstCloudIterator<PointT> &cloud_iterator,
  #                    Eigen::Matrix<Scalar, 4, 1> &centroid);
  # 
  # template <typename PointT> inline unsigned int
  # compute3DCentroid (ConstCloudIterator<PointT> &cloud_iterator,
  #                    Eigen::Vector4f &centroid)
  # {
  #   return (compute3DCentroid <PointT, float> (cloud_iterator, centroid));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # compute3DCentroid (ConstCloudIterator<PointT> &cloud_iterator,
  #                    Eigen::Vector4d &centroid)
  # {
  #   return (compute3DCentroid <PointT, double> (cloud_iterator, centroid));
  # }
  # 
  # /** \brief Compute the 3D (X-Y-Z) centroid of a set of points and return it as a 3D vector.
  #   * \param[in] cloud the input point cloud
  #   * \param[out] centroid the output centroid
  #   * \return number of valid point used to determine the centroid. In case of dense point clouds, this is the same as the size of input cloud.
  #   * \note if return value is 0, the centroid is not changed, thus not valid.
  #   * The last compononent of the vector is set to 1, this allow to transform the centroid vector with 4x4 matrices.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # compute3DCentroid (const pcl::PointCloud<PointT> &cloud, 
  #                    Eigen::Matrix<Scalar, 4, 1> &centroid);
  # 
  # template <typename PointT> inline unsigned int
  # compute3DCentroid (const pcl::PointCloud<PointT> &cloud, 
  #                    Eigen::Vector4f &centroid)
  # {
  #   return (compute3DCentroid <PointT, float> (cloud, centroid));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # compute3DCentroid (const pcl::PointCloud<PointT> &cloud, 
  #                    Eigen::Vector4d &centroid)
  # {
  #   return (compute3DCentroid <PointT, double> (cloud, centroid));
  # }
  # 
  # /** \brief Compute the 3D (X-Y-Z) centroid of a set of points using their indices and
  #   * return it as a 3D vector.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] indices the point cloud indices that need to be used
  #   * \param[out] centroid the output centroid
  #   * \return number of valid point used to determine the centroid. In case of dense point clouds, this is the same as the size of input cloud.
  #   * \note if return value is 0, the centroid is not changed, thus not valid.
  #   * The last compononent of the vector is set to 1, this allow to transform the centroid vector with 4x4 matrices.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # compute3DCentroid (const pcl::PointCloud<PointT> &cloud,
  #                    const std::vector<int> &indices, 
  #                    Eigen::Matrix<Scalar, 4, 1> &centroid);
  # 
  # template <typename PointT> inline unsigned int
  # compute3DCentroid (const pcl::PointCloud<PointT> &cloud,
  #                    const std::vector<int> &indices, 
  #                    Eigen::Vector4f &centroid)
  # {
  #   return (compute3DCentroid <PointT, float> (cloud, indices, centroid));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # compute3DCentroid (const pcl::PointCloud<PointT> &cloud,
  #                    const std::vector<int> &indices, 
  #                    Eigen::Vector4d &centroid)
  # {
  #   return (compute3DCentroid <PointT, double> (cloud, indices, centroid));
  # }
  # 
  # /** \brief Compute the 3D (X-Y-Z) centroid of a set of points using their indices and
  #   * return it as a 3D vector.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] indices the point cloud indices that need to be used
  #   * \param[out] centroid the output centroid
  #   * \return number of valid point used to determine the centroid. In case of dense point clouds, this is the same as the size of input cloud.
  #   * \note if return value is 0, the centroid is not changed, thus not valid.
  #   * The last compononent of the vector is set to 1, this allow to transform the centroid vector with 4x4 matrices.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # compute3DCentroid (const pcl::PointCloud<PointT> &cloud,
  #                    const pcl::PointIndices &indices, 
  #                    Eigen::Matrix<Scalar, 4, 1> &centroid);
  # 
  # template <typename PointT> inline unsigned int
  # compute3DCentroid (const pcl::PointCloud<PointT> &cloud,
  #                    const pcl::PointIndices &indices, 
  #                    Eigen::Vector4f &centroid)
  # {
  #   return (compute3DCentroid <PointT, float> (cloud, indices, centroid));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # compute3DCentroid (const pcl::PointCloud<PointT> &cloud,
  #                    const pcl::PointIndices &indices, 
  #                    Eigen::Vector4d &centroid)
  # {
  #   return (compute3DCentroid <PointT, double> (cloud, indices, centroid));
  # }
  # 
  # /** \brief Compute the 3x3 covariance matrix of a given set of points.
  #   * The result is returned as a Eigen::Matrix3f.
  #   * Note: the covariance matrix is not normalized with the number of
  #   * points. For a normalized covariance, please use
  #   * computeNormalizedCovarianceMatrix.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] centroid the centroid of the set of points in the cloud
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \note if return value is 0, the covariance matrix is not changed, thus not valid.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                          Eigen::Matrix<Scalar, 3, 3> &covariance_matrix);
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const Eigen::Vector4f &centroid,
  #                          Eigen::Matrix3f &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, float> (cloud, centroid, covariance_matrix));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const Eigen::Vector4d &centroid,
  #                          Eigen::Matrix3d &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, double> (cloud, centroid, covariance_matrix));
  # }
  # 
  # /** \brief Compute normalized the 3x3 covariance matrix of a given set of points.
  #   * The result is returned as a Eigen::Matrix3f.
  #   * Normalized means that every entry has been divided by the number of points in the point cloud.
  #   * For small number of points, or if you want explicitely the sample-variance, use computeCovarianceMatrix
  #   * and scale the covariance matrix with 1 / (n-1), where n is the number of points used to calculate
  #   * the covariance matrix and is returned by the computeCovarianceMatrix function.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] centroid the centroid of the set of points in the cloud
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
  #                                    const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                                    Eigen::Matrix<Scalar, 3, 3> &covariance_matrix);
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
  #                                    const Eigen::Vector4f &centroid,
  #                                    Eigen::Matrix3f &covariance_matrix)
  # {
  #   return (computeCovarianceMatrixNormalized<PointT, float> (cloud, centroid, covariance_matrix));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
  #                                    const Eigen::Vector4d &centroid,
  #                                    Eigen::Matrix3d &covariance_matrix)
  # {
  #   return (computeCovarianceMatrixNormalized<PointT, double> (cloud, centroid, covariance_matrix));
  # }
  # 
  # /** \brief Compute the 3x3 covariance matrix of a given set of points using their indices.
  #   * The result is returned as a Eigen::Matrix3f.
  #   * Note: the covariance matrix is not normalized with the number of
  #   * points. For a normalized covariance, please use
  #   * computeNormalizedCovarianceMatrix.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] indices the point cloud indices that need to be used
  #   * \param[in] centroid the centroid of the set of points in the cloud
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const std::vector<int> &indices,
  #                          const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                          Eigen::Matrix<Scalar, 3, 3> &covariance_matrix);
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const std::vector<int> &indices,
  #                          const Eigen::Vector4f &centroid,
  #                          Eigen::Matrix3f &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, float> (cloud, indices, centroid, covariance_matrix));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const std::vector<int> &indices,
  #                          const Eigen::Vector4d &centroid,
  #                          Eigen::Matrix3d &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, double> (cloud, indices, centroid, covariance_matrix));
  # }
  # 
  # /** \brief Compute the 3x3 covariance matrix of a given set of points using their indices.
  #   * The result is returned as a Eigen::Matrix3f.
  #   * Note: the covariance matrix is not normalized with the number of
  #   * points. For a normalized covariance, please use
  #   * computeNormalizedCovarianceMatrix.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] indices the point cloud indices that need to be used
  #   * \param[in] centroid the centroid of the set of points in the cloud
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const pcl::PointIndices &indices,
  #                          const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                          Eigen::Matrix<Scalar, 3, 3> &covariance_matrix);
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const pcl::PointIndices &indices,
  #                          const Eigen::Vector4f &centroid,
  #                          Eigen::Matrix3f &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, float> (cloud, indices, centroid, covariance_matrix));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const pcl::PointIndices &indices,
  #                          const Eigen::Vector4d &centroid,
  #                          Eigen::Matrix3d &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, double> (cloud, indices, centroid, covariance_matrix));
  # }
  # 
  # /** \brief Compute the normalized 3x3 covariance matrix of a given set of points using
  #   * their indices.
  #   * The result is returned as a Eigen::Matrix3f.
  #   * Normalized means that every entry has been divided by the number of entries in indices.
  #   * For small number of points, or if you want explicitely the sample-variance, use computeCovarianceMatrix
  #   * and scale the covariance matrix with 1 / (n-1), where n is the number of points used to calculate
  #   * the covariance matrix and is returned by the computeCovarianceMatrix function.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] indices the point cloud indices that need to be used
  #   * \param[in] centroid the centroid of the set of points in the cloud
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
  #                                    const std::vector<int> &indices,
  #                                    const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                                    Eigen::Matrix<Scalar, 3, 3> &covariance_matrix);
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
  #                                    const std::vector<int> &indices,
  #                                    const Eigen::Vector4f &centroid,
  #                                    Eigen::Matrix3f &covariance_matrix)
  # {
  #   return (computeCovarianceMatrixNormalized<PointT, float> (cloud, indices, centroid, covariance_matrix));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
  #                                    const std::vector<int> &indices,
  #                                    const Eigen::Vector4d &centroid,
  #                                    Eigen::Matrix3d &covariance_matrix)
  # {
  #   return (computeCovarianceMatrixNormalized<PointT, double> (cloud, indices, centroid, covariance_matrix));
  # }
  # 
  # /** \brief Compute the normalized 3x3 covariance matrix of a given set of points using
  #   * their indices. The result is returned as a Eigen::Matrix3f.
  #   * Normalized means that every entry has been divided by the number of entries in indices.
  #   * For small number of points, or if you want explicitely the sample-variance, use computeCovarianceMatrix
  #   * and scale the covariance matrix with 1 / (n-1), where n is the number of points used to calculate
  #   * the covariance matrix and is returned by the computeCovarianceMatrix function.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] indices the point cloud indices that need to be used
  #   * \param[in] centroid the centroid of the set of points in the cloud
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
  #                                    const pcl::PointIndices &indices,
  #                                    const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                                    Eigen::Matrix<Scalar, 3, 3> &covariance_matrix);
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
  #                                    const pcl::PointIndices &indices,
  #                                    const Eigen::Vector4f &centroid,
  #                                    Eigen::Matrix3f &covariance_matrix)
  # {
  #   return (computeCovarianceMatrixNormalized<PointT, float> (cloud, indices, centroid, covariance_matrix));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrixNormalized (const pcl::PointCloud<PointT> &cloud,
  #                                    const pcl::PointIndices &indices,
  #                                    const Eigen::Vector4d &centroid,
  #                                    Eigen::Matrix3d &covariance_matrix)
  # {
  #   return (computeCovarianceMatrixNormalized<PointT, double> (cloud, indices, centroid, covariance_matrix));
  # }
  # 
  # /** \brief Compute the normalized 3x3 covariance matrix and the centroid of a given set of points in a single loop.
  #   * Normalized means that every entry has been divided by the number of entries in indices.
  #   * For small number of points, or if you want explicitely the sample-variance, scale the covariance matrix
  #   * with n / (n-1), where n is the number of points used to calculate the covariance matrix and is returned by this function.
  #   * \note This method is theoretically exact. However using float for internal calculations reduces the accuracy but increases the efficiency.
  #   * \param[in] cloud the input point cloud
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \param[out] centroid the centroid of the set of points in the cloud
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                                 Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
  #                                 Eigen::Matrix<Scalar, 4, 1> &centroid);
  # 
  # template <typename PointT> inline unsigned int
  # computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                                 Eigen::Matrix3f &covariance_matrix,
  #                                 Eigen::Vector4f &centroid)
  # {
  #   return (computeMeanAndCovarianceMatrix<PointT, float> (cloud, covariance_matrix, centroid));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                                 Eigen::Matrix3d &covariance_matrix,
  #                                 Eigen::Vector4d &centroid)
  # {
  #   return (computeMeanAndCovarianceMatrix<PointT, double> (cloud, covariance_matrix, centroid));
  # }
  # 
  # /** \brief Compute the normalized 3x3 covariance matrix and the centroid of a given set of points in a single loop.
  #   * Normalized means that every entry has been divided by the number of entries in indices.
  #   * For small number of points, or if you want explicitely the sample-variance, scale the covariance matrix
  #   * with n / (n-1), where n is the number of points used to calculate the covariance matrix and is returned by this function.
  #   * \note This method is theoretically exact. However using float for internal calculations reduces the accuracy but increases the efficiency.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] indices subset of points given by their indices
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \param[out] centroid the centroid of the set of points in the cloud
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                                 const std::vector<int> &indices,
  #                                 Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
  #                                 Eigen::Matrix<Scalar, 4, 1> &centroid);
  # 
  # template <typename PointT> inline unsigned int
  # computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                                 const std::vector<int> &indices,
  #                                 Eigen::Matrix3f &covariance_matrix,
  #                                 Eigen::Vector4f &centroid)
  # {
  #   return (computeMeanAndCovarianceMatrix<PointT, float> (cloud, indices, covariance_matrix, centroid));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                                 const std::vector<int> &indices,
  #                                 Eigen::Matrix3d &covariance_matrix,
  #                                 Eigen::Vector4d &centroid)
  # {
  #   return (computeMeanAndCovarianceMatrix<PointT, double> (cloud, indices, covariance_matrix, centroid));
  # }
  # 
  # /** \brief Compute the normalized 3x3 covariance matrix and the centroid of a given set of points in a single loop.
  #   * Normalized means that every entry has been divided by the number of entries in indices.
  #   * For small number of points, or if you want explicitely the sample-variance, scale the covariance matrix
  #   * with n / (n-1), where n is the number of points used to calculate the covariance matrix and is returned by this function.
  #   * \note This method is theoretically exact. However using float for internal calculations reduces the accuracy but increases the efficiency.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] indices subset of points given by their indices
  #   * \param[out] centroid the centroid of the set of points in the cloud
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                                 const pcl::PointIndices &indices,
  #                                 Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
  #                                 Eigen::Matrix<Scalar, 4, 1> &centroid);
  # 
  # template <typename PointT> inline unsigned int
  # computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                                 const pcl::PointIndices &indices,
  #                                 Eigen::Matrix3f &covariance_matrix,
  #                                 Eigen::Vector4f &centroid)
  # {
  #   return (computeMeanAndCovarianceMatrix<PointT, float> (cloud, indices, covariance_matrix, centroid));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                                 const pcl::PointIndices &indices,
  #                                 Eigen::Matrix3d &covariance_matrix,
  #                                 Eigen::Vector4d &centroid)
  # {
  #   return (computeMeanAndCovarianceMatrix<PointT, double> (cloud, indices, covariance_matrix, centroid));
  # }
  # 
  # /** \brief Compute the normalized 3x3 covariance matrix for a already demeaned point cloud.
  #   * Normalized means that every entry has been divided by the number of entries in indices.
  #   * For small number of points, or if you want explicitely the sample-variance, scale the covariance matrix
  #   * with n / (n-1), where n is the number of points used to calculate the covariance matrix and is returned by this function.
  #   * \note This method is theoretically exact. However using float for internal calculations reduces the accuracy but increases the efficiency.
  #   * \param[in] cloud the input point cloud
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          Eigen::Matrix<Scalar, 3, 3> &covariance_matrix);
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          Eigen::Matrix3f &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, float> (cloud, covariance_matrix));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          Eigen::Matrix3d &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, double> (cloud, covariance_matrix));
  # }
  # 
  # /** \brief Compute the normalized 3x3 covariance matrix for a already demeaned point cloud.
  #   * Normalized means that every entry has been divided by the number of entries in indices.
  #   * For small number of points, or if you want explicitely the sample-variance, scale the covariance matrix
  #   * with n / (n-1), where n is the number of points used to calculate the covariance matrix and is returned by this function.
  #   * \note This method is theoretically exact. However using float for internal calculations reduces the accuracy but increases the efficiency.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] indices subset of points given by their indices
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const std::vector<int> &indices,
  #                          Eigen::Matrix<Scalar, 3, 3> &covariance_matrix);
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const std::vector<int> &indices,
  #                          Eigen::Matrix3f &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, float> (cloud, indices, covariance_matrix));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const std::vector<int> &indices,
  #                          Eigen::Matrix3d &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, double> (cloud, indices, covariance_matrix));
  # }
  # 
  # /** \brief Compute the normalized 3x3 covariance matrix for a already demeaned point cloud.
  #   * Normalized means that every entry has been divided by the number of entries in indices.
  #   * For small number of points, or if you want explicitely the sample-variance, scale the covariance matrix
  #   * with n / (n-1), where n is the number of points used to calculate the covariance matrix and is returned by this function.
  #   * \note This method is theoretically exact. However using float for internal calculations reduces the accuracy but increases the efficiency.
  #   * \param[in] cloud the input point cloud
  #   * \param[in] indices subset of points given by their indices
  #   * \param[out] covariance_matrix the resultant 3x3 covariance matrix
  #   * \return number of valid point used to determine the covariance matrix.
  #   * In case of dense point clouds, this is the same as the size of input cloud.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const pcl::PointIndices &indices,
  #                          Eigen::Matrix<Scalar, 3, 3> &covariance_matrix);
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const pcl::PointIndices &indices,
  #                          Eigen::Matrix3f &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, float> (cloud, indices, covariance_matrix));
  # }
  # 
  # template <typename PointT> inline unsigned int
  # computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
  #                          const pcl::PointIndices &indices,
  #                          Eigen::Matrix3d &covariance_matrix)
  # {
  #   return (computeCovarianceMatrix<PointT, double> (cloud, indices, covariance_matrix));
  # }
  # 
  # /** \brief Subtract a centroid from a point cloud and return the de-meaned representation
  #   * \param[in] cloud_iterator an iterator over the input point cloud
  #   * \param[in] centroid the centroid of the point cloud
  #   * \param[out] cloud_out the resultant output point cloud
  #   * \param[in] npts the number of samples guaranteed to be left in the input cloud, accessible by the iterator. If not given, it will be calculated.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> void
  # demeanPointCloud (ConstCloudIterator<PointT> &cloud_iterator,
  #                   const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out,
  #                   int npts = 0);
  # 
  # template <typename PointT> void
  # demeanPointCloud (ConstCloudIterator<PointT> &cloud_iterator,
  #                   const Eigen::Vector4f &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out,
  #                   int npts = 0)
  # {
  #   return (demeanPointCloud<PointT, float> (cloud_iterator, centroid, cloud_out, npts));
  # }
  # 
  # template <typename PointT> void
  # demeanPointCloud (ConstCloudIterator<PointT> &cloud_iterator,
  #                   const Eigen::Vector4d &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out,
  #                   int npts = 0)
  # {
  #   return (demeanPointCloud<PointT, double> (cloud_iterator, centroid, cloud_out, npts));
  # }
  # 
  # /** \brief Subtract a centroid from a point cloud and return the de-meaned representation
  #   * \param[in] cloud_in the input point cloud
  #   * \param[in] centroid the centroid of the point cloud
  #   * \param[out] cloud_out the resultant output point cloud
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out);
  # 
  # template <typename PointT> void
  # demeanPointCloud (ConstCloudIterator<PointT> &cloud_iterator,
  #                   const Eigen::Vector4f &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out)
  # {
  #   return (demeanPointCloud<PointT, float> (cloud_iterator, centroid, cloud_out));
  # }
  # 
  # template <typename PointT> void
  # demeanPointCloud (ConstCloudIterator<PointT> &cloud_iterator,
  #                   const Eigen::Vector4d &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out)
  # {
  #   return (demeanPointCloud<PointT, double> (cloud_iterator, centroid, cloud_out));
  # }
  # 
  # /** \brief Subtract a centroid from a point cloud and return the de-meaned representation
  #   * \param[in] cloud_in the input point cloud
  #   * \param[in] indices the set of point indices to use from the input point cloud
  #   * \param[out] centroid the centroid of the point cloud
  #   * \param cloud_out the resultant output point cloud
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const std::vector<int> &indices,
  #                   const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out);
  # 
  # template <typename PointT> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const std::vector<int> &indices,
  #                   const Eigen::Vector4f &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out)
  # {
  #   return (demeanPointCloud<PointT, float> (cloud_in, indices, centroid, cloud_out));
  # }
  # 
  # template <typename PointT> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const std::vector<int> &indices,
  #                   const Eigen::Vector4d &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out)
  # {
  #   return (demeanPointCloud<PointT, double> (cloud_in, indices, centroid, cloud_out));
  # }
  # 
  # /** \brief Subtract a centroid from a point cloud and return the de-meaned representation
  #   * \param[in] cloud_in the input point cloud
  #   * \param[in] indices the set of point indices to use from the input point cloud
  #   * \param[out] centroid the centroid of the point cloud
  #   * \param cloud_out the resultant output point cloud
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const pcl::PointIndices& indices,
  #                   const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out);
  # 
  # template <typename PointT> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const pcl::PointIndices& indices,
  #                   const Eigen::Vector4f &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out)
  # {
  #   return (demeanPointCloud<PointT, float> (cloud_in, indices, centroid, cloud_out));
  # }
  # 
  # template <typename PointT> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const pcl::PointIndices& indices,
  #                   const Eigen::Vector4d &centroid,
  #                   pcl::PointCloud<PointT> &cloud_out)
  # {
  #   return (demeanPointCloud<PointT, double> (cloud_in, indices, centroid, cloud_out));
  # }
  # 
  # /** \brief Subtract a centroid from a point cloud and return the de-meaned
  #   * representation as an Eigen matrix
  #   * \param[in] cloud_iterator an iterator over the input point cloud
  #   * \param[in] centroid the centroid of the point cloud
  #   * \param[out] cloud_out the resultant output XYZ0 dimensions of \a cloud_in as
  #   * an Eigen matrix (4 rows, N pts columns)
  #   * \param[in] npts the number of samples guaranteed to be left in the input cloud, accessible by the iterator. If not given, it will be calculated.
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> void
  # demeanPointCloud (ConstCloudIterator<PointT> &cloud_iterator,
  #                   const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                   Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &cloud_out,
  #                   int npts = 0);
  # 
  # template <typename PointT> void
  # demeanPointCloud (ConstCloudIterator<PointT> &cloud_iterator,
  #                   const Eigen::Vector4f &centroid,
  #                   Eigen::MatrixXf &cloud_out,
  #                   int npts = 0)
  # {
  #   return (demeanPointCloud<PointT, float> (cloud_iterator, centroid, cloud_out, npts));
  # }
  # 
  # template <typename PointT> void
  # demeanPointCloud (ConstCloudIterator<PointT> &cloud_iterator,
  #                   const Eigen::Vector4d &centroid,
  #                   Eigen::MatrixXd &cloud_out,
  #                   int npts = 0)
  # {
  #   return (demeanPointCloud<PointT, double> (cloud_iterator, centroid, cloud_out, npts));
  # }
  # 
  # /** \brief Subtract a centroid from a point cloud and return the de-meaned
  #   * representation as an Eigen matrix
  #   * \param[in] cloud_in the input point cloud
  #   * \param[in] centroid the centroid of the point cloud
  #   * \param[out] cloud_out the resultant output XYZ0 dimensions of \a cloud_in as
  #   * an Eigen matrix (4 rows, N pts columns)
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                   Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &cloud_out);
  # 
  # template <typename PointT> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const Eigen::Vector4f &centroid,
  #                   Eigen::MatrixXf &cloud_out)
  # {
  #   return (demeanPointCloud<PointT, float> (cloud_in, centroid, cloud_out));
  # }
  # 
  # template <typename PointT> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const Eigen::Vector4d &centroid,
  #                   Eigen::MatrixXd &cloud_out)
  # {
  #   return (demeanPointCloud<PointT, double> (cloud_in, centroid, cloud_out));
  # }
# 
  # /** \brief Subtract a centroid from a point cloud and return the de-meaned
  #   * representation as an Eigen matrix
  #   * \param[in] cloud_in the input point cloud
  #   * \param[in] indices the set of point indices to use from the input point cloud
  #   * \param[in] centroid the centroid of the point cloud
  #   * \param[out] cloud_out the resultant output XYZ0 dimensions of \a cloud_in as
  #   * an Eigen matrix (4 rows, N pts columns)
  #   * \ingroup common
  #   */
  # template <typename PointT, typename Scalar> void
  # demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
  #                   const std::vector<int> &indices,
  #                   const Eigen::Matrix<Scalar, 4, 1> &centroid,
  #                   Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &cloud_out);
# 
#   template <typename PointT> void
#   demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
#                     const std::vector<int> &indices,
#                     const Eigen::Vector4f &centroid,
#                     Eigen::MatrixXf &cloud_out)
#   {
#     return (demeanPointCloud<PointT, float> (cloud_in, indices, centroid, cloud_out));
#   }
# 
#   template <typename PointT> void
#   demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
#                     const std::vector<int> &indices,
#                     const Eigen::Vector4d &centroid,
#                     Eigen::MatrixXd &cloud_out)
#   {
#     return (demeanPointCloud<PointT, double> (cloud_in, indices, centroid, cloud_out));
#   }
# 
#   /** \brief Subtract a centroid from a point cloud and return the de-meaned
#     * representation as an Eigen matrix
#     * \param[in] cloud_in the input point cloud
#     * \param[in] indices the set of point indices to use from the input point cloud
#     * \param[in] centroid the centroid of the point cloud
#     * \param[out] cloud_out the resultant output XYZ0 dimensions of \a cloud_in as
#     * an Eigen matrix (4 rows, N pts columns)
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> void
#   demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
#                     const pcl::PointIndices& indices,
#                     const Eigen::Matrix<Scalar, 4, 1> &centroid,
#                     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &cloud_out);
# 
#   template <typename PointT> void
#   demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
#                     const pcl::PointIndices& indices,
#                     const Eigen::Vector4f &centroid,
#                     Eigen::MatrixXf &cloud_out)
#   {
#     return (demeanPointCloud<PointT, float> (cloud_in, indices, centroid, cloud_out));
#   }
# 
#   template <typename PointT> void
#   demeanPointCloud (const pcl::PointCloud<PointT> &cloud_in,
#                     const pcl::PointIndices& indices,
#                     const Eigen::Vector4d &centroid,
#                     Eigen::MatrixXd &cloud_out)
#   {
#     return (demeanPointCloud<PointT, double> (cloud_in, indices, centroid, cloud_out));
#   }
# 
#   /** \brief Helper functor structure for n-D centroid estimation. */
#   template<typename PointT, typename Scalar>
#   struct NdCentroidFunctor
#   {
#     typedef typename traits::POD<PointT>::type Pod;
# 
#     NdCentroidFunctor (const PointT &p, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &centroid)
#       : f_idx_ (0),
#         centroid_ (centroid),
#         p_ (reinterpret_cast<const Pod&>(p)) { }
# 
#     template<typename Key> inline void operator() ()
#     {
#       typedef typename pcl::traits::datatype<PointT, Key>::type T;
#       const uint8_t* raw_ptr = reinterpret_cast<const uint8_t*>(&p_) + pcl::traits::offset<PointT, Key>::value;
#       const T* data_ptr = reinterpret_cast<const T*>(raw_ptr);
# 
#       // Check if the value is invalid
#       if (!pcl_isfinite (*data_ptr))
#       {
#         f_idx_++;
#         return;
#       }
# 
#       centroid_[f_idx_++] += *data_ptr;
#     }
# 
#     private:
#       int f_idx_;
#       Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &centroid_;
#       const Pod &p_;
#   };
# 
#   /** \brief General, all purpose nD centroid estimation for a set of points using their
#     * indices.
#     * \param cloud the input point cloud
#     * \param centroid the output centroid
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> inline void
#   computeNDCentroid (const pcl::PointCloud<PointT> &cloud, 
#                      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &centroid);
# 
#   template <typename PointT> inline void
#   computeNDCentroid (const pcl::PointCloud<PointT> &cloud, 
#                      Eigen::VectorXf &centroid)
#   {
#     return (computeNDCentroid<PointT, float> (cloud, centroid));
#   }
# 
#   template <typename PointT> inline void
#   computeNDCentroid (const pcl::PointCloud<PointT> &cloud, 
#                      Eigen::VectorXd &centroid)
#   {
#     return (computeNDCentroid<PointT, double> (cloud, centroid));
#   }
# 
#   /** \brief General, all purpose nD centroid estimation for a set of points using their
#     * indices.
#     * \param cloud the input point cloud
#     * \param indices the point cloud indices that need to be used
#     * \param centroid the output centroid
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> inline void
#   computeNDCentroid (const pcl::PointCloud<PointT> &cloud,
#                      const std::vector<int> &indices, 
#                      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &centroid);
# 
#   template <typename PointT> inline void
#   computeNDCentroid (const pcl::PointCloud<PointT> &cloud, 
#                      const std::vector<int> &indices, 
#                      Eigen::VectorXf &centroid)
#   {
#     return (computeNDCentroid<PointT, float> (cloud, indices, centroid));
#   }
# 
#   template <typename PointT> inline void
#   computeNDCentroid (const pcl::PointCloud<PointT> &cloud, 
#                      const std::vector<int> &indices, 
#                      Eigen::VectorXd &centroid)
#   {
#     return (computeNDCentroid<PointT, double> (cloud, indices, centroid));
#   }
# 
#   /** \brief General, all purpose nD centroid estimation for a set of points using their
#     * indices.
#     * \param cloud the input point cloud
#     * \param indices the point cloud indices that need to be used
#     * \param centroid the output centroid
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> inline void
#   computeNDCentroid (const pcl::PointCloud<PointT> &cloud,
#                      const pcl::PointIndices &indices, 
#                      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &centroid);
# 
#   template <typename PointT> inline void
#   computeNDCentroid (const pcl::PointCloud<PointT> &cloud, 
#                      const pcl::PointIndices &indices, 
#                      Eigen::VectorXf &centroid)
#   {
#     return (computeNDCentroid<PointT, float> (cloud, indices, centroid));
#   }
# 
#   template <typename PointT> inline void
#   computeNDCentroid (const pcl::PointCloud<PointT> &cloud, 
#                      const pcl::PointIndices &indices, 
#                      Eigen::VectorXd &centroid)
#   {
#     return (computeNDCentroid<PointT, double> (cloud, indices, centroid));
#   }
# 
# }
# 
# # #include <pcl/common/impl/accumulators.hpp>
# ###
# 
# namespace pcl
# {
# 
#   /** A generic class that computes the centroid of points fed to it.
#     *
#     * Here by "centroid" we denote not just the mean of 3D point coordinates,
#     * but also mean of values in the other data fields. The general-purpose
#     * \ref computeNDCentroid() function also implements this sort of
#     * functionality, however it does it in a "dumb" way, i.e. regardless of the
#     * semantics of the data inside a field it simply averages the values. In
#     * certain cases (e.g. for \c x, \c y, \c z, \c intensity fields) this
#     * behavior is reasonable, however in other cases (e.g. \c rgb, \c rgba,
#     * \c label fields) this does not lead to meaningful results.
#     *
#     * This class is capable of computing the centroid in a "smart" way, i.e.
#     * taking into account the meaning of the data inside fields. Currently the
#     * following fields are supported:
#     *
#     * - XYZ (\c x, \c y, \c z)
#     *
#     *   Separate average for each field.
#     *
#     * - Normal (\c normal_x, \c normal_y, \c normal_z)
#     *
#     *   Separate average for each field, and the resulting vector is normalized.
#     *
#     * - Curvature (\c curvature)
#     *
#     *   Average.
#     *
#     * - RGB/RGBA (\c rgb or \c rgba)
#     *
#     *   Separate average for R, G, B, and alpha channels.
#     *
#     * - Intensity (\c intensity)
#     *
#     *   Average.
#     *
#     * - Label (\c label)
#     *
#     *   Majority vote. If several labels have the same largest support then the
#     *   smaller label wins.
#     *
#     * The template parameter defines the type of points that may be accumulated
#     * with this class. This may be an arbitrary PCL point type, and centroid
#     * computation will happen only for the fields that are present in it and are
#     * supported.
#     *
#     * Current centroid may be retrieved at any time using get(). Note that the
#     * function is templated on point type, so it is possible to fetch the
#     * centroid into a point type that differs from the type of points that are
#     * being accumulated. All the "extra" fields for which the centroid is not
#     * being calculated will be left untouched.
#     *
#     * Example usage:
#     *
#     * \code
#     * // Create and accumulate points
#     * CentroidPoint<pcl::PointXYZ> centroid;
#     * centroid.add (pcl::PointXYZ (1, 2, 3);
#     * centroid.add (pcl::PointXYZ (5, 6, 7);
#     * // Fetch centroid using `get()`
#     * pcl::PointXYZ c1;
#     * centroid.get (c1);
#     * // The expected result is: c1.x == 3, c1.y == 4, c1.z == 5
#     * // It is also okay to use `get()` with a different point type
#     * pcl::PointXYZRGB c2;
#     * centroid.get (c2);
#     * // The expected result is: c2.x == 3, c2.y == 4, c2.z == 5,
#     * // and c2.rgb is left untouched
#     * \endcode
#     *
#     * \note Assumes that the points being inserted are valid.
#     *
#     * \note This class template can be successfully instantiated for *any*
#     * PCL point type. Of course, each of the field averages is computed only if
#     * the point type has the corresponding field.
#     *
#     * \ingroup common
#     * \author Sergey Alexandrov */
#   template <typename PointT>
#   class CentroidPoint
#   {
# 
#     public:
# 
#       CentroidPoint ()
#       : num_points_ (0)
#       {
#       }
# 
#       /** Add a new point to the centroid computation.
#         *
#         * In this function only the accumulators and point counter are updated,
#         * actual centroid computation does not happen until get() is called. */
#       void
#       add (const PointT& point)
#       {
#         // Invoke add point on each accumulator
#         boost::fusion::for_each (accumulators_, detail::AddPoint<PointT> (point));
#         ++num_points_;
#       }
# 
#       /** Retrieve the current centroid.
#         *
#         * Computation (division of accumulated values by the number of points
#         * and normalization where applicable) happens here. The result is not
#         * cached, so any subsequent call to this function will trigger
#         * re-computation.
#         *
#         * If the number of accumulated points is zero, then the point will be
#         * left untouched. */
#       template <typename PointOutT> void
#       get (PointOutT& point) const
#       {
#         if (num_points_ != 0)
#         {
#           // Filter accumulators so that only those that are compatible with
#           // both PointT and requested point type remain
#           typename pcl::detail::Accumulators<PointT, PointOutT>::type ca (accumulators_);
#           // Invoke get point on each accumulator in filtered list
#           boost::fusion::for_each (ca, detail::GetPoint<PointOutT> (point, num_points_));
#         }
#       }
# 
#       /** Get the total number of points that were added. */
#       size_t
#       getSize () const
#       {
#         return (num_points_);
#       }
# 
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
# 
#     private:
# 
#       size_t num_points_;
#       typename pcl::detail::Accumulators<PointT>::type accumulators_;
# 
#   };
# 
#   /** Compute the centroid of a set of points and return it as a point.
#     *
#     * Implementation leverages \ref CentroidPoint class and therefore behaves
#     * differently from \ref compute3DCentroid() and \ref computeNDCentroid().
#     * See \ref CentroidPoint documentation for explanation.
#     *
#     * \param[in] cloud input point cloud
#     * \param[out] centroid output centroid
#     *
#     * \return number of valid points used to determine the centroid (will be the
#     * same as the size of the cloud if it is dense)
#     *
#     * \note If return value is \c 0, then the centroid is not changed, thus is
#     * not valid.
#     *
#     * \ingroup common */
#   template <typename PointInT, typename PointOutT> size_t
#   computeCentroid (const pcl::PointCloud<PointInT>& cloud,
#                    PointOutT& centroid);
# 
#   /** Compute the centroid of a set of points and return it as a point.
#     * \param[in] cloud
#     * \param[in] indices point cloud indices that need to be used
#     * \param[out] centroid
#     * This is an overloaded function provided for convenience. See the
#     * documentation for computeCentroid().
#     *
#     * \ingroup common */
#   template <typename PointInT, typename PointOutT> size_t
#   computeCentroid (const pcl::PointCloud<PointInT>& cloud,
#                    const std::vector<int>& indices,
#                    PointOutT& centroid);
# 
# }
# /*@}*/
# # #include <pcl/common/impl/centroid.hpp>
# # #endif  //#ifndef PCL_COMMON_CENTROID_H_
# ###
# 
# # common.h
# namespace pcl
# {
#   /** \brief Compute the smallest angle between two vectors in the [ 0, PI ) interval in 3D.
#     * \param v1 the first 3D vector (represented as a \a Eigen::Vector4f)
#     * \param v2 the second 3D vector (represented as a \a Eigen::Vector4f)
#     * \return the angle between v1 and v2
#     * \ingroup common
#     */
#   inline double 
#   getAngle3D (const Eigen::Vector4f &v1, const Eigen::Vector4f &v2);
# 
#   /** \brief Compute both the mean and the standard deviation of an array of values
#     * \param values the array of values
#     * \param mean the resultant mean of the distribution
#     * \param stddev the resultant standard deviation of the distribution
#     * \ingroup common
#     */
#   inline void 
#   getMeanStd (const std::vector<float> &values, double &mean, double &stddev);
# 
#   /** \brief Get a set of points residing in a box given its bounds
#     * \param cloud the point cloud data message
#     * \param min_pt the minimum bounds
#     * \param max_pt the maximum bounds
#     * \param indices the resultant set of point indices residing in the box
#     * \ingroup common
#     */
#   template <typename PointT> inline void 
#   getPointsInBox (const pcl::PointCloud<PointT> &cloud, Eigen::Vector4f &min_pt,
#                   Eigen::Vector4f &max_pt, std::vector<int> &indices);
# 
#   /** \brief Get the point at maximum distance from a given point and a given pointcloud
#     * \param cloud the point cloud data message
#     * \param pivot_pt the point from where to compute the distance
#     * \param max_pt the point in cloud that is the farthest point away from pivot_pt
#     * \ingroup common
#     */
#   template<typename PointT> inline void
#   getMaxDistance (const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt);
# 
#   /** \brief Get the point at maximum distance from a given point and a given pointcloud
#     * \param cloud the point cloud data message
#     * \param pivot_pt the point from where to compute the distance
#     * \param indices the vector of point indices to use from \a cloud
#     * \param max_pt the point in cloud that is the farthest point away from pivot_pt
#     * \ingroup common
#     */
#   template<typename PointT> inline void
#   getMaxDistance (const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices, 
#                   const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt);
# 
#   /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud
#     * \param cloud the point cloud data message
#     * \param min_pt the resultant minimum bounds
#     * \param max_pt the resultant maximum bounds
#     * \ingroup common
#     */
#   template <typename PointT> inline void 
#   getMinMax3D (const pcl::PointCloud<PointT> &cloud, PointT &min_pt, PointT &max_pt);
#   
#   /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud
#     * \param cloud the point cloud data message
#     * \param min_pt the resultant minimum bounds
#     * \param max_pt the resultant maximum bounds
#     * \ingroup common
#     */
#   template <typename PointT> inline void 
#   getMinMax3D (const pcl::PointCloud<PointT> &cloud, 
#                Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt);
# 
#   /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud
#     * \param cloud the point cloud data message
#     * \param indices the vector of point indices to use from \a cloud
#     * \param min_pt the resultant minimum bounds
#     * \param max_pt the resultant maximum bounds
#     * \ingroup common
#     */
#   template <typename PointT> inline void 
#   getMinMax3D (const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices, 
#                Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt);
# 
#   /** \brief Get the minimum and maximum values on each of the 3 (x-y-z) dimensions in a given pointcloud
#     * \param cloud the point cloud data message
#     * \param indices the vector of point indices to use from \a cloud
#     * \param min_pt the resultant minimum bounds
#     * \param max_pt the resultant maximum bounds
#     * \ingroup common
#     */
#   template <typename PointT> inline void 
#   getMinMax3D (const pcl::PointCloud<PointT> &cloud, const pcl::PointIndices &indices, 
#                Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt);
# 
#   /** \brief Compute the radius of a circumscribed circle for a triangle formed of three points pa, pb, and pc
#     * \param pa the first point
#     * \param pb the second point
#     * \param pc the third point
#     * \return the radius of the circumscribed circle
#     * \ingroup common
#     */
#   template <typename PointT> inline double 
#   getCircumcircleRadius (const PointT &pa, const PointT &pb, const PointT &pc);
# 
#   /** \brief Get the minimum and maximum values on a point histogram
#     * \param histogram the point representing a multi-dimensional histogram
#     * \param len the length of the histogram
#     * \param min_p the resultant minimum 
#     * \param max_p the resultant maximum 
#     * \ingroup common
#     */
#   template <typename PointT> inline void 
#   getMinMax (const PointT &histogram, int len, float &min_p, float &max_p);
# 
#   /** \brief Calculate the area of a polygon given a point cloud that defines the polygon 
#       * \param polygon point cloud that contains those vertices that comprises the polygon. Vertices are stored in counterclockwise.
#       * \return the polygon area 
#       * \ingroup common
#       */
#   template<typename PointT> inline float
#   calculatePolygonArea (const pcl::PointCloud<PointT> &polygon);
# 
#   /** \brief Get the minimum and maximum values on a point histogram
#     * \param cloud the cloud containing multi-dimensional histograms
#     * \param idx point index representing the histogram that we need to compute min/max for
#     * \param field_name the field name containing the multi-dimensional histogram
#     * \param min_p the resultant minimum 
#     * \param max_p the resultant maximum 
#     * \ingroup common
#     */
#   PCL_EXPORTS void 
#   getMinMax (const pcl::PCLPointCloud2 &cloud, int idx, const std::string &field_name,
#              float &min_p, float &max_p);
# 
#   /** \brief Compute both the mean and the standard deviation of an array of values
#     * \param values the array of values
#     * \param mean the resultant mean of the distribution
#     * \param stddev the resultant standard deviation of the distribution
#     * \ingroup common
#     */
#   PCL_EXPORTS void
#   getMeanStdDev (const std::vector<float> &values, double &mean, double &stddev);
# 
# }
# /*@}*/
# #include <pcl/common/impl/common.hpp>
# 
# #endif  //#ifndef PCL_COMMON_H_
# ###
# 
# # common_headers.h
# # #include <pcl/common/common.h>
# # #include <pcl/common/norms.h>
# # #include <pcl/common/angles.h>
# # #include <pcl/common/time.h>
# # #include <pcl/common/file_io.h>
# # #include <pcl/common/eigen.h>
# ###
# 
# # concatenate.h
# // We're doing a lot of black magic with Boost here, so disable warnings in Maintainer mode, as we will never
# // be able to fix them anyway
# #ifdef BUILD_Maintainer
# #  if defined __GNUC__
# #    if __GNUC__ == 4 && __GNUC_MINOR__ > 3
# #      pragma GCC diagnostic ignored "-Weffc++"
# #      pragma GCC diagnostic ignored "-pedantic"
# #    else
# #      pragma GCC system_header 
# #    endif
# #  elif defined _MSC_VER
# #    pragma warning(push, 1)
# #  endif
# #endif
# 
# namespace pcl
# {
#   /** \brief Helper functor structure for concatenate. 
#     * \ingroup common
#     */
#   template<typename PointInT, typename PointOutT>
#   struct NdConcatenateFunctor
#   {
#     typedef typename traits::POD<PointInT>::type PodIn;
#     typedef typename traits::POD<PointOutT>::type PodOut;
#     
#     NdConcatenateFunctor (const PointInT &p1, PointOutT &p2)
#       : p1_ (reinterpret_cast<const PodIn&> (p1))
#       , p2_ (reinterpret_cast<PodOut&> (p2)) { }
# 
#     template<typename Key> inline void 
#     operator () ()
#     {
#       // This sucks without Fusion :(
#       //boost::fusion::at_key<Key> (p2_) = boost::fusion::at_key<Key> (p1_);
#       typedef typename pcl::traits::datatype<PointInT, Key>::type InT;
#       typedef typename pcl::traits::datatype<PointOutT, Key>::type OutT;
#       // Note: don't currently support different types for the same field (e.g. converting double to float)
#       BOOST_MPL_ASSERT_MSG ((boost::is_same<InT, OutT>::value),
#                             POINT_IN_AND_POINT_OUT_HAVE_DIFFERENT_TYPES_FOR_FIELD,
#                             (Key, PointInT&, InT, PointOutT&, OutT));
#       memcpy (reinterpret_cast<uint8_t*>(&p2_) + pcl::traits::offset<PointOutT, Key>::value,
#               reinterpret_cast<const uint8_t*>(&p1_) + pcl::traits::offset<PointInT, Key>::value,
#               sizeof (InT));
#     }
# 
#     private:
#       const PodIn &p1_;
#       PodOut &p2_;
#   };
# }
# 
# #ifdef BUILD_Maintainer
# #  if defined __GNUC__
# #    if __GNUC__ == 4 && __GNUC_MINOR__ > 3
# #      pragma GCC diagnostic warning "-Weffc++"
# #      pragma GCC diagnostic warning "-pedantic"
# #    endif
# #  elif defined _MSC_VER
# #    pragma warning(pop)
# #  endif
# #endif
# 
# #endif // PCL_COMMON_CONCATENATE_H_
# ###
# 
# # conversions.h
# namespace pcl
# {
#   namespace detail
#   {
#     // For converting template point cloud to message.
#     template<typename PointT>
#     struct FieldAdder
#     {
#       FieldAdder (std::vector<pcl::PCLPointField>& fields) : fields_ (fields) {};
# 
#       template<typename U> void operator() ()
#       {
#         pcl::PCLPointField f;
#         f.name = traits::name<PointT, U>::value;
#         f.offset = traits::offset<PointT, U>::value;
#         f.datatype = traits::datatype<PointT, U>::value;
#         f.count = traits::datatype<PointT, U>::size;
#         fields_.push_back (f);
#       }
# 
#       std::vector<pcl::PCLPointField>& fields_;
#     };
# 
#     // For converting message to template point cloud.
#     template<typename PointT>
#     struct FieldMapper
#     {
#       FieldMapper (const std::vector<pcl::PCLPointField>& fields,
#                    std::vector<FieldMapping>& map)
#         : fields_ (fields), map_ (map)
#       {
#       }
# 
#       template<typename Tag> void
#       operator () ()
#       {
#         BOOST_FOREACH (const pcl::PCLPointField& field, fields_)
#         {
#           if (FieldMatches<PointT, Tag>()(field))
#           {
#             FieldMapping mapping;
#             mapping.serialized_offset = field.offset;
#             mapping.struct_offset = traits::offset<PointT, Tag>::value;
#             mapping.size = sizeof (typename traits::datatype<PointT, Tag>::type);
#             map_.push_back (mapping);
#             return;
#           }
#         }
#         // Disable thrown exception per #595: http://dev.pointclouds.org/issues/595
#         PCL_WARN ("Failed to find match for field '%s'.\n", traits::name<PointT, Tag>::value);
#         //throw pcl::InvalidConversionException (ss.str ());
#       }
# 
#       const std::vector<pcl::PCLPointField>& fields_;
#       std::vector<FieldMapping>& map_;
#     };
# 
#     inline bool
#     fieldOrdering (const FieldMapping& a, const FieldMapping& b)
#     {
#       return (a.serialized_offset < b.serialized_offset);
#     }
# 
#   } //namespace detail
# 
#   template<typename PointT> void
#   createMapping (const std::vector<pcl::PCLPointField>& msg_fields, MsgFieldMap& field_map)
#   {
#     // Create initial 1-1 mapping between serialized data segments and struct fields
#     detail::FieldMapper<PointT> mapper (msg_fields, field_map);
#     for_each_type< typename traits::fieldList<PointT>::type > (mapper);
# 
#     // Coalesce adjacent fields into single memcpy's where possible
#     if (field_map.size() > 1)
#     {
#       std::sort(field_map.begin(), field_map.end(), detail::fieldOrdering);
#       MsgFieldMap::iterator i = field_map.begin(), j = i + 1;
#       while (j != field_map.end())
#       {
#         // This check is designed to permit padding between adjacent fields.
#         /// @todo One could construct a pathological case where the struct has a
#         /// field where the serialized data has padding
#         if (j->serialized_offset - i->serialized_offset == j->struct_offset - i->struct_offset)
#         {
#           i->size += (j->struct_offset + j->size) - (i->struct_offset + i->size);
#           j = field_map.erase(j);
#         }
#         else
#         {
#           ++i;
#           ++j;
#         }
#       }
#     }
#   }
# 
#   /** \brief Convert a PCLPointCloud2 binary data blob into a pcl::PointCloud<T> object using a field_map.
#     * \param[in] msg the PCLPointCloud2 binary blob
#     * \param[out] cloud the resultant pcl::PointCloud<T>
#     * \param[in] field_map a MsgFieldMap object
#     *
#     * \note Use fromPCLPointCloud2 (PCLPointCloud2, PointCloud<T>) directly or create you
#     * own MsgFieldMap using:
#     *
#     * \code
#     * MsgFieldMap field_map;
#     * createMapping<PointT> (msg.fields, field_map);
#     * \endcode
#     */
#   template <typename PointT> void
#   fromPCLPointCloud2 (const pcl::PCLPointCloud2& msg, pcl::PointCloud<PointT>& cloud,
#               const MsgFieldMap& field_map)
#   {
#     // Copy info fields
#     cloud.header   = msg.header;
#     cloud.width    = msg.width;
#     cloud.height   = msg.height;
#     cloud.is_dense = msg.is_dense == 1;
# 
#     // Copy point data
#     uint32_t num_points = msg.width * msg.height;
#     cloud.points.resize (num_points);
#     uint8_t* cloud_data = reinterpret_cast<uint8_t*>(&cloud.points[0]);
# 
#     // Check if we can copy adjacent points in a single memcpy
#     if (field_map.size() == 1 &&
#         field_map[0].serialized_offset == 0 &&
#         field_map[0].struct_offset == 0 &&
#         msg.point_step == sizeof(PointT))
#     {
#       uint32_t cloud_row_step = static_cast<uint32_t> (sizeof (PointT) * cloud.width);
#       const uint8_t* msg_data = &msg.data[0];
#       // Should usually be able to copy all rows at once
#       if (msg.row_step == cloud_row_step)
#       {
#         memcpy (cloud_data, msg_data, msg.data.size ());
#       }
#       else
#       {
#         for (uint32_t i = 0; i < msg.height; ++i, cloud_data += cloud_row_step, msg_data += msg.row_step)
#           memcpy (cloud_data, msg_data, cloud_row_step);
#       }
# 
#     }
#     else
#     {
#       // If not, memcpy each group of contiguous fields separately
#       for (uint32_t row = 0; row < msg.height; ++row)
#       {
#         const uint8_t* row_data = &msg.data[row * msg.row_step];
#         for (uint32_t col = 0; col < msg.width; ++col)
#         {
#           const uint8_t* msg_data = row_data + col * msg.point_step;
#           BOOST_FOREACH (const detail::FieldMapping& mapping, field_map)
#           {
#             memcpy (cloud_data + mapping.struct_offset, msg_data + mapping.serialized_offset, mapping.size);
#           }
#           cloud_data += sizeof (PointT);
#         }
#       }
#     }
#   }
# 
#   /** \brief Convert a PCLPointCloud2 binary data blob into a pcl::PointCloud<T> object.
#     * \param[in] msg the PCLPointCloud2 binary blob
#     * \param[out] cloud the resultant pcl::PointCloud<T>
#     */
#   template<typename PointT> void
#   fromPCLPointCloud2 (const pcl::PCLPointCloud2& msg, pcl::PointCloud<PointT>& cloud)
#   {
#     MsgFieldMap field_map;
#     createMapping<PointT> (msg.fields, field_map);
#     fromPCLPointCloud2 (msg, cloud, field_map);
#   }
# 
#   /** \brief Convert a pcl::PointCloud<T> object to a PCLPointCloud2 binary data blob.
#     * \param[in] cloud the input pcl::PointCloud<T>
#     * \param[out] msg the resultant PCLPointCloud2 binary blob
#     */
#   template<typename PointT> void
#   toPCLPointCloud2 (const pcl::PointCloud<PointT>& cloud, pcl::PCLPointCloud2& msg)
#   {
#     // Ease the user's burden on specifying width/height for unorganized datasets
#     if (cloud.width == 0 && cloud.height == 0)
#     {
#       msg.width  = static_cast<uint32_t>(cloud.points.size ());
#       msg.height = 1;
#     }
#     else
#     {
#       assert (cloud.points.size () == cloud.width * cloud.height);
#       msg.height = cloud.height;
#       msg.width  = cloud.width;
#     }
# 
#     // Fill point cloud binary data (padding and all)
#     size_t data_size = sizeof (PointT) * cloud.points.size ();
#     msg.data.resize (data_size);
#     memcpy (&msg.data[0], &cloud.points[0], data_size);
# 
#     // Fill fields metadata
#     msg.fields.clear ();
#     for_each_type<typename traits::fieldList<PointT>::type> (detail::FieldAdder<PointT>(msg.fields));
# 
#     msg.header     = cloud.header;
#     msg.point_step = sizeof (PointT);
#     msg.row_step   = static_cast<uint32_t> (sizeof (PointT) * msg.width);
#     msg.is_dense   = cloud.is_dense;
#     /// @todo msg.is_bigendian = ?;
#   }
# 
#    /** \brief Copy the RGB fields of a PointCloud into pcl::PCLImage format
#      * \param[in] cloud the point cloud message
#      * \param[out] msg the resultant pcl::PCLImage
#      * CloudT cloud type, CloudT should be akin to pcl::PointCloud<pcl::PointXYZRGBA>
#      * \note will throw std::runtime_error if there is a problem
#      */
#   template<typename CloudT> void
#   toPCLPointCloud2 (const CloudT& cloud, pcl::PCLImage& msg)
#   {
#     // Ease the user's burden on specifying width/height for unorganized datasets
#     if (cloud.width == 0 && cloud.height == 0)
#       throw std::runtime_error("Needs to be a dense like cloud!!");
#     else
#     {
#       if (cloud.points.size () != cloud.width * cloud.height)
#         throw std::runtime_error("The width and height do not match the cloud size!");
#       msg.height = cloud.height;
#       msg.width = cloud.width;
#     }
# 
#     // ensor_msgs::image_encodings::BGR8;
#     msg.encoding = "bgr8";
#     msg.step = msg.width * sizeof (uint8_t) * 3;
#     msg.data.resize (msg.step * msg.height);
#     for (size_t y = 0; y < cloud.height; y++)
#     {
#       for (size_t x = 0; x < cloud.width; x++)
#       {
#         uint8_t * pixel = &(msg.data[y * msg.step + x * 3]);
#         memcpy (pixel, &cloud (x, y).rgb, 3 * sizeof(uint8_t));
#       }
#     }
#   }
# 
#   /** \brief Copy the RGB fields of a PCLPointCloud2 msg into pcl::PCLImage format
#     * \param cloud the point cloud message
#     * \param msg the resultant pcl::PCLImage
#     * will throw std::runtime_error if there is a problem
#     */
#   inline void
#   toPCLPointCloud2 (const pcl::PCLPointCloud2& cloud, pcl::PCLImage& msg)
#   {
#     int rgb_index = -1;
#     // Get the index we need
#     for (size_t d = 0; d < cloud.fields.size (); ++d)
#       if (cloud.fields[d].name == "rgb")
#       {
#         rgb_index = static_cast<int>(d);
#         break;
#       }
# 
#     if(rgb_index == -1)
#       throw std::runtime_error ("No rgb field!!");
#     if (cloud.width == 0 && cloud.height == 0)
#       throw std::runtime_error ("Needs to be a dense like cloud!!");
#     else
#     {
#       msg.height = cloud.height;
#       msg.width = cloud.width;
#     }
#     int rgb_offset = cloud.fields[rgb_index].offset;
#     int point_step = cloud.point_step;
# 
#     // pcl::image_encodings::BGR8;
#     msg.encoding = "bgr8";
#     msg.step = static_cast<uint32_t>(msg.width * sizeof (uint8_t) * 3);
#     msg.data.resize (msg.step * msg.height);
# 
#     for (size_t y = 0; y < cloud.height; y++)
#     {
#       for (size_t x = 0; x < cloud.width; x++, rgb_offset += point_step)
#       {
#         uint8_t * pixel = &(msg.data[y * msg.step + x * 3]);
#         memcpy (pixel, &(cloud.data[rgb_offset]), 3 * sizeof (uint8_t));
#       }
#     }
#   }
# }
# 
# #endif  //#ifndef PCL_CONVERSIONS_H_
# ###
# 
# # copy_point.h
# namespace pcl
# {
# 
#   /** \brief Copy the fields of a source point into a target point.
#     *
#     * If the source and the target point types are the same, then a complete
#     * copy is made. Otherwise only those fields that the two point types share
#     * in common are copied.
#     *
#     * \param[in]  point_in the source point
#     * \param[out] point_out the target point
#     *
#     * \ingroup common */
#   template <typename PointInT, typename PointOutT> void
#   copyPoint (const PointInT& point_in, PointOutT& point_out);
# 
# }
# 
# #include <pcl/common/impl/copy_point.hpp>
# #endif // PCL_COMMON_COPY_POINT_H_
# ###
# 
# 
# # distances.h
# namespace pcl
# {
#   /** \brief Get the shortest 3D segment between two 3D lines
#     * \param line_a the coefficients of the first line (point, direction)
#     * \param line_b the coefficients of the second line (point, direction)
#     * \param pt1_seg the first point on the line segment
#     * \param pt2_seg the second point on the line segment
#     * \ingroup common
#     */
#   PCL_EXPORTS void
#   lineToLineSegment (const Eigen::VectorXf &line_a, const Eigen::VectorXf &line_b, 
#                      Eigen::Vector4f &pt1_seg, Eigen::Vector4f &pt2_seg);
# 
#   /** \brief Get the square distance from a point to a line (represented by a point and a direction)
#     * \param pt a point
#     * \param line_pt a point on the line (make sure that line_pt[3] = 0 as there are no internal checks!)
#     * \param line_dir the line direction
#     * \ingroup common
#     */
#   double inline
#   sqrPointToLineDistance (const Eigen::Vector4f &pt, const Eigen::Vector4f &line_pt, const Eigen::Vector4f &line_dir)
#   {
#     // Calculate the distance from the point to the line
#     // D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p1-p0)) / norm(p2-p1)
#     return (line_dir.cross3 (line_pt - pt)).squaredNorm () / line_dir.squaredNorm ();
#   }
# 
#   /** \brief Get the square distance from a point to a line (represented by a point and a direction)
#     * \note This one is useful if one has to compute many distances to a fixed line, so the vector length can be pre-computed
#     * \param pt a point
#     * \param line_pt a point on the line (make sure that line_pt[3] = 0 as there are no internal checks!)
#     * \param line_dir the line direction
#     * \param sqr_length the squared norm of the line direction
#     * \ingroup common
#     */
#   double inline
#   sqrPointToLineDistance (const Eigen::Vector4f &pt, const Eigen::Vector4f &line_pt, const Eigen::Vector4f &line_dir, const double sqr_length)
#   {
#     // Calculate the distance from the point to the line
#     // D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p1-p0)) / norm(p2-p1)
#     return (line_dir.cross3 (line_pt - pt)).squaredNorm () / sqr_length;
#   }
# 
#   /** \brief Obtain the maximum segment in a given set of points, and return the minimum and maximum points.
#     * \param[in] cloud the point cloud dataset
#     * \param[out] pmin the coordinates of the "minimum" point in \a cloud (one end of the segment)
#     * \param[out] pmax the coordinates of the "maximum" point in \a cloud (the other end of the segment)
#     * \return the length of segment length
#     * \ingroup common
#     */
#   template <typename PointT> double inline
#   getMaxSegment (const pcl::PointCloud<PointT> &cloud, 
#                  PointT &pmin, PointT &pmax)
#   {
#     double max_dist = std::numeric_limits<double>::min ();
#     int i_min = -1, i_max = -1;
# 
#     for (size_t i = 0; i < cloud.points.size (); ++i)
#     {
#       for (size_t j = i; j < cloud.points.size (); ++j)
#       {
#         // Compute the distance 
#         double dist = (cloud.points[i].getVector4fMap () - 
#                        cloud.points[j].getVector4fMap ()).squaredNorm ();
#         if (dist <= max_dist)
#           continue;
# 
#         max_dist = dist;
#         i_min = i;
#         i_max = j;
#       }
#     }
# 
#     if (i_min == -1 || i_max == -1)
#       return (max_dist = std::numeric_limits<double>::min ());
# 
#     pmin = cloud.points[i_min];
#     pmax = cloud.points[i_max];
#     return (std::sqrt (max_dist));
#   }
#  
#   /** \brief Obtain the maximum segment in a given set of points, and return the minimum and maximum points.
#     * \param[in] cloud the point cloud dataset
#     * \param[in] indices a set of point indices to use from \a cloud
#     * \param[out] pmin the coordinates of the "minimum" point in \a cloud (one end of the segment)
#     * \param[out] pmax the coordinates of the "maximum" point in \a cloud (the other end of the segment)
#     * \return the length of segment length
#     * \ingroup common
#     */
#   template <typename PointT> double inline
#   getMaxSegment (const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices,
#                  PointT &pmin, PointT &pmax)
#   {
#     double max_dist = std::numeric_limits<double>::min ();
#     int i_min = -1, i_max = -1;
# 
#     for (size_t i = 0; i < indices.size (); ++i)
#     {
#       for (size_t j = i; j < indices.size (); ++j)
#       {
#         // Compute the distance 
#         double dist = (cloud.points[indices[i]].getVector4fMap () - 
#                        cloud.points[indices[j]].getVector4fMap ()).squaredNorm ();
#         if (dist <= max_dist)
#           continue;
# 
#         max_dist = dist;
#         i_min = i;
#         i_max = j;
#       }
#     }
# 
#     if (i_min == -1 || i_max == -1)
#       return (max_dist = std::numeric_limits<double>::min ());
# 
#     pmin = cloud.points[indices[i_min]];
#     pmax = cloud.points[indices[i_max]];
#     return (std::sqrt (max_dist));
#   }
# 
#   /** \brief Calculate the squared euclidean distance between the two given points.
#     * \param[in] p1 the first point
#     * \param[in] p2 the second point
#     */
#   template<typename PointType1, typename PointType2> inline float
#   squaredEuclideanDistance (const PointType1& p1, const PointType2& p2)
#   {
#     float diff_x = p2.x - p1.x, diff_y = p2.y - p1.y, diff_z = p2.z - p1.z;
#     return (diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);
#   }
# 
#   /** \brief Calculate the squared euclidean distance between the two given points.
#     * \param[in] p1 the first point
#     * \param[in] p2 the second point
#     */
#   template<> inline float
#   squaredEuclideanDistance (const PointXY& p1, const PointXY& p2)
#   {
#     float diff_x = p2.x - p1.x, diff_y = p2.y - p1.y;
#     return (diff_x*diff_x + diff_y*diff_y);
#   }
# 
#    /** \brief Calculate the euclidean distance between the two given points.
#     * \param[in] p1 the first point
#     * \param[in] p2 the second point
#     */
#   template<typename PointType1, typename PointType2> inline float
#   euclideanDistance (const PointType1& p1, const PointType2& p2)
#   {
#     return (sqrtf (squaredEuclideanDistance (p1, p2)));
#   }
# }
# /*@*/
# #endif  //#ifndef PCL_DISTANCES_H_
# ###
# 
# 
# # eigen.h
# #ifndef NOMINMAX
# #define NOMINMAX
# #endif
# 
# #if defined __GNUC__
# #  pragma GCC system_header
# #elif defined __SUNPRO_CC
# #  pragma disable_warn
# #endif
# 
# #include <cmath>
# #include <pcl/ModelCoefficients.h>
# 
# #include <Eigen/StdVector>
# #include <Eigen/Core>
# #include <Eigen/Eigenvalues>
# #include <Eigen/Geometry>
# #include <Eigen/SVD>
# #include <Eigen/LU>
# #include <Eigen/Dense>
# #include <Eigen/Eigenvalues>
# 
# namespace pcl
# {
#   /** \brief Compute the roots of a quadratic polynom x^2 + b*x + c = 0
#     * \param[in] b linear parameter
#     * \param[in] c constant parameter
#     * \param[out] roots solutions of x^2 + b*x + c = 0
#     */
#   template <typename Scalar, typename Roots> void
#   computeRoots2 (const Scalar &b, const Scalar &c, Roots &roots);
# 
#   /** \brief computes the roots of the characteristic polynomial of the input matrix m, which are the eigenvalues
#     * \param[in] m input matrix
#     * \param[out] roots roots of the characteristic polynomial of the input matrix m, which are the eigenvalues
#     */
#   template <typename Matrix, typename Roots> void
#   computeRoots (const Matrix &m, Roots &roots);
# 
#   /** \brief determine the smallest eigenvalue and its corresponding eigenvector
#     * \param[in] mat input matrix that needs to be symmetric and positive semi definite
#     * \param[out] eigenvalue the smallest eigenvalue of the input matrix
#     * \param[out] eigenvector the corresponding eigenvector to the smallest eigenvalue of the input matrix
#     * \ingroup common
#     */
#   template <typename Matrix, typename Vector> void
#   eigen22 (const Matrix &mat, typename Matrix::Scalar &eigenvalue, Vector &eigenvector);
# 
#   /** \brief determine the smallest eigenvalue and its corresponding eigenvector
#     * \param[in] mat input matrix that needs to be symmetric and positive semi definite
#     * \param[out] eigenvectors the corresponding eigenvector to the smallest eigenvalue of the input matrix
#     * \param[out] eigenvalues the smallest eigenvalue of the input matrix
#     * \ingroup common
#     */
#   template <typename Matrix, typename Vector> void
#   eigen22 (const Matrix &mat, Matrix &eigenvectors, Vector &eigenvalues);
# 
#   /** \brief determines the corresponding eigenvector to the given eigenvalue of the symmetric positive semi definite input matrix
#     * \param[in] mat symmetric positive semi definite input matrix
#     * \param[in] eigenvalue the eigenvalue which corresponding eigenvector is to be computed
#     * \param[out] eigenvector the corresponding eigenvector for the input eigenvalue
#     * \ingroup common
#     */
#   template <typename Matrix, typename Vector> void
#   computeCorrespondingEigenVector (const Matrix &mat, const typename Matrix::Scalar &eigenvalue, Vector &eigenvector);
#   
#   /** \brief determines the eigenvector and eigenvalue of the smallest eigenvalue of the symmetric positive semi definite input matrix
#     * \param[in] mat symmetric positive semi definite input matrix
#     * \param[out] eigenvalue smallest eigenvalue of the input matrix
#     * \param[out] eigenvector the corresponding eigenvector for the input eigenvalue
#     * \note if the smallest eigenvalue is not unique, this function may return any eigenvector that is consistent to the eigenvalue.
#     * \ingroup common
#     */
#   template <typename Matrix, typename Vector> void
#   eigen33 (const Matrix &mat, typename Matrix::Scalar &eigenvalue, Vector &eigenvector);
# 
#   /** \brief determines the eigenvalues of the symmetric positive semi definite input matrix
#     * \param[in] mat symmetric positive semi definite input matrix
#     * \param[out] evals resulting eigenvalues in ascending order
#     * \ingroup common
#     */
#   template <typename Matrix, typename Vector> void
#   eigen33 (const Matrix &mat, Vector &evals);
# 
#   /** \brief determines the eigenvalues and corresponding eigenvectors of the symmetric positive semi definite input matrix
#     * \param[in] mat symmetric positive semi definite input matrix
#     * \param[out] evecs resulting eigenvalues in ascending order
#     * \param[out] evals corresponding eigenvectors in correct order according to eigenvalues
#     * \ingroup common
#     */
#   template <typename Matrix, typename Vector> void
#   eigen33 (const Matrix &mat, Matrix &evecs, Vector &evals);
# 
#   /** \brief Calculate the inverse of a 2x2 matrix
#     * \param[in] matrix matrix to be inverted
#     * \param[out] inverse the resultant inverted matrix
#     * \note only the upper triangular part is taken into account => non symmetric matrices will give wrong results
#     * \return determinant of the original matrix => if 0 no inverse exists => result is invalid
#     * \ingroup common
#     */
#   template <typename Matrix> typename Matrix::Scalar
#   invert2x2 (const Matrix &matrix, Matrix &inverse);
# 
#   /** \brief Calculate the inverse of a 3x3 symmetric matrix.
#     * \param[in] matrix matrix to be inverted
#     * \param[out] inverse the resultant inverted matrix
#     * \note only the upper triangular part is taken into account => non symmetric matrices will give wrong results
#     * \return determinant of the original matrix => if 0 no inverse exists => result is invalid
#     * \ingroup common
#     */
#   template <typename Matrix> typename Matrix::Scalar
#   invert3x3SymMatrix (const Matrix &matrix, Matrix &inverse);
# 
#   /** \brief Calculate the inverse of a general 3x3 matrix.
#     * \param[in] matrix matrix to be inverted
#     * \param[out] inverse the resultant inverted matrix
#     * \return determinant of the original matrix => if 0 no inverse exists => result is invalid
#     * \ingroup common
#     */
#   template <typename Matrix> typename Matrix::Scalar
#   invert3x3Matrix (const Matrix &matrix, Matrix &inverse);
# 
#   /** \brief Calculate the determinant of a 3x3 matrix.
#     * \param[in] matrix matrix
#     * \return determinant of the matrix
#     * \ingroup common
#     */
#   template <typename Matrix> typename Matrix::Scalar
#   determinant3x3Matrix (const Matrix &matrix);
#   
#   /** \brief Get the unique 3D rotation that will rotate \a z_axis into (0,0,1) and \a y_direction into a vector
#     * with x=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
#     * \param[in] z_axis the z-axis
#     * \param[in] y_direction the y direction
#     * \param[out] transformation the resultant 3D rotation
#     * \ingroup common
#     */
#   inline void
#   getTransFromUnitVectorsZY (const Eigen::Vector3f& z_axis, 
#                              const Eigen::Vector3f& y_direction,
#                              Eigen::Affine3f& transformation);
# 
#   /** \brief Get the unique 3D rotation that will rotate \a z_axis into (0,0,1) and \a y_direction into a vector
#     * with x=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
#     * \param[in] z_axis the z-axis
#     * \param[in] y_direction the y direction
#     * \return the resultant 3D rotation
#     * \ingroup common
#     */
#   inline Eigen::Affine3f
#   getTransFromUnitVectorsZY (const Eigen::Vector3f& z_axis, 
#                              const Eigen::Vector3f& y_direction);
# 
#   /** \brief Get the unique 3D rotation that will rotate \a x_axis into (1,0,0) and \a y_direction into a vector
#     * with z=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
#     * \param[in] x_axis the x-axis
#     * \param[in] y_direction the y direction
#     * \param[out] transformation the resultant 3D rotation
#     * \ingroup common
#     */
#   inline void
#   getTransFromUnitVectorsXY (const Eigen::Vector3f& x_axis, 
#                              const Eigen::Vector3f& y_direction,
#                              Eigen::Affine3f& transformation);
# 
#   /** \brief Get the unique 3D rotation that will rotate \a x_axis into (1,0,0) and \a y_direction into a vector
#     * with z=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
#     * \param[in] x_axis the x-axis
#     * \param[in] y_direction the y direction
#     * \return the resulting 3D rotation
#     * \ingroup common
#     */
#   inline Eigen::Affine3f
#   getTransFromUnitVectorsXY (const Eigen::Vector3f& x_axis, 
#                              const Eigen::Vector3f& y_direction);
# 
#   /** \brief Get the unique 3D rotation that will rotate \a z_axis into (0,0,1) and \a y_direction into a vector
#     * with x=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
#     * \param[in] y_direction the y direction
#     * \param[in] z_axis the z-axis
#     * \param[out] transformation the resultant 3D rotation
#     * \ingroup common
#     */
#   inline void
#   getTransformationFromTwoUnitVectors (const Eigen::Vector3f& y_direction, 
#                                        const Eigen::Vector3f& z_axis,
#                                        Eigen::Affine3f& transformation);
# 
#   /** \brief Get the unique 3D rotation that will rotate \a z_axis into (0,0,1) and \a y_direction into a vector
#     * with x=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
#     * \param[in] y_direction the y direction
#     * \param[in] z_axis the z-axis
#     * \return transformation the resultant 3D rotation
#     * \ingroup common
#     */
#   inline Eigen::Affine3f
#   getTransformationFromTwoUnitVectors (const Eigen::Vector3f& y_direction, 
#                                        const Eigen::Vector3f& z_axis);
# 
#   /** \brief Get the transformation that will translate \a orign to (0,0,0) and rotate \a z_axis into (0,0,1)
#     * and \a y_direction into a vector with x=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
#     * \param[in] y_direction the y direction
#     * \param[in] z_axis the z-axis
#     * \param[in] origin the origin
#     * \param[in] transformation the resultant transformation matrix
#     * \ingroup common
#     */
#   inline void
#   getTransformationFromTwoUnitVectorsAndOrigin (const Eigen::Vector3f& y_direction, 
#                                                 const Eigen::Vector3f& z_axis,
#                                                 const Eigen::Vector3f& origin, 
#                                                 Eigen::Affine3f& transformation);
# 
#   /** \brief Extract the Euler angles (XYZ-convention) from the given transformation
#     * \param[in] t the input transformation matrix
#     * \param[in] roll the resulting roll angle
#     * \param[in] pitch the resulting pitch angle
#     * \param[in] yaw the resulting yaw angle
#     * \ingroup common
#     */
#   template <typename Scalar> void
#   getEulerAngles (const Eigen::Transform<Scalar, 3, Eigen::Affine> &t, Scalar &roll, Scalar &pitch, Scalar &yaw);
# 
#   inline void
#   getEulerAngles (const Eigen::Affine3f &t, float &roll, float &pitch, float &yaw)
#   {
#     getEulerAngles<float> (t, roll, pitch, yaw);
#   }
# 
#   inline void
#   getEulerAngles (const Eigen::Affine3d &t, double &roll, double &pitch, double &yaw)
#   {
#     getEulerAngles<double> (t, roll, pitch, yaw);
#   }
# 
#   /** Extract x,y,z and the Euler angles (XYZ-convention) from the given transformation
#     * \param[in] t the input transformation matrix
#     * \param[out] x the resulting x translation
#     * \param[out] y the resulting y translation
#     * \param[out] z the resulting z translation
#     * \param[out] roll the resulting roll angle
#     * \param[out] pitch the resulting pitch angle
#     * \param[out] yaw the resulting yaw angle
#     * \ingroup common
#     */
#   template <typename Scalar> void
#   getTranslationAndEulerAngles (const Eigen::Transform<Scalar, 3, Eigen::Affine> &t,
#                                 Scalar &x, Scalar &y, Scalar &z,
#                                 Scalar &roll, Scalar &pitch, Scalar &yaw);
# 
#   inline void
#   getTranslationAndEulerAngles (const Eigen::Affine3f &t,
#                                 float &x, float &y, float &z,
#                                 float &roll, float &pitch, float &yaw)
#   {
#     getTranslationAndEulerAngles<float> (t, x, y, z, roll, pitch, yaw);
#   }
# 
#   inline void
#   getTranslationAndEulerAngles (const Eigen::Affine3d &t,
#                                 double &x, double &y, double &z,
#                                 double &roll, double &pitch, double &yaw)
#   {
#     getTranslationAndEulerAngles<double> (t, x, y, z, roll, pitch, yaw);
#   }
# 
#   /** \brief Create a transformation from the given translation and Euler angles (XYZ-convention)
#     * \param[in] x the input x translation
#     * \param[in] y the input y translation
#     * \param[in] z the input z translation
#     * \param[in] roll the input roll angle
#     * \param[in] pitch the input pitch angle
#     * \param[in] yaw the input yaw angle
#     * \param[out] t the resulting transformation matrix
#     * \ingroup common
#     */
#   template <typename Scalar> void
#   getTransformation (Scalar x, Scalar y, Scalar z, Scalar roll, Scalar pitch, Scalar yaw, 
#                      Eigen::Transform<Scalar, 3, Eigen::Affine> &t);
# 
#   inline void
#   getTransformation (float x, float y, float z, float roll, float pitch, float yaw, 
#                      Eigen::Affine3f &t)
#   {
#     return (getTransformation<float> (x, y, z, roll, pitch, yaw, t));
#   }
# 
#   inline void
#   getTransformation (double x, double y, double z, double roll, double pitch, double yaw, 
#                      Eigen::Affine3d &t)
#   {
#     return (getTransformation<double> (x, y, z, roll, pitch, yaw, t));
#   }
# 
#   /** \brief Create a transformation from the given translation and Euler angles (XYZ-convention)
#     * \param[in] x the input x translation
#     * \param[in] y the input y translation
#     * \param[in] z the input z translation
#     * \param[in] roll the input roll angle
#     * \param[in] pitch the input pitch angle
#     * \param[in] yaw the input yaw angle
#     * \return the resulting transformation matrix
#     * \ingroup common
#     */
#   inline Eigen::Affine3f
#   getTransformation (float x, float y, float z, float roll, float pitch, float yaw)
#   {
#     Eigen::Affine3f t;
#     getTransformation<float> (x, y, z, roll, pitch, yaw, t);
#     return (t);
#   }
# 
#   /** \brief Write a matrix to an output stream
#     * \param[in] matrix the matrix to output
#     * \param[out] file the output stream
#     * \ingroup common
#     */
#   template <typename Derived> void
#   saveBinary (const Eigen::MatrixBase<Derived>& matrix, std::ostream& file);
# 
#   /** \brief Read a matrix from an input stream
#     * \param[out] matrix the resulting matrix, read from the input stream
#     * \param[in,out] file the input stream
#     * \ingroup common
#     */
#   template <typename Derived> void
#   loadBinary (Eigen::MatrixBase<Derived> const& matrix, std::istream& file);
# 
# // PCL_EIGEN_SIZE_MIN_PREFER_DYNAMIC gives the min between compile-time sizes. 0 has absolute priority, followed by 1,
# // followed by Dynamic, followed by other finite values. The reason for giving Dynamic the priority over
# // finite values is that min(3, Dynamic) should be Dynamic, since that could be anything between 0 and 3.
# #define PCL_EIGEN_SIZE_MIN_PREFER_DYNAMIC(a,b) ((int (a) == 0 || int (b) == 0) ? 0 \
#                            : (int (a) == 1 || int (b) == 1) ? 1 \
#                            : (int (a) == Eigen::Dynamic || int (b) == Eigen::Dynamic) ? Eigen::Dynamic \
#                            : (int (a) <= int (b)) ? int (a) : int (b))
# 
#   /** \brief Returns the transformation between two point sets. 
#     * The algorithm is based on: 
#     * "Least-squares estimation of transformation parameters between two point patterns",
#     * Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
#     *
#     * It estimates parameters \f$ c, \mathbf{R}, \f$ and \f$ \mathbf{t} \f$ such that
#     * \f{align*}
#     *   \frac{1}{n} \sum_{i=1}^n \vert\vert y_i - (c\mathbf{R}x_i + \mathbf{t}) \vert\vert_2^2
#     * \f}
#     * is minimized.
#     *
#     * The algorithm is based on the analysis of the covariance matrix
#     * \f$ \Sigma_{\mathbf{x}\mathbf{y}} \in \mathbb{R}^{d \times d} \f$
#     * of the input point sets \f$ \mathbf{x} \f$ and \f$ \mathbf{y} \f$ where
#     * \f$d\f$ is corresponding to the dimension (which is typically small).
#     * The analysis is involving the SVD having a complexity of \f$O(d^3)\f$
#     * though the actual computational effort lies in the covariance
#     * matrix computation which has an asymptotic lower bound of \f$O(dm)\f$ when
#     * the input point sets have dimension \f$d \times m\f$.
#     *
#     * \param[in] src Source points \f$ \mathbf{x} = \left( x_1, \hdots, x_n \right) \f$
#     * \param[in] dst Destination points \f$ \mathbf{y} = \left( y_1, \hdots, y_n \right) \f$.
#     * \param[in] with_scaling Sets \f$ c=1 \f$ when <code>false</code> is passed. (default: false)
#     * \return The homogeneous transformation 
#     * \f{align*}
#     *   T = \begin{bmatrix} c\mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix}
#     * \f}
#     * minimizing the resudiual above. This transformation is always returned as an
#     * Eigen::Matrix.
#     */
#   template <typename Derived, typename OtherDerived> 
#   typename Eigen::internal::umeyama_transform_matrix_type<Derived, OtherDerived>::type
#   umeyama (const Eigen::MatrixBase<Derived>& src, const Eigen::MatrixBase<OtherDerived>& dst, bool with_scaling = false);
# 
# /** \brief Transform a point using an affine matrix
#   * \param[in] point_in the vector to be transformed
#   * \param[out] point_out the transformed vector
#   * \param[in] transformation the transformation matrix
#   *
#   * \note Can be used with \c point_in = \c point_out
#   */
#   template<typename Scalar> inline void
#   transformPoint (const Eigen::Matrix<Scalar, 3, 1> &point_in,
#                         Eigen::Matrix<Scalar, 3, 1> &point_out,
#                   const Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation)
#   {
#     Eigen::Matrix<Scalar, 4, 1> point;
#     point << point_in, 1.0;
#     point_out = (transformation * point).template head<3> ();
#   }
# 
#   inline void
#   transformPoint (const Eigen::Vector3f &point_in,
#                         Eigen::Vector3f &point_out,
#                   const Eigen::Affine3f &transformation)
#   {
#     transformPoint<float> (point_in, point_out, transformation);
#   }
# 
#   inline void
#   transformPoint (const Eigen::Vector3d &point_in,
#                         Eigen::Vector3d &point_out,
#                   const Eigen::Affine3d &transformation)
#   {
#     transformPoint<double> (point_in, point_out, transformation);
#   }
# 
# /** \brief Transform a vector using an affine matrix
#   * \param[in] vector_in the vector to be transformed
#   * \param[out] vector_out the transformed vector
#   * \param[in] transformation the transformation matrix
#   *
#   * \note Can be used with \c vector_in = \c vector_out
#   */
#   template <typename Scalar> inline void
#   transformVector (const Eigen::Matrix<Scalar, 3, 1> &vector_in,
#                          Eigen::Matrix<Scalar, 3, 1> &vector_out,
#                    const Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation)
#   {
#     vector_out = transformation.linear () * vector_in;
#   }
# 
#   inline void
#   transformVector (const Eigen::Vector3f &vector_in,
#                          Eigen::Vector3f &vector_out,
#                    const Eigen::Affine3f &transformation)
#   {
#     transformVector<float> (vector_in, vector_out, transformation);
#   }
# 
#   inline void
#   transformVector (const Eigen::Vector3d &vector_in,
#                          Eigen::Vector3d &vector_out,
#                    const Eigen::Affine3d &transformation)
#   {
#     transformVector<double> (vector_in, vector_out, transformation);
#   }
# 
# /** \brief Transform a line using an affine matrix
#   * \param[in] line_in the line to be transformed
#   * \param[out] line_out the transformed line
#   * \param[in] transformation the transformation matrix
#   *
#   * Lines must be filled in this form:\n
#   * line[0-2] = Origin coordinates of the vector\n
#   * line[3-5] = Direction vector
#   *
#   * \note Can be used with \c line_in = \c line_out
#   */
#   template <typename Scalar> bool
#   transformLine (const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &line_in,
#                        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &line_out,
#                  const Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation);
# 
#   inline bool
#   transformLine (const Eigen::VectorXf &line_in,
#                        Eigen::VectorXf &line_out,
#                  const Eigen::Affine3f &transformation)
#   {
#     return (transformLine<float> (line_in, line_out, transformation));
#   }
# 
#   inline bool
#   transformLine (const Eigen::VectorXd &line_in,
#                        Eigen::VectorXd &line_out,
#                  const Eigen::Affine3d &transformation)
#   {
#     return (transformLine<double> (line_in, line_out, transformation));
#   }
# 
# /** \brief Transform plane vectors using an affine matrix
#   * \param[in] plane_in the plane coefficients to be transformed
#   * \param[out] plane_out the transformed plane coefficients to fill
#   * \param[in] transformation the transformation matrix
#   *
#   * The plane vectors are filled in the form ax+by+cz+d=0
#   * Can be used with non Hessian form planes coefficients
#   * Can be used with \c plane_in = \c plane_out
#   */
#   template <typename Scalar> void
#   transformPlane (const Eigen::Matrix<Scalar, 4, 1> &plane_in,
#                         Eigen::Matrix<Scalar, 4, 1> &plane_out,
#                   const Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation);
# 
#   inline void
#   transformPlane (const Eigen::Matrix<double, 4, 1> &plane_in,
#                         Eigen::Matrix<double, 4, 1> &plane_out,
#                   const Eigen::Transform<double, 3, Eigen::Affine> &transformation)
#   {
#     transformPlane<double> (plane_in, plane_out, transformation);
#   }
# 
#   inline void
#   transformPlane (const Eigen::Matrix<float, 4, 1> &plane_in,
#                         Eigen::Matrix<float, 4, 1> &plane_out,
#                   const Eigen::Transform<float, 3, Eigen::Affine> &transformation)
#   {
#     transformPlane<float> (plane_in, plane_out, transformation);
#   }
# 
# /** \brief Transform plane vectors using an affine matrix
#   * \param[in] plane_in the plane coefficients to be transformed
#   * \param[out] plane_out the transformed plane coefficients to fill
#   * \param[in] transformation the transformation matrix
#   *
#   * The plane vectors are filled in the form ax+by+cz+d=0
#   * Can be used with non Hessian form planes coefficients
#   * Can be used with \c plane_in = \c plane_out
#   * \warning ModelCoefficients stores floats only !
#   */
#   template<typename Scalar> void
#   transformPlane (const pcl::ModelCoefficients::Ptr plane_in,
#                         pcl::ModelCoefficients::Ptr plane_out,
#                   const Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation);
# 
#   inline void
#   transformPlane (const pcl::ModelCoefficients::Ptr plane_in,
#                         pcl::ModelCoefficients::Ptr plane_out,
#                   const Eigen::Transform<double, 3, Eigen::Affine> &transformation)
#   {
#     transformPlane<double> (plane_in, plane_out, transformation);
#   }
# 
#   inline void
#   transformPlane (const pcl::ModelCoefficients::Ptr plane_in,
#                         pcl::ModelCoefficients::Ptr plane_out,
#                   const Eigen::Transform<float, 3, Eigen::Affine> &transformation)
#   {
#     transformPlane<float> (plane_in, plane_out, transformation);
#   }
# 
# /** \brief Check coordinate system integrity
#   * \param[in] line_x the first axis
#   * \param[in] line_y the second axis
#   * \param[in] norm_limit the limit to ignore norm rounding errors
#   * \param[in] dot_limit the limit to ignore dot product rounding errors
#   * \return True if the coordinate system is consistent, false otherwise.
#   *
#   * Lines must be filled in this form:\n
#   * line[0-2] = Origin coordinates of the vector\n
#   * line[3-5] = Direction vector
#   *
#   * Can be used like this :\n
#   * line_x = X axis and line_y = Y axis\n
#   * line_x = Z axis and line_y = X axis\n
#   * line_x = Y axis and line_y = Z axis\n
#   * Because X^Y = Z, Z^X = Y and Y^Z = X.
#   * Do NOT invert line order !
#   *
#   * Determine whether a coordinate system is consistent or not by checking :\n
#   * Line origins: They must be the same for the 2 lines\n
#   * Norm: The 2 lines must be normalized\n
#   * Dot products: Must be 0 or perpendicular vectors
#   */
#   template<typename Scalar> bool
#   checkCoordinateSystem (const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &line_x,
#                          const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &line_y,
#                          const Scalar norm_limit = 1e-3,
#                          const Scalar dot_limit = 1e-3);
# 
#   inline bool
#   checkCoordinateSystem (const Eigen::Matrix<double, Eigen::Dynamic, 1> &line_x,
#                          const Eigen::Matrix<double, Eigen::Dynamic, 1> &line_y,
#                          const double norm_limit = 1e-3,
#                          const double dot_limit = 1e-3)
#   {
#     return (checkCoordinateSystem<double> (line_x, line_y, norm_limit, dot_limit));
#   }
# 
#   inline bool
#   checkCoordinateSystem (const Eigen::Matrix<float, Eigen::Dynamic, 1> &line_x,
#                          const Eigen::Matrix<float, Eigen::Dynamic, 1> &line_y,
#                          const float norm_limit = 1e-3,
#                          const float dot_limit = 1e-3)
#   {
#     return (checkCoordinateSystem<float> (line_x, line_y, norm_limit, dot_limit));
#   }
# 
# /** \brief Check coordinate system integrity
#   * \param[in] origin the origin of the coordinate system
#   * \param[in] x_direction the first axis
#   * \param[in] y_direction the second axis
#   * \param[in] norm_limit the limit to ignore norm rounding errors
#   * \param[in] dot_limit the limit to ignore dot product rounding errors
#   * \return True if the coordinate system is consistent, false otherwise.
#   *
#   * Read the other variant for more information
#   */
#   template <typename Scalar> inline bool
#   checkCoordinateSystem (const Eigen::Matrix<Scalar, 3, 1> &origin,
#                          const Eigen::Matrix<Scalar, 3, 1> &x_direction,
#                          const Eigen::Matrix<Scalar, 3, 1> &y_direction,
#                          const Scalar norm_limit = 1e-3,
#                          const Scalar dot_limit = 1e-3)
#   {
#     Eigen::Matrix<Scalar, Eigen::Dynamic, 1> line_x;
#     Eigen::Matrix<Scalar, Eigen::Dynamic, 1> line_y;
#     line_x << origin, x_direction;
#     line_y << origin, y_direction;
#     return (checkCoordinateSystem<Scalar> (line_x, line_y, norm_limit, dot_limit));
#   }
# 
#   inline bool
#   checkCoordinateSystem (const Eigen::Matrix<double, 3, 1> &origin,
#                          const Eigen::Matrix<double, 3, 1> &x_direction,
#                          const Eigen::Matrix<double, 3, 1> &y_direction,
#                          const double norm_limit = 1e-3,
#                          const double dot_limit = 1e-3)
#   {
#     Eigen::Matrix<double, Eigen::Dynamic, 1> line_x;
#     Eigen::Matrix<double, Eigen::Dynamic, 1> line_y;
#     line_x.resize (6);
#     line_y.resize (6);
#     line_x << origin, x_direction;
#     line_y << origin, y_direction;
#     return (checkCoordinateSystem<double> (line_x, line_y, norm_limit, dot_limit));
#   }
# 
#   inline bool
#   checkCoordinateSystem (const Eigen::Matrix<float, 3, 1> &origin,
#                          const Eigen::Matrix<float, 3, 1> &x_direction,
#                          const Eigen::Matrix<float, 3, 1> &y_direction,
#                          const float norm_limit = 1e-3,
#                          const float dot_limit = 1e-3)
#   {
#     Eigen::Matrix<float, Eigen::Dynamic, 1> line_x;
#     Eigen::Matrix<float, Eigen::Dynamic, 1> line_y;
#     line_x.resize (6);
#     line_y.resize (6);
#     line_x << origin, x_direction;
#     line_y << origin, y_direction;
#     return (checkCoordinateSystem<float> (line_x, line_y, norm_limit, dot_limit));
#   }
# 
# /** \brief Compute the transformation between two coordinate systems
#   * \param[in] from_line_x X axis from the origin coordinate system
#   * \param[in] from_line_y Y axis from the origin coordinate system
#   * \param[in] to_line_x X axis from the destination coordinate system
#   * \param[in] to_line_y Y axis from the destination coordinate system
#   * \param[out] transformation the transformation matrix to fill
#   * \return true if transformation was filled, false otherwise.
#   *
#   * Line must be filled in this form:\n
#   * line[0-2] = Coordinate system origin coordinates \n
#   * line[3-5] = Direction vector (norm doesn't matter)
#   */
#   template <typename Scalar> bool
#   transformBetween2CoordinateSystems (const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> from_line_x,
#                                       const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> from_line_y,
#                                       const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> to_line_x,
#                                       const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> to_line_y,
#                                       Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation);
# 
#   inline bool
#   transformBetween2CoordinateSystems (const Eigen::Matrix<double, Eigen::Dynamic, 1> from_line_x,
#                                       const Eigen::Matrix<double, Eigen::Dynamic, 1> from_line_y,
#                                       const Eigen::Matrix<double, Eigen::Dynamic, 1> to_line_x,
#                                       const Eigen::Matrix<double, Eigen::Dynamic, 1> to_line_y,
#                                       Eigen::Transform<double, 3, Eigen::Affine> &transformation)
#   {
#     return (transformBetween2CoordinateSystems<double> (from_line_x, from_line_y, to_line_x, to_line_y, transformation));
#   }
# 
#   inline bool
#   transformBetween2CoordinateSystems (const Eigen::Matrix<float, Eigen::Dynamic, 1> from_line_x,
#                                       const Eigen::Matrix<float, Eigen::Dynamic, 1> from_line_y,
#                                       const Eigen::Matrix<float, Eigen::Dynamic, 1> to_line_x,
#                                       const Eigen::Matrix<float, Eigen::Dynamic, 1> to_line_y,
#                                       Eigen::Transform<float, 3, Eigen::Affine> &transformation)
#   {
#     return (transformBetween2CoordinateSystems<float> (from_line_x, from_line_y, to_line_x, to_line_y, transformation));
#   }
# 
# }
# 
# #include <pcl/common/impl/eigen.hpp>
# 
# #if defined __SUNPRO_CC
# #  pragma enable_warn
# #endif
# 
# #endif  //PCL_COMMON_EIGEN_H_
# ###
# 
# # file_io.h
# namespace pcl
# {
#   /** \brief Find all *.pcd files in the directory and return them sorted
#     * \param directory the directory to be searched
#     * \param file_names the resulting (sorted) list of .pcd files
#     */
#   inline void 
#   getAllPcdFilesInDirectory (const std::string& directory, std::vector<std::string>& file_names);
#   
#   /** \brief Remove the path from the given string and return only the filename (the remaining string after the 
#     * last '/')
#     * \param input the input filename (with full path)
#     * \return the resulting filename, stripped of the path
#     */
#   inline std::string 
#   getFilenameWithoutPath (const std::string& input);
# 
#   /** \brief Remove the extension from the given string and return only the filename (everything before the last '.')
#     * \param input the input filename (with the file extension)
#     * \return the resulting filename, stripped of its extension
#     */
#   inline std::string 
#   getFilenameWithoutExtension (const std::string& input);
# 
#   /** \brief Get the file extension from the given string (the remaining string after the last '.')
#     * \param input the input filename
#     * \return \a input 's file extension
#     */
#   inline std::string 
#   getFileExtension (const std::string& input);
# }  // namespace end
# /*@}*/
# #include <pcl/common/impl/file_io.hpp>
# 
# #endif  //#ifndef PCL_FILE_IO_H_
# ###
# 
# # gaussian.h
# namespace pcl
# {
#   /** Class GaussianKernel assembles all the method for computing, 
#     * convolving, smoothing, gradients computing an image using
#     * a gaussian kernel. The image is stored in point cloud elements 
#     * intensity member or rgb or...
#     * \author Nizar Sallem
#     * \ingroup common
#     */
#   class PCL_EXPORTS GaussianKernel
#   {
#     public:
# 
#       GaussianKernel () {}
# 
#       static const unsigned MAX_KERNEL_WIDTH = 71;
#       /** Computes the gaussian kernel and dervative assiociated to sigma.
#         * The kernel and derivative width are adjusted according.
#         * \param[in] sigma
#         * \param[out] kernel the computed gaussian kernel
#         * \param[in] kernel_width the desired kernel width upper bond
#         * \throws pcl::KernelWidthTooSmallException
#         */
#       void
#       compute (float sigma, 
#                Eigen::VectorXf &kernel,
#                unsigned kernel_width = MAX_KERNEL_WIDTH) const;
# 
#       /** Computes the gaussian kernel and dervative assiociated to sigma.
#         * The kernel and derivative width are adjusted according.
#         * \param[in] sigma
#         * \param[out] kernel the computed gaussian kernel
#         * \param[out] derivative the computed kernel derivative
#         * \param[in] kernel_width the desired kernel width upper bond
#         * \throws pcl::KernelWidthTooSmallException
#         */
#       void
#       compute (float sigma, 
#                Eigen::VectorXf &kernel, 
#                Eigen::VectorXf &derivative, 
#                unsigned kernel_width = MAX_KERNEL_WIDTH) const;
# 
#       /** Convolve a float image rows by a given kernel.
#         * \param[in] kernel convolution kernel
#         * \param[in] input the image to convolve
#         * \param[out] output the convolved image
#         * \note if output doesn't fit in input i.e. output.rows () < input.rows () or
#         * output.cols () < input.cols () then output is resized to input sizes.
#         */
#       void
#       convolveRows (const pcl::PointCloud<float> &input,
#                     const Eigen::VectorXf &kernel,
#                     pcl::PointCloud<float> &output) const;
# 
#       /** Convolve a float image rows by a given kernel.
#         * \param[in] input the image to convolve
#         * \param[in] field_accessor a field accessor
#         * \param[in] kernel convolution kernel
#         * \param[out] output the convolved image
#         * \note if output doesn't fit in input i.e. output.rows () < input.rows () or
#         * output.cols () < input.cols () then output is resized to input sizes.
#         */
#      template <typename PointT> void
#      convolveRows (const pcl::PointCloud<PointT> &input,
#                    boost::function <float (const PointT& p)> field_accessor,
#                    const Eigen::VectorXf &kernel,
#                    pcl::PointCloud<float> &output) const;
# 
#       /** Convolve a float image columns by a given kernel.
#         * \param[in] input the image to convolve
#         * \param[in] kernel convolution kernel
#         * \param[out] output the convolved image
#         * \note if output doesn't fit in input i.e. output.rows () < input.rows () or
#         * output.cols () < input.cols () then output is resized to input sizes.
#         */
#       void
#       convolveCols (const pcl::PointCloud<float> &input,
#                     const Eigen::VectorXf &kernel,
#                     pcl::PointCloud<float> &output) const;
# 
#       /** Convolve a float image columns by a given kernel.
#         * \param[in] input the image to convolve
#         * \param[in] field_accessor a field accessor
#         * \param[in] kernel convolution kernel
#         * \param[out] output the convolved image
#         * \note if output doesn't fit in input i.e. output.rows () < input.rows () or
#         * output.cols () < input.cols () then output is resized to input sizes.
#         */
#       template <typename PointT> void
#       convolveCols (const pcl::PointCloud<PointT> &input,
#                     boost::function <float (const PointT& p)> field_accessor,
#                     const Eigen::VectorXf &kernel,
#                     pcl::PointCloud<float> &output) const;
# 
#       /** Convolve a float image in the 2 directions
#         * \param[in] horiz_kernel kernel for convolving rows
#         * \param[in] vert_kernel kernel for convolving columns
#         * \param[in] input image to convolve
#         * \param[out] output the convolved image
#         * \note if output doesn't fit in input i.e. output.rows () < input.rows () or
#         * output.cols () < input.cols () then output is resized to input sizes.
#         */
#       inline void
#       convolve (const pcl::PointCloud<float> &input,
#                 const Eigen::VectorXf &horiz_kernel,
#                 const Eigen::VectorXf &vert_kernel,
#                 pcl::PointCloud<float> &output) const
#       {
#         std::cout << ">>> convolve cpp" << std::endl;
#         pcl::PointCloud<float> tmp (input.width, input.height) ;
#         convolveRows (input, horiz_kernel, tmp);        
#         convolveCols (tmp, vert_kernel, output);
#         std::cout << "<<< convolve cpp" << std::endl;
#       }
# 
#       /** Convolve a float image in the 2 directions
#         * \param[in] input image to convolve
#         * \param[in] field_accessor a field accessor
#         * \param[in] horiz_kernel kernel for convolving rows
#         * \param[in] vert_kernel kernel for convolving columns
#         * \param[out] output the convolved image
#         * \note if output doesn't fit in input i.e. output.rows () < input.rows () or
#         * output.cols () < input.cols () then output is resized to input sizes.
#         */
#       template <typename PointT> inline void
#       convolve (const pcl::PointCloud<PointT> &input,
#                 boost::function <float (const PointT& p)> field_accessor,
#                 const Eigen::VectorXf &horiz_kernel,
#                 const Eigen::VectorXf &vert_kernel,
#                 pcl::PointCloud<float> &output) const
#       {
#         std::cout << ">>> convolve hpp" << std::endl;
#         pcl::PointCloud<float> tmp (input.width, input.height);
#         convolveRows<PointT>(input, field_accessor, horiz_kernel, tmp);
#         convolveCols(tmp, vert_kernel, output);
#         std::cout << "<<< convolve hpp" << std::endl;
#       }
#       
#       /** Computes float image gradients using a gaussian kernel and gaussian kernel
#         * derivative.
#         * \param[in] input image to compute gardients for
#         * \param[in] gaussian_kernel the gaussian kernel to be used
#         * \param[in] gaussian_kernel_derivative the associated derivative
#         * \param[out] grad_x gradient along X direction
#         * \param[out] grad_y gradient along Y direction
#         * \note if output doesn't fit in input i.e. output.rows () < input.rows () or
#         * output.cols () < input.cols () then output is resized to input sizes.
#         */
#       inline void
#       computeGradients (const pcl::PointCloud<float> &input,
#                         const Eigen::VectorXf &gaussian_kernel,
#                         const Eigen::VectorXf &gaussian_kernel_derivative,
#                         pcl::PointCloud<float> &grad_x,
#                         pcl::PointCloud<float> &grad_y) const
#       {
#         convolve (input, gaussian_kernel_derivative, gaussian_kernel, grad_x);
#         convolve (input, gaussian_kernel, gaussian_kernel_derivative, grad_y);
#       }
# 
#       /** Computes float image gradients using a gaussian kernel and gaussian kernel
#         * derivative.
#         * \param[in] input image to compute gardients for
#         * \param[in] field_accessor a field accessor
#         * \param[in] gaussian_kernel the gaussian kernel to be used
#         * \param[in] gaussian_kernel_derivative the associated derivative
#         * \param[out] grad_x gradient along X direction
#         * \param[out] grad_y gradient along Y direction
#         * \note if output doesn't fit in input i.e. output.rows () < input.rows () or
#         * output.cols () < input.cols () then output is resized to input sizes.
#         */
#       template <typename PointT> inline void
#       computeGradients (const pcl::PointCloud<PointT> &input,
#                         boost::function <float (const PointT& p)> field_accessor,
#                         const Eigen::VectorXf &gaussian_kernel,
#                         const Eigen::VectorXf &gaussian_kernel_derivative,
#                         pcl::PointCloud<float> &grad_x,
#                         pcl::PointCloud<float> &grad_y) const
#       {
#         convolve<PointT> (input, field_accessor, gaussian_kernel_derivative, gaussian_kernel, grad_x);
#         convolve<PointT> (input, field_accessor, gaussian_kernel, gaussian_kernel_derivative, grad_y);
#       }
#       
#       /** Smooth image using a gaussian kernel.
#         * \param[in] input image
#         * \param[in] gaussian_kernel the gaussian kernel to be used
#         * \param[out] output the smoothed image
#         * \note if output doesn't fit in input i.e. output.rows () < input.rows () or
#         * output.cols () < input.cols () then output is resized to input sizes.
#         */
#       inline void
#       smooth (const pcl::PointCloud<float> &input,
#               const Eigen::VectorXf &gaussian_kernel,
#               pcl::PointCloud<float> &output) const
#       {
#         convolve (input, gaussian_kernel, gaussian_kernel, output);
#       }
# 
#       /** Smooth image using a gaussian kernel.
#         * \param[in] input image
#         * \param[in] field_accessor a field accessor
#         * \param[in] gaussian_kernel the gaussian kernel to be used
#         * \param[out] output the smoothed image
#         * \note if output doesn't fit in input i.e. output.rows () < input.rows () or
#         * output.cols () < input.cols () then output is resized to input sizes.
#         */
#       template <typename PointT> inline void
#       smooth (const pcl::PointCloud<PointT> &input,
#               boost::function <float (const PointT& p)> field_accessor,
#               const Eigen::VectorXf &gaussian_kernel,
#               pcl::PointCloud<float> &output) const
#       {
#         convolve<PointT> (input, field_accessor, gaussian_kernel, gaussian_kernel, output);
#       }
#   };
# }
# 
# # #include <pcl/common/impl/gaussian.hpp>
# # #endif // PCL_GAUSSIAN_KERNEL
# ###
# 
# # generate.h
# namespace pcl
# {
# 
#   namespace common
#   {
#     /** \brief CloudGenerator class generates a point cloud using some randoom number generator.
#       * Generators can be found in \file common/random.h and easily extensible.
#       *  
#       * \ingroup common
#       * \author Nizar Sallem
#       */
#     template <typename PointT, typename GeneratorT>
#     class CloudGenerator
#     {
#       public:
#       typedef typename GeneratorT::Parameters GeneratorParameters;
# 
#       /// Default constructor
#       CloudGenerator ();
# 
#       /** Consttructor with single generator to ensure all X, Y and Z values are within same range
#         * \param params paramteres for X, Y and Z values generation. Uniqueness is ensured through
#         * seed incrementation
#         */
#       CloudGenerator (const GeneratorParameters& params);
# 
#       /** Constructor with independant generators per axis
#         * \param x_params parameters for x values generation
#         * \param y_params parameters for y values generation
#         * \param z_params parameters for z values generation
#         */
#       CloudGenerator (const GeneratorParameters& x_params,
#                       const GeneratorParameters& y_params,
#                       const GeneratorParameters& z_params);
# 
#       /** Set parameters for x, y and z values. Uniqueness is ensured through seed incrementation.
#         * \param params parameteres for X, Y and Z values generation. 
#         */
#       void
#       setParameters (const GeneratorParameters& params);
#       
#       /** Set parameters for x values generation
#         * \param x_params paramters for x values generation
#         */
#       void
#       setParametersForX (const GeneratorParameters& x_params);
# 
#       /** Set parameters for y values generation
#         * \param y_params paramters for y values generation
#         */
#       void
#       setParametersForY (const GeneratorParameters& y_params);
#       
#       /** Set parameters for z values generation
#         * \param z_params paramters for z values generation
#         */
#       void
#       setParametersForZ (const GeneratorParameters& z_params);
# 
#       /// \return x values generation parameters
#       const GeneratorParameters& 
#       getParametersForX () const;
# 
#       /// \return y values generation parameters
#       const GeneratorParameters& 
#       getParametersForY () const;
# 
#       /// \return z values generation parameters
#       const GeneratorParameters& 
#       getParametersForZ () const;
#       
#       /// \return a single random generated point 
#       PointT 
#       get ();
#         
#       /** Generates a cloud with X Y Z picked within given ranges. This function assumes that
#         * cloud is properly defined else it raises errors and does nothing.
#         * \param[out] cloud cloud to generate coordinates for
#         * \return 0 if generation went well else -1.
#         */
#       int
#       fill (pcl::PointCloud<PointT>& cloud);
# 
#       /** Generates a cloud of specified dimensions with X Y Z picked within given ranges. 
#         * \param[in] width width of generated cloud
#         * \param[in] height height of generated cloud
#         * \param[out] cloud output cloud
#         * \return 0 if generation went well else -1.
#         */
#       int 
#       fill (int width, int height, pcl::PointCloud<PointT>& cloud);
#       
#       private:
#         GeneratorT x_generator_, y_generator_, z_generator_;
#     };
# 
#     template <typename GeneratorT>
#     class CloudGenerator<pcl::PointXY, GeneratorT>
#     {
#       public:
#       typedef typename GeneratorT::Parameters GeneratorParameters;
#       
#       CloudGenerator ();
#       
#       CloudGenerator (const GeneratorParameters& params);
# 
#       CloudGenerator (const GeneratorParameters& x_params,
#                       const GeneratorParameters& y_params);
#       
#       void
#       setParameters (const GeneratorParameters& params);
# 
#       void
#       setParametersForX (const GeneratorParameters& x_params);
# 
#       void
#       setParametersForY (const GeneratorParameters& y_params);
# 
#       const GeneratorParameters& 
#       getParametersForX () const;
# 
#       const GeneratorParameters& 
#       getParametersForY () const;
# 
#       pcl::PointXY
#       get ();
# 
#       int 
#       fill (pcl::PointCloud<pcl::PointXY>& cloud);
# 
#       int 
#       fill (int width, int height, pcl::PointCloud<pcl::PointXY>& cloud);
#       
#       private:
#         GeneratorT x_generator_;
#         GeneratorT y_generator_;
#     };
#   }
# }
# 
# # #include <pcl/common/impl/generate.hpp>
# # #endif
# ###
# 
# # geometry.h
# namespace pcl
# {
#   namespace geometry
#   {
#     /** @return the euclidean distance between 2 points */
#     template <typename PointT> inline float 
#     distance (const PointT& p1, const PointT& p2)
#     {
#       Eigen::Vector3f diff = p1 -p2;
#       return (diff.norm ());
#     }
# 
#     /** @return the squared euclidean distance between 2 points */
#     template<typename PointT> inline float 
#     squaredDistance (const PointT& p1, const PointT& p2)
#     {
#       Eigen::Vector3f diff = p1 -p2;
#       return (diff.squaredNorm ());
#     }
# 
#     /** @return the point projection on a plane defined by its origin and normal vector 
#       * \param[in] point Point to be projected
#       * \param[in] plane_origin The plane origin
#       * \param[in] plane_normal The plane normal 
#       * \param[out] projected The returned projected point
#       */
#     template<typename PointT, typename NormalT> inline void 
#     project (const PointT& point, const PointT &plane_origin, 
#              const NormalT& plane_normal, PointT& projected)
#     {
#       Eigen::Vector3f po = point - plane_origin;
#       const Eigen::Vector3f normal = plane_normal.getVector3fMapConst ();
#       float lambda = normal.dot(po);
#       projected.getVector3fMap () = point.getVector3fMapConst () - (lambda * normal);
#     }
# 
#     /** @return the point projection on a plane defined by its origin and normal vector 
#       * \param[in] point Point to be projected
#       * \param[in] plane_origin The plane origin
#       * \param[in] plane_normal The plane normal 
#       * \param[out] projected The returned projected point
#       */
#     inline void 
#     project (const Eigen::Vector3f& point, const Eigen::Vector3f &plane_origin, 
#              const Eigen::Vector3f& plane_normal, Eigen::Vector3f& projected)
#     {
#       Eigen::Vector3f po = point - plane_origin;
#       float lambda = plane_normal.dot(po);
#       projected = point - (lambda * plane_normal);
#     }
#   }
# }
# 
# /*@}*/
# #endif  //#ifndef PCL_GEOMETRY_H_
# ###
# 
# # intensity.h
# namespace pcl
# {
#   namespace common
#   {
#     /** \brief Intensity field accessor provides access to the inetnsity filed of a PoinT
#       * implementation for specific types should be done in \file pcl/common/impl/intensity.hpp
#       */
#     template<typename PointT>
#     struct IntensityFieldAccessor
#     {
#       /** \brief get intensity field
#         * \param[in] p point
#         * \return p.intensity
#         */
#       inline float
#       operator () (const PointT &p) const
#       {
#         return p.intensity;
#       }
#       /** \brief gets the intensity value of a point
#         * \param p point for which intensity to be get
#         * \param[in] intensity value of the intensity field
#         */
#       inline void
#       get (const PointT &p, float &intensity) const
#       {
#         intensity = p.intensity;
#       }
#       /** \brief sets the intensity value of a point
#         * \param p point for which intensity to be set
#         * \param[in] intensity value of the intensity field
#         */
#       inline void
#       set (PointT &p, float intensity) const
#       {
#         p.intensity = intensity;
#       }
#       /** \brief subtract value from intensity field
#         * \param p point for which to modify inetnsity
#         * \param[in] value value to be subtracted from point intensity
#         */
#       inline void
#       demean (PointT& p, float value) const
#       {
#         p.intensity -= value;
#       }
#       /** \brief add value to intensity field
#         * \param p point for which to modify inetnsity
#         * \param[in] value value to be added to point intensity
#         */
#       inline void
#       add (PointT& p, float value) const
#       {
#         p.intensity += value;
#       }
#     };
#   }
# }
# 
# # #include <pcl/common/impl/intensity.hpp>
# # #endif
# ###
# 
# # intersections.h
# namespace pcl
# {
#   /** \brief Get the intersection of a two 3D lines in space as a 3D point
#     * \param[in] line_a the coefficients of the first line (point, direction)
#     * \param[in] line_b the coefficients of the second line (point, direction)
#     * \param[out] point holder for the computed 3D point
#     * \param[in] sqr_eps maximum allowable squared distance to the true solution
#     * \ingroup common
#     */
#   PCL_EXPORTS inline bool
#   lineWithLineIntersection (const Eigen::VectorXf &line_a, 
#                             const Eigen::VectorXf &line_b, 
#                             Eigen::Vector4f &point,
#                             double sqr_eps = 1e-4);
# 
#   /** \brief Get the intersection of a two 3D lines in space as a 3D point
#     * \param[in] line_a the coefficients of the first line (point, direction)
#     * \param[in] line_b the coefficients of the second line (point, direction)
#     * \param[out] point holder for the computed 3D point
#     * \param[in] sqr_eps maximum allowable squared distance to the true solution
#     * \ingroup common
#     */
# 
#   PCL_EXPORTS inline bool
#   lineWithLineIntersection (const pcl::ModelCoefficients &line_a, 
#                             const pcl::ModelCoefficients &line_b, 
#                             Eigen::Vector4f &point,
#                             double sqr_eps = 1e-4);
# 
#   /** \brief Determine the line of intersection of two non-parallel planes using lagrange multipliers
#     * \note Described in: "Intersection of Two Planes, John Krumm, Microsoft Research, Redmond, WA, USA"
#     * \param[in] plane_a coefficients of plane A and plane B in the form ax + by + cz + d = 0
#     * \param[in] plane_b coefficients of line where line.tail<3>() = direction vector and
#     * line.head<3>() the point on the line clossest to (0, 0, 0)
#     * \param[out] line the intersected line to be filled
#     * \param[in] angular_tolerance tolerance in radians
#     * \return true if succeeded/planes aren't parallel
#     */
#   PCL_EXPORTS template <typename Scalar> bool
#   planeWithPlaneIntersection (const Eigen::Matrix<Scalar, 4, 1> &plane_a,
#                               const Eigen::Matrix<Scalar, 4, 1> &plane_b,
#                               Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &line,
#                               double angular_tolerance = 0.1);
# 
#   PCL_EXPORTS inline bool
#   planeWithPlaneIntersection (const Eigen::Vector4f &plane_a,
#                               const Eigen::Vector4f &plane_b,
#                               Eigen::VectorXf &line,
#                               double angular_tolerance = 0.1)
#   {
#     return (planeWithPlaneIntersection<float> (plane_a, plane_b, line, angular_tolerance));
#   }
# 
#   PCL_EXPORTS inline bool
#   planeWithPlaneIntersection (const Eigen::Vector4d &plane_a,
#                               const Eigen::Vector4d &plane_b,
#                               Eigen::VectorXd &line,
#                               double angular_tolerance = 0.1)
#   {
#     return (planeWithPlaneIntersection<double> (plane_a, plane_b, line, angular_tolerance));
#   }
# 
#   /** \brief Determine the point of intersection of three non-parallel planes by solving the equations.
#     * \note If using nearly parralel planes you can lower the determinant_tolerance value. This can
#     * lead to inconsistent results.
#     * If the three planes intersects in a line the point will be anywhere on the line.
#     * \param[in] plane_a are the coefficients of the first plane in the form ax + by + cz + d = 0
#     * \param[in] plane_b are the coefficients of the second plane
#     * \param[in] plane_c are the coefficients of the third plane
#     * \param[in] determinant_tolerance is a limit to determine whether planes are parallel or not
#     * \param[out] intersection_point the three coordinates x, y, z of the intersection point
#     * \return true if succeeded/planes aren't parallel
#     */
#   PCL_EXPORTS template <typename Scalar> bool
#   threePlanesIntersection (const Eigen::Matrix<Scalar, 4, 1> &plane_a,
#                            const Eigen::Matrix<Scalar, 4, 1> &plane_b,
#                            const Eigen::Matrix<Scalar, 4, 1> &plane_c,
#                            Eigen::Matrix<Scalar, 3, 1> &intersection_point,
#                            double determinant_tolerance = 1e-6);
# 
# 
#   PCL_EXPORTS inline bool
#   threePlanesIntersection (const Eigen::Vector4f &plane_a,
#                            const Eigen::Vector4f &plane_b,
#                            const Eigen::Vector4f &plane_c,
#                            Eigen::Vector3f &intersection_point,
#                            double determinant_tolerance = 1e-6)
#   {
#     return (threePlanesIntersection<float> (plane_a, plane_b, plane_c,
#                                             intersection_point, determinant_tolerance));
#   }
# 
#   PCL_EXPORTS inline bool
#   threePlanesIntersection (const Eigen::Vector4d &plane_a,
#                            const Eigen::Vector4d &plane_b,
#                            const Eigen::Vector4d &plane_c,
#                            Eigen::Vector3d &intersection_point,
#                            double determinant_tolerance = 1e-6)
#   {
#     return (threePlanesIntersection<double> (plane_a, plane_b, plane_c,
#                                             intersection_point, determinant_tolerance));
#   }
# 
# }
# /*@}*/
# 
# #include <pcl/common/impl/intersections.hpp>
# 
# #endif  //#ifndef PCL_INTERSECTIONS_H_
# ###
# 
# # io.h
# namespace pcl
# {
#   /** \brief Get the index of a specified field (i.e., dimension/channel)
#     * \param[in] cloud the the point cloud message
#     * \param[in] field_name the string defining the field name
#     * \ingroup common
#     */
#   inline int
#   getFieldIndex (const pcl::PCLPointCloud2 &cloud, const std::string &field_name)
#   {
#     // Get the index we need
#     for (size_t d = 0; d < cloud.fields.size (); ++d)
#       if (cloud.fields[d].name == field_name)
#         return (static_cast<int>(d));
#     return (-1);
#   }
# 
#   /** \brief Get the index of a specified field (i.e., dimension/channel)
#     * \param[in] cloud the the point cloud message
#     * \param[in] field_name the string defining the field name
#     * \param[out] fields a vector to the original \a PCLPointField vector that the raw PointCloud message contains
#     * \ingroup common
#     */
#   template <typename PointT> inline int 
#   getFieldIndex (const pcl::PointCloud<PointT> &cloud, const std::string &field_name, 
#                  std::vector<pcl::PCLPointField> &fields);
# 
#   /** \brief Get the index of a specified field (i.e., dimension/channel)
#     * \param[in] field_name the string defining the field name
#     * \param[out] fields a vector to the original \a PCLPointField vector that the raw PointCloud message contains
#     * \ingroup common
#     */
#   template <typename PointT> inline int 
#   getFieldIndex (const std::string &field_name, 
#                  std::vector<pcl::PCLPointField> &fields);
# 
#   /** \brief Get the list of available fields (i.e., dimension/channel)
#     * \param[in] cloud the point cloud message
#     * \param[out] fields a vector to the original \a PCLPointField vector that the raw PointCloud message contains
#     * \ingroup common
#     */
#   template <typename PointT> inline void 
#   getFields (const pcl::PointCloud<PointT> &cloud, std::vector<pcl::PCLPointField> &fields);
# 
#   /** \brief Get the list of available fields (i.e., dimension/channel)
#     * \param[out] fields a vector to the original \a PCLPointField vector that the raw PointCloud message contains
#     * \ingroup common
#     */
#   template <typename PointT> inline void 
#   getFields (std::vector<pcl::PCLPointField> &fields);
# 
#   /** \brief Get the list of all fields available in a given cloud
#     * \param[in] cloud the the point cloud message
#     * \ingroup common
#     */
#   template <typename PointT> inline std::string 
#   getFieldsList (const pcl::PointCloud<PointT> &cloud);
# 
#   /** \brief Get the available point cloud fields as a space separated string
#     * \param[in] cloud a pointer to the PointCloud message
#     * \ingroup common
#     */
#   inline std::string
#   getFieldsList (const pcl::PCLPointCloud2 &cloud)
#   {
#     std::string result;
#     for (size_t i = 0; i < cloud.fields.size () - 1; ++i)
#       result += cloud.fields[i].name + " ";
#     result += cloud.fields[cloud.fields.size () - 1].name;
#     return (result);
#   }
# 
#   /** \brief Obtains the size of a specific field data type in bytes
#     * \param[in] datatype the field data type (see PCLPointField.h)
#     * \ingroup common
#     */
#   inline int
#   getFieldSize (const int datatype)
#   {
#     switch (datatype)
#     {
#       case pcl::PCLPointField::INT8:
#       case pcl::PCLPointField::UINT8:
#         return (1);
# 
#       case pcl::PCLPointField::INT16:
#       case pcl::PCLPointField::UINT16:
#         return (2);
# 
#       case pcl::PCLPointField::INT32:
#       case pcl::PCLPointField::UINT32:
#       case pcl::PCLPointField::FLOAT32:
#         return (4);
# 
#       case pcl::PCLPointField::FLOAT64:
#         return (8);
# 
#       default:
#         return (0);
#     }
#   }
# 
#   /** \brief Obtain a vector with the sizes of all valid fields (e.g., not "_")
#     * \param[in] fields the input vector containing the fields
#     * \param[out] field_sizes the resultant field sizes in bytes
#     */
#   PCL_EXPORTS void
#   getFieldsSizes (const std::vector<pcl::PCLPointField> &fields,
#                   std::vector<int> &field_sizes);
# 
#   /** \brief Obtains the type of the PCLPointField from a specific size and type
#     * \param[in] size the size in bytes of the data field
#     * \param[in] type a char describing the type of the field  ('F' = float, 'I' = signed, 'U' = unsigned)
#     * \ingroup common
#     */
#   inline int
#   getFieldType (const int size, char type)
#   {
#     type = std::toupper (type, std::locale::classic ());
#     switch (size)
#     {
#       case 1:
#         if (type == 'I')
#           return (pcl::PCLPointField::INT8);
#         if (type == 'U')
#           return (pcl::PCLPointField::UINT8);
# 
#       case 2:
#         if (type == 'I')
#           return (pcl::PCLPointField::INT16);
#         if (type == 'U')
#           return (pcl::PCLPointField::UINT16);
# 
#       case 4:
#         if (type == 'I')
#           return (pcl::PCLPointField::INT32);
#         if (type == 'U')
#           return (pcl::PCLPointField::UINT32);
#         if (type == 'F')
#           return (pcl::PCLPointField::FLOAT32);
# 
#       case 8:
#         return (pcl::PCLPointField::FLOAT64);
# 
#       default:
#         return (-1);
#     }
#   }
# 
#   /** \brief Obtains the type of the PCLPointField from a specific PCLPointField as a char
#     * \param[in] type the PCLPointField field type
#     * \ingroup common
#     */
#   inline char
#   getFieldType (const int type)
#   {
#     switch (type)
#     {
#       case pcl::PCLPointField::INT8:
#       case pcl::PCLPointField::INT16:
#       case pcl::PCLPointField::INT32:
#         return ('I');
# 
#       case pcl::PCLPointField::UINT8:
#       case pcl::PCLPointField::UINT16:
#       case pcl::PCLPointField::UINT32:
#         return ('U');
# 
#       case pcl::PCLPointField::FLOAT32:
#       case pcl::PCLPointField::FLOAT64:
#         return ('F');
#       default:
#         return ('?');
#     }
#   }
# 
#   typedef enum
#   {
#     BORDER_CONSTANT = 0, BORDER_REPLICATE = 1,
#     BORDER_REFLECT = 2, BORDER_WRAP = 3,
#     BORDER_REFLECT_101 = 4, BORDER_TRANSPARENT = 5,
#     BORDER_DEFAULT = BORDER_REFLECT_101
#   } InterpolationType;
# 
#   /** \brief \return the right index according to the interpolation type.
#     * \note this is adapted from OpenCV
#     * \param p the index of point to interpolate
#     * \param length the top/bottom row or left/right column index
#     * \param type the requested interpolation
#     * \throws pcl::BadArgumentException if type is unknown
#     */
#   PCL_EXPORTS int
#   interpolatePointIndex (int p, int length, InterpolationType type);
# 
#   /** \brief Concatenate two pcl::PCLPointCloud2.
#     * \param[in] cloud1 the first input point cloud dataset
#     * \param[in] cloud2 the second input point cloud dataset
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \return true if successful, false if failed (e.g., name/number of fields differs)
#     * \ingroup common
#     */
#   PCL_EXPORTS bool 
#   concatenatePointCloud (const pcl::PCLPointCloud2 &cloud1,
#                          const pcl::PCLPointCloud2 &cloud2,
#                          pcl::PCLPointCloud2 &cloud_out);
# 
#   /** \brief Extract the indices of a given point cloud as a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] indices the vector of indices representing the points to be copied from \a cloud_in
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \note Assumes unique indices.
#     * \ingroup common
#     */
#   PCL_EXPORTS void 
#   copyPointCloud (const pcl::PCLPointCloud2 &cloud_in,
#                   const std::vector<int> &indices, 
#                   pcl::PCLPointCloud2 &cloud_out);
# 
#   /** \brief Extract the indices of a given point cloud as a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] indices the vector of indices representing the points to be copied from \a cloud_in
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \note Assumes unique indices.
#     * \ingroup common
#     */
#   PCL_EXPORTS void 
#   copyPointCloud (const pcl::PCLPointCloud2 &cloud_in,
#                   const std::vector<int, Eigen::aligned_allocator<int> > &indices, 
#                   pcl::PCLPointCloud2 &cloud_out);
# 
#   /** \brief Copy fields and point cloud data from \a cloud_in to \a cloud_out
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \ingroup common
#     */
#   PCL_EXPORTS void 
#   copyPointCloud (const pcl::PCLPointCloud2 &cloud_in,
#                   pcl::PCLPointCloud2 &cloud_out);
# 
#   /** \brief Check if two given point types are the same or not. */
#   template <typename Point1T, typename Point2T> inline bool
#   isSamePointType ()
#   {
#     return (typeid (Point1T) == typeid (Point2T));
#   }
# 
#   /** \brief Extract the indices of a given point cloud as a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] indices the vector of indices representing the points to be copied from \a cloud_in
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \note Assumes unique indices.
#     * \ingroup common
#     */
#   template <typename PointT> void 
#   copyPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                   const std::vector<int> &indices, 
#                   pcl::PointCloud<PointT> &cloud_out);
#  
#   /** \brief Extract the indices of a given point cloud as a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] indices the vector of indices representing the points to be copied from \a cloud_in
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \note Assumes unique indices.
#     * \ingroup common
#     */
#   template <typename PointT> void 
#   copyPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                   const std::vector<int, Eigen::aligned_allocator<int> > &indices, 
#                   pcl::PointCloud<PointT> &cloud_out);
# 
#   /** \brief Extract the indices of a given point cloud as a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] indices the PointIndices structure representing the points to be copied from cloud_in
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \note Assumes unique indices.
#     * \ingroup common
#     */
#   template <typename PointT> void 
#   copyPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                   const PointIndices &indices, 
#                   pcl::PointCloud<PointT> &cloud_out);
# 
#   /** \brief Extract the indices of a given point cloud as a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] indices the vector of indices representing the points to be copied from \a cloud_in
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \note Assumes unique indices.
#     * \ingroup common
#     */
#   template <typename PointT> void 
#   copyPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                   const std::vector<pcl::PointIndices> &indices, 
#                   pcl::PointCloud<PointT> &cloud_out);
# 
#   /** \brief Copy all the fields from a given point cloud into a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \ingroup common
#     */
#   template <typename PointInT, typename PointOutT> void 
#   copyPointCloud (const pcl::PointCloud<PointInT> &cloud_in, 
#                   pcl::PointCloud<PointOutT> &cloud_out);
# 
#   /** \brief Extract the indices of a given point cloud as a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] indices the vector of indices representing the points to be copied from \a cloud_in
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \note Assumes unique indices.
#     * \ingroup common
#     */
#   template <typename PointInT, typename PointOutT> void 
#   copyPointCloud (const pcl::PointCloud<PointInT> &cloud_in, 
#                   const std::vector<int> &indices, 
#                   pcl::PointCloud<PointOutT> &cloud_out);
# 
#   /** \brief Extract the indices of a given point cloud as a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] indices the vector of indices representing the points to be copied from \a cloud_in
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \note Assumes unique indices.
#     * \ingroup common
#     */
#   template <typename PointInT, typename PointOutT> void 
#   copyPointCloud (const pcl::PointCloud<PointInT> &cloud_in, 
#                   const std::vector<int, Eigen::aligned_allocator<int> > &indices, 
#                   pcl::PointCloud<PointOutT> &cloud_out);
# 
#   /** \brief Extract the indices of a given point cloud as a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] indices the PointIndices structure representing the points to be copied from cloud_in
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \note Assumes unique indices.
#     * \ingroup common
#     */
#   template <typename PointInT, typename PointOutT> void 
#   copyPointCloud (const pcl::PointCloud<PointInT> &cloud_in, 
#                   const PointIndices &indices, 
#                   pcl::PointCloud<PointOutT> &cloud_out);
# 
#   /** \brief Extract the indices of a given point cloud as a new point cloud
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[in] indices the vector of indices representing the points to be copied from cloud_in
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \note Assumes unique indices.
#     * \ingroup common
#     */
#   template <typename PointInT, typename PointOutT> void 
#   copyPointCloud (const pcl::PointCloud<PointInT> &cloud_in, 
#                   const std::vector<pcl::PointIndices> &indices, 
#                   pcl::PointCloud<PointOutT> &cloud_out);
# 
#   /** \brief Copy a point cloud inside a larger one interpolating borders.
#     * \param[in] cloud_in the input point cloud dataset
#     * \param[out] cloud_out the resultant output point cloud dataset
#     * \param top
#     * \param bottom
#     * \param left
#     * \param right
#     * Position of cloud_in inside cloud_out is given by \a top, \a left, \a bottom \a right.
#     * \param[in] border_type the interpolating method (pcl::BORDER_XXX)
#     *  BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
#     *  BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
#     *  BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
#     *  BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
#     *  BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
#     *  BORDER_TRANSPARENT:   mnopqr|abcdefgh|tuvwxyz  where m-r and t-z are orignal values of cloud_out
#     * \param value
#     * \throw pcl::BadArgumentException if any of top, bottom, left or right is negative.
#     * \ingroup common
#     */
#   template <typename PointT> void
#   copyPointCloud (const pcl::PointCloud<PointT> &cloud_in,
#                   pcl::PointCloud<PointT> &cloud_out,
#                   int top, int bottom, int left, int right,
#                   pcl::InterpolationType border_type, const PointT& value);
# 
#   /** \brief Concatenate two datasets representing different fields.
#     *
#     * \note If the input datasets have overlapping fields (i.e., both contain
#     * the same fields), then the data in the second cloud (cloud2_in) will
#     * overwrite the data in the first (cloud1_in).
#     *
#     * \param[in] cloud1_in the first input dataset
#     * \param[in] cloud2_in the second input dataset (overwrites the fields of the first dataset for those that are shared)
#     * \param[out] cloud_out the resultant output dataset created by the concatenation of all the fields in the input datasets
#     * \ingroup common
#     */
#   template <typename PointIn1T, typename PointIn2T, typename PointOutT> void 
#   concatenateFields (const pcl::PointCloud<PointIn1T> &cloud1_in, 
#                      const pcl::PointCloud<PointIn2T> &cloud2_in, 
#                      pcl::PointCloud<PointOutT> &cloud_out);
# 
#   /** \brief Concatenate two datasets representing different fields.
#     *
#     * \note If the input datasets have overlapping fields (i.e., both contain
#     * the same fields), then the data in the second cloud (cloud2_in) will
#     * overwrite the data in the first (cloud1_in).
#     *
#     * \param[in] cloud1_in the first input dataset
#     * \param[in] cloud2_in the second input dataset (overwrites the fields of the first dataset for those that are shared)
#     * \param[out] cloud_out the output dataset created by concatenating all the fields in the input datasets
#     * \ingroup common
#     */
#   PCL_EXPORTS bool
#   concatenateFields (const pcl::PCLPointCloud2 &cloud1_in,
#                      const pcl::PCLPointCloud2 &cloud2_in,
#                      pcl::PCLPointCloud2 &cloud_out);
# 
#   /** \brief Copy the XYZ dimensions of a pcl::PCLPointCloud2 into Eigen format
#     * \param[in] in the point cloud message
#     * \param[out] out the resultant Eigen MatrixXf format containing XYZ0 / point
#     * \ingroup common
#     */
#   PCL_EXPORTS bool 
#   getPointCloudAsEigen (const pcl::PCLPointCloud2 &in, Eigen::MatrixXf &out);
# 
#   /** \brief Copy the XYZ dimensions from an Eigen MatrixXf into a pcl::PCLPointCloud2 message
#     * \param[in] in the Eigen MatrixXf format containing XYZ0 / point
#     * \param[out] out the resultant point cloud message
#     * \note the method assumes that the PCLPointCloud2 message already has the fields set up properly !
#     * \ingroup common
#     */
#   PCL_EXPORTS bool 
#   getEigenAsPointCloud (Eigen::MatrixXf &in, pcl::PCLPointCloud2 &out);
#   
#   namespace io 
#   {
#     /** \brief swap bytes order of a char array of length N
#       * \param bytes char array to swap
#       * \ingroup common
#       */
#     template <std::size_t N> void 
#     swapByte (char* bytes);
# 
#    /** \brief specialization of swapByte for dimension 1
#      * \param bytes char array to swap
#      */
#     template <> inline void 
#     swapByte<1> (char* bytes) { bytes[0] = bytes[0]; }
# 
#   
#    /** \brief specialization of swapByte for dimension 2
#      * \param bytes char array to swap
#      */
#     template <> inline void 
#     swapByte<2> (char* bytes) { std::swap (bytes[0], bytes[1]); }
#   
#    /** \brief specialization of swapByte for dimension 4
#      * \param bytes char array to swap
#      */
#     template <> inline void 
#     swapByte<4> (char* bytes)
#     {
#       std::swap (bytes[0], bytes[3]);
#       std::swap (bytes[1], bytes[2]);
#     }
#   
#    /** \brief specialization of swapByte for dimension 8
#      * \param bytes char array to swap
#      */
#     template <> inline void 
#     swapByte<8> (char* bytes)
#     {
#       std::swap (bytes[0], bytes[7]);
#       std::swap (bytes[1], bytes[6]);
#       std::swap (bytes[2], bytes[5]);
#       std::swap (bytes[3], bytes[4]);
#     }
#   
#     /** \brief swaps byte of an arbitrary type T casting it to char*
#       * \param value the data you want its bytes swapped
#       */
#     template <typename T> void 
#     swapByte (T& value)
#     {
#       pcl::io::swapByte<sizeof(T)> (reinterpret_cast<char*> (&value));
#     }
#   }
# }
# 
# #include <pcl/common/impl/io.hpp>
# 
# #endif  //#ifndef PCL_COMMON_IO_H_
# ###
# 
# # norms.h
# namespace pcl
# {
#   /** \brief Enum that defines all the types of norms available.
#    * \note Any new norm type should have its own enum value and its own case in the selectNorm () method
#    * \ingroup common
#    */
#   enum NormType {L1, L2_SQR, L2, LINF, JM, B, SUBLINEAR, CS, DIV, PF, K, KL, HIK};
# 
#   /** \brief Method that calculates any norm type available, based on the norm_type variable
#    * \note FloatVectorT is any type of vector with its values accessible via [ ]
#    * \ingroup common
#    * */
#   template <typename FloatVectorT> inline float
#   selectNorm (FloatVectorT A, FloatVectorT B, int dim, NormType norm_type);
# 
#   /** \brief Compute the L1 norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   L1_Norm (FloatVectorT A, FloatVectorT B, int dim);
#   
#   /** \brief Compute the squared L2 norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   L2_Norm_SQR (FloatVectorT A, FloatVectorT B, int dim);
#   
#   /** \brief Compute the L2 norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   L2_Norm (FloatVectorT A, FloatVectorT B, int dim);
# 
#   /** \brief Compute the L-infinity norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */  
#   template <typename FloatVectorT> inline float
#   Linf_Norm (FloatVectorT A, FloatVectorT B, int dim);
# 
#   /** \brief Compute the JM norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   JM_Norm (FloatVectorT A, FloatVectorT B, int dim);
# 
#   /** \brief Compute the B norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   B_Norm (FloatVectorT A, FloatVectorT B, int dim);
# 
#   /** \brief Compute the sublinear norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   Sublinear_Norm (FloatVectorT A, FloatVectorT B, int dim);
# 
#   /** \brief Compute the CS norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   CS_Norm (FloatVectorT A, FloatVectorT B, int dim);
# 
#   /** \brief Compute the div norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   Div_Norm (FloatVectorT A, FloatVectorT B, int dim);
# 
#   /** \brief Compute the PF norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \param P1 the first parameter
#     * \param P2 the second parameter
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   PF_Norm (FloatVectorT A, FloatVectorT B, int dim, float P1, float P2);
# 
#   /** \brief Compute the K norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \param P1 the first parameter
#     * \param P2 the second parameter
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   K_Norm (FloatVectorT A, FloatVectorT B, int dim, float P1, float P2);
# 
#   /** \brief Compute the KL between two discrete probability density functions
#     * \param A the first discrete PDF
#     * \param B the second discrete PDF
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   KL_Norm (FloatVectorT A, FloatVectorT B, int dim);
# 
#   /** \brief Compute the HIK norm of the vector between two points
#     * \param A the first point
#     * \param B the second point
#     * \param dim the number of dimensions in \a A and \a B (dimensions must match)
#     * \note FloatVectorT is any type of vector with its values accessible via [ ]
#     * \ingroup common
#     */
#   template <typename FloatVectorT> inline float
#   HIK_Norm (FloatVectorT A, FloatVectorT B, int dim);
# }
# /*@}*/
# #include <pcl/common/impl/norms.hpp>
# 
# #endif  //#ifndef PCL_NORMS_H_
# ###
# 
# # pca.h
# namespace pcl 
# {
#   /** Principal Component analysis (PCA) class.\n
#     *  Principal components are extracted by singular values decomposition on the 
#     * covariance matrix of the centered input cloud. Available data after pca computation 
#     * are the mean of the input data, the eigenvalues (in descending order) and 
#     * corresponding eigenvectors.\n
#     * Other methods allow projection in the eigenspace, reconstruction from eigenspace and 
#     *  update of the eigenspace with a new datum (according Matej Artec, Matjaz Jogan and 
#     * Ales Leonardis: "Incremental PCA for On-line Visual Learning and Recognition").
#     *
#     * \author Nizar Sallem
#     * \ingroup common
#     */
#   template <typename PointT>
#   class PCA : public pcl::PCLBase <PointT>
#   {
#     public:
#       typedef pcl::PCLBase <PointT> Base;
#       typedef typename Base::PointCloud PointCloud;
#       typedef typename Base::PointCloudPtr PointCloudPtr;
#       typedef typename Base::PointCloudConstPtr PointCloudConstPtr;
#       typedef typename Base::PointIndicesPtr PointIndicesPtr;
#       typedef typename Base::PointIndicesConstPtr PointIndicesConstPtr;
# 
#       using Base::input_;
#       using Base::indices_;
#       using Base::initCompute;
#       using Base::setInputCloud;
# 
#       /** Updating method flag */
#       enum FLAG 
#       {
#         /** keep the new basis vectors if possible */
#         increase, 
#         /** preserve subspace dimension */
#         preserve
#       };
#     
#       /** \brief Default Constructor
#         * \param basis_only flag to compute only the PCA basis
#         */
#       PCA (bool basis_only = false)
#         : Base ()
#         , compute_done_ (false)
#         , basis_only_ (basis_only) 
#         , eigenvectors_ ()
#         , coefficients_ ()
#         , mean_ ()
#         , eigenvalues_  ()
#       {}
#       
#       /** \brief Constructor with direct computation
#         * X input m*n matrix (ie n vectors of R(m))
#         * basis_only flag to compute only the PCA basis
#         */
#       PCL_DEPRECATED ("Use PCA (bool basis_only); setInputCloud (X.makeShared ()); instead")
#       PCA (const pcl::PointCloud<PointT>& X, bool basis_only = false);
# 
#       /** Copy Constructor
#         * \param[in] pca PCA object
#         */
#       PCA (PCA const & pca) 
#         : Base (pca)
#         , compute_done_ (pca.compute_done_)
#         , basis_only_ (pca.basis_only_) 
#         , eigenvectors_ (pca.eigenvectors_)
#         , coefficients_ (pca.coefficients_)
#         , mean_ (pca.mean_)
#         , eigenvalues_  (pca.eigenvalues_)
#       {}
# 
#       /** Assignment operator
#         * \param[in] pca PCA object
#         */
#       inline PCA& 
#       operator= (PCA const & pca) 
#       {
#         eigenvectors_ = pca.eigenvectors;
#         coefficients_ = pca.coefficients;
#         eigenvalues_  = pca.eigenvalues;
#         mean_         = pca.mean;
#         return (*this);
#       }
#       
#       /** \brief Provide a pointer to the input dataset
#         * \param cloud the const boost shared pointer to a PointCloud message
#         */
#       inline void 
#       setInputCloud (const PointCloudConstPtr &cloud) 
#       { 
#         Base::setInputCloud (cloud);
#         compute_done_ = false;
#       }
# 
#       /** \brief Mean accessor
#         * \throw InitFailedException
#         */
#       inline Eigen::Vector4f& 
#       getMean () 
#       {
#         if (!compute_done_)
#           initCompute ();
#         if (!compute_done_)
#           PCL_THROW_EXCEPTION (InitFailedException, 
#                                "[pcl::PCA::getMean] PCA initCompute failed");
#         return (mean_);
#       }
# 
#       /** Eigen Vectors accessor
#         * \throw InitFailedException
#         */
#       inline Eigen::Matrix3f& 
#       getEigenVectors () 
#       {
#         if (!compute_done_)
#           initCompute ();
#         if (!compute_done_)
#           PCL_THROW_EXCEPTION (InitFailedException, 
#                                "[pcl::PCA::getEigenVectors] PCA initCompute failed");
#         return (eigenvectors_);
#       }
#       
#       /** Eigen Values accessor
#         * \throw InitFailedException
#         */
#       inline Eigen::Vector3f& 
#       getEigenValues ()
#       {
#         if (!compute_done_)
#           initCompute ();
#         if (!compute_done_)
#           PCL_THROW_EXCEPTION (InitFailedException, 
#                                "[pcl::PCA::getEigenVectors] PCA getEigenValues failed");
#         return (eigenvalues_);
#       }
#       
#       /** Coefficients accessor
#         * \throw InitFailedException
#         */
#       inline Eigen::MatrixXf& 
#       getCoefficients () 
#       {
#         if (!compute_done_)
#           initCompute ();
#         if (!compute_done_)
#           PCL_THROW_EXCEPTION (InitFailedException, 
#                                "[pcl::PCA::getEigenVectors] PCA getCoefficients failed");
#         return (coefficients_);
#       }
#             
#       /** update PCA with a new point
#         * \param[in] input input point 
#         * \param[in] flag update flag
#         * \throw InitFailedException
#         */
#       inline void 
#       update (const PointT& input, FLAG flag = preserve);
#       
#       /** Project point on the eigenspace.
#         * \param[in] input point from original dataset
#         * \param[out] projection the point in eigen vectors space
#         * \throw InitFailedException
#         */
#       inline void 
#       project (const PointT& input, PointT& projection);
# 
#       /** Project cloud on the eigenspace.
#         * \param[in] input cloud from original dataset
#         * \param[out] projection the cloud in eigen vectors space
#         * \throw InitFailedException
#         */
#       inline void
#       project (const PointCloud& input, PointCloud& projection);
#       
#       /** Reconstruct point from its projection
#         * \param[in] projection point from eigenvector space
#         * \param[out] input reconstructed point
#         * \throw InitFailedException
#         */
#       inline void 
#       reconstruct (const PointT& projection, PointT& input);
# 
#       /** Reconstruct cloud from its projection
#         * \param[in] projection cloud from eigenvector space
#         * \param[out] input reconstructed cloud
#         * \throw InitFailedException
#         */
#       inline void
#       reconstruct (const PointCloud& projection, PointCloud& input);
# 
#     private:
#       inline bool
#       initCompute ();
# 
#       bool compute_done_;
#       bool basis_only_;
#       Eigen::Matrix3f eigenvectors_;
#       Eigen::MatrixXf coefficients_;
#       Eigen::Vector4f mean_;
#       Eigen::Vector3f eigenvalues_;
#   }; // class PCA
# } // namespace pcl
# 
# #include <pcl/common/impl/pca.hpp>
# 
# #endif // PCL_PCA_H
# ###
# 
# 
# # piecewise_linear_function.h
# namespace pcl 
# {
#   /**
#     * \brief This provides functionalities to efficiently return values for piecewise linear function
#     * \ingroup common
#     */
#   class PiecewiseLinearFunction
#   {
#     public:
#       // =====CONSTRUCTOR & DESTRUCTOR=====
#       //! Constructor
#       PiecewiseLinearFunction (float factor, float offset);
#       
#       // =====PUBLIC METHODS=====
#       //! Get the list of known data points
#       std::vector<float>& 
#       getDataPoints () 
#       { 
#         return data_points_;
#       }
#       
#       //! Get the value of the function at the given point
#       inline float 
#       getValue (float point) const;
#       
#       // =====PUBLIC MEMBER VARIABLES=====
#       
#     protected:
#       // =====PROTECTED MEMBER VARIABLES=====
#       std::vector<float> data_points_;
#       float factor_;
#       float offset_;
#   };
# 
# }  // end namespace pcl
# 
# #include <pcl/common/impl/piecewise_linear_function.hpp>
# 
# #endif  //#ifndef PCL_PIECEWISE_LINEAR_FUNCTION_H_
# ###
# 
# 
# # point_operators.h
# ###
# 
# # point_tests.h
# namespace pcl
# {
#   /** Tests if the 3D components of a point are all finite
#     * param[in] pt point to be tested
#     */
#   template <typename PointT> inline bool
#   isFinite (const PointT &pt)
#   {
#     return (pcl_isfinite (pt.x) && pcl_isfinite (pt.y) && pcl_isfinite (pt.z));
#   }
# 
# #ifdef _MSC_VER
#   template <typename PointT> inline bool
#   isFinite (const Eigen::internal::workaround_msvc_stl_support<PointT> &pt)
#   {
#     return isFinite<PointT> (static_cast<const PointT&> (pt));
#   }
# #endif
# 
#   template<> inline bool isFinite<pcl::RGB> (const pcl::RGB&) { return (true); }
#   template<> inline bool isFinite<pcl::Label> (const pcl::Label&) { return (true); }
#   template<> inline bool isFinite<pcl::Axis> (const pcl::Axis&) { return (true); }
#   template<> inline bool isFinite<pcl::Intensity> (const pcl::Intensity&) { return (true); }
#   template<> inline bool isFinite<pcl::MomentInvariants> (const pcl::MomentInvariants&) { return (true); }
#   template<> inline bool isFinite<pcl::PrincipalRadiiRSD> (const pcl::PrincipalRadiiRSD&) { return (true); }
#   template<> inline bool isFinite<pcl::Boundary> (const pcl::Boundary&) { return (true); }
#   template<> inline bool isFinite<pcl::PrincipalCurvatures> (const pcl::PrincipalCurvatures&) { return (true); }
#   template<> inline bool isFinite<pcl::SHOT352> (const pcl::SHOT352&) { return (true); }
#   template<> inline bool isFinite<pcl::SHOT1344> (const pcl::SHOT1344&) { return (true); }
#   template<> inline bool isFinite<pcl::ReferenceFrame> (const pcl::ReferenceFrame&) { return (true); }
#   template<> inline bool isFinite<pcl::ShapeContext1980> (const pcl::ShapeContext1980&) { return (true); }
#   template<> inline bool isFinite<pcl::PFHSignature125> (const pcl::PFHSignature125&) { return (true); }
#   template<> inline bool isFinite<pcl::PFHRGBSignature250> (const pcl::PFHRGBSignature250&) { return (true); }
#   template<> inline bool isFinite<pcl::PPFSignature> (const pcl::PPFSignature&) { return (true); }
#   template<> inline bool isFinite<pcl::PPFRGBSignature> (const pcl::PPFRGBSignature&) { return (true); }
#   template<> inline bool isFinite<pcl::NormalBasedSignature12> (const pcl::NormalBasedSignature12&) { return (true); }
#   template<> inline bool isFinite<pcl::FPFHSignature33> (const pcl::FPFHSignature33&) { return (true); }
#   template<> inline bool isFinite<pcl::VFHSignature308> (const pcl::VFHSignature308&) { return (true); }
#   template<> inline bool isFinite<pcl::ESFSignature640> (const pcl::ESFSignature640&) { return (true); }
#   template<> inline bool isFinite<pcl::IntensityGradient> (const pcl::IntensityGradient&) { return (true); }
# 
#   // specification for pcl::PointXY
#   template <> inline bool
#   isFinite<pcl::PointXY> (const pcl::PointXY &p)
#   {
#     return (pcl_isfinite (p.x) && pcl_isfinite (p.y));
#   }
# 
#   // specification for pcl::BorderDescription
#   template <> inline bool
#   isFinite<pcl::BorderDescription> (const pcl::BorderDescription &p)
#   {
#     return (pcl_isfinite (p.x) && pcl_isfinite (p.y));
#   }
# 
#   // specification for pcl::Normal
#   template <> inline bool
#   isFinite<pcl::Normal> (const pcl::Normal &n)
#   {
#     return (pcl_isfinite (n.normal_x) && pcl_isfinite (n.normal_y) && pcl_isfinite (n.normal_z));
#   }
# }
# 
# #endif    // PCL_COMMON_POINT_TESTS_H_
# ###
# 
# # polynomial_calculations.h
# namespace pcl 
# {
#   /** \brief This provides some functionality for polynomials,
#     *         like finding roots or approximating bivariate polynomials
#     *  \author Bastian Steder 
#     *  \ingroup common
#     */
#   template <typename real>
#   class PolynomialCalculationsT 
#   {
#     public:
#       // =====CONSTRUCTOR & DESTRUCTOR=====
#       PolynomialCalculationsT ();
#       ~PolynomialCalculationsT ();
#       
#       // =====PUBLIC STRUCTS=====
#       //! Parameters used in this class
#       struct Parameters
#       {
#         Parameters () : zero_value (), sqr_zero_value () { setZeroValue (1e-6);}
#         //! Set zero_value
#         void
#         setZeroValue (real new_zero_value);
# 
#         real zero_value;       //!< Every value below this is considered to be zero
#         real sqr_zero_value;   //!< sqr of the above
#       };
#       
#       // =====PUBLIC METHODS=====
#       /** Solves an equation of the form ax^4 + bx^3 + cx^2 +dx + e = 0
#        *  See http://en.wikipedia.org/wiki/Quartic_equation#Summary_of_Ferrari.27s_method */
#       inline void
#       solveQuarticEquation (real a, real b, real c, real d, real e, std::vector<real>& roots) const;
# 
#       /** Solves an equation of the form ax^3 + bx^2 + cx + d = 0
#        *  See http://en.wikipedia.org/wiki/Cubic_equation */
#       inline void
#       solveCubicEquation (real a, real b, real c, real d, std::vector<real>& roots) const;
# 
#       /** Solves an equation of the form ax^2 + bx + c = 0
#        *  See http://en.wikipedia.org/wiki/Quadratic_equation */
#       inline void
#       solveQuadraticEquation (real a, real b, real c, std::vector<real>& roots) const;
# 
#       /** Solves an equation of the form ax + b = 0 */
#       inline void
#       solveLinearEquation (real a, real b, std::vector<real>& roots) const;
#       
#       /** Get the bivariate polynomial approximation for Z(X,Y) from the given sample points.
#        *  The parameters a,b,c,... for the polynom are returned.
#        *  The order is, e.g., for degree 1: ax+by+c and for degree 2: ax2+bxy+cx+dy2+ey+f.
#        *  error is set to true if the approximation did not work for any reason
#        *  (not enough points, matrix not invertible, etc.) */
#       inline BivariatePolynomialT<real>
#       bivariatePolynomialApproximation (std::vector<Eigen::Matrix<real, 3, 1> >& samplePoints,
#                                         unsigned int polynomial_degree, bool& error) const;
#       
#       //! Same as above, using a reference for the return value
#       inline bool
#       bivariatePolynomialApproximation (std::vector<Eigen::Matrix<real, 3, 1> >& samplePoints,
#                                         unsigned int polynomial_degree, BivariatePolynomialT<real>& ret) const;
# 
#       //! Set the minimum value under which values are considered zero
#       inline void
#       setZeroValue (real new_zero_value) { parameters_.setZeroValue(new_zero_value); }
#       
#     protected:  
#       // =====PROTECTED METHODS=====
#       //! check if fabs(d)<zeroValue
#       inline bool
#       isNearlyZero (real d) const 
#       { 
#         return (fabs (d) < parameters_.zero_value);
#       }
#       
#       //! check if sqrt(fabs(d))<zeroValue
#       inline bool
#       sqrtIsNearlyZero (real d) const 
#       { 
#         return (fabs (d) < parameters_.sqr_zero_value);
#       }
#       
#       // =====PROTECTED MEMBERS=====
#       Parameters parameters_;
#   };
# 
#   typedef PolynomialCalculationsT<double> PolynomialCalculationsd;
#   typedef PolynomialCalculationsT<float>  PolynomialCalculations;
# 
# }  // end namespace
# 
# #include <pcl/common/impl/polynomial_calculations.hpp>
# #endif
# ###
# 
# # poses_from_matches.h
# 
# namespace pcl
# {
#   /**
#     * \brief calculate 3D transformation based on point correspondencdes
#     * \author Bastian Steder
#     * \ingroup common
#     */
#   class PCL_EXPORTS PosesFromMatches
#   {
#     public:
#       // =====CONSTRUCTOR & DESTRUCTOR=====
#       //! Constructor
#       PosesFromMatches();
#       //! Destructor
#       ~PosesFromMatches();
#       
#       // =====STRUCTS=====
#       //! Parameters used in this class
#       struct PCL_EXPORTS Parameters
#       {
#         Parameters() : max_correspondence_distance_error(0.2f) {}
#         float max_correspondence_distance_error;  // As a fraction
#       };
# 
#       //! A result of the pose estimation process
#       struct PoseEstimate
#       {
#         PoseEstimate () : 
#           transformation (Eigen::Affine3f::Identity ()),
#           score (0),
#           correspondence_indices (0) 
#         {}
# 
#         Eigen::Affine3f transformation;   //!< The estimated transformation between the two coordinate systems
#         float score;                         //!< An estimate in [0,1], how good the estimated pose is 
#         std::vector<int> correspondence_indices;  //!< The indices of the used correspondences
# 
#         struct IsBetter 
#         {
#           bool operator()(const PoseEstimate& pe1, const PoseEstimate& pe2) const { return pe1.score>pe2.score;}
#         };
#         public:
#           EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#       };
#       
#       // =====TYPEDEFS=====
#       typedef std::vector<PoseEstimate, Eigen::aligned_allocator<PoseEstimate> > PoseEstimatesVector;
# 
#       
#       // =====STATIC METHODS=====
#       
#       // =====PUBLIC METHODS=====
#       /** Use single 6DOF correspondences to estimate transformations between the coordinate systems.
#        *  Use max_no_of_results=-1 to use all.
#        *  It is assumed, that the correspondences are sorted from good to bad. */
#       void 
#       estimatePosesUsing1Correspondence (
#           const PointCorrespondences6DVector& correspondences,
#           int max_no_of_results, PoseEstimatesVector& pose_estimates) const;
# 
#       /** Use pairs of 6DOF correspondences to estimate transformations between the coordinate systems.
#        *  It is assumed, that the correspondences are sorted from good to bad. */
#       void 
#       estimatePosesUsing2Correspondences (
#           const PointCorrespondences6DVector& correspondences,
#           int max_no_of_tested_combinations, int max_no_of_results,
#           PoseEstimatesVector& pose_estimates) const;
#       
#       /** Use triples of 6DOF correspondences to estimate transformations between the coordinate systems.
#        *  It is assumed, that the correspondences are sorted from good to bad. */
#       void 
#       estimatePosesUsing3Correspondences (
#           const PointCorrespondences6DVector& correspondences,
#           int max_no_of_tested_combinations, int max_no_of_results,
#           PoseEstimatesVector& pose_estimates) const;
# 
#       /// Get a reference to the parameters struct
#       Parameters& 
#       getParameters () { return parameters_; }
# 
#     protected:
#       // =====PROTECTED MEMBER VARIABLES=====
#       Parameters parameters_;
# 
#   };
# 
# }  // end namespace pcl
# 
# #endif  //#ifndef PCL_POSES_FROM_MATCHES_H_
# ###
# 
# # projection_matrix.h
# namespace pcl
# {
#   template <typename T> class PointCloud;
# 
#   /** \brief Estimates the projection matrix P = K * (R|-R*t) from organized point clouds, with
#     *        K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]]
#     *        R = rotation matrix and
#     *        t = translation vector  
#     * 
#     * \param[in] cloud input cloud. Must be organized and from a projective device. e.g. stereo or kinect, ...
#     * \param[out] projection_matrix output projection matrix
#     * \param[in] indices The indices to be used to determine the projection matrix 
#     * \return the resudial error. A high residual indicates, that the point cloud was not from a projective device.
#     */
#   template<typename PointT> double
#   estimateProjectionMatrix (typename pcl::PointCloud<PointT>::ConstPtr cloud, Eigen::Matrix<float, 3, 4, Eigen::RowMajor>& projection_matrix, const std::vector<int>& indices = std::vector<int> ());
#   
#   /** \brief Determines the camera matrix from the given projection matrix.
#     * \note This method does NOT use a RQ decomposition, but uses the fact that the left 3x3 matrix P' of P squared eliminates the rotational part.
#     *       P' = K * R -> P' * P'^T = K * R * R^T * K = K * K^T
#     * \param[in] projection_matrix
#     * \param[out] camera_matrix
#     */
#   PCL_EXPORTS void
#   getCameraMatrixFromProjectionMatrix (const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>& projection_matrix, Eigen::Matrix3f& camera_matrix);  
# }
# 
# #include <pcl/common/impl/projection_matrix.hpp>
# 
# #endif // __PCL_ORGANIZED_PROJECTION_MATRIX_H__
# ###
# 
# 
# # random.h
# namespace pcl 
# {
#   namespace common 
#   {
#     /// uniform distribution dummy struct
#     template <typename T> struct uniform_distribution;
#     /// uniform distribution int specialized
#     template<> 
#     struct uniform_distribution<int> 
#     {
#       typedef boost::uniform_int<int> type;
#     };
#     /// uniform distribution float specialized
#     template<> 
#     struct uniform_distribution<float> 
#     {
#       typedef boost::uniform_real<float> type;
#     };
#     ///  normal distribution
#     template<typename T> 
#     struct normal_distribution
#     {
#       typedef boost::normal_distribution<T> type;
#     };
# 
#     /** \brief UniformGenerator class generates a random number from range [min, max] at each run picked
#       * according to a uniform distribution i.e eaach number within [min, max] has almost the same 
#       * probability of being drawn.
#       *
#       * \author Nizar Sallem
#       */
#     template<typename T>
#     class UniformGenerator 
#     {
#       public:
#         struct Parameters
#         {
#             Parameters (T _min = 0, T _max = 1, pcl::uint32_t _seed = 1)
#             : min (_min)
#             , max (_max)
#             , seed (_seed)
#           {}
# 
#           T min;
#           T max;
#           pcl::uint32_t seed;
#         };
# 
#         /** Constructor
#           * \param min: included lower bound
#           * \param max: included higher bound
#           * \param seed: seeding value
#           */
#         UniformGenerator(T min = 0, T max = 1, pcl::uint32_t seed = -1);
# 
#         /** Constructor
#           * \param parameters uniform distribution parameters and generator seed
#           */
#         UniformGenerator(const Parameters& parameters);
# 
#         /** Change seed value
#           * \param[in] seed new generator seed value
#           */
#         void 
#         setSeed (pcl::uint32_t seed);
# 
#         /** Set the uniform number generator parameters
#           * \param[in] min minimum allowed value
#           * \param[in] max maximum allowed value
#           * \param[in] seed random number generator seed (applied if != -1)
#           */
#         void 
#         setParameters (T min, T max, pcl::uint32_t seed = -1);
# 
#         /** Set generator parameters
#           * \param parameters uniform distribution parameters and generator seed
#           */
#         void
#         setParameters (const Parameters& parameters);
# 
#         /// \return uniform distribution parameters and generator seed
#         const Parameters&
#         getParameters () { return (parameters_); }
# 
#         /// \return a randomly generated number in the interval [min, max]
#         inline T 
#         run () { return (generator_ ()); }
# 
#       private:
#         typedef boost::mt19937 EngineType;
#         typedef typename uniform_distribution<T>::type DistributionType;
#         /// parameters
#         Parameters parameters_;
#         /// uniform distribution
#         DistributionType distribution_;
#         /// random number generator
#         EngineType rng_;
#         /// generator of random number from a uniform distribution
#         boost::variate_generator<EngineType&, DistributionType> generator_;
#     };
# 
#     /** \brief NormalGenerator class generates a random number from a normal distribution specified
#       * by (mean, sigma).
#       *
#       * \author Nizar Sallem
#       */
#     template<typename T>
#     class NormalGenerator 
#     {
#       public:
#         struct Parameters
#         {
#             Parameters (T _mean = 0, T _sigma = 1, pcl::uint32_t _seed = 1)
#             : mean (_mean)
#             , sigma (_sigma)
#             , seed (_seed)
#           {}
# 
#           T mean;
#           T sigma;
#           pcl::uint32_t seed;
#         };
# 
#         /** Constructor
#           * \param[in] mean normal mean
#           * \param[in] sigma normal variation
#           * \param[in] seed seeding value
#           */
#         NormalGenerator(T mean = 0, T sigma = 1, pcl::uint32_t seed = -1);
# 
#         /** Constructor
#           * \param parameters normal distribution parameters and seed
#           */
#         NormalGenerator(const Parameters& parameters);
# 
#         /** Change seed value
#           * \param[in] seed new seed value
#           */
#         void 
#         setSeed (pcl::uint32_t seed);
# 
#         /** Set the normal number generator parameters
#           * \param[in] mean mean of the normal distribution
#           * \param[in] sigma standard variation of the normal distribution
#           * \param[in] seed random number generator seed (applied if != -1)
#           */
#         void 
#         setParameters (T mean, T sigma, pcl::uint32_t seed = -1);
# 
#         /** Set generator parameters
#           * \param parameters normal distribution parameters and seed
#           */
#         void
#         setParameters (const Parameters& parameters);
# 
#         /// \return normal distribution parameters and generator seed
#         const Parameters&
#         getParameters () { return (parameters_); }
# 
#         /// \return a randomly generated number in the normal distribution (mean, sigma)
#         inline T 
#         run () { return (generator_ ()); }
# 
#         typedef boost::mt19937 EngineType;
#         typedef typename normal_distribution<T>::type DistributionType;
#         /// parameters
#         Parameters parameters_;
#         /// normal distribution
#         DistributionType distribution_;
#         /// random number generator
#         EngineType rng_;
#         /// generator of random number from a normal distribution
#         boost::variate_generator<EngineType&, DistributionType > generator_;
#     };
#   }
# }
# 
# #include <pcl/common/impl/random.hpp>
# #endif
# ###
# 
# # register_point_struct.h
# #include <pcl/pcl_macros.h>
# #include <pcl/point_traits.h>
# #include <boost/mpl/vector.hpp>
# #include <boost/preprocessor/seq/enum.hpp>
# #include <boost/preprocessor/seq/for_each.hpp>
# #include <boost/preprocessor/seq/transform.hpp>
# #include <boost/preprocessor/cat.hpp>
# #include <boost/preprocessor/comparison.hpp>
# #include <boost/utility.hpp>
# //https://bugreports.qt-project.org/browse/QTBUG-22829
# #ifndef Q_MOC_RUN
# #include <boost/type_traits.hpp>
# #endif
# #include <stddef.h> //offsetof
# 
# // Must be used in global namespace with name fully qualified
# #define POINT_CLOUD_REGISTER_POINT_STRUCT(name, fseq)               \
#   POINT_CLOUD_REGISTER_POINT_STRUCT_I(name,                         \
#     BOOST_PP_CAT(POINT_CLOUD_REGISTER_POINT_STRUCT_X fseq, 0))      \
#   /***/
# 
# #define POINT_CLOUD_REGISTER_POINT_WRAPPER(wrapper, pod)    \
#   BOOST_MPL_ASSERT_MSG(sizeof(wrapper) == sizeof(pod), POINT_WRAPPER_AND_POD_TYPES_HAVE_DIFFERENT_SIZES, (wrapper&, pod&)); \
#   namespace pcl {                                           \
#     namespace traits {                                      \
#       template<> struct POD<wrapper> { typedef pod type; }; \
#     }                                                       \
#   }                                                         \
#   /***/
# 
# // These macros help transform the unusual data structure (type, name, tag)(type, name, tag)...
# // into a proper preprocessor sequence of 3-tuples ((type, name, tag))((type, name, tag))...
# #define POINT_CLOUD_REGISTER_POINT_STRUCT_X(type, name, tag)            \
#   ((type, name, tag)) POINT_CLOUD_REGISTER_POINT_STRUCT_Y
# #define POINT_CLOUD_REGISTER_POINT_STRUCT_Y(type, name, tag)            \
#   ((type, name, tag)) POINT_CLOUD_REGISTER_POINT_STRUCT_X
# #define POINT_CLOUD_REGISTER_POINT_STRUCT_X0
# #define POINT_CLOUD_REGISTER_POINT_STRUCT_Y0
# 
# namespace pcl
# {
#   namespace traits
#   {
#     template<typename T> inline
#     typename boost::disable_if_c<boost::is_array<T>::value>::type
#     plus (T &l, const T &r)
#     {
#       l += r;
#     }
# 
#     template<typename T> inline
#     typename boost::enable_if_c<boost::is_array<T>::value>::type
#     plus (typename boost::remove_const<T>::type &l, const T &r)
#     {
#       typedef typename boost::remove_all_extents<T>::type type;
#       static const uint32_t count = sizeof (T) / sizeof (type);
#       for (int i = 0; i < count; ++i)
#         l[i] += r[i];
#     }
# 
#     template<typename T1, typename T2> inline
#     typename boost::disable_if_c<boost::is_array<T1>::value>::type
#     plusscalar (T1 &p, const T2 &scalar)
#     {
#       p += scalar;
#     }
# 
#     template<typename T1, typename T2> inline
#     typename boost::enable_if_c<boost::is_array<T1>::value>::type
#     plusscalar (T1 &p, const T2 &scalar)
#     {
#       typedef typename boost::remove_all_extents<T1>::type type;
#       static const uint32_t count = sizeof (T1) / sizeof (type);
#       for (int i = 0; i < count; ++i)
#         p[i] += scalar;
#     }
# 
#     template<typename T> inline
#     typename boost::disable_if_c<boost::is_array<T>::value>::type
#     minus (T &l, const T &r)
#     {
#       l -= r;
#     }
# 
#     template<typename T> inline
#     typename boost::enable_if_c<boost::is_array<T>::value>::type
#     minus (typename boost::remove_const<T>::type &l, const T &r)
#     {
#       typedef typename boost::remove_all_extents<T>::type type;
#       static const uint32_t count = sizeof (T) / sizeof (type);
#       for (int i = 0; i < count; ++i)
#         l[i] -= r[i];
#     }
# 
#     template<typename T1, typename T2> inline
#     typename boost::disable_if_c<boost::is_array<T1>::value>::type
#     minusscalar (T1 &p, const T2 &scalar)
#     {
#       p -= scalar;
#     }
# 
#     template<typename T1, typename T2> inline
#     typename boost::enable_if_c<boost::is_array<T1>::value>::type
#     minusscalar (T1 &p, const T2 &scalar)
#     {
#       typedef typename boost::remove_all_extents<T1>::type type;
#       static const uint32_t count = sizeof (T1) / sizeof (type);
#       for (int i = 0; i < count; ++i)
#         p[i] -= scalar;
#     }
# 
#     template<typename T1, typename T2> inline
#     typename boost::disable_if_c<boost::is_array<T1>::value>::type
#     mulscalar (T1 &p, const T2 &scalar)
#     {
#       p *= scalar;
#     }
# 
#     template<typename T1, typename T2> inline
#     typename boost::enable_if_c<boost::is_array<T1>::value>::type
#     mulscalar (T1 &p, const T2 &scalar)
#     {
#       typedef typename boost::remove_all_extents<T1>::type type;
#       static const uint32_t count = sizeof (T1) / sizeof (type);
#       for (int i = 0; i < count; ++i)
#         p[i] *= scalar;
#     }
# 
#     template<typename T1, typename T2> inline
#     typename boost::disable_if_c<boost::is_array<T1>::value>::type
#     divscalar (T1 &p, const T2 &scalar)
#     {
#       p /= scalar;
#     }
# 
#     template<typename T1, typename T2> inline
#     typename boost::enable_if_c<boost::is_array<T1>::value>::type
#     divscalar (T1 &p, const T2 &scalar)
#     {
#       typedef typename boost::remove_all_extents<T1>::type type;
#       static const uint32_t count = sizeof (T1) / sizeof (type);
#       for (int i = 0; i < count; ++i)
#         p[i] /= scalar;
#     }
#   }
# }
# 
# // Point operators
# #define PCL_PLUSEQ_POINT_TAG(r, data, elem)                \
#   pcl::traits::plus (lhs.BOOST_PP_TUPLE_ELEM(3, 1, elem),  \
#                      rhs.BOOST_PP_TUPLE_ELEM(3, 1, elem)); \
#   /***/
# 
# #define PCL_PLUSEQSC_POINT_TAG(r, data, elem)                 \
#   pcl::traits::plusscalar (p.BOOST_PP_TUPLE_ELEM(3, 1, elem), \
#                            scalar);                           \
#   /***/
#    //p.BOOST_PP_TUPLE_ELEM(3, 1, elem) += scalar;  \
# 
# #define PCL_MINUSEQ_POINT_TAG(r, data, elem)                \
#   pcl::traits::minus (lhs.BOOST_PP_TUPLE_ELEM(3, 1, elem),  \
#                       rhs.BOOST_PP_TUPLE_ELEM(3, 1, elem)); \
#   /***/
# 
# #define PCL_MINUSEQSC_POINT_TAG(r, data, elem)                 \
#   pcl::traits::minusscalar (p.BOOST_PP_TUPLE_ELEM(3, 1, elem), \
#                             scalar);                           \
#   /***/
#    //p.BOOST_PP_TUPLE_ELEM(3, 1, elem) -= scalar;   \
# 
# #define PCL_MULEQSC_POINT_TAG(r, data, elem)                 \
#   pcl::traits::mulscalar (p.BOOST_PP_TUPLE_ELEM(3, 1, elem), \
#                             scalar);                         \
#   /***/
# 
# #define PCL_DIVEQSC_POINT_TAG(r, data, elem)   \
#   pcl::traits::divscalar (p.BOOST_PP_TUPLE_ELEM(3, 1, elem), \
#                             scalar);                         \
#   /***/
# 
# // Construct type traits given full sequence of (type, name, tag) triples
# //  BOOST_MPL_ASSERT_MSG(boost::is_pod<name>::value,
# //                       REGISTERED_POINT_TYPE_MUST_BE_PLAIN_OLD_DATA, (name));
# #define POINT_CLOUD_REGISTER_POINT_STRUCT_I(name, seq)                           \
#   namespace pcl                                                                  \
#   {                                                                              \
#     namespace fields                                                             \
#     {                                                                            \
#       BOOST_PP_SEQ_FOR_EACH(POINT_CLOUD_REGISTER_FIELD_TAG, name, seq)           \
#     }                                                                            \
#     namespace traits                                                             \
#     {                                                                            \
#       BOOST_PP_SEQ_FOR_EACH(POINT_CLOUD_REGISTER_FIELD_NAME, name, seq)          \
#       BOOST_PP_SEQ_FOR_EACH(POINT_CLOUD_REGISTER_FIELD_OFFSET, name, seq)        \
#       BOOST_PP_SEQ_FOR_EACH(POINT_CLOUD_REGISTER_FIELD_DATATYPE, name, seq)      \
#       POINT_CLOUD_REGISTER_POINT_FIELD_LIST(name, POINT_CLOUD_EXTRACT_TAGS(seq)) \
#     }                                                                            \
#     namespace common                                           \
#     {                                                          \
#       inline const name&                                       \
#       operator+= (name& lhs, const name& rhs)                  \
#       {                                                        \
#         BOOST_PP_SEQ_FOR_EACH(PCL_PLUSEQ_POINT_TAG, _, seq)    \
#         return (lhs);                                          \
#       }                                                        \
#       inline const name&                                       \
#       operator+= (name& p, const float& scalar)                \
#       {                                                        \
#         BOOST_PP_SEQ_FOR_EACH(PCL_PLUSEQSC_POINT_TAG, _, seq)  \
#         return (p);                                            \
#       }                                                        \
#       inline const name operator+ (const name& lhs, const name& rhs)   \
#       { name result = lhs; result += rhs; return (result); }           \
#       inline const name operator+ (const float& scalar, const name& p) \
#       { name result = p; result += scalar; return (result); }          \
#       inline const name operator+ (const name& p, const float& scalar) \
#       { name result = p; result += scalar; return (result); }          \
#       inline const name&                                       \
#       operator-= (name& lhs, const name& rhs)                  \
#       {                                                        \
#         BOOST_PP_SEQ_FOR_EACH(PCL_MINUSEQ_POINT_TAG, _, seq)   \
#         return (lhs);                                          \
#       }                                                        \
#       inline const name&                                       \
#       operator-= (name& p, const float& scalar)                \
#       {                                                        \
#         BOOST_PP_SEQ_FOR_EACH(PCL_MINUSEQSC_POINT_TAG, _, seq) \
#         return (p);                                            \
#       }                                                        \
#       inline const name operator- (const name& lhs, const name& rhs)   \
#       { name result = lhs; result -= rhs; return (result); }           \
#       inline const name operator- (const float& scalar, const name& p) \
#       { name result = p; result -= scalar; return (result); }          \
#       inline const name operator- (const name& p, const float& scalar) \
#       { name result = p; result -= scalar; return (result); }          \
#       inline const name&                                       \
#       operator*= (name& p, const float& scalar)                \
#       {                                                        \
#         BOOST_PP_SEQ_FOR_EACH(PCL_MULEQSC_POINT_TAG, _, seq)   \
#         return (p);                                            \
#       }                                                        \
#       inline const name operator* (const float& scalar, const name& p) \
#       { name result = p; result *= scalar; return (result); }          \
#       inline const name operator* (const name& p, const float& scalar) \
#       { name result = p; result *= scalar; return (result); }          \
#       inline const name&                                       \
#       operator/= (name& p, const float& scalar)                \
#       {                                                        \
#         BOOST_PP_SEQ_FOR_EACH(PCL_DIVEQSC_POINT_TAG, _, seq)   \
#         return (p);                                            \
#       }                                                        \
#       inline const name operator/ (const float& scalar, const name& p) \
#       { name result = p; result /= scalar; return (result); }          \
#       inline const name operator/ (const name& p, const float& scalar) \
#       { name result = p; result /= scalar; return (result); }          \
#     }                                                          \
#   }                                                            \
#   /***/
# 
# #define POINT_CLOUD_REGISTER_FIELD_TAG(r, name, elem)   \
#   struct BOOST_PP_TUPLE_ELEM(3, 2, elem);               \
#   /***/
# 
# #define POINT_CLOUD_REGISTER_FIELD_NAME(r, point, elem)                 \
#   template<int dummy>                                                   \
#   struct name<point, pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem), dummy> \
#   {                                                                     \
#     static const char value[];                                          \
#   };                                                                    \
#                                                                         \
#   template<int dummy>                                                   \
#   const char name<point,                                                \
#                   pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem),         \
#                   dummy>::value[] =                                     \
#     BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(3, 2, elem));                \
#   /***/
# 
# #define POINT_CLOUD_REGISTER_FIELD_OFFSET(r, name, elem)                \
#   template<> struct offset<name, pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem)> \
#   {                                                                     \
#     static const size_t value = offsetof(name, BOOST_PP_TUPLE_ELEM(3, 1, elem)); \
#   };                                                                    \
#   /***/
# 
# // \note: the mpl::identity weirdness is to support array types without requiring the
# // user to wrap them. The basic problem is:
# // typedef float[81] type; // SYNTAX ERROR!
# // typedef float type[81]; // OK, can now use "type" as a synonym for float[81]
# #define POINT_CLOUD_REGISTER_FIELD_DATATYPE(r, name, elem)              \
#   template<> struct datatype<name, pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem)> \
#   {                                                                     \
#     typedef boost::mpl::identity<BOOST_PP_TUPLE_ELEM(3, 0, elem)>::type type; \
#     typedef decomposeArray<type> decomposed;                            \
#     static const uint8_t value = asEnum<decomposed::type>::value;       \
#     static const uint32_t size = decomposed::value;                     \
#   };                                                                    \
#   /***/
# 
# #define POINT_CLOUD_TAG_OP(s, data, elem) pcl::fields::BOOST_PP_TUPLE_ELEM(3, 2, elem)
# 
# #define POINT_CLOUD_EXTRACT_TAGS(seq) BOOST_PP_SEQ_TRANSFORM(POINT_CLOUD_TAG_OP, _, seq)
# 
# #define POINT_CLOUD_REGISTER_POINT_FIELD_LIST(name, seq)        \
#   template<> struct fieldList<name>                             \
#   {                                                             \
#     typedef boost::mpl::vector<BOOST_PP_SEQ_ENUM(seq)> type;    \
#   };                                                            \
#   /***/
# 
# #if defined _MSC_VER
#   #pragma warning (pop)
# #endif
# 
# #endif  //#ifndef PCL_REGISTER_POINT_STRUCT_H_
# ###
# 
# # spring.h
# 
# namespace pcl
# {
#   namespace common
#   {
#     /** expand point cloud inserting \a amount rows at the 
#      * top and the bottom of a point cloud and filling them with 
#      * custom values.
#      * \param[in] input the input point cloud
#      * \param[out] output the output point cloud
#      * \param[in] val the point value to be insterted
#      * \param[in] amount the amount of rows to be added
#      */
#     template <typename PointT> void
#     expandRows (const PointCloud<PointT>& input, PointCloud<PointT>& output, 
#                 const PointT& val, const size_t& amount);
# 
#     /** expand point cloud inserting \a amount columns at 
#       * the right and the left of a point cloud and filling them with 
#       * custom values.
#       * \param[in] input the input point cloud
#       * \param[out] output the output point cloud
#       * \param[in] val the point value to be insterted
#       * \param[in] amount the amount of columns to be added
#       */
#     template <typename PointT> void
#     expandColumns (const PointCloud<PointT>& input, PointCloud<PointT>& output, 
#                    const PointT& val, const size_t& amount);
# 
#     /** expand point cloud duplicating the \a amount top and bottom rows times.
#       * \param[in] input the input point cloud
#       * \param[out] output the output point cloud
#       * \param[in] amount the amount of rows to be added
#       */
#     template <typename PointT> void
#     duplicateRows (const PointCloud<PointT>& input, PointCloud<PointT>& output, 
#                    const size_t& amount);
# 
#     /** expand point cloud duplicating the \a amount right and left columns
#       * times.
#       * \param[in] input the input point cloud
#       * \param[out] output the output point cloud
#       * \param[in] amount the amount of cilumns to be added
#       */
#     template <typename PointT> void
#     duplicateColumns (const PointCloud<PointT>& input, PointCloud<PointT>& output, 
#                       const size_t& amount);
# 
#     /** expand point cloud mirroring \a amount top and bottom rows. 
#       * \param[in] input the input point cloud
#       * \param[out] output the output point cloud
#       * \param[in] amount the amount of rows to be added
#       */
#     template <typename PointT> void
#     mirrorRows (const PointCloud<PointT>& input, PointCloud<PointT>& output, 
#                 const size_t& amount);
# 
#     /** expand point cloud mirroring \a amount right and left columns.
#       * \param[in] input the input point cloud
#       * \param[out] output the output point cloud
#       * \param[in] amount the amount of rows to be added
#       */
#     template <typename PointT> void
#     mirrorColumns (const PointCloud<PointT>& input, PointCloud<PointT>& output, 
#                    const size_t& amount);
# 
#     /** delete \a amount rows in top and bottom of point cloud 
#       * \param[in] input the input point cloud
#       * \param[out] output the output point cloud
#       * \param[in] amount the amount of rows to be added
#       */
#     template <typename PointT> void
#     deleteRows (const PointCloud<PointT>& input, PointCloud<PointT>& output, 
#                 const size_t& amount);
# 
#     /** delete \a amount columns in top and bottom of point cloud
#       * \param[in] input the input point cloud
#       * \param[out] output the output point cloud
#       * \param[in] amount the amount of rows to be added
#       */
#     template <typename PointT> void
#     deleteCols (const PointCloud<PointT>& input, PointCloud<PointT>& output, 
#                 const size_t& amount);
#   };
# }
# 
# #include <pcl/common/impl/spring.hpp>
# 
# #endif
# ###
# 
# 
# # synchronizer.h
# namespace pcl
# {
#   /** /brief This template class synchronizes two data streams of different types.
#    *         The data can be added using add0 and add1 methods which expects also a timestamp of type unsigned long.
#    *         If two matching data objects are found, registered callback functions are invoked with the objects and the time stamps.
#    *         The only assumption of the timestamp is, that they are in the same unit, linear and strictly monotonic increasing.
#    *         If filtering is desired, e.g. thresholding of time differences, the user can do that in the callback method.
#    *         This class is thread safe.
#    * /ingroup common
#    */
#   template <typename T1, typename T2>
#   class Synchronizer
#   {
#     typedef std::pair<unsigned long, T1> T1Stamped;
#     typedef std::pair<unsigned long, T2> T2Stamped;
#     boost::mutex mutex1_;
#     boost::mutex mutex2_;
#     boost::mutex publish_mutex_;
#     std::deque<T1Stamped> queueT1;
#     std::deque<T2Stamped> queueT2;
# 
#     typedef boost::function<void(T1, T2, unsigned long, unsigned long) > CallbackFunction;
# 
#     std::map<int, CallbackFunction> cb_;
#     int callback_counter;
#   public:
# 
#     Synchronizer () : mutex1_ (), mutex2_ (), publish_mutex_ (), queueT1 (), queueT2 (), cb_ (), callback_counter (0) { };
# 
#     int
#     addCallback (const CallbackFunction& callback)
#     {
#       boost::unique_lock<boost::mutex> publish_lock (publish_mutex_);
#       cb_[callback_counter] = callback;
#       return callback_counter++;
#     }
# 
#     void
#     removeCallback (int i)
#     {
#       boost::unique_lock<boost::mutex> publish_lock (publish_mutex_);
#       cb_.erase (i);
#     }
# 
#     void
#     add0 (const T1& t, unsigned long time)
#     {
#       mutex1_.lock ();
#       queueT1.push_back (T1Stamped (time, t));
#       mutex1_.unlock ();
#       publish ();
#     }
# 
#     void
#     add1 (const T2& t, unsigned long time)
#     {
#       mutex2_.lock ();
#       queueT2.push_back (T2Stamped (time, t));
#       mutex2_.unlock ();
#       publish ();
#     }
# 
#   private:
# 
#     void
#     publishData ()
#     {
#       boost::unique_lock<boost::mutex> lock1 (mutex1_);
#       boost::unique_lock<boost::mutex> lock2 (mutex2_);
# 
#       for (typename std::map<int, CallbackFunction>::iterator cb = cb_.begin (); cb != cb_.end (); ++cb)
#       {
#         if (!cb->second.empty ())
#         {
#           cb->second.operator()(queueT1.front ().second, queueT2.front ().second, queueT1.front ().first, queueT2.front ().first);
#         }
#       }
# 
#       queueT1.pop_front ();
#       queueT2.pop_front ();
#     }
# 
#     void
#     publish ()
#     {
#       // only one publish call at once allowed
#       boost::unique_lock<boost::mutex> publish_lock (publish_mutex_);
# 
#       boost::unique_lock<boost::mutex> lock1 (mutex1_);
#       if (queueT1.empty ())
#         return;
#       T1Stamped t1 = queueT1.front ();
#       lock1.unlock ();
# 
#       boost::unique_lock<boost::mutex> lock2 (mutex2_);
#       if (queueT2.empty ())
#         return;
#       T2Stamped t2 = queueT2.front ();
#       lock2.unlock ();
# 
#       bool do_publish = false;
# 
#       if (t1.first <= t2.first)
#       { // iterate over queue1
#         lock1.lock ();
#         while (queueT1.size () > 1 && queueT1[1].first <= t2.first)
#           queueT1.pop_front ();
# 
#         if (queueT1.size () > 1)
#         { // we have at least 2 measurements; first in past and second in future -> find out closer one!
#           if ( (t2.first << 1) > (queueT1[0].first + queueT1[1].first) )
#             queueT1.pop_front ();
# 
#           do_publish = true;
#         }
#         lock1.unlock ();
#       }
#       else
#       { // iterate over queue2
#         lock2.lock ();
#         while (queueT2.size () > 1 && (queueT2[1].first <= t1.first) )
#           queueT2.pop_front ();
# 
#         if (queueT2.size () > 1)
#         { // we have at least 2 measurements; first in past and second in future -> find out closer one!
#           if ( (t1.first << 1) > queueT2[0].first + queueT2[1].first )
#             queueT2.pop_front ();
# 
#           do_publish = true;
#         }
#         lock2.unlock ();
#       }
# 
#       if (do_publish)
#         publishData ();
#     }
#   } ;
# } // namespace
# 
# #endif // __PCL_SYNCHRONIZER__
# ###
# 
# 
# # time.h
# namespace pcl
# {
#   /** \brief Simple stopwatch.
#     * \ingroup common
#     */
#   class StopWatch
#   {
#     public:
#       /** \brief Constructor. */
#       StopWatch () : start_time_ (boost::posix_time::microsec_clock::local_time ())
#       {
#       }
# 
#       /** \brief Destructor. */
#       virtual ~StopWatch () {}
# 
#       /** \brief Retrieve the time in milliseconds spent since the last call to \a reset(). */
#       inline double
#       getTime ()
#       {
#         boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time ();
#         return (static_cast<double> (((end_time - start_time_).total_milliseconds ())));
#       }
# 
#       /** \brief Retrieve the time in seconds spent since the last call to \a reset(). */
#       inline double
#       getTimeSeconds ()
#       {
#         return (getTime () * 0.001f);
#       }
# 
#       /** \brief Reset the stopwatch to 0. */
#       inline void
#       reset ()
#       {
#         start_time_ = boost::posix_time::microsec_clock::local_time ();
#       }
# 
#     protected:
#       boost::posix_time::ptime start_time_;
#   };
# 
#   /** \brief Class to measure the time spent in a scope
#     *
#     * To use this class, e.g. to measure the time spent in a function,
#     * just create an instance at the beginning of the function. Example:
#     *
#     * \code
#     * {
#     *   pcl::ScopeTime t1 ("calculation");
#     *
#     *   // ... perform calculation here
#     * }
#     * \endcode
#     *
#     * \ingroup common
#     */
#   class ScopeTime : public StopWatch
#   {
#     public:
#       inline ScopeTime (const char* title) : 
#         title_ (std::string (title))
#       {
#         start_time_ = boost::posix_time::microsec_clock::local_time ();
#       }
# 
#       inline ScopeTime () :
#         title_ (std::string (""))
#       {
#         start_time_ = boost::posix_time::microsec_clock::local_time ();
#       }
# 
#       inline ~ScopeTime ()
#       {
#         double val = this->getTime ();
#         std::cerr << title_ << " took " << val << "ms.\n";
#       }
# 
#     private:
#       std::string title_;
#   };
# 
# 
# #ifndef MEASURE_FUNCTION_TIME
# #define MEASURE_FUNCTION_TIME \
#   ScopeTime scopeTime(__func__)
# #endif
# 
# inline double 
# getTime ()
# {
#   boost::posix_time::ptime epoch_time (boost::gregorian::date (1970, 1, 1));
#   boost::posix_time::ptime current_time = boost::posix_time::microsec_clock::local_time ();
#   return (static_cast<double>((current_time - epoch_time).total_nanoseconds ()) * 1.0e-9);
# }
# 
# /// Executes code, only if secs are gone since last exec.
# #ifndef DO_EVERY_TS
# #define DO_EVERY_TS(secs, currentTime, code) \
# if (1) {\
#   static double s_lastDone_ = 0.0; \
#   double s_now_ = (currentTime); \
#   if (s_lastDone_ > s_now_) \
#     s_lastDone_ = s_now_; \
#   if ((s_now_ - s_lastDone_) > (secs)) {        \
#     code; \
#     s_lastDone_ = s_now_; \
#   }\
# } else \
#   (void)0
# #endif
# 
# /// Executes code, only if secs are gone since last exec.
# #ifndef DO_EVERY
# #define DO_EVERY(secs, code) \
#   DO_EVERY_TS(secs, pcl::getTime(), code)
# #endif
# 
# }  // end namespace
# /*@}*/
# 
# #endif  //#ifndef PCL_NORMS_H_
# ###
# 
# # time_trigger.h
# namespace pcl
# {
#   /** \brief Timer class that invokes registered callback methods periodically.
#     * \ingroup common
#     */
#   class PCL_EXPORTS TimeTrigger
#   {
#     public:
#       typedef boost::function<void() > callback_type;
# 
#       /** \brief Timer class that calls a callback method periodically. Due to possible blocking calls, only one callback method can be registered per instance.
#         * \param[in] interval_seconds interval in seconds
#         * \param[in] callback callback to be invoked periodically
#         */
#       TimeTrigger (double interval_seconds, const callback_type& callback);
# 
#       /** \brief Timer class that calls a callback method periodically. Due to possible blocking calls, only one callback method can be registered per instance.
#         * \param[in] interval_seconds interval in seconds
#         */
#       TimeTrigger (double interval_seconds = 1.0);
# 
#       /** \brief Destructor. */
#       ~TimeTrigger ();
# 
#       /** \brief registeres a callback
#         * \param[in] callback callback function to the list of callbacks. signature has to be boost::function<void()>
#         * \return connection the connection, which can be used to disable/enable and remove callback from list
#         */
#       boost::signals2::connection registerCallback (const callback_type& callback);
# 
#       /** \brief Resets the timer interval
#         * \param[in] interval_seconds interval in seconds
#         */
#       void 
#       setInterval (double interval_seconds);
# 
#       /** \brief Start the Trigger. */
#       void 
#       start ();
# 
#       /** \brief Stop the Trigger. */
#       void 
#       stop ();
#     private:
#       void 
#       thread_function ();
#       boost::signals2::signal <void() > callbacks_;
# 
#       double interval_;
# 
#       bool quit_;
#       bool running_;
# 
#       boost::thread timer_thread_;
#       boost::condition_variable condition_;
#       boost::mutex condition_mutex_;
#   };
# }
# 
# #endif
# ###
# 
# # transformation_from_correspondences.h
# namespace pcl 
# {
#   /**
#     * \brief Calculates a transformation based on corresponding 3D points
#     * \author Bastian Steder
#     * \ingroup common
#     */
#   class TransformationFromCorrespondences 
#   {
#      public:
#         //-----CONSTRUCTOR&DESTRUCTOR-----
#         /** Constructor - dimension gives the size of the vectors to work with. */
#         TransformationFromCorrespondences () : 
#           no_of_samples_ (0), accumulated_weight_ (0), 
#           mean1_ (Eigen::Vector3f::Identity ()),
#           mean2_ (Eigen::Vector3f::Identity ()),
#           covariance_ (Eigen::Matrix<float, 3, 3>::Identity ())
#         { reset (); }
# 
#         /** Destructor */
#         ~TransformationFromCorrespondences () { };
#         
#         //-----METHODS-----
#         /** Reset the object to work with a new data set */
#         inline void 
#         reset ();
#         
#         /** Get the summed up weight of all added vectors */
#         inline float 
#         getAccumulatedWeight () const { return accumulated_weight_;}
#         
#         /** Get the number of added vectors */
#         inline unsigned int 
#         getNoOfSamples () { return no_of_samples_;}
#         
#         /** Add a new sample */
#         inline void 
#         add (const Eigen::Vector3f& point, const Eigen::Vector3f& corresponding_point, float weight=1.0);
#         
#         /** Calculate the transformation that will best transform the points into their correspondences */
#         inline Eigen::Affine3f 
#         getTransformation ();
#         
#         //-----VARIABLES-----
#         
#      protected:
#         //-----METHODS-----
#         //-----VARIABLES-----
#         unsigned int no_of_samples_;
#         float accumulated_weight_;
#         Eigen::Vector3f mean1_, mean2_;
#         Eigen::Matrix<float, 3, 3> covariance_;
#   };
# 
# }  // END namespace
# 
# #include <pcl/common/impl/transformation_from_correspondences.hpp>
# 
# #endif  // #ifndef PCL_TRANSFORMATION_FROM_CORRESPONDENCES_H
# ###
# 
# 
# # transforms.h
# namespace pcl
# {
#   /** \brief Apply an affine transform defined by an Eigen Transform
#     * \param[in] cloud_in the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform an affine transformation (typically a rigid transformation)
#     * \note Can be used with cloud_in equal to cloud_out
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Transform<Scalar, 3, Eigen::Affine> &transform);
# 
#   template <typename PointT> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Affine3f &transform)
#   {
#     return (transformPointCloud<PointT, float> (cloud_in, cloud_out, transform));
#   }
# 
#   /** \brief Apply an affine transform defined by an Eigen Transform
#     * \param[in] cloud_in the input point cloud
#     * \param[in] indices the set of point indices to use from the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform an affine transformation (typically a rigid transformation)
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        const std::vector<int> &indices, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Transform<Scalar, 3, Eigen::Affine> &transform);
# 
#   template <typename PointT> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        const std::vector<int> &indices, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Affine3f &transform)
#   {
#     return (transformPointCloud<PointT, float> (cloud_in, indices, cloud_out, transform));
#   }
# 
#   /** \brief Apply an affine transform defined by an Eigen Transform
#     * \param[in] cloud_in the input point cloud
#     * \param[in] indices the set of point indices to use from the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform an affine transformation (typically a rigid transformation)
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        const pcl::PointIndices &indices, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Transform<Scalar, 3, Eigen::Affine> &transform)
#   {
#     return (transformPointCloud<PointT, Scalar> (cloud_in, indices.indices, cloud_out, transform));
#   }
# 
#   template <typename PointT> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        const pcl::PointIndices &indices, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Affine3f &transform)
#   {
#     return (transformPointCloud<PointT, float> (cloud_in, indices, cloud_out, transform));
#   }
# 
#   /** \brief Transform a point cloud and rotate its normals using an Eigen transform.
#     * \param[in] cloud_in the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform an affine transformation (typically a rigid transformation)
#     * \note Can be used with cloud_in equal to cloud_out
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Transform<Scalar, 3, Eigen::Affine> &transform);
# 
#   template <typename PointT> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Affine3f &transform)
#   {
#     return (transformPointCloudWithNormals<PointT, float> (cloud_in, cloud_out, transform));
#   }
# 
#   /** \brief Transform a point cloud and rotate its normals using an Eigen transform.
#     * \param[in] cloud_in the input point cloud
#     * \param[in] indices the set of point indices to use from the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform an affine transformation (typically a rigid transformation)
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   const std::vector<int> &indices, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Transform<Scalar, 3, Eigen::Affine> &transform);
# 
#   template <typename PointT> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   const std::vector<int> &indices, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Affine3f &transform)
#   {
#     return (transformPointCloudWithNormals<PointT, float> (cloud_in, indices, cloud_out, transform));
#   }
# 
#   /** \brief Transform a point cloud and rotate its normals using an Eigen transform.
#     * \param[in] cloud_in the input point cloud
#     * \param[in] indices the set of point indices to use from the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform an affine transformation (typically a rigid transformation)
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   const pcl::PointIndices &indices, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Transform<Scalar, 3, Eigen::Affine> &transform)
#   {
#     return (transformPointCloudWithNormals<PointT, Scalar> (cloud_in, indices.indices, cloud_out, transform));
#   }
# 
# 
#   template <typename PointT> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   const pcl::PointIndices &indices, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Affine3f &transform)
#   {
#     return (transformPointCloudWithNormals<PointT, float> (cloud_in, indices, cloud_out, transform));
#   }
# 
#   /** \brief Apply a rigid transform defined by a 4x4 matrix
#     * \param[in] cloud_in the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform a rigid transformation 
#     * \note Can be used with cloud_in equal to cloud_out
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Matrix<Scalar, 4, 4> &transform)
#   {
#     Eigen::Transform<Scalar, 3, Eigen::Affine> t (transform);
#     return (transformPointCloud<PointT, Scalar> (cloud_in, cloud_out, t));
#   }
# 
#   template <typename PointT> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Matrix4f &transform)
#   {
#     return (transformPointCloud<PointT, float> (cloud_in, cloud_out, transform));
#   }
# 
#   /** \brief Apply a rigid transform defined by a 4x4 matrix
#     * \param[in] cloud_in the input point cloud
#     * \param[in] indices the set of point indices to use from the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform a rigid transformation 
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        const std::vector<int> &indices, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Matrix<Scalar, 4, 4> &transform)
#   {
#     Eigen::Transform<Scalar, 3, Eigen::Affine> t (transform);
#     return (transformPointCloud<PointT, Scalar> (cloud_in, indices, cloud_out, t));
#   }
# 
#   template <typename PointT> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        const std::vector<int> &indices, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Matrix4f &transform)
#   {
#     return (transformPointCloud<PointT, float> (cloud_in, indices, cloud_out, transform));
#   }
# 
#   /** \brief Apply a rigid transform defined by a 4x4 matrix
#     * \param[in] cloud_in the input point cloud
#     * \param[in] indices the set of point indices to use from the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform a rigid transformation 
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        const pcl::PointIndices &indices, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Matrix<Scalar, 4, 4> &transform)
#   {
#     return (transformPointCloud<PointT, Scalar> (cloud_in, indices.indices, cloud_out, transform));
#   }
# 
#   template <typename PointT> void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        const pcl::PointIndices &indices, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Matrix4f &transform)
#   {
#     return (transformPointCloud<PointT, float> (cloud_in, indices, cloud_out, transform));
#   }
# 
#   /** \brief Transform a point cloud and rotate its normals using an Eigen transform.
#     * \param[in] cloud_in the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform an affine transformation (typically a rigid transformation)
#     * \note Can be used with cloud_in equal to cloud_out
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Matrix<Scalar, 4, 4> &transform)
#   {
#     Eigen::Transform<Scalar, 3, Eigen::Affine> t (transform);
#     return (transformPointCloudWithNormals<PointT, Scalar> (cloud_in, cloud_out, t));
#   }
# 
# 
#   template <typename PointT> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Matrix4f &transform)
#   {
#     return (transformPointCloudWithNormals<PointT, float> (cloud_in, cloud_out, transform));
#   }
# 
#   /** \brief Transform a point cloud and rotate its normals using an Eigen transform.
#     * \param[in] cloud_in the input point cloud
#     * \param[in] indices the set of point indices to use from the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform an affine transformation (typically a rigid transformation)
#     * \note Can be used with cloud_in equal to cloud_out
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   const std::vector<int> &indices, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Matrix<Scalar, 4, 4> &transform)
#   {
#     Eigen::Transform<Scalar, 3, Eigen::Affine> t (transform);
#     return (transformPointCloudWithNormals<PointT, Scalar> (cloud_in, indices, cloud_out, t));
#   }
# 
# 
#   template <typename PointT> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   const std::vector<int> &indices, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Matrix4f &transform)
#   {
#     return (transformPointCloudWithNormals<PointT, float> (cloud_in, indices, cloud_out, transform));
#   }
# 
#   /** \brief Transform a point cloud and rotate its normals using an Eigen transform.
#     * \param[in] cloud_in the input point cloud
#     * \param[in] indices the set of point indices to use from the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] transform an affine transformation (typically a rigid transformation)
#     * \note Can be used with cloud_in equal to cloud_out
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   const pcl::PointIndices &indices, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Matrix<Scalar, 4, 4> &transform)
#   {
#     Eigen::Transform<Scalar, 3, Eigen::Affine> t (transform);
#     return (transformPointCloudWithNormals<PointT, Scalar> (cloud_in, indices, cloud_out, t));
#   }
# 
# 
#   template <typename PointT> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   const pcl::PointIndices &indices, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Matrix4f &transform)
#   {
#     return (transformPointCloudWithNormals<PointT, float> (cloud_in, indices, cloud_out, transform));
#   }
# 
#   /** \brief Apply a rigid transform defined by a 3D offset and a quaternion
#     * \param[in] cloud_in the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] offset the translation component of the rigid transformation
#     * \param[in] rotation the rotation component of the rigid transformation
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> inline void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Matrix<Scalar, 3, 1> &offset, 
#                        const Eigen::Quaternion<Scalar> &rotation);
# 
#   template <typename PointT> inline void 
#   transformPointCloud (const pcl::PointCloud<PointT> &cloud_in, 
#                        pcl::PointCloud<PointT> &cloud_out, 
#                        const Eigen::Vector3f &offset, 
#                        const Eigen::Quaternionf &rotation)
#   {
#     return (transformPointCloud<PointT, float> (cloud_in, cloud_out, offset, rotation));
#   }
# 
#   /** \brief Transform a point cloud and rotate its normals using an Eigen transform.
#     * \param[in] cloud_in the input point cloud
#     * \param[out] cloud_out the resultant output point cloud
#     * \param[in] offset the translation component of the rigid transformation
#     * \param[in] rotation the rotation component of the rigid transformation
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> inline void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Matrix<Scalar, 3, 1> &offset, 
#                                   const Eigen::Quaternion<Scalar> &rotation);
# 
#   template <typename PointT> void 
#   transformPointCloudWithNormals (const pcl::PointCloud<PointT> &cloud_in, 
#                                   pcl::PointCloud<PointT> &cloud_out, 
#                                   const Eigen::Vector3f &offset, 
#                                   const Eigen::Quaternionf &rotation)
#   {
#     return (transformPointCloudWithNormals<PointT, float> (cloud_in, cloud_out, offset, rotation));
#   }
# 
#   /** \brief Transform a point with members x,y,z
#     * \param[in] point the point to transform
#     * \param[out] transform the transformation to apply
#     * \return the transformed point
#     * \ingroup common
#     */
#   template <typename PointT, typename Scalar> inline PointT
#   transformPoint (const PointT &point, 
#                   const Eigen::Transform<Scalar, 3, Eigen::Affine> &transform);
#   
#   template <typename PointT> inline PointT
#   transformPoint (const PointT &point, 
#                   const Eigen::Affine3f &transform)
#   {
#     return (transformPoint<PointT, float> (point, transform));
#   }
# 
#   /** \brief Calculates the principal (PCA-based) alignment of the point cloud
#     * \param[in] cloud the input point cloud
#     * \param[out] transform the resultant transform
#     * \return the ratio lambda1/lambda2 or lambda2/lambda3, whatever is closer to 1.
#     * \note If the return value is close to one then the transformation might be not unique -> two principal directions have
#     * almost same variance (extend)
#     */
#   template <typename PointT, typename Scalar> inline double
#   getPrincipalTransformation (const pcl::PointCloud<PointT> &cloud, 
#                               Eigen::Transform<Scalar, 3, Eigen::Affine> &transform);
# 
#   template <typename PointT> inline double
#   getPrincipalTransformation (const pcl::PointCloud<PointT> &cloud, 
#                               Eigen::Affine3f &transform)
#   {
#     return (getPrincipalTransformation<PointT, float> (cloud, transform));
#   }
# }
# 
# #include <pcl/common/impl/transforms.hpp>
# 
# #endif // PCL_TRANSFORMS_H_
# ###
# 
# 
# # utils.h
# namespace pcl
# {
#   namespace utils
#   {
#     /** \brief Check if val1 and val2 are equals to an epsilon extent
#       * \param[in] val1 first number to check
#       * \param[in] val2 second number to check
#       * \param[in] eps epsilon
#       * \return true if val1 is equal to val2, false otherwise.
#       */
#     template<typename T> bool 
#     equal (T val1, T val2, T eps = std::numeric_limits<T>::min ())
#     {
#       return (fabs (val1 - val2) < eps);
#     }
#   }
# }
# 
# #endif
# ###
# 
# 
# # vector_average.h
# namespace pcl 
# {
#   /** \brief Calculates the weighted average and the covariance matrix
#     *
#     * A class to calculate the weighted average and the covariance matrix of a set of vectors with given weights.
#     * The original data is not saved. Mean and covariance are calculated iteratively.
#     * \author Bastian Steder
#     * \ingroup common
#     */
#   template <typename real, int dimension>
#   class VectorAverage 
#   {
#      public:
#         //-----CONSTRUCTOR&DESTRUCTOR-----
#         /** Constructor - dimension gives the size of the vectors to work with. */
#         VectorAverage ();
#         /** Destructor */
#         ~VectorAverage () {}
#         
#         //-----METHODS-----
#         /** Reset the object to work with a new data set */
#         inline void 
#         reset ();
#         
#         /** Get the mean of the added vectors */
#         inline const
#         Eigen::Matrix<real, dimension, 1>& getMean () const { return mean_;}
#         
#         /** Get the covariance matrix of the added vectors */
#         inline const
#         Eigen::Matrix<real, dimension, dimension>& getCovariance () const { return covariance_;}
#         
#         /** Get the summed up weight of all added vectors */
#         inline real
#         getAccumulatedWeight () const { return accumulatedWeight_;}
#         
#         /** Get the number of added vectors */
#         inline unsigned int
#         getNoOfSamples () { return noOfSamples_;}
#         
#         /** Add a new sample */
#         inline void
#         add (const Eigen::Matrix<real, dimension, 1>& sample, real weight=1.0);
# 
#         /** Do Principal component analysis */
#         inline void
#         doPCA (Eigen::Matrix<real, dimension, 1>& eigen_values, Eigen::Matrix<real, dimension, 1>& eigen_vector1,
#                Eigen::Matrix<real, dimension, 1>& eigen_vector2, Eigen::Matrix<real, dimension, 1>& eigen_vector3) const;
#         
#         /** Do Principal component analysis */
#         inline void
#         doPCA (Eigen::Matrix<real, dimension, 1>& eigen_values) const;
#         
#         /** Get the eigenvector corresponding to the smallest eigenvalue */
#         inline void
#         getEigenVector1 (Eigen::Matrix<real, dimension, 1>& eigen_vector1) const;
#         
#         //-----VARIABLES-----
#         
#      protected:
#         //-----METHODS-----
#         //-----VARIABLES-----
#         unsigned int noOfSamples_;
#         real accumulatedWeight_;
#         Eigen::Matrix<real, dimension, 1> mean_;
#         Eigen::Matrix<real, dimension, dimension> covariance_;
#   };
# 
#   typedef VectorAverage<float, 2> VectorAverage2f;
#   typedef VectorAverage<float, 3> VectorAverage3f;
#   typedef VectorAverage<float, 4> VectorAverage4f;
# }  // END namespace
# 
# #include <pcl/common/impl/vector_average.hpp>
# 
# #endif  // #ifndef PCL_VECTOR_AVERAGE_H
###


