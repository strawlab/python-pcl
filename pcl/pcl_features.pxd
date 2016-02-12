from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from boost_shared_ptr cimport shared_ptr

# main
cimport pcl_defs as cpp

###############################################################################
# Types
###############################################################################

# class ShapeContext3DEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/3dsc.h" namespace "pcl":
    cdef cppclass ShapeContext3DEstimation[T, N]:
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

        # brief Set the number of bins along the azimuth to \a bins.
        # param[in] bins the number of bins along the azimuth
        void setAzimuthBins (size_t)

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
        void setMinimalRadius (double )

        # return The minimal sphere radius
        double getMinimalRadius ()

        # brief This radius is used to compute local point density 
        # density = number of points within this radius
        # param[in] radius value of the point density search radius
        void setPointDensityRadius (double )

        # return The point density search radius
        double getPointDensityRadius ()
        
        # protected:
        # brief Initialize computation by allocating all the intervals and the volume lookup table. */
        # bool initCompute ();

        # brief Estimate a descriptor for a given point.
        # param[in] index the index of the point to estimate a descriptor for
        # param[in] normals a pointer to the set of normals
        # param[in] rf the reference frame
        # param[out] desc the resultant estimated descriptor
        # return true if the descriptor was computed successfully, false if there was an error 
        # e.g. the nearest neighbor didn't return any neighbors)
        # bool computePoint (size_t index, const pcl::PointCloud<PointNT> &normals, float rf[9], std::vector<float> &desc);

        # brief Estimate the actual feature. 
        # param[out] output the resultant feature 
        # void computeFeature (PointCloudOut &output);

        # brief Values of the radii interval
        # vector<float> radii_interval_

        # brief Theta divisions interval
        # std::vector<float> theta_divisions_;

        # brief Phi divisions interval
        # std::vector<float> phi_divisions_;

        # brief Volumes look up table
        # vector<float> volume_lut_;

        # brief Bins along the azimuth dimension
        # size_t azimuth_bins_;

        # brief Bins along the elevation dimension
        # size_t elevation_bins_;

        # brief Bins along the radius dimension
        # size_t radius_bins_;

        # brief Minimal radius value
        # double min_radius_;

        # brief Point density radius
        # double point_density_radius_;

        # brief Descriptor length
        # size_t descriptor_length_;

        # brief Boost-based random number generator algorithm.
        # boost::mt19937 rng_alg_;

        # brief Boost-based random number generator distribution.
        # boost::shared_ptr<boost::uniform_01<boost::mt19937> > rng_;

        # brief Shift computed descriptor "L" times along the azimuthal direction
        # param[in] block_size the size of each azimuthal block
        # param[in] desc at input desc == original descriptor and on output it contains 
        # shifted descriptor resized descriptor_length_ * azimuth_bins_
        # void shiftAlongAzimuth (size_t block_size, std::vector<float>& desc);

        # brief Boost-based random number generator.
        # inline double rnd ()
    
        # private:
        # void computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &) {}

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
#         # private:
#         # void computeFeatureEigen (pcl::PointCloud<Eigen::MatrixXf> &output);
#         # void compute (pcl::PointCloud<pcl::SHOT> &) {}
###

# class BoundaryEstimation: public FeatureFromNormals<PointInT, PointNT, PointOutT>
cdef extern from "pcl/features/boundary.h" namespace "pcl":
    cdef cppclass BoundaryEstimation[I, N, O]:
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

        # brief Check whether a point is a boundary point in a planar patch of projected points given by indices.
        # note A coordinate system u-v-n must be computed a-priori using \a getCoordinateSystemOnPlane
        # param[in] cloud a pointer to the input point cloud
        # param[in] q_idx the index of the query point in \a cloud
        # param[in] indices the estimated point neighbors of the query point
        # param[in] u the u direction
        # param[in] v the v direction
        # param[in] angle_threshold the threshold angle (default \f$\pi / 2.0\f$)
        # bool isBoundaryPoint (const pcl::PointCloud<PointInT> &cloud, 
        #                int q_idx, const std::vector<int> &indices, 
        #                const Eigen::Vector4f &u, const Eigen::Vector4f &v, const float angle_threshold);

        # brief Check whether a point is a boundary point in a planar patch of projected points given by indices.
        # note A coordinate system u-v-n must be computed a-priori using \a getCoordinateSystemOnPlane
        # param[in] cloud a pointer to the input point cloud
        # param[in] q_point a pointer to the querry point
        # param[in] indices the estimated point neighbors of the query point
        # param[in] u the u direction
        # param[in] v the v direction
        # param[in] angle_threshold the threshold angle (default \f$\pi / 2.0\f$)
        # bool isBoundaryPoint (const pcl::PointCloud<PointInT> &cloud, 
        #                const PointInT &q_point, 
        #                const std::vector<int> &indices, 
        #                const Eigen::Vector4f &u, const Eigen::Vector4f &v, const float angle_threshold);

        # brief Set the decision boundary (angle threshold) that marks points as boundary or regular. 
        # (default \f$\pi / 2.0\f$) 
        # param[in] angle the angle threshold
        # inline void setAngleThreshold (float angle)

        # inline float getAngleThreshold ()

        # brief Get a u-v-n coordinate system that lies on a plane defined by its normal
        # param[in] p_coeff the plane coefficients (containing the plane normal)
        # param[out] u the resultant u direction
        # param[out] v the resultant v direction
        # inline void getCoordinateSystemOnPlane (const PointNT &p_coeff, 
        #                           Eigen::Vector4f &u, Eigen::Vector4f &v)

        # protected:
        # void computeFeature (PointCloudOut &output);
        # float angle_threshold_;

###

cdef extern from "pcl/features/normal_3d.h" namespace "pcl":
    cdef cppclass NormalEstimation[T, N]:
        NormalEstimation()

###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################
