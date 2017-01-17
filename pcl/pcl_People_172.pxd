from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp
cimport eigen as eigen3

# boost
from boost_shared_ptr cimport shared_ptr

### base class ###

# person_classifier.h
# namespace pcl
# namespace people
# template <typename PointT> class PersonClassifier;
# template <typename PointT>
# class PersonClassifier
cdef extern from "pcl/people/person_classifier.h" namespace "pcl::people":
    cdef cppclass PersonClassifier:
        PersonClassifier()
        # protected:
        # /** \brief Height of the image patch to classify. */
        # int window_height_;          
        # /** \brief Width of the image patch to classify. */
        # int window_width_;          
        # /** \brief SVM offset. */
        # float SVM_offset_;          
        # /** \brief SVM weights vector. */
        # std::vector<float> SVM_weights_;  
        # public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef boost::shared_ptr<PointCloud> PointCloudPtr;
        # /** \brief Load SVM parameters from a text file. 
        # * \param[in] svm_filename Filename containing SVM parameters.
        # * \return true if SVM has been correctly set, false otherwise.
        bool loadSVMFromFile (string svm_filename)
        # * \brief Set trained SVM for person confidence estimation.
        # * \param[in] window_height Detection window height.
        # * \param[in] window_width Detection window width.
        # * \param[in] SVM_weights SVM weights vector.
        # * \param[in] SVM_offset SVM offset.
        void setSVM (int window_height, int window_width, vector[float] SVM_weights, float SVM_offset)
        # * \brief Get trained SVM for person confidence estimation.
        # * \param[out] window_height Detection window height.
        # * \param[out] window_width Detection window width.
        # * \param[out] SVM_weights SVM weights vector.
        # * \param[out] SVM_offset SVM offset.
        # void getSVM (int& window_height, int& window_width, vector[float]& SVM_weights, float& SVM_offset);
        # * \brief Resize an image represented by a pointcloud containing RGB information.
        # * \param[in] input_image A pointer to a pointcloud containing RGB information.
        # * \param[out] output_image A pointer to the output pointcloud.
        # * \param[in] width Output width.
        # * \param[in] height Output height.
        # void resize (PointCloudPtr& input_image, PointCloudPtr& output_image, int width, int height)
        # * \brief Copies an image and makes a black border around it, where the source image is not present.
        # * \param[in] input_image A pointer to a pointcloud containing RGB information.
        # * \param[out] output_image A pointer to the output pointcloud.
        # * \param[in] xmin x coordinate of the top-left point of the bbox to copy from the input image.
        # * \param[in] ymin y coordinate of the top-left point of the bbox to copy from the input image.
        # * \param[in] width Output width.
        # * \param[in] height Output height.
        # void copyMakeBorder (PointCloudPtr& input_image, PointCloudPtr& output_image, int xmin, int ymin, int width, int height)
        # * \brief Classify the given portion of image.
        # * \param[in] height The height of the image patch to classify, in pixels.
        # * \param[in] xc The x-coordinate of the center of the image patch to classify, in pixels.
        # * \param[in] yc The y-coordinate of the center of the image patch to classify, in pixels.
        # * \param[in] image The whole image (pointer to a point cloud containing RGB information) containing the object to classify.
        # * \return The classification score given by the SVM.
        # double evaluate (float height, float xc, float yc, PointCloudPtr& image)
        # * \brief Compute person confidence for a given PersonCluster.
        # * \param[in] image The input image (pointer to a point cloud containing RGB information).
        # * \param[in] bottom Theoretical bottom point of the cluster projected to the image.
        # * \param[in] top Theoretical top point of the cluster projected to the image.
        # * \param[in] centroid Theoretical centroid point of the cluster projected to the image.
        # * \param[in] vertical If true, the sensor is considered to be vertically placed (portrait mode).
        # * \return The person confidence.
        # double evaluate (PointCloudPtr& image, Eigen::Vector3f& bottom, Eigen::Vector3f& top, Eigen::Vector3f& centroid, bool vertical)
###


# person_cluster.h
# namespace pcl
# namespace people
# /** \brief @b PersonCluster represents a class for representing information about a cluster containing a person.
#  * \author Filippo Basso, Matteo Munaro
#  * \ingroup people
#  */
# template <typename PointT> class PersonCluster;
# template <typename PointT> bool operator<(const PersonCluster<PointT>& c1, const PersonCluster<PointT>& c2);
# 
# template <typename PointT>
# class PersonCluster
cdef extern from "pcl/people/person_cluster.h" namespace "pcl::people":
    cdef cppclass PersonCluster:
        PersonClassifier()
        # PersonCluster (
        # const PointCloudPtr& input_cloud,
        # const pcl::PointIndices& indices,
        # const Eigen::VectorXf& ground_coeffs,
        # float sqrt_ground_coeffs,
        # bool head_centroid,
        # bool vertical = false);
        # protected:
        # bool head_centroid_;
        # /** \brief Minimum x coordinate of the cluster points. */
        # float min_x_;
        # /** \brief Minimum y coordinate of the cluster points. */
        # float min_y_;
        # /** \brief Minimum z coordinate of the cluster points. */
        # float min_z_;
        # /** \brief Maximum x coordinate of the cluster points. */
        # float max_x_;
        # /** \brief Maximum y coordinate of the cluster points. */
        # float max_y_;
        # /** \brief Maximum z coordinate of the cluster points. */
        # float max_z_;
        # /** \brief Sum of x coordinates of the cluster points. */
        # float sum_x_;
        # /** \brief Sum of y coordinates of the cluster points. */
        # float sum_y_;
        # /** \brief Sum of z coordinates of the cluster points. */
        # float sum_z_;
        # /** \brief Number of cluster points. */
        # int n_;
        # /** \brief x coordinate of the cluster centroid. */
        # float c_x_;
        # /** \brief y coordinate of the cluster centroid. */
        # float c_y_;
        # /** \brief z coordinate of the cluster centroid. */
        # float c_z_;
        # /** \brief Cluster height from the ground plane. */
        # float height_;
        # /** \brief Cluster distance from the sensor. */
        # float distance_;
        # /** \brief Cluster centroid horizontal angle with respect to z axis. */
        # float angle_;
        # /** \brief Maximum angle of the cluster points. */
        # float angle_max_;
        # /** \brief Minimum angle of the cluster points. */
        # float angle_min_;
        # /** \brief Cluster top point. */
        # Eigen::Vector3f top_;
        # /** \brief Cluster bottom point. */
        # Eigen::Vector3f bottom_;
        # /** \brief Cluster centroid. */
        # Eigen::Vector3f center_;
        # /** \brief Theoretical cluster top. */
        # Eigen::Vector3f ttop_;
        # /** \brief Theoretical cluster bottom (lying on the ground plane). */
        # Eigen::Vector3f tbottom_;
        # /** \brief Theoretical cluster center (between ttop_ and tbottom_). */
        # Eigen::Vector3f tcenter_;
        # /** \brief Vector containing the minimum coordinates of the cluster. */
        # Eigen::Vector3f min_;
        # /** \brief Vector containing the maximum coordinates of the cluster. */
        # Eigen::Vector3f max_;
        # /** \brief Point cloud indices of the cluster points. */
        # pcl::PointIndices points_indices_;
        # /** \brief If true, the sensor is considered to be vertically placed (portrait mode). */
        # bool vertical_;
        # /** \brief PersonCluster HOG confidence. */
        # float person_confidence_;
        # public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef boost::shared_ptr<PointCloud> PointCloudPtr;
        # typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
        # * \brief Returns the height of the cluster.
        # * \return the height of the cluster.
        float getHeight ();
        # * \brief Update the height of the cluster.
        # * \param[in] ground_coeffs The coefficients of the ground plane.
        # * \return the height of the cluster.
        # float updateHeight (const Eigen::VectorXf& ground_coeffs);
        # * \brief Update the height of the cluster.
        # * \param[in] ground_coeffs The coefficients of the ground plane.
        # * \param[in] sqrt_ground_coeffs The norm of the vector [a, b, c] where a, b and c are the first
        # * three coefficients of the ground plane (ax + by + cz + d = 0).
        # * \return the height of the cluster.
        # float updateHeight (const Eigen::VectorXf& ground_coeffs, float sqrt_ground_coeffs);
        # * \brief Returns the distance of the cluster from the sensor.
        # * \return the distance of the cluster (its centroid) from the sensor without considering the
        # * y dimension.
        float getDistance ()
        # * \brief Returns the angle formed by the cluster's centroid with respect to the sensor (in radians).
        # * \return the angle formed by the cluster's centroid with respect to the sensor (in radians).
        float getAngle ()
        # * \brief Returns the minimum angle formed by the cluster with respect to the sensor (in radians).
        # * \return the minimum angle formed by the cluster with respect to the sensor (in radians).
        float getAngleMin ()
        # * \brief Returns the maximum angle formed by the cluster with respect to the sensor (in radians).
        # * \return the maximum angle formed by the cluster with respect to the sensor (in radians).
        float getAngleMax ()
        # * \brief Returns the indices of the point cloud points corresponding to the cluster.
        # * \return the indices of the point cloud points corresponding to the cluster.
        # pcl::PointIndices& getIndices ();
        # * \brief Returns the theoretical top point.
        # * \return the theoretical top point.
        # Eigen::Vector3f& getTTop ();
        # * \brief Returns the theoretical bottom point.
        # * \return the theoretical bottom point.
        # Eigen::Vector3f& getTBottom ();
        # * \brief Returns the theoretical centroid (at half height).
        # * \return the theoretical centroid (at half height).
        # Eigen::Vector3f& getTCenter ();
        # * \brief Returns the top point.
        # * \return the top point.
        # Eigen::Vector3f& getTop ();
        # * \brief Returns the bottom point.
        # * \return the bottom point.
        # Eigen::Vector3f& getBottom ();
        # * \brief Returns the centroid.
        # * \return the centroid.
        # Eigen::Vector3f& getCenter ();  
        # //Eigen::Vector3f& getTMax();
        # * \brief Returns the point formed by min x, min y and min z.
        # * \return the point formed by min x, min y and min z.
        # Eigen::Vector3f& getMin ();
        # * \brief Returns the point formed by max x, max y and max z.
        # * \return the point formed by max x, max y and max z.
        # Eigen::Vector3f& getMax ();
        # /**
        # * \brief Returns the HOG confidence.
        # * \return the HOG confidence.
        # */
        # float
        # getPersonConfidence ();
        # /**
        # * \brief Returns the number of points of the cluster.
        # * \return the number of points of the cluster.
        # */
        # int getNumberPoints ();
        # /**
        # * \brief Sets the cluster height.
        # * \param[in] height
        # */
        # void setHeight (float height);
        # /**
        # * \brief Sets the HOG confidence.
        # * \param[in] confidence
        # */
        # void setPersonConfidence (float confidence);
        # /**
        # * \brief Draws the theoretical 3D bounding box of the cluster in the PCL visualizer.
        # * \param[in] viewer PCL visualizer.
        # * \param[in] person_number progressive number representing the person.
        # */
        # void drawTBoundingBox (pcl::visualization::PCLVisualizer& viewer, int person_number);
        # /**
        # * \brief For sorting purpose: sort by distance.
        # */
        # friend bool operator< <>(const PersonCluster<PointT>& c1, const PersonCluster<PointT>& c2);
        # protected:
        # /**
        # * \brief PersonCluster initialization.
        # */
        # void init(
        #   const PointCloudPtr& input_cloud,
        #   const pcl::PointIndices& indices,
        #   const Eigen::VectorXf& ground_coeffs,
        #   float sqrt_ground_coeffs,
        #   bool head_centroid,
        #   bool vertical);
###


# ground_based_people_detection_app.h
# namespace pcl
# namespace people
# template <typename PointT>
# class GroundBasedPeopleDetectionApp
# public:
cdef extern from "pcl/people/ground_based_people_detection_app.h" namespace "pcl::people":
    cdef cppclass GroundBasedPeopleDetectionApp[PointT]:
        GroundBasedPeopleDetectionApp()
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef boost::shared_ptr<PointCloud> PointCloudPtr;
        # typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
        # * \brief Set the pointer to the input cloud.
        # * \param[in] cloud A pointer to the input cloud.
        # void setInputCloud (PointCloudPtr& cloud);
        void setInputCloud (sharedPtr[cpp.PointCloud[PointT]] cloud)
        #  * \brief Set the ground coefficients.
        #  * \param[in] ground_coeffs Vector containing the four plane coefficients.
        # void setGround (Eigen::VectorXf& ground_coeffs);
        #  * \brief Set the transformation matrix, which is used in order to transform the given point cloud, the ground plane and the intrinsics matrix to the internal coordinate frame.
        #  * \param[in] cloud A pointer to the input cloud.
        void setTransformation (eigen3.Matrix3f& transformation);
        #  * \brief Set sampling factor. 
        #  * \param[in] sampling_factor Value of the downsampling factor (in each dimension) which is applied to the raw point cloud (default = 1.).
        void setSamplingFactor (int sampling_factor);
        #  * \brief Set voxel size. 
        #  * \param[in] voxel_size Value of the voxel dimension (default = 0.06m.).
        void setVoxelSize (float voxel_size);
        #  * \brief Set intrinsic parameters of the RGB camera.
        #  * \param[in] intrinsics_matrix RGB camera intrinsic parameters matrix.
        void setIntrinsics (eigen3.Matrix3f intrinsics_matrix);
        #  * \brief Set SVM-based person classifier.
        #  * \param[in] person_classifier Needed for people detection on RGB data.
        # void setClassifier (pcl::people::PersonClassifier<pcl::RGB> person_classifier);
        #  * \brief Set the field of view of the point cloud in z direction.
        #  * \param[in] min The beginning of the field of view in z-direction, should be usually set to zero.
        #  * \param[in] max The end of the field of view in z-direction.
        void setFOV (float min, float max)
        #  * \brief Set sensor orientation (vertical = true means portrait mode, vertical = false means landscape mode).
        #  * \param[in] vertical Set landscape/portait camera orientation (default = false).
        void setSensorPortraitOrientation (bool vertical)
        #  * \brief Set head_centroid_ to true (person centroid is in the head) or false (person centroid is the whole body centroid).
        #  * \param[in] head_centroid Set the location of the person centroid (head or body center) (default = true).
        void setHeadCentroid (bool head_centroid)
        #  * \brief Set minimum and maximum allowed height and width for a person cluster.
        #  * \param[in] min_height Minimum allowed height for a person cluster (default = 1.3).
        #  * \param[in] max_height Maximum allowed height for a person cluster (default = 2.3).
        #  * \param[in] min_width Minimum width for a person cluster (default = 0.1).
        #  * \param[in] max_width Maximum width for a person cluster (default = 8.0).
        void setPersonClusterLimits (float min_height, float max_height, float min_width, float max_width)
        #  * \brief Set minimum distance between persons' heads.
        #  * \param[in] heads_minimum_distance Minimum allowed distance between persons' heads (default = 0.3).
        void setMinimumDistanceBetweenHeads (float heads_minimum_distance)
        #  * \brief Get the minimum and maximum allowed height and width for a person cluster.
        #  * \param[out] min_height Minimum allowed height for a person cluster.
        #  * \param[out] max_height Maximum allowed height for a person cluster.
        #  * \param[out] min_width Minimum width for a person cluster.
        #  * \param[out] max_width Maximum width for a person cluster.
        void getPersonClusterLimits (float& min_height, float& max_height, float& min_width, float& max_width);
        #   * \brief Get minimum and maximum allowed number of points for a person cluster.
        #   * \param[out] min_points Minimum allowed number of points for a person cluster.
        #   * \param[out] max_points Maximum allowed number of points for a person cluster.
        void getDimensionLimits (int& min_points, int& max_points)
        #   * \brief Get minimum distance between persons' heads.
        float getMinimumDistanceBetweenHeads ()
        #   * \brief Get floor coefficients.
        # Eigen::VectorXf getGround ();
        #   * \brief Get the filtered point cloud.
        # PointCloudPtr getFilteredCloud ();
        #   * \brief Get pointcloud after voxel grid filtering and ground removal.
        # PointCloudPtr getNoGroundCloud ();
        #   * \brief Extract RGB information from a point cloud and output the corresponding RGB point cloud.
        #   * \param[in] input_cloud A pointer to a point cloud containing also RGB information.
        #   * \param[out] output_cloud A pointer to a RGB point cloud.
        # void extractRGBFromPointCloud (PointCloudPtr input_cloud, pcl::PointCloud<pcl::RGB>::Ptr& output_cloud);
        #   * \brief Swap rows/cols dimensions of a RGB point cloud (90 degrees counterclockwise rotation).
        #   * \param[in,out] cloud A pointer to a RGB point cloud.
        # void swapDimensions (pcl::PointCloud<pcl::RGB>::Ptr& cloud);
        #   * \brief Estimates min_points_ and max_points_ based on the minimal and maximal cluster size and the voxel size.
        void updateMinMaxPoints ()
        #   * \brief Applies the transformation to the input point cloud.
        void applyTransformationPointCloud ()
        #   * \brief Applies the transformation to the ground plane.
        void applyTransformationGround ()
        #   * \brief Applies the transformation to the intrinsics matrix.
        void applyTransformationIntrinsics ()
        #   * \brief Reduces the input cloud to one point per voxel and limits the field of view.
        void filter ()
        #   * \brief Perform people detection on the input data and return people clusters information.
        #   * \param[out] clusters Vector of PersonCluster.
        #   * \return true if the compute operation is successful, false otherwise.
        # bool compute (std::vector<pcl::people::PersonCluster<PointT> >& clusters);
        # protected:
        # /** \brief sampling factor used to downsample the point cloud */
        # int sampling_factor_; 
        # /** \brief voxel size */
        # float voxel_size_;                  
        # /** \brief ground plane coefficients */
        # Eigen::VectorXf ground_coeffs_;
        # /** \brief flag stating whether the ground coefficients have been set or not */
        # bool ground_coeffs_set_;
        # /** \brief the transformed ground coefficients */
        # Eigen::VectorXf ground_coeffs_transformed_;
        # /** \brief ground plane normalization factor */
        # float sqrt_ground_coeffs_;
        # /** \brief rotation matrix which transforms input point cloud to internal people tracker coordinate frame */
        # Eigen::Matrix3f transformation_;
        # /** \brief flag stating whether the transformation matrix has been set or not */
        # bool transformation_set_;
        # /** \brief pointer to the input cloud */
        # PointCloudPtr cloud_;
        # /** \brief pointer to the filtered cloud */
        # PointCloudPtr cloud_filtered_;
        # /** \brief pointer to the cloud after voxel grid filtering and ground removal */
        # PointCloudPtr no_ground_cloud_;              
        # /** \brief pointer to a RGB cloud corresponding to cloud_ */
        # pcl::PointCloud<pcl::RGB>::Ptr rgb_image_;      
        # /** \brief person clusters maximum height from the ground plane */
        # float max_height_;                  
        # /** \brief person clusters minimum height from the ground plane */
        # float min_height_;
        # /** \brief person clusters maximum width, used to estimate how many points maximally represent a person cluster */
        # float max_width_;
        # /** \brief person clusters minimum width, used to estimate how many points minimally represent a person cluster */
        # float min_width_;
        # /** \brief the beginning of the field of view in z-direction, should be usually set to zero */
        # float min_fov_;
        # /** \brief the end of the field of view in z-direction */
        # float max_fov_;
        # /** \brief if true, the sensor is considered to be vertically placed (portrait mode) */
        # bool vertical_;                    
        # /** \brief if true, the person centroid is computed as the centroid of the cluster points belonging to the head;  
        # * if false, the person centroid is computed as the centroid of the whole cluster points (default = true) */
        # bool head_centroid_;    // if true, the person centroid is computed as the centroid of the cluster points belonging to the head (default = true)
        #                       // if false, the person centroid is computed as the centroid of the whole cluster points 
        # /** \brief maximum number of points for a person cluster */
        # int max_points_;                  
        # /** \brief minimum number of points for a person cluster */
        # int min_points_;                  
        # /** \brief minimum distance between persons' heads */
        # float heads_minimum_distance_;            
        # /** \brief intrinsic parameters matrix of the RGB camera */
        # Eigen::Matrix3f intrinsics_matrix_;
        # /** \brief flag stating whether the intrinsics matrix has been set or not */
        # bool intrinsics_matrix_set_;
        # /** \brief the transformed intrinsics matrix */
        # Eigen::Matrix3f intrinsics_matrix_transformed_;
        # /** \brief SVM-based person classifier */
        # pcl::people::PersonClassifier<pcl::RGB> person_classifier_;  
        # /** \brief flag stating if the classifier has been set or not */
        # bool person_classifier_set_flag_;
###

# head_based_subcluster.h
# namespace pcl
# namespace people
# /** \brief @b HeadBasedSubclustering represents a class for searching for people inside a HeightMap2D based on a 3D head detection algorithm
#   * \author Matteo Munaro
#   * \ingroup people
# */
# template <typename PointT> class HeadBasedSubclustering;
# 
# template <typename PointT>
# class HeadBasedSubclustering
cdef extern from "pcl/people/head_based_subcluster.h" namespace "pcl::people":
    cdef cppclass HeadBasedSubclustering[PointT]:
        HeadBasedSubclustering()
        # public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef boost::shared_ptr<PointCloud> PointCloudPtr;
        # typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
        #  * \brief Compute subclusters and return them into a vector of PersonCluster.
        #  * \param[in] clusters Vector of PersonCluster.
        # void subcluster (std::vector<pcl::people::PersonCluster<PointT> >& clusters);
        # /**
        # * \brief Merge clusters close in floor coordinates.
        # * 
        # * \param[in] input_clusters Input vector of PersonCluster.
        # * \param[in] output_clusters Output vector of PersonCluster (after merging).
        # */
        # void
        # mergeClustersCloseInFloorCoordinates (std::vector<pcl::people::PersonCluster<PointT> >& input_clusters,
        #   std::vector<pcl::people::PersonCluster<PointT> >& output_clusters);
        # /**
        # * \brief Create subclusters centered on the heads position from the current cluster.
        # * 
        # * \param[in] cluster A PersonCluster.
        # * \param[in] maxima_number_after_filtering Number of local maxima to use as centers of the new cluster.
        # * \param[in] maxima_cloud_indices_filtered Cloud indices of local maxima to use as centers of the new cluster.
        # * \param[out] subclusters Output vector of PersonCluster objects derived from the input cluster.
        # */
        # void
        # createSubClusters (pcl::people::PersonCluster<PointT>& cluster, int maxima_number_after_filtering,  std::vector<int>& maxima_cloud_indices_filtered,
        #   std::vector<pcl::people::PersonCluster<PointT> >& subclusters);
        # /**
        # * \brief Set input cloud.
        # * 
        # * \param[in] cloud A pointer to the input point cloud.
        # */
        # void
        # setInputCloud (PointCloudPtr& cloud);
        # /**
        # * \brief Set the ground coefficients.
        # * 
        # * \param[in] ground_coeffs The ground plane coefficients.
        # */
        # void
        # setGround (Eigen::VectorXf& ground_coeffs);
        # /**
        # * \brief Set sensor orientation to landscape mode (false) or portrait mode (true).
        # * 
        # * \param[in] vertical Landscape (false) or portrait (true) mode (default = false).
        # */
        # void
        # setSensorPortraitOrientation (bool vertical);
        # /**
        # * \brief Set head_centroid_ to true (person centroid is in the head) or false (person centroid is the whole body centroid).
        # *
        # * \param[in] head_centroid Set the location of the person centroid (head or body center) (default = true).
        # */
        # void
        # setHeadCentroid (bool head_centroid);
        # /**
        # * \brief Set initial cluster indices.
        # * 
        # * \param[in] cluster_indices Point cloud indices corresponding to the initial clusters (before subclustering).
        # */
        # void
        # setInitialClusters (std::vector<pcl::PointIndices>& cluster_indices);
        # /**
        # * \brief Set minimum and maximum allowed height for a person cluster.
        # *
        # * \param[in] min_height Minimum allowed height for a person cluster (default = 1.3).
        # * \param[in] max_height Maximum allowed height for a person cluster (default = 2.3).
        # */
        # void
        # setHeightLimits (float min_height, float max_height);
        # /**
        # * \brief Set minimum and maximum allowed number of points for a person cluster.
        # *
        # * \param[in] min_points Minimum allowed number of points for a person cluster.
        # * \param[in] max_points Maximum allowed number of points for a person cluster.
        # */
        # void
        # setDimensionLimits (int min_points, int max_points);
        # /**
        # * \brief Set minimum distance between persons' heads.
        # *
        # * \param[in] heads_minimum_distance Minimum allowed distance between persons' heads (default = 0.3).
        # */
        # void
        # setMinimumDistanceBetweenHeads (float heads_minimum_distance);
        # /**
        # * \brief Get minimum and maximum allowed height for a person cluster.
        # *
        # * \param[out] min_height Minimum allowed height for a person cluster.
        # * \param[out] max_height Maximum allowed height for a person cluster.
        # */
        # void
        # getHeightLimits (float& min_height, float& max_height);
        # /**
        # * \brief Get minimum and maximum allowed number of points for a person cluster.
        # *
        # * \param[out] min_points Minimum allowed number of points for a person cluster.
        # * \param[out] max_points Maximum allowed number of points for a person cluster.
        # */
        # void
        # getDimensionLimits (int& min_points, int& max_points);
        # /**
        # * \brief Get minimum distance between persons' heads.
        # */
        # float
        # getMinimumDistanceBetweenHeads ();
        # protected:
        # /** \brief ground plane coefficients */
        # Eigen::VectorXf ground_coeffs_;            
        # /** \brief ground plane normalization factor */
        # float sqrt_ground_coeffs_;              
        # /** \brief initial clusters indices */
        # std::vector<pcl::PointIndices> cluster_indices_;   
        # /** \brief pointer to the input cloud */
        # PointCloudPtr cloud_;                
        # /** \brief person clusters maximum height from the ground plane */
        # float max_height_;                  
        # /** \brief person clusters minimum height from the ground plane */
        # float min_height_;                  
        # /** \brief if true, the sensor is considered to be vertically placed (portrait mode) */
        # bool vertical_;                   
        # /** \brief if true, the person centroid is computed as the centroid of the cluster points belonging to the head 
        #          if false, the person centroid is computed as the centroid of the whole cluster points (default = true) */
        # bool head_centroid_;                                            
        # /** \brief maximum number of points for a person cluster */
        # int max_points_;                  
        # /** \brief minimum number of points for a person cluster */
        # int min_points_;                  
        # /** \brief minimum distance between persons' heads */
        # float heads_minimum_distance_;           
###

# height_map_2d.h
# namespace pcl
# namespace people
# /** \brief @b HeightMap2D represents a class for creating a 2D height map from a point cloud and searching for its local maxima
#   * \author Matteo Munaro
#   * \ingroup people
# */
# template <typename PointT> class HeightMap2D;
# template <typename PointT>
cdef extern from "pcl/people/height_map_2d.h" namespace "pcl::people":
    cdef cppclass HeightMap2D:
        HeightMap2D()
        # public:
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef boost::shared_ptr<PointCloud> PointCloudPtr;
        # typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
        # /**
        # * \brief Compute the height map with the projection of cluster points onto the ground plane.
        # * \param[in] cluster The PersonCluster used to compute the height map.
        # */
        # void
        # compute (pcl::people::PersonCluster<PointT>& cluster);
        # /**
        # * \brief Compute local maxima of the height map.
        # */
        # void
        # searchLocalMaxima ();
        # /**
        # * \brief Filter maxima of the height map by imposing a minimum distance between them.
        # */
        # void
        # filterMaxima ();
        # /**
        # * \brief Set initial cluster indices.
        # * 
        # * \param[in] cloud A pointer to the input cloud.
        # */
        # void
        # setInputCloud (PointCloudPtr& cloud);
        # /**
        # * \brief Set the ground coefficients.
        # * 
        # * \param[in] ground_coeffs The ground plane coefficients.
        # */
        # void
        # setGround (Eigen::VectorXf& ground_coeffs);
        # /**
        # * \brief Set bin size for the height map. 
        # * 
        # * \param[in] bin_size Bin size for the height map (default = 0.06).
        # */
        # void
        # setBinSize (float bin_size);
        # /**
        # * \brief Set minimum distance between maxima. 
        # * 
        # * \param[in] minimum_distance_between_maxima Minimum allowed distance between maxima (default = 0.3).
        # */
        # void
        # setMinimumDistanceBetweenMaxima (float minimum_distance_between_maxima);
        # /**
        # * \brief Set sensor orientation to landscape mode (false) or portrait mode (true).
        # * 
        # * \param[in] vertical Landscape (false) or portrait (true) mode (default = false).
        # */
        # void
        # setSensorPortraitOrientation (bool vertical);
        # /**
        # * \brief Get the height map as a vector of int.
        # */
        # std::vector<int>& getHeightMap ();
        # /**
        # * \brief Get bin size for the height map. 
        # */
        # float getBinSize ();
        # /**
        # * \brief Get minimum distance between maxima of the height map. 
        # */
        # float getMinimumDistanceBetweenMaxima ();
        # /**
        # * \brief Return the maxima number after the filterMaxima method.
        # */
        # int& getMaximaNumberAfterFiltering ();
        # /**
        # * \brief Return the point cloud indices corresponding to the maxima computed after the filterMaxima method.
        # */
        # std::vector<int>& getMaximaCloudIndicesFiltered ();
        # protected:
        # /** \brief ground plane coefficients */
        # Eigen::VectorXf ground_coeffs_;            
        # /** \brief ground plane normalization factor */
        # float sqrt_ground_coeffs_;              
        # /** \brief pointer to the input cloud */
        # PointCloudPtr cloud_;                
        # /** \brief if true, the sensor is considered to be vertically placed (portrait mode) */
        # bool vertical_;                    
        # /** \brief vector with maximum height values for every bin (height map) */
        # std::vector<int> buckets_;              
        # /** \brief indices of the pointcloud points with maximum height for every bin */
        # std::vector<int> buckets_cloud_indices_;      
        # /** \brief bin dimension */
        # float bin_size_;                  
        # /** \brief number of local maxima in the height map */
        # int maxima_number_;                  
        # /** \brief contains the position of the maxima in the buckets vector */
        # std::vector<int> maxima_indices_;          
        # /** \brief contains the point cloud position of the maxima (indices of the point cloud) */
        # std::vector<int> maxima_cloud_indices_;        
        # /** \brief number of local maxima after filtering */
        # int maxima_number_after_filtering_;          
        # /** \brief contains the position of the maxima in the buckets array after filtering */
        # std::vector<int> maxima_indices_filtered_;      
        # /** \brief contains the point cloud position of the maxima after filtering */
        # std::vector<int> maxima_cloud_indices_filtered_;  
        # /** \brief minimum allowed distance between maxima */
        # float min_dist_between_maxima_;            
###

# hog.h
# namespace pcl
# namespace people
# /** \brief @b HOG represents a class for computing the HOG descriptor described in 
# * Dalal, N. and Triggs, B., "Histograms of oriented gradients for human detection", CVPR 2005.
# * \author Matteo Munaro, Stefano Ghidoni, Stefano Michieletto
# * \ingroup people
# */
# class PCL_EXPORTS HOG
cdef extern from "pcl/people/hog.h" namespace "pcl::people":
    cdef cppclass HOG:
        HOG()
        # public:
        # /** 
        # * \brief Compute gradient magnitude and orientation at each location (uses sse). 
        # * 
        # * \param[in] I Image as array of float.
        # * \param[in] h Image height.
        # * \param[in] w Image width.
        # * \param[in] d Image number of channels.
        # * \param[out] M Gradient magnitude for each image point.
        # * \param[out] O Gradient orientation for each image point.
        # */
        # void gradMag ( float *I, int h, int w, int d, float *M, float *O ) const;
        # /** 
        # * \brief Compute n_orients gradient histograms per bin_size x bin_size block of pixels.  
        # * 
        # * \param[in] M Gradient magnitude for each image point.
        # * \param[in] O Gradient orientation for each image point.
        # * \param[in] h Image height.
        # * \param[in] w Image width.
        # * \param[in] bin_size Spatial bin size.
        # * \param[in] n_orients Number of orientation bins.
        # * \param[in] soft_bin If true, each pixel can contribute to multiple spatial bins (using bilinear interpolation).
        # * \param[out] H Gradient histograms.
        # */
        # void gradHist ( float *M, float *O, int h, int w, int bin_size, int n_orients, bool soft_bin, float *H) const;
        # /** 
        # * \brief Normalize histogram of gradients. 
        # * 
        # * \param[in] H Gradient histograms.
        # * \param[in] h Image height.
        # * \param[in] w Image width.
        # * \param[in] bin_size Spatial bin size.
        # * \param[in] n_orients Number of orientation bins.  
        # * \param[in] clip Value at which to clip histogram bins.      
        # * \param[out] G Normalized gradient histograms.
        # */
        # void normalization ( float *H, int h, int w, int bin_size, int n_orients, float clip, float *G ) const;
        # /**
        # * \brief Compute HOG descriptor.
        # * 
        # * \param[in] I Image as array of float between 0 and 1.
        # * \param[in] h Image height.
        # * \param[in] w Image width.
        # * \param[in] n_channels Image number of channels.
        # * \param[in] bin_size Spatial bin size.  
        # * \param[in] n_orients Number of orientation bins.     
        # * \param[in] soft_bin If true, each pixel can contribute to multiple spatial bins (using bilinear interpolation).
        # * \param[out] descriptor HOG descriptor.
        # */
        # void compute (float *I, int h, int w, int n_channels, int bin_size, int n_orients, bool soft_bin, float *descriptor);
        # /**
        # * \brief Compute HOG descriptor with default parameters.
        # * \param[in] I Image as array of float between 0 and 1.
        # * \param[out] descriptor HOG descriptor.
        # */
        # void compute (float *I, float *descriptor) const;
        # protected:
        # /** \brief image height (default = 128) */
        # int h_;
        # /** \brief image width (default = 64) */
        # int w_;
        # /** \brief image number of channels (default = 3) */
        # int n_channels_;
        # /** \brief spatial bin size (default = 8) */
        # int bin_size_; 
        # /** \brief number of orientation bins (default = 9) */
        # int n_orients_;
        # /** \brief if true, each pixel can contribute to multiple spatial bins (using bilinear interpolation) (default = true) */
        # bool soft_bin_;   
        # /** \brief value at which to clip histogram bins (default = 0.2) */
        # float clip_; 
###


