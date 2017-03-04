# -*- coding: utf-8 -*-
from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

# boost
from boost_shared_ptr cimport shared_ptr

cimport eigen as eigen3

# FW: Link time errors in RangeImage (with /clr)
# http://www.pcl-users.org/FW-Link-time-errors-in-RangeImage-with-clr-td3581422.html
# Linking errors using RangeImagePlanar (no use /clr)
# http://www.pcl-users.org/Linking-errors-using-RangeImagePlanar-td4036511.html
# range_image.h
# class RangeImage : public pcl::PointCloud<PointWithRange>
cdef extern from "pcl/range_image/range_image.h" namespace "pcl":
    cdef cppclass RangeImage(cpp.PointCloud[cpp.PointWithRange]):
        RangeImage()
        # public:
        
        # // =====STATIC METHODS=====
        # brief Get the size of a certain area when seen from the given pose
        # param viewer_pose an affine matrix defining the pose of the viewer
        # param center the center of the area
        # param radius the radius of the area
        # return the size of the area as viewed according to \a viewer_pose
        # static inline float getMaxAngleSize (const Eigen::Affine3f& viewer_pose, const Eigen::Vector3f& center, float radius);
        float getMaxAngleSize (eigen3.Affine3f& viewer_pose, const eigen3.Vector3f& center, float radius)
        
        # brief Get Eigen::Vector3f from PointWithRange
        # param point the input point
        # return an Eigen::Vector3f representation of the input point
        # static inline Eigen::Vector3f getEigenVector3f (const PointWithRange& point);
        eigen3.Vector3f getEigenVector3f (const cpp.PointWithRange& point)
        
        # brief Get the transformation that transforms the given coordinate frame into CAMERA_FRAME
        # param coordinate_frame the input coordinate frame
        # param transformation the resulting transformation that warps \a coordinate_frame into CAMERA_FRAME
        # PCL_EXPORTS static void getCoordinateFrameTransformation (RangeImage::CoordinateFrame coordinate_frame, Eigen::Affine3f& transformation);
        void getCoordinateFrameTransformation (CoordinateFrame2 coordinate_frame, eigen3.Affine3f& transformation)
        
        # brief Get the average viewpoint of a point cloud where each point carries viewpoint information as 
        # vp_x, vp_y, vp_z
        # param point_cloud the input point cloud
        # return the average viewpoint (as an Eigen::Vector3f)
        # template <typename PointCloudTypeWithViewpoints> static Eigen::Vector3f getAverageViewPoint (const PointCloudTypeWithViewpoints& point_cloud);
        eigen3.Vector3f getAverageViewPoint (const cpp.PointCloud[cpp.PointWithRange]& point_cloud)
        
        # brief Check if the provided data includes far ranges and add them to far_ranges
        # param point_cloud_data a PointCloud2 message containing the input cloud
        # param far_ranges the resulting cloud containing those points with far ranges
        # PCL_EXPORTS static void extractFarRanges (const sensor_msgs::PointCloud2& point_cloud_data, PointCloud<PointWithViewpoint>& far_ranges);
        # void extractFarRanges (const sensor_msgs::PointCloud2& point_cloud_data, PointCloud<PointWithViewpoint>& far_ranges)
        
        # // =====METHODS=====
        # brief Get a boost shared pointer of a copy of this
        # inline Ptr makeShared () { return Ptr (new RangeImage (*this)); } 
        # RangeImagePtr_t makeShared ()
        
        # brief Reset all values to an empty range image
        # PCL_EXPORTS void reset ();
        void reset ()
        
        ###
        # brief Create the depth image from a point cloud
        # param point_cloud the input point cloud
        # param angular_resolution the angular difference (in radians) between the individual pixels in the image
        # param max_angle_width an angle (in radians) defining the horizontal bounds of the sensor
        # param max_angle_height an angle (in radians) defining the vertical bounds of the sensor
        # param sensor_pose an affine matrix defining the pose of the sensor (defaults to Eigen::Affine3f::Identity () )
        # param coordinate_frame the coordinate frame (defaults to CAMERA_FRAME)
        # param noise_level - The distance in meters inside of which the z-buffer will not use the minimum,
        #                     but the mean of the points. If 0.0 it is equivalent to a normal z-buffer and
        #                     will always take the minimum per cell.
        # param min_range the minimum visible range (defaults to 0)
        # param border_size the border size (defaults to 0)
        #
        # template <typename PointCloudType> void
        # createFromPointCloud (const PointCloudType& point_cloud, float angular_resolution=pcl::deg2rad (0.5f),
        #     float max_angle_width=pcl::deg2rad (360.0f), float max_angle_height=pcl::deg2rad (180.0f),
        #     const Eigen::Affine3f& sensor_pose = Eigen::Affine3f::Identity (),
        #     CoordinateFrame coordinate_frame=CAMERA_FRAME, float noise_level=0.0f,
        #     float min_range=0.0f, int border_size=0);
        # use Template
        void createFromPointCloud [PointCloudType](
            const PointCloudType& point_cloud, 
            float angular_resolution,
            float max_angle_width, 
            float max_angle_height,
            const eigen3.Affine3f& sensor_pose,
            CoordinateFrame2 coordinate_frame, 
            float noise_level,
            float min_range, 
            int border_size)
        ###
        
        # brief Create the depth image from a point cloud
        # param point_cloud the input point cloud
        # param angular_resolution_x the angular difference (in radians) between the
        #                            individual pixels in the image in the x-direction
        # param angular_resolution_y the angular difference (in radians) between the
        #                            individual pixels in the image in the y-direction
        # param max_angle_width an angle (in radians) defining the horizontal bounds of the sensor
        # param max_angle_height an angle (in radians) defining the vertical bounds of the sensor
        # param sensor_pose an affine matrix defining the pose of the sensor (defaults to
        #                   Eigen::Affine3f::Identity () )
        # param coordinate_frame the coordinate frame (defaults to CAMERA_FRAME)
        # param noise_level - The distance in meters inside of which the z-buffer will not use the minimum,
        #                     but the mean of the points. If 0.0 it is equivalent to a normal z-buffer and
        #                     will always take the minimum per cell.
        # param min_range the minimum visible range (defaults to 0)
        # param border_size the border size (defaults to 0)
        # -- 
        # template <typename PointCloudType> void
        # createFromPointCloud (const PointCloudType& point_cloud,
        #   float angular_resolution_x=pcl::deg2rad (0.5f), float angular_resolution_y=pcl::deg2rad (0.5f),
        #   float max_angle_width=pcl::deg2rad (360.0f), float max_angle_height=pcl::deg2rad (180.0f),
        #   const Eigen::Affine3f& sensor_pose = Eigen::Affine3f::Identity (),
        #   CoordinateFrame coordinate_frame=CAMERA_FRAME,
        #   float noise_level=0.0f, float min_range=0.0f, int border_size=0);
        ## 
        void createFromPointCloud (
            cpp.PointCloud_t& point_cloud, 
            float angular_resolution_x,
            float angular_resolution_y,
            float max_angle_width, 
            float max_angle_height,
            const eigen3.Affine3f& sensor_pose,
            CoordinateFrame2 coordinate_frame, 
            float noise_level,
            float min_range, 
            int border_size)
        
        # void createFromPointCloud [PointCloudType](
        #     cpp.PointCloud[PointCloudType]& point_cloud, 
        #     float angular_resolution_x,
        #     float angular_resolution_y,
        #     float max_angle_width, 
        #     float max_angle_height,
        #     const eigen3.Affine3f& sensor_pose,
        #     CoordinateFrame2 coordinate_frame, 
        #     float noise_level,
        #     float min_range, 
        #     int border_size)
        
        # brief Create the depth image from a point cloud, getting a hint about the size of the scene for aster calculation.
        # param point_cloud the input point cloud
        # param angular_resolution the angle (in radians) between each sample in the depth image
        # param point_cloud_center the center of bounding sphere
        # param point_cloud_radius the radius of the bounding sphere
        # param sensor_pose an affine matrix defining the pose of the sensor (defaults to
        #                    Eigen::Affine3f::Identity () )
        # param coordinate_frame the coordinate frame (defaults to CAMERA_FRAME)
        # param noise_level - The distance in meters inside of which the z-buffer will not use the minimum,
        #                     but the mean of the points. If 0.0 it is equivalent to a normal z-buffer and
        #                     will always take the minimum per cell.
        # param min_range the minimum visible range (defaults to 0)
        # param border_size the border size (defaults to 0)
        # -- 
        # template <typename PointCloudType> void
        # createFromPointCloudWithKnownSize (const PointCloudType& point_cloud, float angular_resolution,
        #                                  const Eigen::Vector3f& point_cloud_center, float point_cloud_radius,
        #                                  const Eigen::Affine3f& sensor_pose = Eigen::Affine3f::Identity (),
        #                                  CoordinateFrame coordinate_frame=CAMERA_FRAME,
        #                                  float noise_level=0.0f, float min_range=0.0f, int border_size=0);
        ## 
        void createFromPointCloudWithKnownSize [PointCloudType](
            PointCloudType& point_cloud,
            float angular_resolution,
            const eigen3.Vector3f& point_cloud_center, 
            float point_cloud_radius,
            const eigen3.Affine3f& sensor_pose,
            CoordinateFrame2 coordinate_frame,
            float noise_level, 
            float min_range, 
            int border_size)
        
        # brief Create the depth image from a point cloud, getting a hint about the size of the scene for 
        # aster calculation.
        # param point_cloud the input point cloud
        # param angular_resolution_x the angular difference (in radians) between the
        #                            individual pixels in the image in the x-direction
        # param angular_resolution_y the angular difference (in radians) between the
        #                            individual pixels in the image in the y-direction
        # param angular_resolution the angle (in radians) between each sample in the depth image
        # param point_cloud_center the center of bounding sphere
        # param point_cloud_radius the radius of the bounding sphere
        # param sensor_pose an affine matrix defining the pose of the sensor (defaults to
        #                    Eigen::Affine3f::Identity () )
        # param coordinate_frame the coordinate frame (defaults to CAMERA_FRAME)
        # param noise_level - The distance in meters inside of which the z-buffer will not use the minimum,
        #                     but the mean of the points. If 0.0 it is equivalent to a normal z-buffer and
        #                     will always take the minimum per cell.
        # param min_range the minimum visible range (defaults to 0)
        # param border_size the border size (defaults to 0)
        # -- 
        # template <typename PointCloudType> void
        # createFromPointCloudWithKnownSize (const PointCloudType& point_cloud,
        #                                  float angular_resolution_x, float angular_resolution_y,
        #                                  const Eigen::Vector3f& point_cloud_center, float point_cloud_radius,
        #                                  const Eigen::Affine3f& sensor_pose = Eigen::Affine3f::Identity (),
        #                                  CoordinateFrame coordinate_frame=CAMERA_FRAME,
        #                                  float noise_level=0.0f, float min_range=0.0f, int border_size=0);
        ## 
        # createFromPointCloudWithKnownSize (
        #     cpp.PointCloud_t& point_cloud, 
        #     float angular_resolution_x,
        #     float angular_resolution_y,
        #     const eigen3.Vector3f& point_cloud_center,
        #     float point_cloud_radius,
        #     const eigen3.Affine3f& sensor_pose,
        #     CoordinateFrame2 coordinate_frame,
        #     float noise_level, 
        #     float min_range, 
        #     int border_size)
        void createFromPointCloudWithKnownSize [PointCloudType](
            cpp.PointCloud[PointCloudType]& point_cloud, 
            float angular_resolution_x, float angular_resolution_y,
            const eigen3.Vector3f& point_cloud_center,
            float point_cloud_radius,
            const eigen3.Affine3f& sensor_pose,
            CoordinateFrame2 coordinate_frame,
            float noise_level, 
            float min_range, 
            int border_size)
        
        # brief Create the depth image from a point cloud, using the average viewpoint of the points 
        # (vp_x,vp_y,vp_z in the point type) in the point cloud as sensor pose (assuming a rotation of (0,0,0)).
        # param point_cloud the input point cloud
        # param angular_resolution the angle (in radians) between each sample in the depth image
        # param max_angle_width an angle (in radians) defining the horizontal bounds of the sensor
        # param max_angle_height an angle (in radians) defining the vertical bounds of the sensor
        # param coordinate_frame the coordinate frame (defaults to CAMERA_FRAME)
        # param noise_level - The distance in meters inside of which the z-buffer will not use the minimum,
        #                     but the mean of the points. If 0.0 it is equivalent to a normal z-buffer and
        #                     will always take the minimum per cell.
        # param min_range the minimum visible range (defaults to 0)
        # param border_size the border size (defaults to 0)
        # note If wrong_coordinate_system is true, the sensor pose will be rotated to change from a coordinate frame
        # with x to the front, y to the left and z to the top to the coordinate frame we use here (x to the right, y 
        # to the bottom and z to the front)
        # template <typename PointCloudTypeWithViewpoints> 
        # void createFromPointCloudWithViewpoints (const PointCloudTypeWithViewpoints& point_cloud, float angular_resolution,
        #                                   float max_angle_width, float max_angle_height,
        #                                   CoordinateFrame coordinate_frame=CAMERA_FRAME, float noise_level=0.0f,
        #                                   float min_range=0.0f, int border_size=0);
        ## 
        void createFromPointCloudWithViewpoints (
            const cpp.PointCloud_PointWithViewpoint_t& point_cloud,
            float angular_resolution,
            float max_angle_width, 
            float max_angle_height,
            CoordinateFrame2 coordinate_frame,
            float noise_level,
            float min_range,
            int border_size)
        
        # brief Create the depth image from a point cloud, using the average viewpoint of the points 
        # (vp_x,vp_y,vp_z in the point type) in the point cloud as sensor pose (assuming a rotation of (0,0,0)).
        # param point_cloud the input point cloud
        # param angular_resolution_x the angular difference (in radians) between the
        #                            individual pixels in the image in the x-direction
        # param angular_resolution_y the angular difference (in radians) between the
        #                            individual pixels in the image in the y-direction
        # param max_angle_width an angle (in radians) defining the horizontal bounds of the sensor
        # param max_angle_height an angle (in radians) defining the vertical bounds of the sensor
        # param coordinate_frame the coordinate frame (defaults to CAMERA_FRAME)
        # param noise_level - The distance in meters inside of which the z-buffer will not use the minimum,
        #                     but the mean of the points. If 0.0 it is equivalent to a normal z-buffer and
        #                     will always take the minimum per cell.
        # param min_range the minimum visible range (defaults to 0)
        # param border_size the border size (defaults to 0)
        # note If wrong_coordinate_system is true, the sensor pose will be rotated to change from a coordinate frame
        # with x to the front, y to the left and z to the top to the coordinate frame we use here (x to the right, y 
        # to the bottom and z to the front)
        # template <typename PointCloudTypeWithViewpoints> 
        # void createFromPointCloudWithViewpoints (const PointCloudTypeWithViewpoints& point_cloud,
        #                                   float angular_resolution_x, float angular_resolution_y,
        #                                   float max_angle_width, float max_angle_height,
        #                                   CoordinateFrame coordinate_frame=CAMERA_FRAME, float noise_level=0.0f,
        #                                   float min_range=0.0f, int border_size=0);
        ##
        void createFromPointCloudWithViewpoints (
            const cpp.PointCloud_PointWithViewpoint_t& point_cloud,
            float angular_resolution_x,
            float angular_resolution_y,
            float max_angle_width,
            float max_angle_height,
            CoordinateFrame2 coordinate_frame,
            float noise_level,
            float min_range, 
            int border_size)
        
        # brief Create an empty depth image (filled with unobserved points)
        # param[in] angular_resolution the angle (in radians) between each sample in the depth image
        # param[in] sensor_pose an affine matrix defining the pose of the sensor (defaults to  Eigen::Affine3f::Identity () )
        # param[in] coordinate_frame the coordinate frame (defaults to CAMERA_FRAME)
        # param[in] angle_width an angle (in radians) defining the horizontal bounds of the sensor (defaults to 2*pi (360deg))
        # param[in] angle_height an angle (in radians) defining the vertical bounds of the sensor (defaults to pi (180deg))
        # void createEmpty (float angular_resolution, const Eigen::Affine3f& sensor_pose=Eigen::Affine3f::Identity (),
        #            RangeImage::CoordinateFrame coordinate_frame=CAMERA_FRAME, float angle_width=pcl::deg2rad (360.0f),
        #            float angle_height=pcl::deg2rad (180.0f));
        ##
        void createEmpty (
            float angular_resolution, 
            const eigen3.Affine3f& sensor_pose,
            CoordinateFrame2 coordinate_frame,
            float angle_width,
            float angle_height)
        
        # brief Create an empty depth image (filled with unobserved points)
        # param angular_resolution_x the angular difference (in radians) between the
        #                            individual pixels in the image in the x-direction
        # param angular_resolution_y the angular difference (in radians) between the
        #                            individual pixels in the image in the y-direction
        # param[in] sensor_pose an affine matrix defining the pose of the sensor (defaults to  Eigen::Affine3f::Identity () )
        # param[in] coordinate_frame the coordinate frame (defaults to CAMERA_FRAME)
        # param[in] angle_width an angle (in radians) defining the horizontal bounds of the sensor (defaults to 2*pi (360deg))
        # param[in] angle_height an angle (in radians) defining the vertical bounds of the sensor (defaults to pi (180deg))
        # 
        # void createEmpty (float angular_resolution_x, float angular_resolution_y,
        #            const Eigen::Affine3f& sensor_pose=Eigen::Affine3f::Identity (),
        #            RangeImage::CoordinateFrame coordinate_frame=CAMERA_FRAME, float angle_width=pcl::deg2rad (360.0f),
        #            float angle_height=pcl::deg2rad (180.0f));
        ##
        void createEmpty (
            float angular_resolution_x, 
            float angular_resolution_y,
            const eigen3.Affine3f& sensor_pose,
            CoordinateFrame2 coordinate_frame,
            float angle_width,
            float angle_height)
        
        # brief Integrate the given point cloud into the current range image using a z-buffer
        # param point_cloud the input point cloud
        # param noise_level - The distance in meters inside of which the z-buffer will not use the minimum,
        #                     but the mean of the points. If 0.0 it is equivalent to a normal z-buffer and
        #                     will always take the minimum per cell.
        # param min_range the minimum visible range
        # param top    returns the minimum y pixel position in the image where a point was added
        # param right  returns the maximum x pixel position in the image where a point was added
        # param bottom returns the maximum y pixel position in the image where a point was added
        # param top returns the minimum y position in the image where a point was added
        # param left   returns the minimum x pixel position in the image where a point was added
        # 
        # template <typename PointCloudType> void
        # doZBuffer (const PointCloudType& point_cloud, float noise_level,
        #            float min_range, int& top, int& right, int& bottom, int& left);
        ##
        void doZBuffer [PointCloudType](
            cpp.PointCloud[PointCloudType]& point_cloud,
            float noise_level,
            float min_range, 
            int& top, 
            int& right, 
            int& bottom,
            int& left)
        
        # brief Integrates the given far range measurements into the range image */
        # PCL_EXPORTS void integrateFarRanges (const PointCloud<PointWithViewpoint>& far_ranges);
        # integrateFarRanges (const cpp.PointCloud_PointWithViewpoint_t far_ranges)
        # integrateFarRanges (const cpp.PointCloud_PointWithViewpoint_Ptr_t &far_ranges)
        void integrateFarRanges (const cpp.PointCloud_PointWithViewpoint_t &far_ranges)
        
        # brief Cut the range image to the minimal size so that it still contains all actual range readings.
        # param border_size allows increase from the minimal size by the specified number of pixels (defaults to 0)
        # param top if positive, this value overrides the position of the top edge (defaults to -1)
        # param right if positive, this value overrides the position of the right edge (defaults to -1)
        # param bottom if positive, this value overrides the position of the bottom edge (defaults to -1)
        # param left if positive, this value overrides the position of the left edge (defaults to -1)
        # 
        # PCL_EXPORTS void cropImage (int border_size=0, int top=-1, int right=-1, int bottom=-1, int left=-1);
        void cropImage (int border_size, int top, int right, int bottom, int left)
        
        # brief Get all the range values in one float array of size width*height  
        # return a pointer to a new float array containing the range values
        # note This method allocates a new float array; the caller is responsible for freeing this memory.
        # PCL_EXPORTS float* getRangesArray () const;
        float* getRangesArray ()
        
        # Getter for the transformation from the world system into the range image system
        # (the sensor coordinate frame)
        # inline const Eigen::Affine3f& getTransformationToRangeImageSystem () const { return (to_range_image_system_); }
        const eigen3.Affine3f& getTransformationToRangeImageSystem ()
        
        # Setter for the transformation from the range image system
        # (the sensor coordinate frame) into the world system
        # inline void setTransformationToRangeImageSystem (const Eigen::Affine3f& to_range_image_system);
        void setTransformationToRangeImageSystem (eigen3.Affine3f& to_range_image_system)
        
        # Getter for the transformation from the range image system
        # (the sensor coordinate frame) into the world system
        # inline const Eigen::Affine3f& getTransformationToWorldSystem () const { return to_world_system_;}
        const eigen3.Affine3f& getTransformationToWorldSystem ()
        
        # Getter for the angular resolution of the range image in x direction in radians per pixel.
        # Provided for downwards compatability */
        # inline float getAngularResolution () const { return angular_resolution_x_;}
        float getAngularResolution ()
        
        # Getter for the angular resolution of the range image in x direction in radians per pixel.
        # inline float getAngularResolutionX () const { return angular_resolution_x_;}
        float getAngularResolutionX ()
        
        # Getter for the angular resolution of the range image in y direction in radians per pixel.
        # inline float getAngularResolutionY () const { return angular_resolution_y_;}
        float getAngularResolutionY ()
        
        # Getter for the angular resolution of the range image in x and y direction (in radians).
        # inline void getAngularResolution (float& angular_resolution_x, float& angular_resolution_y) const;
        void getAngularResolution (float& angular_resolution_x, float& angular_resolution_y)
        
        # brief Set the angular resolution of the range image
        # param angular_resolution the new angular resolution in x and y direction (in radians per pixel)
        # inline void setAngularResolution (float angular_resolution);
        void setAngularResolution (float angular_resolution)
        
        # brief Set the angular resolution of the range image
        # param angular_resolution_x the new angular resolution in x direction (in radians per pixel)
        # param angular_resolution_y the new angular resolution in y direction (in radians per pixel)
        # inline void setAngularResolution (float angular_resolution_x, float angular_resolution_y)
        void setAngularResolution (float angular_resolution_x, float angular_resolution_y)
        
        # brief Return the 3D point with range at the given image position
        # param image_x the x coordinate
        # param image_y the y coordinate
        # return the point at the specified location (returns unobserved_point if outside of the image bounds)
        # inline const PointWithRange& getPoint (int image_x, int image_y) const;
        const cpp.PointWithRange& getPoint (int image_x, int image_y)
        
        # brief Non-const-version of getPoint
        # inline PointWithRange& getPoint (int image_x, int image_y);
        
        # Return the 3d point with range at the given image position
        # inline const PointWithRange& getPoint (float image_x, float image_y) const;
        const cpp.PointWithRange& getPoint (float image_x, float image_y)
        
        # Non-const-version of the above
        # inline PointWithRange& getPoint (float image_x, float image_y);
        
        # brief Return the 3D point with range at the given image position.  This methd performs no error checking
        # to make sure the specified image position is inside of the image!
        # param image_x the x coordinate
        # param image_y the y coordinate
        # return the point at the specified location (program may fail if the location is outside of the image bounds)
        # inline const PointWithRange& getPointNoCheck (int image_x, int image_y) const;
        const cpp.PointWithRange& getPointNoCheck (int image_x, int image_y)
        
        # Non-const-version of getPointNoCheck
        # inline PointWithRange& getPointNoCheck (int image_x, int image_y);
        
        # Same as above
        # inline void getPoint (int image_x, int image_y, Eigen::Vector3f& point) const;
        
        # Same as above
        # inline void getPoint (int index, Eigen::Vector3f& point) const;
        
        # Same as above
        # inline const Eigen::Map<const Eigen::Vector3f>
        # getEigenVector3f (int x, int y) const;
        
        # Same as above
        # inline const Eigen::Map<const Eigen::Vector3f>
        # getEigenVector3f (int index) const;
        
        # Return the 3d point with range at the given index (whereas index=y*width+x)
        # inline const PointWithRange& getPoint (int index) const;
        const cpp.PointWithRange& getPoint (int index)
        
        # Calculate the 3D point according to the given image point and range
        # inline void calculate3DPoint (float image_x, float image_y, float range, PointWithRange& point) const;
        void calculate3DPoint (float image_x, float image_y, float range, cpp.PointWithRange& point)
        
        # Calculate the 3D point according to the given image point and the range value at the closest pixel
        # inline void calculate3DPoint (float image_x, float image_y, PointWithRange& point) const;
        inline void calculate3DPoint (float image_x, float image_y, cpp.PointWithRange& point)
        
        # Calculate the 3D point according to the given image point and range
        # virtual inline void calculate3DPoint (float image_x, float image_y, float range, Eigen::Vector3f& point) const;
        
        # Calculate the 3D point according to the given image point and the range value at the closest pixel
        # inline void calculate3DPoint (float image_x, float image_y, Eigen::Vector3f& point) const;
        void calculate3DPoint (float image_x, float image_y, eigen3.Vector3f& point)
        
        # Recalculate all 3D point positions according to their pixel position and range
        # PCL_EXPORTS void recalculate3DPointPositions ();
        void recalculate3DPointPositions ()
        
        # Get imagePoint from 3D point in world coordinates
        # inline virtual void getImagePoint (const Eigen::Vector3f& point, float& image_x, float& image_y, float& range) const;
        # void getImagePoint (const Eigen::Vector3f& point, float& image_x, float& image_y, float& range)
        
        # Same as above
        # inline void getImagePoint (const Eigen::Vector3f& point, int& image_x, int& image_y, float& range) const;
        void getImagePoint (const eigen3.Vector3f& point, int& image_x, int& image_y, float& range)
        
        # Same as above
        # inline void getImagePoint (const Eigen::Vector3f& point, float& image_x, float& image_y) const;
        void getImagePoint (const eigen3.Vector3f& point, float& image_x, float& image_y)
        
        # Same as above
        # inline void getImagePoint (const Eigen::Vector3f& point, int& image_x, int& image_y) const;
        void getImagePoint (const eigen3.Vector3f& point, int& image_x, int& image_y)
        
        # Same as above
        # inline void getImagePoint (float x, float y, float z, float& image_x, float& image_y, float& range) const;
        void getImagePoint (float x, float y, float z, float& image_x, float& image_y, float& range)
        
        # Same as above
        # inline void getImagePoint (float x, float y, float z, float& image_x, float& image_y) const;
        void getImagePoint (float x, float y, float z, float& image_x, float& image_y)
        
        # Same as above
        # inline void getImagePoint (float x, float y, float z, int& image_x, int& image_y) const;
        void getImagePoint (float x, float y, float z, int& image_x, int& image_y)
        
        # point_in_image will be the point in the image at the position the given point would be. Returns
        # the range of the given point.
        # inline float checkPoint (const Eigen::Vector3f& point, PointWithRange& point_in_image) const;
        float checkPoint (const eigen3.Vector3f& point, cpp.PointWithRange& point_in_image)
        
        # Returns the difference in range between the given point and the range of the point in the image
        # at the position the given point would be.
        # (Return value is point_in_image.range-given_point.range)
        # inline float getRangeDifference (const Eigen::Vector3f& point) const;
        float getRangeDifference (const eigen3.Vector3f& point)
        
        # Get the image point corresponding to the given angles
        # inline void getImagePointFromAngles (float angle_x, float angle_y, float& image_x, float& image_y) const;
        void getImagePointFromAngles (float angle_x, float angle_y, float& image_x, float& image_y)
        
        # Get the angles corresponding to the given image point
        # inline void getAnglesFromImagePoint (float image_x, float image_y, float& angle_x, float& angle_y) const;
        void getAnglesFromImagePoint (float image_x, float image_y, float& angle_x, float& angle_y)
        
        # Transforms an image point in float values to an image point in int values
        # inline void real2DToInt2D (float x, float y, int& xInt, int& yInt) const;
        void real2DToInt2D (float x, float y, int& xInt, int& yInt)
        
        # Check if a point is inside of the image
        # inline bool isInImage (int x, int y) const;
        bool isInImage (int x, int y)
        
        # Check if a point is inside of the image and has a finite range
        # inline bool isValid (int x, int y) const;
        bool isValid (int x, int y)
        
        # Check if a point has a finite range
        # inline bool isValid (int index) const;
        bool isValid (int index)
        
        # Check if a point is inside of the image and has either a finite range or a max reading (range=INFINITY)
        # inline bool isObserved (int x, int y) const;
        bool isObserved (int x, int y)
        
        # Check if a point is a max range (range=INFINITY) - please check isInImage or isObserved first!
        # inline bool isMaxRange (int x, int y) const;
        bool isMaxRange (int x, int y)
        
        # Calculate the normal of an image point using the neighbors with a maximum pixel distance of radius.
        # step_size determines how many pixels are used. 1 means all, 2 only every second, etc..
        # Returns false if it was unable to calculate a normal.
        # inline bool getNormal (int x, int y, int radius, Eigen::Vector3f& normal, int step_size=1) const;
        bool getNormal (int x, int y, int radius, eigen3.Vector3f& normal, int step_size)
        
        # Same as above, but only the no_of_nearest_neighbors points closest to the given point are considered.
        # inline bool getNormalForClosestNeighbors (int x, int y, int radius, const PointWithRange& point, int no_of_nearest_neighbors, Eigen::Vector3f& normal, int step_size=1) const;
        bool getNormalForClosestNeighbors (int x, int y, int radius, const cpp.PointWithRange& point,
                                           int no_of_nearest_neighbors, eigen3.Vector3f& normal, int step_size)
        
        # Same as above
        # inline bool getNormalForClosestNeighbors (int x, int y, int radius, const Eigen::Vector3f& point, int no_of_nearest_neighbors, Eigen::Vector3f& normal, Eigen::Vector3f* point_on_plane=NULL, int step_size=1) const;
        bool getNormalForClosestNeighbors (int x, int y, int radius, const eigen3.Vector3f& point, int no_of_nearest_neighbors, eigen3.Vector3f& normal, eigen3.Vector3f* point_on_plane, int step_size)
        
        # Same as above, using default values
        # inline bool getNormalForClosestNeighbors (int x, int y, Eigen::Vector3f& normal, int radius=2) const;
        bool getNormalForClosestNeighbors (int x, int y, eigen3.Vector3f& normal, int radius)
        
        # Same as above but extracts some more data and can also return the extracted
        # information for all neighbors in radius if normal_all_neighbors is not NULL
        # inline bool getSurfaceInformation (int x, int y, int radius, const Eigen::Vector3f& point,
        #                        int no_of_closest_neighbors, int step_size,
        #                        float& max_closest_neighbor_distance_squared,
        #                        Eigen::Vector3f& normal, Eigen::Vector3f& mean, Eigen::Vector3f& eigen_values,
        #                        Eigen::Vector3f* normal_all_neighbors=NULL,
        #                        Eigen::Vector3f* mean_all_neighbors=NULL,
        #                        Eigen::Vector3f* eigen_values_all_neighbors=NULL) const;
        ##
        bool getSurfaceInformation (
            int x,
            int y,
            int radius,
            const eigen3.Vector3f& point,
            int no_of_closest_neighbors,
            int step_size,
            float& max_closest_neighbor_distance_squared,
            eigen3.Vector3f& normal, 
            eigen3.Vector3f& mean,
            eigen3.Vector3f& eigen_values,
            eigen3.Vector3f* normal_all_neighbors,
            eigen3.Vector3f* mean_all_neighbors,
            eigen3.Vector3f* eigen_values_all_neighbors) const;
        
        # // Return the squared distance to the n-th neighbors of the point at x,y
        # inline float getSquaredDistanceOfNthNeighbor (int x, int y, int radius, int n, int step_size) const;
        float getSquaredDistanceOfNthNeighbor (
            int x, int y,
            int radius,
            int n, 
            int step_size)
        
        # /** Calculate the impact angle based on the sensor position and the two given points - will return
        #  * -INFINITY if one of the points is unobserved */
        # inline float getImpactAngle (const PointWithRange& point1, const PointWithRange& point2) const;
        float getImpactAngle (
            const cpp.PointWithRange& point1, 
            const cpp.PointWithRange& point2)
        
        # Same as above
        # inline float getImpactAngle (int x1, int y1, int x2, int y2) const;
        float getImpactAngle (int x1, int y1, int x2, int y2)
        
        # /** Extract a local normal (with a heuristic not to include background points) and calculate the impact
        #  *  angle based on this */
        # inline float getImpactAngleBasedOnLocalNormal (int x, int y, int radius) const;
        float getImpactAngleBasedOnLocalNormal (int x, int y, int radius)
        
        # /** Uses the above function for every point in the image */
        # PCL_EXPORTS float* getImpactAngleImageBasedOnLocalNormals (int radius) const;
        float* getImpactAngleImageBasedOnLocalNormals (int radius)
        
        # /** Calculate a score [0,1] that tells how acute the impact angle is (1.0f - getImpactAngle/90deg)
        #  *  This uses getImpactAngleBasedOnLocalNormal
        #  *  Will return -INFINITY if no normal could be calculated */
        # inline float getNormalBasedAcutenessValue (int x, int y, int radius) const;
        float getNormalBasedAcutenessValue (int x, int y, int radius)
        
        # /** Calculate a score [0,1] that tells how acute the impact angle is (1.0f - getImpactAngle/90deg)
        #  *  will return -INFINITY if one of the points is unobserved */
        # inline float getAcutenessValue (const PointWithRange& point1, const PointWithRange& point2) const;
        float getAcutenessValue (const cpp.PointWithRange& point1, const cpp.PointWithRange& point2)
        
        # Same as above
        # inline float getAcutenessValue (int x1, int y1, int x2, int y2) const;
        float getAcutenessValue (int x1, int y1, int x2, int y2)
        
        # /** Calculate getAcutenessValue for every point */
        # PCL_EXPORTS void getAcutenessValueImages (int pixel_distance, float*& acuteness_value_image_x, float*& acuteness_value_image_y) const;
        void getAcutenessValueImages (
            int pixel_distance,
            float*& acuteness_value_image_x,
            float*& acuteness_value_image_y)
        
        # Calculates, how much the surface changes at a point. Pi meaning a flat suface and 0.0f would be a needle point
        # //inline float
        # //  getSurfaceChange (const PointWithRange& point, const PointWithRange& neighbor1,
        # //                   const PointWithRange& neighbor2) const;
        
        # Calculates, how much the surface changes at a point. 1 meaning a 90deg angle and 0 a flat suface
        # PCL_EXPORTS float getSurfaceChange (int x, int y, int radius) const;
        float getSurfaceChange (int x, int y, int radius)
        
        # Uses the above function for every point in the image
        # PCL_EXPORTS float* getSurfaceChangeImage (int radius) const;
        float* getSurfaceChangeImage (int radius)
        
        # Calculates, how much the surface changes at a point. Returns an angle [0.0f, PI] for x and y direction.
        # A return value of -INFINITY means that a point was unobserved.
        # inline void getSurfaceAngleChange (int x, int y, int radius, float& angle_change_x, float& angle_change_y) const;
        void getSurfaceAngleChange (int x, int y, int radius, float& angle_change_x, float& angle_change_y)
        
        # Uses the above function for every point in the image
        # PCL_EXPORTS void getSurfaceAngleChangeImages (int radius, float*& angle_change_image_x, float*& angle_change_image_y) const;
        void getSurfaceAngleChangeImages (int radius, float*& angle_change_image_x, float*& angle_change_image_y)
        
        # Calculates the curvature in a point using pca
        # inline float getCurvature (int x, int y, int radius, int step_size) const;
        float getCurvature (int x, int y, int radius, int step_size)
        
        # Get the sensor position
        # inline const Eigen::Vector3f getSensorPos () const;
        eigen3.Vector3f getSensorPos ()
        
        # Sets all -INFINITY values to INFINITY
        # PCL_EXPORTS void setUnseenToMaxRange ();
        void setUnseenToMaxRange ()
        
        # Getter for image_offset_x_
        # inline int getImageOffsetX () const
        # Getter for image_offset_y_
        # inline int getImageOffsetY () const
        int getImageOffsetX ()
        int getImageOffsetY ()
        
        # Setter for image offsets
        # inline void setImageOffsets (int offset_x, int offset_y)
        # Get a sub part of the complete image as a new range image.
        # param sub_image_image_offset_x - The x coordinate of the top left pixel of the sub image.
        #                        This is always according to absolute 0,0 meaning -180,-90
        #                        and it is already in the system of the new image, so the
        #                        actual pixel used in the original image is
        #                        combine_pixels* (image_offset_x-image_offset_x_)
        # param sub_image_image_offset_y - Same as image_offset_x for the y coordinate
        # param sub_image_width - width of the new image
        # param sub_image_height - height of the new image
        # param combine_pixels - shrinking factor, meaning the new angular resolution
        #                        is combine_pixels times the old one
        # param sub_image - the output image
        # virtual void getSubImage (int sub_image_image_offset_x, int sub_image_image_offset_y, int sub_image_width, int sub_image_height, int combine_pixels, RangeImage& sub_image) const;
        # NG(LNK2001)
        # void getSubImage (int sub_image_image_offset_x, int sub_image_image_offset_y, int sub_image_width, int sub_image_height, int combine_pixels, RangeImage& sub_image)
        
        # Get a range image with half the resolution
        # virtual void getHalfImage (RangeImage& half_image) const;
        # NG(LNK2001)
        # void getHalfImage (RangeImage& half_image)
        
        # Find the minimum and maximum range in the image
        # PCL_EXPORTS void getMinMaxRanges (float& min_range, float& max_range) const;
        void getMinMaxRanges (float& min_range, float& max_range)
        
        # This function sets the sensor pose to 0 and transforms all point positions to this local coordinate frame
        # PCL_EXPORTS void change3dPointsToLocalCoordinateFrame ();
        void change3dPointsToLocalCoordinateFrame ()
        
        # /** Calculate a range patch as the z values of the coordinate frame given by pose.
        #  *  The patch will have size pixel_size x pixel_size and each pixel
        #  *  covers world_size/pixel_size meters in the world
        #  *  You are responsible for deleting the structure afterwards! */
        # PCL_EXPORTS float* getInterpolatedSurfaceProjection (const Eigen::Affine3f& pose, int pixel_size, float world_size) const;
        float* getInterpolatedSurfaceProjection (const eigen3.Affine3f& pose, int pixel_size, float world_size)
        
        # Same as above, but using the local coordinate frame defined by point and the viewing direction
        # PCL_EXPORTS float* getInterpolatedSurfaceProjection (const Eigen::Vector3f& point, int pixel_size, float world_size) const;
        float* getInterpolatedSurfaceProjection (const eigen3.Vector3f& point, int pixel_size, float world_size)
        
        # Get the local coordinate frame with 0,0,0 in point, upright and Z as the viewing direction
        # inline Eigen::Affine3f getTransformationToViewerCoordinateFrame (const Eigen::Vector3f& point) const;
        eigen3.Affine3f getTransformationToViewerCoordinateFrame (const eigen3.Vector3f& point)
        
        # Same as above, using a reference for the retrurn value
        # inline void getTransformationToViewerCoordinateFrame (const Eigen::Vector3f& point, Eigen::Affine3f& transformation) const;
        void getTransformationToViewerCoordinateFrame (const eigen3.Vector3f& point, eigen3.Affine3f& transformation)
        
        # Same as above, but only returning the rotation
        # inline void getRotationToViewerCoordinateFrame (const Eigen::Vector3f& point, Eigen::Affine3f& transformation) const;
        void getRotationToViewerCoordinateFrame (const eigen3.Vector3f& point, eigen3.Affine3f& transformation)
        
        # Get a local coordinate frame at the given point based on the normal.
        # PCL_EXPORTS bool getNormalBasedUprightTransformation (const Eigen::Vector3f& point, float max_dist, Eigen::Affine3f& transformation) const;
        bool getNormalBasedUprightTransformation (const eigen3.Vector3f& point, float max_dist, eigen3.Affine3f& transformation)
        
        # Get the integral image of the range values (used for fast blur operations).
        # You are responsible for deleting it after usage!
        # PCL_EXPORTS void getIntegralImage (float*& integral_image, int*& valid_points_num_image) const;
        # void getIntegralImage (float*& integral_image, int*& valid_points_num_image)
        
        # /** Get a blurred version of the range image using box filters on the provided integral image*/
        # PCL_EXPORTS void getBlurredImageUsingIntegralImage (int blur_radius, float* integral_image, int* valid_points_num_image, RangeImage& range_image) const;
        # void getBlurredImageUsingIntegralImage (int blur_radius, float* integral_image, int* valid_points_num_image, RangeImage& range_image)
        
        # /** Get a blurred version of the range image using box filters */
        # PCL_EXPORTS void getBlurredImage (int blur_radius, RangeImage& range_image) const;
        # void getBlurredImage (int blur_radius, RangeImage& range_image)
        
        # /** Get the squared euclidean distance between the two image points.
        #  *  Returns -INFINITY if one of the points was not observed */
        # inline float getEuclideanDistanceSquared (int x1, int y1, int x2, int y2) const;
        # float getEuclideanDistanceSquared (int x1, int y1, int x2, int y2)
        
        # Doing the above for some steps in the given direction and averaging
        # inline float getAverageEuclideanDistance (int x, int y, int offset_x, int offset_y, int max_steps) const;
        float getAverageEuclideanDistance (int x, int y, int offset_x, int offset_y, int max_steps)
        
        # Project all points on the local plane approximation, thereby smoothing the surface of the scan
        # PCL_EXPORTS void getRangeImageWithSmoothedSurface (int radius, RangeImage& smoothed_range_image) const;
        # void getRangeImageWithSmoothedSurface (int radius, RangeImage& smoothed_range_image)
        
        # //void getLocalNormals (int radius) const;
        
        # /** Calculates the average 3D position of the no_of_points points described by the start
        #  *  point x,y in the direction delta.
        #  *  Returns a max range point (range=INFINITY) if the first point is max range and an
        #  *  unobserved point (range=-INFINITY) if non of the points is observed. */
        # inline void get1dPointAverage (int x, int y, int delta_x, int delta_y, int no_of_points, PointWithRange& average_point) const;
        void get1dPointAverage (int x, int y, int delta_x, int delta_y, int no_of_points, cpp.PointWithRange& average_point)
        
        # /** Calculates the overlap of two range images given the relative transformation
        #  *  (from the given image to *this) */
        # PCL_EXPORTS float getOverlap (const RangeImage& other_range_image, const Eigen::Affine3f& relative_transformation, int search_radius, float max_distance, int pixel_step=1) const;
        
        # /** Get the viewing direction for the given point */
        # inline bool getViewingDirection (int x, int y, Eigen::Vector3f& viewing_direction) const;
        # bool getViewingDirection (int x, int y, Eigen::Vector3f& viewing_direction) const;
        
        # /** Get the viewing direction for the given point */
        # inline void getViewingDirection (const Eigen::Vector3f& point, Eigen::Vector3f& viewing_direction) const;
        # void getViewingDirection (const Eigen::Vector3f& point, Eigen::Vector3f& viewing_direction) const;
        
        # /** Return a newly created Range image.
        #  *  Can be reimplmented in derived classes like RangeImagePlanar to return an image of the same type. */
        # virtual RangeImage* getNew () const { return new RangeImage; }
        
        # // =====MEMBER VARIABLES=====
        # // BaseClass:
        # //   roslib::Header header;
        # //   std::vector<PointT> points;
        # //   uint32_t width;
        # //   uint32_t height;
        # //   bool is_dense;
        # static bool debug; /**< Just for... well... debugging purposes. :-) */


ctypedef RangeImage RangeImage_t
ctypedef shared_ptr[RangeImage] RangeImagePtr_t
ctypedef shared_ptr[const RangeImage] RangeImageConstPtr_t
###

# range_image_planar.h
# class RangeImagePlanar : public RangeImage
cdef extern from "pcl/range_image/range_image_planar.h" namespace "pcl":
    cdef cppclass RangeImagePlanar(RangeImage):
        RangeImagePlanar()
        # public:
        # // =====TYPEDEFS=====
        # typedef RangeImage BaseClass;
        # typedef boost::shared_ptr<RangeImagePlanar> Ptr;
        # typedef boost::shared_ptr<const RangeImagePlanar> ConstPtr;
        # // =====CONSTRUCTOR & DESTRUCTOR=====
        # /** Constructor */
        # PCL_EXPORTS RangeImagePlanar ();
        # /** Destructor */
        # PCL_EXPORTS ~RangeImagePlanar ();
        # /** Return a newly created RangeImagePlanar.
        #  *  Reimplmentation to return an image of the same type. */
        # virtual RangeImage*  getNew () const { return new RangeImagePlanar; }
        
        # // =====PUBLIC METHODS=====
        # brief Get a boost shared pointer of a copy of this
        # inline Ptr makeShared () { return Ptr (new RangeImagePlanar (*this)); } 
        
        # brief Create the image from an existing disparity image.
        # param disparity_image the input disparity image data
        # param di_width the disparity image width
        # param di_height the disparity image height
        # param focal_length the focal length of the primary camera that generated the disparity image
        # param base_line the baseline of the stereo pair that generated the disparity image
        # param desired_angular_resolution If this is set, the system will skip as many pixels as necessary to get as
        #        close to this angular resolution as possible while not going over this value (the density will not be
        #        lower than this value). The value is in radians per pixel. 
        # 
        # PCL_EXPORTS void setDisparityImage (const float* disparity_image, int di_width, int di_height, float focal_length, float base_line, float desired_angular_resolution=-1);
        ##
        
        # Create the image from an existing depth image.
        # param depth_image the input depth image data as float values
        # param di_width the disparity image width 
        # param di_height the disparity image height
        # param di_center_x the x-coordinate of the camera's center of projection
        # param di_center_y the y-coordinate of the camera's center of projection
        # param di_focal_length_x the camera's focal length in the horizontal direction
        # param di_focal_length_y the camera's focal length in the vertical direction
        # param desired_angular_resolution If this is set, the system will skip as many pixels as necessary to get as
        #        close to this angular resolution as possible while not going over this value (the density will not be
        #        lower than this value). The value is in radians per pixel.
        # 
        # PCL_EXPORTS void
        # setDepthImage (const float* depth_image, int di_width, int di_height, float di_center_x, float di_center_y,
        #                float di_focal_length_x, float di_focal_length_y, float desired_angular_resolution=-1);
        ##
        
        # Create the image from an existing depth image.
        # param depth_image the input disparity image data as short values describing millimeters
        # param di_width the disparity image width 
        # param di_height the disparity image height
        # param di_center_x the x-coordinate of the camera's center of projection
        # param di_center_y the y-coordinate of the camera's center of projection
        # param di_focal_length_x the camera's focal length in the horizontal direction
        # param di_focal_length_y the camera's focal length in the vertical direction
        # param desired_angular_resolution If this is set, the system will skip as many pixels as necessary to get as
        #        close to this angular resolution as possible while not going over this value (the density will not be
        #        lower than this value). The value is in radians per pixel.
        # 
        # PCL_EXPORTS void
        # setDepthImage (const unsigned short* depth_image, int di_width, int di_height, float di_center_x, float di_center_y,
        #                float di_focal_length_x, float di_focal_length_y, float desired_angular_resolution=-1);
        ##
        
        # Create the image from an existing point cloud.
        # param point_cloud the source point cloud
        # param di_width the disparity image width 
        # param di_height the disparity image height
        # param di_center_x the x-coordinate of the camera's center of projection
        # param di_center_y the y-coordinate of the camera's center of projection
        # param di_focal_length_x the camera's focal length in the horizontal direction
        # param di_focal_length_y the camera's focal length in the vertical direction
        # param sensor_pose the pose of the virtual depth camera
        # param coordinate_frame the used coordinate frame of the point cloud
        # param noise_level what is the typical noise of the sensor - is used for averaging in the z-buffer
        # param min_range minimum range to consifder points
        # 
        # template <typename PointCloudType> void
        # createFromPointCloudWithFixedSize (const PointCloudType& point_cloud,
        #                                    int di_width, int di_height, float di_center_x, float di_center_y,
        #                                    float di_focal_length_x, float di_focal_length_y,
        #                                    const Eigen::Affine3f& sensor_pose,
        #                                    CoordinateFrame coordinate_frame=CAMERA_FRAME, float noise_level=0.0f,
        #                                    float min_range=0.0f);
        ##
        
        # // Since we reimplement some of these overloaded functions, we have to do the following:
        # using RangeImage::calculate3DPoint;
        # using RangeImage::getImagePoint;
        
        # brief Calculate the 3D point according to the given image point and range
        # param image_x the x image position
        # param image_y the y image position
        # param range the range
        # param point the resulting 3D point
        # note Implementation according to planar range images (compared to spherical as in the original)
        # 
        # virtual inline void calculate3DPoint (float image_x, float image_y, float range, Eigen::Vector3f& point) const;
        ##
        
        # brief Calculate the image point and range from the given 3D point
        # param point the resulting 3D point
        # param image_x the resulting x image position
        # param image_y the resulting y image position
        # param range the resulting range
        # note Implementation according to planar range images (compared to spherical as in the original)
        # 
        # virtual inline void  getImagePoint (const Eigen::Vector3f& point, float& image_x, float& image_y, float& range) const;
        ##
        
        # Get a sub part of the complete image as a new range image.
        # param sub_image_image_offset_x - The x coordinate of the top left pixel of the sub image.
        #                         This is always according to absolute 0,0 meaning(-180, -90)
        #                         and it is already in the system of the new image, so the
        #                         actual pixel used in the original image is
        #                         combine_pixels* (image_offset_x-image_offset_x_)
        # param sub_image_image_offset_y - Same as image_offset_x for the y coordinate
        # param sub_image_width - width of the new image
        # param sub_image_height - height of the new image
        # param combine_pixels - shrinking factor, meaning the new angular resolution
        #                        is combine_pixels times the old one
        # param sub_image - the output image
        # 
        # PCL_EXPORTS virtual void
        # getSubImage (int sub_image_image_offset_x, int sub_image_image_offset_y, int sub_image_width,
        #              int sub_image_height, int combine_pixels, RangeImage& sub_image) const;
        ##
        
        # Get a range image with half the resolution
        # PCL_EXPORTS virtual void getHalfImage (RangeImage& half_image) const;


ctypedef RangeImagePlanar RangeImagePlanar_t
ctypedef shared_ptr[RangeImagePlanar] RangeImagePlanarPtr_t
ctypedef shared_ptr[const RangeImagePlanar] RangeImagePlanarConstPtr_t
###


###############################################################################
# Enum
###############################################################################

# enum CoordinateFrame
# CAMERA_FRAME = 0,
# LASER_FRAME = 1
cdef extern from "pcl/range_image/range_image.h" namespace "pcl":
    ctypedef enum CoordinateFrame2 "pcl::RangeImage::CoordinateFrame":
        COORDINATEFRAME_CAMERA "pcl::RangeImage::CAMERA_FRAME"
        COORDINATEFRAME_LASER "pcl::RangeImage::LASER_FRAME"


# bearing_angle_image.h
# namespace pcl
# /** \brief class BearingAngleImage is used as an interface to generate Bearing Angle(BA) image.
# * \author: Qinghua Li (qinghua__li@163.com)
# */
# class BearingAngleImage : public pcl::PointCloud<PointXYZRGBA>
        # public:
        # // ===== TYPEDEFS =====
        # typedef pcl::PointCloud<PointXYZRGBA> BaseClass;
        # 
        # // =====CONSTRUCTOR & DESTRUCTOR=====
        # /** Constructor */
        # BearingAngleImage ();
        # /** Destructor */
        # virtual ~BearingAngleImage ();
        # 
        # public:
        # /** \brief Reset all values to an empty Bearing Angle image */
        # void reset ();
        # 
        # /** \brief Calculate the angle between the laser beam and the segment joining two consecutive
        #  * measurement points.
        #  * \param point1
        #  * \param point2
        #  */
        # double getAngle (const PointXYZ &point1, const PointXYZ &point2);
        # 
        # /** \brief Transform 3D point cloud into a 2D Bearing Angle(BA) image */
        # void generateBAImage (PointCloud<PointXYZ>& point_cloud);


###

# range_image_spherical.h
# namespace pcl
# /** \brief @b RangeImageSpherical is derived from the original range image and uses a slightly different
# * spherical projection. In the original range image, the image will appear more and more
# * "scaled down" along the y axis, the further away from the mean line of the image a point is.
# * This class removes this scaling, which makes it especially suitable for spinning LIDAR sensors
# * that capure a 360 view, since a rotation of the sensor will now simply correspond to a shift of the
# * range image. (This class is similar to RangeImagePlanar, but changes less of the behaviour of the base class.)
# * \author Andreas Muetzel
# * \ingroup range_image
# */
# class RangeImageSpherical : public RangeImage
        # public:
        # // =====TYPEDEFS=====
        # typedef RangeImage BaseClass;
        # typedef boost::shared_ptr<RangeImageSpherical> Ptr;
        # typedef boost::shared_ptr<const RangeImageSpherical> ConstPtr;
        # 
        # // =====CONSTRUCTOR & DESTRUCTOR=====
        # /** Constructor */
        # PCL_EXPORTS RangeImageSpherical () {}
        # /** Destructor */
        # PCL_EXPORTS virtual ~RangeImageSpherical () {}
        # 
        # /** Return a newly created RangeImagePlanar.
        # *  Reimplmentation to return an image of the same type. */
        # virtual RangeImage* getNew () const { return new RangeImageSpherical; }
        # 
        # // =====PUBLIC METHODS=====
        # /** \brief Get a boost shared pointer of a copy of this */
        # inline Ptr makeShared () { return Ptr (new RangeImageSpherical (*this)); }
        # 
        # // Since we reimplement some of these overloaded functions, we have to do the following:
        # using RangeImage::calculate3DPoint;
        # using RangeImage::getImagePoint;
        # 
        # /** \brief Calculate the 3D point according to the given image point and range
        #  * \param image_x the x image position
        #  * \param image_y the y image position
        #  * \param range the range
        #  * \param point the resulting 3D point
        #  * \note Implementation according to planar range images (compared to spherical as in the original)
        #  */
        # virtual inline void calculate3DPoint (float image_x, float image_y, float range, Eigen::Vector3f& point) const;
        # 
        # /** \brief Calculate the image point and range from the given 3D point
        #  * \param point the resulting 3D point
        #  * \param image_x the resulting x image position
        #  * \param image_y the resulting y image position
        #  * \param range the resulting range
        #  * \note Implementation according to planar range images (compared to spherical as in the original)
        #  */
        # virtual inline void getImagePoint (const Eigen::Vector3f& point, float& image_x, float& image_y, float& range) const;
        # 
        # /** Get the angles corresponding to the given image point */
        # inline void getAnglesFromImagePoint (float image_x, float image_y, float& angle_x, float& angle_y) const;
        # 
        # /** Get the image point corresponding to the given ranges */
        # inline void getImagePointFromAngles (float angle_x, float angle_y, float& image_x, float& image_y) const;


###

