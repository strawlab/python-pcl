# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_range_image as pcl_r_img

cimport eigen as eigen3

cdef class RangeImage:
    """
    rangeImage
    """
    cdef pcl_r_img.RangeImage_t *me
    cdef cpp.PointCloudPtr_t point
    
    def __cinit__(self):
        self.me = new pcl_r_img.RangeImage_t()

    def __cinit__(self, PointCloud pc not None):
        self.me = new pcl_r_img.RangeImage_t()
        point = pc.thisptr_shared

    def __dealloc__(self):
        del self.me
    
    def CreateFromPointCloud(
        self,
        float angular_resolution,
        float max_angle_width,
        float max_angle_height,
        # eigen3.Affine3f sensor_pose,
        pcl_r_img.CoordinateFrame2 coordinate_frame,
        float noise_level,
        float min_range,
        int border_size):
        """
        """
        # 
        cdef eigen3.Affine3f sensor_pose
        cdef float *data = sensor_pose.data()
        data[0] = 0
        data[1] = 0
        data[2] = 0
        
        # cdef cpp.PointXYZ pointXYZ
        # pointXYZ.x = 0
        # pointXYZ.y = 0
        # pointXYZ.z = 0
        
        # self.me.createFromPointCloud(
        #     point.thisptr()[0],
        #     angular_resolution,
        #     max_angle_width,
        #     max_angle_height,
        #     sensor_pose, 
        #     coordinate_frame, 
        #     noise_level, 
        #     min_range, 
        #     border_size)
    
    def SetAngularResolution(self, float angular_resolution_x, float angular_resolution_y):
        self.me.setAngularResolution(angular_resolution_x, angular_resolution_y)
    
    # def IntegrateFarRanges(self, PointCloud_PointWithViewpoint viewpoint):
    #     self.me.integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t> viewpoint.thisptr()[0])
    # 
    # def IntegrateFarRanges(self, PointCloud_PointWithViewpoint viewpoint):
    #     self.me.integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_Ptr_t> viewpoint.thisptr()[0])

    def IntegrateFarRanges(self, PointCloud_PointWithViewpoint viewpoint):
        self.me.integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t&> viewpoint.thisptr()[0])
        # self.me.integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t&> deref(viewpoint.thisptr()))
    
###

