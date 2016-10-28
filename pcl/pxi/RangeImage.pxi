# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_range_image as pcl_r_img

cimport eigen as eigen3

cdef class RangeImage:
    """
    rangeImage
    """
    cdef pcl_r_img.RangeImage_t *me
    
    def __cinit__(self):
        self.me = new pcl_r_img.RangeImage_t()
    
    def __dealloc__(self):
        del self.me
    
    # def CreateFromPointCloud(
    #     self,
    #     PointCloud point_cloud,
    #     float angular_resolution,
    #     float max_angle_width,
    #     float max_angle_height,
    #     eigen3.Affine3f& sensor_pose,
    #     pcl_r_img.CoordinateFrame2 coordinate_frame,
    #     float noise_level,
    #     float min_range,
    #     int border_size):
    #     """
    #     """
    #     self.me.createFromPointCloud(
    #         <cpp.shared_ptr[cpp.PointCloud[cpp.PointXYZ]]> cloud.thisptr_shared,
    #         angular_resolution,
    #         max_angle_width,
    #         max_angle_height,
    #         sensor_pose, coordinate_frame, noise_level, min_range, border_size)
    # 
    #     return rangeImg
    
    def SetAngularResolution(self,
                             float angular_resolution_x, float angular_resolution_y):
        self.me.setAngularResolution(angular_resolution_x, angular_resolution_y)

###

