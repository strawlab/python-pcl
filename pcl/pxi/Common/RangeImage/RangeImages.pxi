# -*- coding: utf-8 -*-
from _pcl cimport PointCloud
from _pcl cimport PointCloudWrapper_PointWithViewpoint
cimport pcl_defs as cpp
cimport pcl_range_image as pcl_r_img

cimport eigen as eigen3
from boost_shared_ptr cimport sp_assign

# from cython.operator import dereference as deref
from cython.operator cimport dereference as deref, preincrement as inc

cdef class RangeImages:
    """
    rangeImage
    """
    # cdef pcl_r_img.RangeImage_t *me
    # cdef pcl_r_img.RangeImage_t *point
    # cdef cpp.PointCloud[cpp.PointWithRange] *point
    
    
    def __cinit__(self):
        # self.me = new pcl_r_img.RangeImage_t()
        # sp_assign(self.thisptr_shared, new pcl_r_img.RangeImage())
        # NG
        # self.thisptr_shared = <pcl_r_img.RangeImage *> new pcl_r_img.RangeImage()
        print('__cinit__')
    
    # def __cinit__(self, PointCloud pc not None):
    #     # self.me = new pcl_r_img.RangeImage_t()
    #     # self.point = pc.thisptr_shared
    #     sp_assign(self.thisptr_shared,  new pcl_r_img.RangeImage())
    
    
    def CreateFromPointCloud(self, PointCloud cloud, float angular_resolution, float max_angle_width, float max_angle_height, 
        pcl_r_img.CoordinateFrame2 coordinate_frame, float noise_level, float min_range, int border_size):
        
        print('CreateFromPointCloud enter')
        
        # cdef cpp.PointCloudPtr_t point
        # point = cloud.thisptr_shared
        # Eigen::Translation3f
        cdef eigen3.Affine3f sensor_pose
        cdef float *data = sensor_pose.data()
        # print('sensor_pose size = ' + str( len(data) ) )
        data[0] = 0.0
        data[1] = 0.0
        data[2] = 0.0
        data[3] = 0.0
        # print('sensor_pose = ' + data)
        
        print('angular_resolution = ' + str(angular_resolution) )
        print('max_angle_width = ' + str(max_angle_width) )
        print('max_angle_height = ' + str(max_angle_height) )
        print('noise_level = ' + str(noise_level) )
        print('min_range = ' + str(min_range) )
        print('border_size = ' + str(border_size) )
        
        print('call createFromPointCloud')
        
        # self.thisprt().createFromPointCloud(
        #     cloud.thisptr()[0],
        #     angular_resolution,
        #     max_angle_width,
        #     max_angle_height,
        #     sensor_pose, 
        #     coordinate_frame, 
        #     noise_level, 
        #     min_range, 
        #     border_size)
        
        cdef pcl_r_img.RangeImage_t *user
        # user = <pcl_r_img.RangeImage_t *> self.thisptr_shared
        user = <pcl_r_img.RangeImage *> self.thisptr()
        
        # <pcl_r_img.RangeImage_t *> self.thisprt().createFromPointCloud(
        user.createFromPointCloud(
            cloud.thisptr()[0],
            angular_resolution,
            max_angle_width,
            max_angle_height,
            sensor_pose, 
            coordinate_frame, 
            noise_level, 
            min_range, 
            border_size)
    
    
    def SetAngularResolution(self, float angular_resolution_x, float angular_resolution_y):
        self.thisprt()[0].setAngularResolution(angular_resolution_x, angular_resolution_y)
    
    
    def IntegrateFarRanges(self, PointCloudWrapper_PointWithViewpoint viewpoint):
    #   self.me.integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t> viewpoint.thisptr()[0])
    #   self.me.integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_Ptr_t> viewpoint.thisptr()[0])
        cdef pcl_r_img.RangeImage_t *user
        # user = <pcl_r_img.RangeImage_t *> self.thisptr_shared
        user = <pcl_r_img.RangeImage *> self.thisptr()
        user.integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t&> viewpoint.thisptr()[0])
        # self.thisprt()[0].integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t&> viewpoint.thisptr()[0])
        # self.me.integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t&> deref(viewpoint.thisptr()))
    
    
    def SetUnseenToMaxRange(self):
        self.thisprt()[0].setUnseenToMaxRange()
    
    
###

