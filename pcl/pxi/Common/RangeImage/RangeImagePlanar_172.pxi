# -*- coding: utf-8 -*-
from _pcl cimport PointCloud
from _pcl cimport PointCloud_PointWithViewpoint
cimport pcl_defs as cpp
cimport pcl_range_image_172 as pcl_r_img

cimport eigen as eigen3
from boost_shared_ptr cimport sp_assign

from cython.operator cimport dereference as deref, preincrement as inc

cdef class RangeImagePlanar:
    """
    RangeImagePlanar
    """

    def __cinit__(self):
        # self.me = new pcl_r_img.RangeImagePlanar_t()
        sp_assign(self.thisptr_shared, new pcl_r_img.RangeImagePlanar_t())
        pass

    def __cinit__(self, PointCloud pc not None):
        # self.me = new pcl_r_img.RangeImagePlanar_t()
        # self.point = pc.thisptr_shared
        sp_assign(self.thisptr_shared,  new pcl_r_img.RangeImagePlanar_t())
        self.thisptr().setInputCloud(pc.thisptr_shared)

    def CreateFromPointCloud(self, PointCloud cloud, float angular_resolution, float max_angle_width, float max_angle_height, 
        pcl_r_img.CoordinateFrame2 coordinate_frame, float noise_level, float min_range, int border_size):
        
        print('CreateFromPointCloud enter')
        
        # cdef cpp.PointCloudPtr_t point
        # point = cloud.thisptr_shared
        # Eigen::Translation3f
        cdef eigen3.Affine3f sensor_pose
        cdef float *data = sensor_pose.data()
        # print('sensor_pose size = ' + str( len(data) ) )
        # cdef eigen3.Translation3f data(0.0, 0.0, 0.0)
        # data[0] = 0.0
        # data[1] = 0.0
        # data[2] = 0.0
        # data[3] = 0.0
        # print('sensor_pose = ' + data)
        # (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);  
        sensor_pose = <eigen3.Affine3f>eigen3.Translation3f(0.0, 0.0, 0.0)
        
        print('angular_resolution = ' + str(angular_resolution) )
        print('max_angle_width = ' + str(max_angle_width) )
        print('max_angle_height = ' + str(max_angle_height) )
        print('noise_level = ' + str(noise_level) )
        print('min_range = ' + str(min_range) )
        print('border_size = ' + str(border_size) )
        
        print('cloud.size = ' + str(cloud.size) )
        print('cloud.width = ' + str(cloud.width) )
        print('cloud.height = ' + str(cloud.height) )
        
        print('call createFromPointCloud')
        
        self.thisptr().createFromPointCloud(
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
        self.thisptr()[0].setAngularResolution(angular_resolution_x, angular_resolution_y)
    
    def IntegrateFarRanges(self, PointCloud_PointWithViewpoint viewpoint):
        # cdef pcl_r_img.RangeImage_t *user
        # (<pcl_r_img.RangeImage *> self.thisptr()).integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t&> viewpoint.thisptr()[0])
        self.thisptr().integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t&> viewpoint.thisptr()[0])
        # self.thisprt()[0].integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t&> viewpoint.thisptr()[0])
        # self.me.integrateFarRanges(<cpp.PointCloud_PointWithViewpoint_t&> deref(viewpoint.thisptr()))
    
    def SetUnseenToMaxRange(self):
        self.thisptr()[0].setUnseenToMaxRange()


###

