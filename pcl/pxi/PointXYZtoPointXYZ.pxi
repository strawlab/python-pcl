# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport numpy as np

cdef cpp.PointXYZ to_point_t(point):
    cdef cpp.PointXYZ p
    # check point datasize
    # TypeError: a float is required
    # http://stackoverflow.com/questions/15003403/typeerror-a-float-is-required
    # TypeError: float() argument must be a string or a number, not 'tuple'
    p.x = np.float(point[0])
    p.y = np.float(point[1])
    p.z = np.float(point[2])
    return p


cdef cpp.PointXYZI to_point2_t(point):
    cdef cpp.PointXYZI p
    # check point datasize
    p.x = point[0]
    p.y = point[1]
    p.z = point[2]
    p.intensity = point[3]
    return p


cdef cpp.PointXYZRGB to_point3_t(point):
    cdef cpp.PointXYZRGB p
    
    # check point datasize
    p.x = point[0]
    p.y = point[1]
    p.z = point[2]
    p.rgb = point[3]
    return p


cdef cpp.PointXYZRGBA to_point4_t(point):
    cdef cpp.PointXYZRGBA p
    
    # check point datasize
    p.x = point[0]
    p.y = point[1]
    p.z = point[2]
    p.rgba = point[3]
    return p
