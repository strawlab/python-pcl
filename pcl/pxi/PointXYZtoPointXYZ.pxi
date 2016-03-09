
cimport pcl_defs as cpp

cdef cpp.PointXYZ to_point_t(point):
    cdef cpp.PointXYZ p
    p.x = point[0]
    p.y = point[1]
    p.z = point[2]
    return p


cdef cpp.PointXYZI to_point2_t(point):
    cdef cpp.PointXYZI p
    p.x = point[0]
    p.y = point[1]
    p.z = point[2]
    p.intensity = point[3]
    return p


cdef cpp.PointXYZRGB to_point3_t(point):
    cdef cpp.PointXYZRGB p
    p.x = point[0]
    p.y = point[1]
    p.z = point[2]
    p.rgb = point[3]
    return p


cdef cpp.PointXYZRGBA to_point4_t(point):
    cdef cpp.PointXYZRGBA p
    p.x = point[0]
    p.y = point[1]
    p.z = point[2]
    p.rgba = point[3]
    return p
