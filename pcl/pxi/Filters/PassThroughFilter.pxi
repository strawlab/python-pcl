# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_filters as pcl_fil

cdef class PassThroughFilter:
    """
    Passes points in a cloud based on constraints for one particular field of the point type.
    """
    cdef pcl_fil.PassThrough_t *me
    def __cinit__(self):
        self.me = new pcl_fil.PassThrough_t()
    def __dealloc__(self):
        del self.me

    def set_filter_field_name(self, field_name):
        """
        Provide the name of the field to be used for filtering data.
        """
        cdef bytes fname_ascii
        if isinstance(field_name, unicode):
            fname_ascii = field_name.encode("ascii")
        elif not isinstance(field_name, bytes):
            raise TypeError("field_name should be a string, got %r"
                            % field_name)
        else:
            fname_ascii = field_name
        self.me.setFilterFieldName(string(fname_ascii))

    def set_filter_limits(self, float filter_min, float filter_max):
        """
        Set the numerical limits for the field for filtering data.
        """
        self.me.setFilterLimits (filter_min, filter_max)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        # cdef cpp.PointCloud_t *cCondAnd = <cpp.PointCloud_t *>pc.thisptr()[0]
        # self.me.filter(<cpp.PointCloud_t*> pc.thisptr()[0])
        # self.me.filter (<cpp.PointCloud_t*> pc.thisptr())
        self.me.c_filter(pc.thisptr()[0])
        return pc


cdef class PassThroughFilter_PointXYZI:
    """
    Passes points in a cloud based on constraints for one particular field of the point type
    """
    cdef pcl_fil.PassThrough_PointXYZI_t *me
    def __cinit__(self):
        self.me = new pcl_fil.PassThrough_PointXYZI_t()
    def __dealloc__(self):
        del self.me

    def set_filter_field_name(self, field_name):
        """
        Provide the name of the field to be used for filtering data.
        """
        cdef bytes fname_ascii
        if isinstance(field_name, unicode):
            fname_ascii = field_name.encode("ascii")
        elif not isinstance(field_name, bytes):
            raise TypeError("field_name should be a string, got %r"
                            % field_name)
        else:
            fname_ascii = field_name
        self.me.setFilterFieldName(string(fname_ascii))

    def set_filter_limits(self, float filter_min, float filter_max):
        """
        Set the numerical limits for the field for filtering data.
        """
        self.me.setFilterLimits (filter_min, filter_max)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new PointCloud_PointXYZI
        """
        cdef PointCloud_PointXYZI pc = PointCloud_PointXYZI()
        self.me.c_filter(pc.thisptr()[0])
        return pc


cdef class PassThroughFilter_PointXYZRGB:
    """
    Passes points in a cloud based on constraints for one particular field of the point type
    """
    cdef pcl_fil.PassThrough_PointXYZRGB_t *me
    def __cinit__(self):
        self.me = new pcl_fil.PassThrough_PointXYZRGB_t()
    def __dealloc__(self):
        del self.me

    def set_filter_field_name(self, field_name):
        """
        Provide the name of the field to be used for filtering data.
        """
        cdef bytes fname_ascii
        if isinstance(field_name, unicode):
            fname_ascii = field_name.encode("ascii")
        elif not isinstance(field_name, bytes):
            raise TypeError("field_name should be a string, got %r"
                            % field_name)
        else:
            fname_ascii = field_name
        self.me.setFilterFieldName(string(fname_ascii))

    def set_filter_limits(self, float filter_min, float filter_max):
        """
        Set the numerical limits for the field for filtering data.
        """
        self.me.setFilterLimits (filter_min, filter_max)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new PointCloud_PointXYZRGB
        """
        cdef PointCloud_PointXYZRGB pc = PointCloud_PointXYZRGB()
        self.me.c_filter(pc.thisptr()[0])
        return pc


cdef class PassThroughFilter_PointXYZRGBA:
    """
    Passes points in a cloud based on constraints for one particular field of the point type
    """
    cdef pcl_fil.PassThrough_PointXYZRGBA_t *me
    def __cinit__(self):
        self.me = new pcl_fil.PassThrough_PointXYZRGBA_t()
    def __dealloc__(self):
        del self.me

    def set_filter_field_name(self, field_name):
        """
        Provide the name of the field to be used for filtering data.
        """
        cdef bytes fname_ascii
        if isinstance(field_name, unicode):
            fname_ascii = field_name.encode("ascii")
        elif not isinstance(field_name, bytes):
            raise TypeError("field_name should be a string, got %r"
                            % field_name)
        else:
            fname_ascii = field_name
        self.me.setFilterFieldName(string(fname_ascii))

    def set_filter_limits(self, float filter_min, float filter_max):
        """
        Set the numerical limits for the field for filtering data.
        """
        self.me.setFilterLimits (filter_min, filter_max)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new PointCloud_PointXYZRGBA
        """
        cdef PointCloud_PointXYZRGBA pc = PointCloud_PointXYZRGBA()
        self.me.c_filter(pc.thisptr()[0])
        return pc
