# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_filters as pclfil

cdef class VoxelGridFilter:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef pclfil.VoxelGrid_t *me
    def __cinit__(self):
        self.me = new pclfil.VoxelGrid_t()
    def __dealloc__(self):
        del self.me

    def set_leaf_size (self, float x, float y, float z):
        """
        Set the voxel grid leaf size.
        """
        self.me.setLeafSize(x,y,z)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc

    def set_DownsampleAllData (self, bool downsample):
        self.me.setDownsampleAllData (downsample)

    def get_DownsampleAllData (self):
        return self.me.getDownsampleAllData ()

    def set_SaveLeafLayout (self, bool save_leaf_layout):
        self.me.setSaveLeafLayout (save_leaf_layout)

    def get_SaveLeafLayout (self):
        return self.me.getSaveLeafLayout ()

    # eigen3.Vector3i
    # def get_MinBoxCoordinates (self):
    #     return self.me.getMinBoxCoordinates ()

    # eigen3.Vector3i
    # def get_MaxBoxCoordinates (self):
    #     return self.me.getMaxBoxCoordinates ()

    # eigen3.Vector3i
    # def get_NrDivisions (self):
    #     return self.me.getNrDivisions ()

    # eigen3.Vector3i
    # def get_DivisionMultiplier (self):
    #     return self.me.getDivisionMultiplier ()

    # int
    # def get_DivisionMultiplier (self, const T &p):
    #     return self.me.getCentroidIndex (p)

    # vector[int]
    # def get_NeighborCentroidIndices (self, const T &reference_point, const eigen3.MatrixXi &relative_coordinates):
    #   return self.me.getNeighborCentroidIndices (reference_point, relative_coordinates)

    # vector[int]
    def get_LeafLayout (self):
        return self.me.getLeafLayout ()

    # Eigen::Vector3i 
    # def get_GridCoordinates (self, float x, float y, float z):
    #     return self.me.getGridCoordinates (x, y, z) 

    # int
    # def get_CentroidIndexAt (self, const eigen3.Vector3i &ijk):
    #     return self.me.getCentroidIndexAt (ijk)

    # def set_FilterFieldName (self, const string &field_name):
    #     self.me.setFilterFieldName (field_name)

    # string
    def get_FilterFieldName (self):
        return self.me.getFilterFieldName ()

    def set_FilterLimits (self, const double &limit_min, const double &limit_max):
        self.me.setFilterLimits (limit_min, limit_max)

    # void
    def get_FilterLimits (self, double &limit_min, double &limit_max):
        self.me.getFilterLimits (limit_min, limit_max)

    def set_FilterLimitsNegative (self, const bool limit_negative):
        self.me.setFilterLimitsNegative (limit_negative)

    # void
    def get_FilterLimitsNegative (self, bool &limit_negative):
        self.me.getFilterLimitsNegative (limit_negative)

    # bool
    def get_FilterLimitsNegative (self):
        return self.me.getFilterLimitsNegative ()


cdef class VoxelGridFilter_PointXYZI:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef pclfil.VoxelGrid_PointXYZI_t *me
    def __cinit__(self):
        self.me = new pclfil.VoxelGrid_PointXYZI_t()
    def __dealloc__(self):
        del self.me

    def set_leaf_size (self, float x, float y, float z):
        """
        Set the voxel grid leaf size.
        """
        self.me.setLeafSize(x,y,z)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZI pc = PointCloud_PointXYZI()
        self.me.filter(pc.thisptr()[0])
        return pc


cdef class VoxelGridFilter_PointXYZRGB:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef pclfil.VoxelGrid_PointXYZRGB_t *me
    def __cinit__(self):
        self.me = new pclfil.VoxelGrid_PointXYZRGB_t()
    def __dealloc__(self):
        del self.me

    def set_leaf_size (self, float x, float y, float z):
        """
        Set the voxel grid leaf size.
        """
        self.me.setLeafSize(x,y,z)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZRGB pc = PointCloud_PointXYZRGB()
        self.me.filter(pc.thisptr()[0])
        return pc


cdef class VoxelGridFilter_PointXYZRGBA:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef pclfil.VoxelGrid_PointXYZRGBA_t *me
    def __cinit__(self):
        self.me = new pclfil.VoxelGrid_PointXYZRGBA_t()
    def __dealloc__(self):
        del self.me

    def set_leaf_size (self, float x, float y, float z):
        """
        Set the voxel grid leaf size.
        """
        self.me.setLeafSize(x,y,z)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud_PointXYZRGBA pc = PointCloud_PointXYZRGBA()
        self.me.filter(pc.thisptr()[0])
        return pc


