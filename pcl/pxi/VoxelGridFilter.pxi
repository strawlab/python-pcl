
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
