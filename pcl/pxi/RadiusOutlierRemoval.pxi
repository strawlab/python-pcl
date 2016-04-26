
cimport pcl_filters as pclfil
cimport pcl_segmentation as pclseg
cimport pcl_defs as cpp

cdef class RadiusOutlierRemoval:
    """
    RadiusOutlierRemoval class for ...
    """
    cdef pclfil.RadiusOutlierRemoval_t *me
    def __cinit__(self):
        self.me = new pclfil.RadiusOutlierRemoval_t()
    def __dealloc__(self):
        del self.me

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc

    def set_radius_search(self, double radius):
        self.me.setRadiusSearch(radius)
    def get_radius_search(self):
        return self.me.getRadiusSearch()
    def set_MinNeighborsInRadius(self, int min_pts):
        self.me.setMinNeighborsInRadius (min_pts)
    def get_MinNeighborsInRadius(self):
        return self.me.getMinNeighborsInRadius ()

