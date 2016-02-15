from libcpp cimport bool
from libcpp.vector cimport vector

# main
cimport pcl_defs as cpp
from boost_shared_ptr cimport shared_ptr

cdef extern from "pcl/surface/mls.h" namespace "pcl":
    cdef cppclass MovingLeastSquares[I,O]:
        MovingLeastSquares()
        void setInputCloud (shared_ptr[cpp.PointCloud[I]])
        void setSearchRadius (double)
        void setPolynomialOrder(bool)
        void setPolynomialFit(int)
        void process(cpp.PointCloud[O] &) except +

ctypedef MovingLeastSquares[cpp.PointXYZ, cpp.PointXYZ] MovingLeastSquares_t
ctypedef MovingLeastSquares[cpp.PointXYZRGBA, cpp.PointXYZRGBA] MovingLeastSquares2_t


###
# allocator.h
# bilateral_upsampling.h
# binary_node.h
# concave_hull.h
# convex_hull.h
# ear_clipping.h
# factor.h
# function_data.h
# geometry.h
# gp3.h
# grid_projection.h
# hash.h
# marching_cubes.h
# marching_cubes_hoppe.h
# marching_cubes_poisson.h
# marching_cubes_rbf.h
# mls.h
# mls_omp.h
# multi_grid_octree_data.h
# octree_poisson.h
# organized_fast_mesh.h
# poisson.h
# polynomial.h
# ppolynomial.h
# processing.h
# qhull.h
# reconstruction.h
# simplification_remove_unused_vertices.h
# sparse_matrix.h
# surfel_smoothing.h
# texture_mapping.h
# vector.h
# vtk.h
# vtk_mesh_smoothing_laplacian.h
# vtk_mesh_smoothing_windowed_sinc.h
# vtk_mesh_subdivision.h
# vtk_utils.h
