# -*- coding: utf-8 -*-

from libcpp cimport bool
from libcpp.vector cimport vector

# main
cimport pcl_defs as cpp
cimport pcl_kdtree as pclkdt
from boost_shared_ptr cimport shared_ptr

###############################################################################
# Types
###############################################################################

### base class ###

# reconstruction.h
# namespace pcl
# brief Pure abstract class. All types of meshing/reconstruction
# algorithms in \b libpcl_surface must inherit from this, in order to make
# sure we have a consistent API. The methods that we care about here are:
#  - \b setSearchMethod(&SearchPtr): passes a search locator
#  - \b reconstruct(&PolygonMesh): creates a PolygonMesh object from the input data
# author Radu B. Rusu, Michael Dixon, Alexandru E. Ichim
# 
# template <typename PointInT>
# class PCLSurfaceBase: public PCLBase<PointInT>
cdef extern from "pcl/surface/reconstruction.h" namespace "pcl":
    cdef cppclass PCLSurfaceBase[In](cpp.PCLBase[In]):
        PCLSurfaceBase()
        
        # brief Provide an optional pointer to a search object.
        # param[in] tree a pointer to the spatial search object.
        # inline void setSearchMethod (const KdTreePtr &tree)
        void setSearchMethod (const pclkdt.KdTreePtr_t &tree)
        
        # brief Get a pointer to the search method used.
        # inline KdTreePtr getSearchMethod ()
        pclkdt.KdTreePtr_t getSearchMethod ()

#       /** \brief Base method for surface reconstruction for all points given in
#         * <setInputCloud (), setIndices ()> 
#         * \param[out] output the resultant reconstructed surface model
#       virtual void reconstruct (pcl::PolygonMesh &output) = 0;

#       protected:
#       /** \brief A pointer to the spatial search object. */
#       KdTreePtr tree_;
#       /** \brief Abstract class get name method. */
#       virtual std::string getClassName () const { return (""); }
###

# /** \brief SurfaceReconstruction represents a base surface reconstruction
#   * class. All \b surface reconstruction methods take in a point cloud and
#   * generate a new surface from it, by either re-sampling the data or
#   * generating new data altogether. These methods are thus \b not preserving
#   * the topology of the original data.
#   * \note Reconstruction methods that always preserve the original input
#   * point cloud data as the surface vertices and simply construct the mesh on
#   * top should inherit from \ref MeshConstruction.
#   * \author Radu B. Rusu, Michael Dixon, Alexandru E. Ichim
#   * \ingroup surface
#   */
# template <typename PointInT>
# class SurfaceReconstruction: public PCLSurfaceBase<PointInT>
cdef extern from "pcl/surface/reconstruction.h" namespace "pcl":
    cdef cppclass SurfaceReconstruction[In](PCLSurfaceBase[In]):
        SurfaceReconstruction()
#       public:
#       using PCLSurfaceBase<PointInT>::input_;
#       using PCLSurfaceBase<PointInT>::indices_;
#       using PCLSurfaceBase<PointInT>::initCompute;
#       using PCLSurfaceBase<PointInT>::deinitCompute;
#       using PCLSurfaceBase<PointInT>::tree_;
#       using PCLSurfaceBase<PointInT>::getClassName;
#       
#       /** \brief Base method for surface reconstruction for all points given in
#        * <setInputCloud (), setIndices ()> 
#        * \param[out] output the resultant reconstructed surface model
#        */
#       virtual void reconstruct (pcl::PolygonMesh &output);
#       /** \brief Base method for surface reconstruction for all points given in
#         * <setInputCloud (), setIndices ()> 
#         * \param[out] points the resultant points lying on the new surface
#         * \param[out] polygons the resultant polygons, as a set of
#         * vertices. The Vertices structure contains an array of point indices.
#         */
#       virtual void 
#       reconstruct (pcl::PointCloud<PointInT> &points,
#                    std::vector<pcl::Vertices> &polygons);
#       protected:
#       /** \brief A flag specifying whether or not the derived reconstruction
#         * algorithm needs the search object \a tree.*/
#       bool check_tree_;
#       /** \brief Abstract surface reconstruction method. 
#         * \param[out] output the output polygonal mesh 
#         */
#       virtual void performReconstruction (pcl::PolygonMesh &output) = 0;
#       /** \brief Abstract surface reconstruction method. 
#         * \param[out] points the resultant points lying on the surface
#         * \param[out] polygons the resultant polygons, as a set of vertices. The Vertices structure contains an array of point indices.
#         */
#       virtual void 
#       performReconstruction (pcl::PointCloud<PointInT> &points, 
#                              std::vector<pcl::Vertices> &polygons) = 0;
###

# brief MeshConstruction represents a base surface reconstruction
# class. All \b mesh constructing methods that take in a point cloud and
# generate a surface that uses the original data as vertices should inherit
# from this class.
# 
# note Reconstruction methods that generate a new surface or create new
# vertices in locations different than the input data should inherit from
# \ref SurfaceReconstruction.
# 
# author Radu B. Rusu, Michael Dixon, Alexandru E. Ichim
# \ingroup surface
# 
# template <typename PointInT>
# class MeshConstruction: public PCLSurfaceBase<PointInT>
cdef extern from "pcl/surface/reconstruction.h" namespace "pcl":
    cdef cppclass MeshConstruction[In](PCLSurfaceBase[In]):
        MeshConstruction()
        # public:
        # using PCLSurfaceBase<PointInT>::input_;
        # using PCLSurfaceBase<PointInT>::indices_;
        # using PCLSurfaceBase<PointInT>::initCompute;
        # using PCLSurfaceBase<PointInT>::deinitCompute;
        # using PCLSurfaceBase<PointInT>::tree_;
        # using PCLSurfaceBase<PointInT>::getClassName;
        
        # brief Base method for surface reconstruction for all points given in <setInputCloud (), setIndices ()> 
        # param[out] output the resultant reconstructed surface model
        # 
        # note This method copies the input point cloud data from
        # PointCloud<T> to PointCloud2, and is implemented here for backwards
        # compatibility only!
        # 
        # virtual void reconstruct (pcl::PolygonMesh &output);
        # brief Base method for mesh construction for all points given in <setInputCloud (), setIndices ()> 
        # param[out] polygons the resultant polygons, as a set of vertices.
        # The Vertices structure contains an array of point indices.
        # 
        # virtual void reconstruct (std::vector<pcl::Vertices> &polygons);
        # 
        # protected:
        # /** \brief A flag specifying whether or not the derived reconstruction
        #   * algorithm needs the search object \a tree.*/
        # bool check_tree_;
        # /** \brief Abstract surface reconstruction method. 
        #   * \param[out] output the output polygonal mesh 
        #   */
        # virtual void performReconstruction (pcl::PolygonMesh &output) = 0;
        # /** \brief Abstract surface reconstruction method. 
        #   * \param[out] polygons the resultant polygons, as a set of vertices. The Vertices structure contains an array of point indices.
        #   */
        # virtual void performReconstruction (std::vector<pcl::Vertices> &polygons) = 0;
###

# processing.h
# namespace pcl
# brief @b CloudSurfaceProcessing represents the base class for algorithms that take a point cloud as an input and
# produce a new output cloud that has been modified towards a better surface representation. These types of
# algorithms include surface smoothing, hole filling, cloud upsampling etc.
# author Alexandru E. Ichim
# ingroup surface
# 
# template <typename PointInT, typename PointOutT>
# class CloudSurfaceProcessing : public PCLBase<PointInT>
cdef extern from "pcl/surface/processing.h" namespace "pcl":
    cdef cppclass CloudSurfaceProcessing[In, Out](cpp.PCLBase[In]):
        CloudSurfaceProcessing()
#       public:
#       using PCLBase<PointInT>::input_;
#       using PCLBase<PointInT>::indices_;
#       using PCLBase<PointInT>::initCompute;
#       using PCLBase<PointInT>::deinitCompute;
#       public:
#       /** \brief Process the input cloud and store the results
#         * \param[out] output the cloud where the results will be stored
#       virtual void process (pcl::PointCloud<PointOutT> &output);
#       protected:
#       /** \brief Abstract cloud processing method */
#       virtual void performProcessing (pcl::PointCloud<PointOutT> &output) = 0;
### 

# /** \brief @b MeshProcessing represents the base class for mesh processing algorithms.
#   * \author Alexandru E. Ichim
#   * \ingroup surface
#   */
# class PCL_EXPORTS MeshProcessing
#       public:
#       typedef PolygonMesh::ConstPtr PolygonMeshConstPtr;
#       /** \brief Constructor. */
#       MeshProcessing () : input_mesh_ () {};
#       /** \brief Destructor. */
#       virtual ~MeshProcessing () {}
#       /** \brief Set the input mesh that we want to process
#         * \param[in] input the input polygonal mesh
#       void setInputMesh (const pcl::PolygonMeshConstPtr &input) 
#       /** \brief Process the input surface mesh and store the results
#         * \param[out] output the resultant processed surface model
#       void process (pcl::PolygonMesh &output);
#       protected:
#       /** \brief Initialize computation. Must be called before processing starts. */
#       virtual bool initCompute ();
#       /** \brief UnInitialize computation. Must be called after processing ends. */
#       virtual void deinitCompute ();
#       /** \brief Abstract surface processing method. */
#       virtual void performProcessing (pcl::PolygonMesh &output) = 0;
#       /** \brief Abstract class get name method. */
#       virtual std::string getClassName () const { return (""); }
#       /** \brief Input polygonal mesh. */
#       pcl::PolygonMeshConstPtr input_mesh_;
###


# (1.6.0)allocator.h
# (1.7.2) -> pcl/surface/3rdparty/poisson4
# namespace pcl 
# namespace poisson 
# class AllocatorState
# cdef extern from "pcl/surface/3rdparty/poisson4/allocator.h" namespace "pcl::poisson":
#     cdef cppclass AllocatorState:
#         AllocatorState()
#         # public:
#         # int index,remains;


# (1.6.0) -> allocator.h
# (1.7.2) -> pcl\surface\3rdparty\poisson4 ?
# template<class T>
# class Allocator
# cdef extern from "pcl/surface/3rdparty/poisson4/allocator.h" namespace "pcl::poisson":
#     cdef cppclass Allocator[T]:
#         Allocator()
        # int blockSize;
        # int index, remains;
        # std::vector<T*> memory;
        # public:
        # /** This method is the allocators destructor. It frees up any of the memory that
        #   * it has allocated. 
        # void reset ()
        # /** This method returns the memory state of the allocator. */
        # AllocatorState getState () const
        # /** This method rolls back the allocator so that it makes all of the memory previously
        #   * allocated available for re-allocation. Note that it does it not call the constructor
        #   * again, so after this method has been called, assumptions about the state of the values
        #   * in memory are no longer valid. 
        # void rollBack ()
        # /** This method rolls back the allocator to the previous memory state and makes all of the memory previously
        #   * allocated available for re-allocation. Note that it does it not call the constructor
        #   * again, so after this method has been called, assumptions about the state of the values
        #   * in memory are no longer valid. 
        # void rollBack (const AllocatorState& state)
        # /** This method initiallizes the constructor and the blockSize variable specifies the
        #   * the number of objects that should be pre-allocated at a time. 
        # void set (const int& blockSize)
        # /** This method returns a pointer to an array of elements objects. If there is left over pre-allocated
        #   * memory, this method simply returns a pointer to the next free piece of memory, otherwise it pre-allocates
        #   * more memory. Note that if the number of objects requested is larger than the value blockSize with which
        #   * the allocator was initialized, the request for memory will fail.
        # T* newElements (const int& elements = 1)
###

# bilateral_upsampling.h
# namespace pcl
# /** \brief Bilateral filtering implementation, based on the following paper:
#   *   * Kopf, Johannes and Cohen, Michael F. and Lischinski, Dani and Uyttendaele, Matt - Joint Bilateral Upsampling,
#   *   * ACM Transations in Graphics, July 2007
#   * Takes in a colored organized point cloud (i.e. PointXYZRGB or PointXYZRGBA), that might contain nan values for the
#   * depth information, and it will returned an upsampled version of this cloud, based on the formula:
#   * \f[
#   *    \tilde{S}_p = \frac{1}{k_p} \sum_{q_d \in \Omega} {S_{q_d} f(||p_d - q_d|| g(||\tilde{I}_p-\tilde{I}_q||})
#   * \f]
#   * where S is the depth image, I is the RGB image and f and g are Gaussian functions centered at 0 and with
#   * standard deviations \f$\sigma_{color}\f$ and \f$\sigma_{depth}\f$
#   */
# template <typename PointInT, typename PointOutT>
# class BilateralUpsampling: public CloudSurfaceProcessing<PointInT, PointOutT>
cdef extern from "pcl/surface/bilateral_upsampling.h" namespace "pcl":
    cdef cppclass BilateralUpsampling[In, Out](CloudSurfaceProcessing[In, Out]):
        BilateralUpsampling()
        # public:
        # using PCLBase<PointInT>::input_;
        # using PCLBase<PointInT>::indices_;
        # using PCLBase<PointInT>::initCompute;
        # using PCLBase<PointInT>::deinitCompute;
        # using CloudSurfaceProcessing<PointInT, PointOutT>::process;
        # typedef pcl::PointCloud<PointOutT> PointCloudOut;
        # Eigen::Matrix3f KinectVGAProjectionMatrix, KinectSXGAProjectionMatrix;
        # /** \brief Method that sets the window size for the filter
        #   * \param[in] window_size the given window size
        # inline void setWindowSize (int window_size)
        # /** \brief Returns the filter window size */
        # inline int getWindowSize () const
        # /** \brief Method that sets the sigma color parameter
        #   * \param[in] sigma_color the new value to be set
        # inline void setSigmaColor (const float &sigma_color)
        # /** \brief Returns the current sigma color value */
        # inline float getSigmaColor () const
        # /** \brief Method that sets the sigma depth parameter
        #   * \param[in] sigma_depth the new value to be set
        # inline void setSigmaDepth (const float &sigma_depth)
        # /** \brief Returns the current sigma depth value */
        # inline float getSigmaDepth () const
        # /** \brief Method that sets the projection matrix to be used when unprojecting the points in the depth image
        #   * back to (x,y,z) positions.
        #   * \note There are 2 matrices already set in the class, used for the 2 modes available for the Kinect. They
        #   * are tuned to be the same as the ones in the OpenNiGrabber
        #   * \param[in] projection_matrix the new projection matrix to be set */
        # inline void setProjectionMatrix (const Eigen::Matrix3f &projection_matrix)
        # /** \brief Returns the current projection matrix */
        # inline Eigen::Matrix3f getProjectionMatrix () const
        # /** \brief Method that does the actual processing on the input cloud.
        #   * \param[out] output the container of the resulting upsampled cloud */
        # void process (pcl::PointCloud<PointOutT> &output)
        # protected:
        # void performProcessing (pcl::PointCloud<PointOutT> &output);
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
###

# binary_node.h (1.6.0)
# pcl/surface/3rdparty\poisson4\binary_node.h (1.7.2)
# namespace pcl
# namespace poisson
# template<class Real>
# class BinaryNode
# cdef extern from "pcl/surface/3rdparty/poisson4/binary_node.h" namespace "pcl::poisson":
#    cdef cppclass BinaryNode[Real]:
#        BinaryNode()
        # public:
        # static inline int CenterCount (int depth){return 1<<depth;}
        # static inline int CumulativeCenterCount (int maxDepth){return  (1<< (maxDepth+1))-1;}
        # static inline int Index (int depth,  int offSet){return  (1<<depth)+offSet-1;}
        # static inline int CornerIndex (int maxDepth, int depth, int offSet, int forwardCorner)
        # static inline Real CornerIndexPosition (int index, int maxDepth)
        # static inline Real Width (int depth)
        # 
        # // Fix for Bug #717 with Visual Studio that generates wrong code for this function
        # // when global optimization is enabled (release mode).
        # #ifdef _MSC_VER
        #   static __declspec(noinline) void CenterAndWidth (int depth, int offset, Real& center, Real& width)
        # #else
        # static inline void CenterAndWidth (int depth, int offset, Real& center, Real& width)
        # # #endif
        # 
        # #ifdef _MSC_VER
        # static __declspec(noinline) void CenterAndWidth (int idx, Real& center, Real& width)
        # #else
        # static inline void CenterAndWidth (int idx, Real& center, Real& width)
        # #endif
        # static inline void DepthAndOffset (int idx,  int& depth, int& offset)
###

# concave_hull.h
# namespace pcl
# template<typename PointInT>
# class ConcaveHull : public MeshConstruction<PointInT>
cdef extern from "pcl/surface/concave_hull.h" namespace "pcl":
    cdef cppclass ConcaveHull[In](MeshConstruction[In]):
        ConcaveHull()
        # public:
        # \brief Compute a concave hull for all points given 
        # \param points the resultant points lying on the concave hull 
        # \param polygons the resultant concave hull polygons, as a set of
        # vertices. The Vertices structure contains an array of point indices.
        # void reconstruct (PointCloud &points, std::vector<pcl::Vertices> &polygons);
        
        # /** \brief Compute a concave hull for all points given 
        #  * \param output the resultant concave hull vertices
        # void reconstruct (PointCloud &output);
        void reconstruct (cpp.PointCloud_t output)
        void reconstruct (cpp.PointCloud_PointXYZI_t output)
        void reconstruct (cpp.PointCloud_PointXYZRGB_t output)
        void reconstruct (cpp.PointCloud_PointXYZRGBA_t output)
        
        # /** \brief Set the alpha value, which limits the size of the resultant
        #   * hull segments (the smaller the more detailed the hull).  
        #   * \param alpha positive, non-zero value, defining the maximum length
        #   * from a vertex to the facet center (center of the voronoi cell).
        # inline void setAlpha (double alpha)
        void setAlpha (double alpha)
        # Returns the alpha parameter, see setAlpha().
        # inline double getAlpha ()
        double getAlpha ()
        
        # If set, the voronoi cells center will be saved in _voronoi_centers_
        # voronoi_centers
        # inline void setVoronoiCenters (PointCloudPtr voronoi_centers)
        
        # \brief If keep_information_is set to true the convex hull
        # points keep other information like rgb, normals, ...
        # \param value where to keep the information or not, default is false
        # void setKeepInformation (bool value)
        
        # brief Returns the dimensionality (2 or 3) of the calculated hull.
        # inline int getDim () const
        # brief Returns the dimensionality (2 or 3) of the calculated hull.
        # inline int getDimension () const
        # brief Sets the dimension on the input data, 2D or 3D.
        # param[in] dimension The dimension of the input data.  If not set, this will be determined automatically.
        # void setDimension (int dimension)
        
        # protected:
        # /** \brief The actual reconstruction method.
        #   * \param points the resultant points lying on the concave hull 
        #   * \param polygons the resultant concave hull polygons, as a set of
        #   * vertices. The Vertices structure contains an array of point indices.
        #   */
        # void performReconstruction (PointCloud &points, 
        #                        std::vector<pcl::Vertices> &polygons);
        # virtual void performReconstruction (PolygonMesh &output);
        # virtual void performReconstruction (std::vector<pcl::Vertices> &polygons);
        # /** \brief The method accepts facets only if the distance from any vertex to the facet->center 
        #   * (center of the voronoi cell) is smaller than alpha 
        # double alpha_;
        # /** \brief If set to true, the reconstructed point cloud describing the hull is obtained from 
        #   * the original input cloud by performing a nearest neighbor search from Qhull output. 
        # bool keep_information_;
        # /** \brief the centers of the voronoi cells */
        # PointCloudPtr voronoi_centers_;
        # /** \brief the dimensionality of the concave hull */
        # int dim_;
        


ctypedef ConcaveHull[cpp.PointXYZ] ConcaveHull_t
ctypedef ConcaveHull[cpp.PointXYZI] ConcaveHull_PointXYZI_t
ctypedef ConcaveHull[cpp.PointXYZRGB] ConcaveHull_PointXYZRGB_t
ctypedef ConcaveHull[cpp.PointXYZRGBA] ConcaveHull_PointXYZRGBA_t

###

# convex_hull.h
# namespace pcl
# /** \brief Sort 2D points in a vector structure
#   * \param p1 the first point
#   * \param p2 the second point
#   * \ingroup surface
#   */
# inline bool
# comparePoints2D (const std::pair<int, Eigen::Vector4f> & p1, const std::pair<int, Eigen::Vector4f> & p2)
# 
# template<typename PointInT>
# class ConvexHull : public MeshConstruction<PointInT>
cdef extern from "pcl/surface/convex_hull.h" namespace "pcl":
    cdef cppclass ConvexHull[In](MeshConstruction[In]):
        ConvexHull()
        # protected:
        # using PCLBase<PointInT>::input_;
        # using PCLBase<PointInT>::indices_;
        # using PCLBase<PointInT>::initCompute;
        # using PCLBase<PointInT>::deinitCompute;
        # public:
        # using MeshConstruction<PointInT>::reconstruct;
        # typedef pcl::PointCloud<PointInT> PointCloud;
        # typedef typename PointCloud::Ptr PointCloudPtr;
        # typedef typename PointCloud::ConstPtr PointCloudConstPtr;
        # 
        # /** \brief Compute a convex hull for all points given 
        #   * \param[out] points the resultant points lying on the convex hull 
        #   * \param[out] polygons the resultant convex hull polygons, as a set of
        #   * vertices. The Vertices structure contains an array of point indices.
        # void reconstruct (PointCloud &points, 
        #              std::vector<pcl::Vertices> &polygons);
        # /** \brief Compute a convex hull for all points given 
        #   * \param[out] output the resultant convex hull vertices
        # void reconstruct (PointCloud &output);
        # /** \brief If set to true, the qhull library is called to compute the total area and volume of the convex hull.
        #   * NOTE: When this option is activated, the qhull library produces output to the console.
        #   * \param[in] value wheter to compute the area and the volume, default is false
        # void setComputeAreaVolume (bool value)
        # /** \brief Returns the total area of the convex hull. */
        # double getTotalArea () const
        # /** \brief Returns the total volume of the convex hull. Only valid for 3-dimensional sets.
        #   *  For 2D-sets volume is zero. 
        # double getTotalVolume () const
        # /** \brief Sets the dimension on the input data, 2D or 3D.
        #   * \param[in] dimension The dimension of the input data.  If not set, this will be determined automatically.
        # void setDimension (int dimension)
        # /** \brief Returns the dimensionality (2 or 3) of the calculated hull. */
        # inline int getDimension () const
        # 
        # protected:
        # /** \brief The actual reconstruction method. 
        #   * \param[out] points the resultant points lying on the convex hull 
        #   * \param[out] polygons the resultant convex hull polygons, as a set of
        #   * vertices. The Vertices structure contains an array of point indices.
        #   * \param[in] fill_polygon_data true if polygons should be filled, false otherwise
        #   */
        # void
        # performReconstruction (PointCloud &points, 
        #                        std::vector<pcl::Vertices> &polygons, 
        #                        bool fill_polygon_data = false);
        # /** \brief The reconstruction method for 2D data.  Does not require dimension to be set. 
        #   * \param[out] points the resultant points lying on the convex hull 
        #   * \param[out] polygons the resultant convex hull polygons, as a set of
        #   * vertices. The Vertices structure contains an array of point indices.
        #   * \param[in] fill_polygon_data true if polygons should be filled, false otherwise
        # void
        # performReconstruction2D (PointCloud &points, 
        #                          std::vector<pcl::Vertices> &polygons, 
        #                          bool fill_polygon_data = false);
        # /** \brief The reconstruction method for 3D data.  Does not require dimension to be set. 
        #   * \param[out] points the resultant points lying on the convex hull 
        #   * \param[out] polygons the resultant convex hull polygons, as a set of
        #   * vertices. The Vertices structure contains an array of point indices.
        #   * \param[in] fill_polygon_data true if polygons should be filled, false otherwise
        # void
        # performReconstruction3D (PointCloud &points, 
        #                          std::vector<pcl::Vertices> &polygons, 
        #                          bool fill_polygon_data = false);
        # /** \brief A reconstruction method that returns a polygonmesh.
        #   *
        #   * \param[out] output a PolygonMesh representing the convex hull of the input data.
        #   */
        # virtual void
        # performReconstruction (PolygonMesh &output);
        # 
        # /** \brief A reconstruction method that returns the polygon of the convex hull.
        #   *
        #   * \param[out] polygons the polygon(s) representing the convex hull of the input data.
        #   */
        # virtual void
        # performReconstruction (std::vector<pcl::Vertices> &polygons);
        # 
        # /** \brief Automatically determines the dimension of input data - 2D or 3D. */
        # void 
        # calculateInputDimension ();
        # 
        # /** \brief Class get name method. */
        # std::string getClassName () const
        # 
        # /* \brief True if we should compute the area and volume of the convex hull. */
        # bool compute_area_;
        # /* \brief The area of the convex hull. */
        # double total_area_;
        # /* \brief The volume of the convex hull (only for 3D hulls, zero for 2D). */
        # double total_volume_;
        # /** \brief The dimensionality of the concave hull (2D or 3D). */
        # int dimension_;
        # /** \brief How close can a 2D plane's normal be to an axis to make projection problematic. */
        # double projection_angle_thresh_;
        # /** \brief Option flag string to be used calling qhull. */
        # std::string qhull_flags;
        # /* \brief x-axis - for checking valid projections. */
        # const Eigen::Vector3f x_axis_;
        # /* \brief y-axis - for checking valid projections. */
        # const Eigen::Vector3f y_axis_;
        # /* \brief z-axis - for checking valid projections. */
        # const Eigen::Vector3f z_axis_;
        # public:
        # EIGEN_MAKE_ALIGNED_OPERATOR_NEW
###

# ear_clipping.h
# namespace pcl
# /** \brief The ear clipping triangulation algorithm.
#   * The code is inspired by Flavien Brebion implementation, which is
#   * in n^3 and does not handle holes.
#   * \author Nicolas Burrus
#   * \ingroup surface
# class PCL_EXPORTS EarClipping : public MeshProcessing
#       public:
#       using MeshProcessing::input_mesh_;
#       using MeshProcessing::initCompute;
#       /** \brief Empty constructor */
#       EarClipping () : MeshProcessing (), points_ ()
#       { 
#       };
# 
#       protected:
#       /** \brief a Pointer to the point cloud data. */
#       pcl::PointCloud<pcl::PointXYZ>::Ptr points_;
# 
#       /** \brief This method should get called before starting the actual computation. */
#       bool initCompute ();
#       /** \brief The actual surface reconstruction method. 
#         * \param[out] output the output polygonal mesh 
#         */
#       void performProcessing (pcl::PolygonMesh &output);
# 
#       /** \brief Triangulate one polygon. 
#         * \param[in] vertices the set of vertices
#         * \param[out] output the resultant polygonal mesh
#         */
#       void triangulate (const Vertices& vertices, PolygonMesh& output);
# 
#       /** \brief Compute the signed area of a polygon. 
#         * \param[in] vertices the vertices representing the polygon 
#         */
#       float area (const std::vector<uint32_t>& vertices);
# 
#       /** \brief Check if the triangle (u,v,w) is an ear. 
#         * \param[in] u the first triangle vertex 
#         * \param[in] v the second triangle vertex 
#         * \param[in] w the third triangle vertex 
#         * \param[in] vertices a set of input vertices
#         */
#       bool isEar (int u, int v, int w, const std::vector<uint32_t>& vertices);
# 
#       /** \brief Check if p is inside the triangle (u,v,w). 
#         * \param[in] u the first triangle vertex 
#         * \param[in] v the second triangle vertex 
#         * \param[in] w the third triangle vertex 
#         * \param[in] p the point to check
#         */
#       bool isInsideTriangle (const Eigen::Vector2f& u,
#                         const Eigen::Vector2f& v,
#                         const Eigen::Vector2f& w,
#                         const Eigen::Vector2f& p);
# 
# 
#       /** \brief Compute the cross product between 2D vectors.
#        * \param[in] p1 the first 2D vector
#        * \param[in] p2 the first 2D vector
#        */
#       float crossProduct (const Eigen::Vector2f& p1, const Eigen::Vector2f& p2) const
###

# factor.h(1.6.0)
# pcl/surface/3rdparty/poisson4/factor.h (1.7.2)
# namespace pcl
# namespace poisson
# 
#     double ArcTan2 (const double& y, const double& x);
#     double Angle (const double in[2]);
#     void Sqrt (const double in[2], double out[2]);
#     void Add (const double in1[2], const double in2[2], double out[2]);
#     void Subtract (const double in1[2], const double in2[2], double out[2]);
#     void Multiply (const double in1[2], const double in2[2], double out[2]);
#     void Divide (const double in1[2], const double in2[2], double out[2]);
# 
#     int Factor (double a1, double a0, double roots[1][2], const double& EPS);
#     int Factor (double a2, double a1, double a0, double roots[2][2], const double& EPS);
#     int Factor (double a3, double a2, double a1, double a0, double roots[3][2], const double& EPS);
#     int Factor (double a4, double a3, double a2, double a1, double a0, double roots[4][2], const double& EPS);
# 
#     int Solve (const double* eqns, const double* values, double* solutions, const int& dim);
###

# function_data.h (1.6.0)
# pcl/surface/3rdparty/poisson4/function_data.h (1.7.2)
# namespace pcl 
# namespace poisson 
# template<int Degree,class Real>
# class FunctionData
# cdef extern from "pcl/surface/function_data.h" namespace "pcl::poisson":
#     cdef cppclass FunctionData:
#         FunctionData()
#         int useDotRatios;
#         int normalize;
#         public:
#         const static int DOT_FLAG;
#         const static int D_DOT_FLAG;
#         const static int D2_DOT_FLAG;
#         const static int VALUE_FLAG;
#         const static int D_VALUE_FLAG;
#         int depth, res, res2;
#         Real *dotTable, *dDotTable, *d2DotTable;
#         Real *valueTables, *dValueTables;
#         PPolynomial<Degree> baseFunction;
#         PPolynomial<Degree-1> dBaseFunction;
#         PPolynomial<Degree+1>* baseFunctions;
#         virtual void setDotTables (const int& flags);
#         virtual void clearDotTables (const int& flags);
#         virtual void setValueTables (const int& flags, const double& smooth = 0);
#         virtual void setValueTables (const int& flags, const double& valueSmooth, const double& normalSmooth);
#         virtual void clearValueTables (void);
#         void set (const int& maxDepth, const PPolynomial<Degree>& F, const int& normalize, const int& useDotRatios = 1);
#         Real dotProduct (const double& center1, const double& width1,
#                          const double& center2, const double& width2) const;
#         Real dDotProduct (const double& center1, const double& width1,
#                           const double& center2, const double& width2) const;
#         Real d2DotProduct (const double& center1, const double& width1,
#                            const double& center2, const double& width2) const;
#         static inline int SymmetricIndex (const int& i1, const int& i2);
#         static inline int SymmetricIndex (const int& i1, const int& i2, int& index);
###

# geometry.h (1.6.0)
# pcl/surface/3rdparty/poisson4/geometry.h (1.7.2)
# namespace pcl
# namespace poisson 
#   {
#     template<class Real>
#     Real Random (void);
# 
#     template<class Real>
#     struct Point3D{Real coords[3];};
# 
#     template<class Real>
#     Point3D<Real> RandomBallPoint (void);
# 
#     template<class Real>
#     Point3D<Real> RandomSpherePoint (void);
# 
#     template<class Real>
#     double Length (const Point3D<Real>& p);
# 
#     template<class Real>
#     double SquareLength (const Point3D<Real>& p);
# 
#     template<class Real>
#     double Distance (const Point3D<Real>& p1, const Point3D<Real>& p2);
# 
#     template<class Real>
#     double SquareDistance (const Point3D<Real>& p1, const Point3D<Real>& p2);
# 
#     template <class Real>
#     void CrossProduct (const Point3D<Real>& p1, const Point3D<Real>& p2, Point3D<Real>& p);
# 
#     class Edge
#     {
#       public:
#         double p[2][2];
#         double Length (void) const
#         {
#           double d[2];
#           d[0]=p[0][0]-p[1][0];
#           d[1]=p[0][1]-p[1][1];
# 
#           return sqrt (d[0]*d[0]+d[1]*d[1]);
#         }
#     };
#     
#     class Triangle
#     {
#       public:
#         double p[3][3];
#         
#         double 
#         Area (void) const
#         {
#           double v1[3], v2[3], v[3];
#           for (int d=0;d<3;d++)
#           {
#             v1[d] = p[1][d]-p[0][d];
#             v2[d] = p[2][d]-p[0][d];
#           }
#           v[0] =  v1[1]*v2[2]-v1[2]*v2[1];
#           v[1] = -v1[0]*v2[2]+v1[2]*v2[0];
#           v[2] =  v1[0]*v2[1]-v1[1]*v2[0];
# 
#           return (sqrt (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / 2);
#         }
# 
#         double 
#         AspectRatio (void) const
#         {
#           double d=0;
#           int i, j;
#           for (i = 0; i < 3; i++)
#           {
#             for (i = 0; i < 3; i++)
#               for (j = 0; j < 3; j++)
#               {
#                 d += (p[(i+1)%3][j]-p[i][j])* (p[(i+1)%3][j]-p[i][j]);
#               }
#           }
#           return (Area () / d);
#         }
#     };
# 
#     class CoredPointIndex
#     {
#       public:
#         int index;
#         char inCore;
# 
#         int operator == (const CoredPointIndex& cpi) const {return (index==cpi.index) && (inCore==cpi.inCore);};
#         int operator != (const CoredPointIndex& cpi) const {return (index!=cpi.index) || (inCore!=cpi.inCore);};
#     };
# 
#     class EdgeIndex
#     {
#       public:
#         int idx[2];
#     };
# 
#     class CoredEdgeIndex
#     {
#       public:
#         CoredPointIndex idx[2];
#     };
# 
#     class TriangleIndex
#     {
#       public:
#         int idx[3];
#     };
# 
#     class TriangulationEdge
#     {
#       public:
#         TriangulationEdge (void);
#         int pIndex[2];
#         int tIndex[2];
#     };
# 
#     class TriangulationTriangle
#     {
#       public:
#         TriangulationTriangle (void);
#         int eIndex[3];
#     };
# 
#     template<class Real>
#     class Triangulation
#     {
#       public:
#         Triangulation () : points (),  edges (),  triangles (),  edgeMap () {}
# 
#         std::vector<Point3D<Real> >        points;
#         std::vector<TriangulationEdge>     edges;
#         std::vector<TriangulationTriangle> triangles;
# 
#         int 
#         factor (const int& tIndex, int& p1, int& p2, int& p3);
#         
#         double 
#         area (void);
# 
#         double 
#         area (const int& tIndex);
# 
#         double 
#         area (const int& p1, const int& p2, const int& p3);
# 
#         int 
#         flipMinimize (const int& eIndex);
# 
#         int 
#         addTriangle (const int& p1, const int& p2, const int& p3);
# 
#       protected:
#         hash_map<long long, int> edgeMap;
#         static long long EdgeIndex (const int& p1, const int& p2);
#         double area (const Triangle& t);
#     };
# 
# 
#     template<class Real> void 
#     EdgeCollapse (const Real& edgeRatio,
#                   std::vector<TriangleIndex>& triangles,
#                   std::vector< Point3D<Real> >& positions,
#                   std::vector<Point3D<Real> >* normals);
#     
#     template<class Real> void 
#     TriangleCollapse (const Real& edgeRatio,
#                       std::vector<TriangleIndex>& triangles,
#                       std::vector<Point3D<Real> >& positions,
#                       std::vector<Point3D<Real> >* normals);
# 
#     struct CoredVertexIndex
#     {
#       int idx;
#       bool inCore;
#     };
# 
#     class CoredMeshData
#     {
#       public:
#         CoredMeshData () : inCorePoints () {}
# 
#         virtual ~CoredMeshData () {}
# 
#         std::vector<Point3D<float> > inCorePoints;
#         
#         virtual void 
#         resetIterator () = 0;
# 
#         virtual int 
#         addOutOfCorePoint (const Point3D<float>& p) = 0;
# 
#         virtual int 
#         addPolygon (const std::vector< CoredVertexIndex >& vertices) = 0;
# 
#         virtual int 
#         nextOutOfCorePoint (Point3D<float>& p) = 0;
#         
#         virtual int 
#         nextPolygon (std::vector<CoredVertexIndex >& vertices) = 0;
# 
#         virtual int 
#         outOfCorePointCount () = 0;
#         
#         virtual int 
#         polygonCount () = 0;
#     };
# 
#     class CoredVectorMeshData : public CoredMeshData
#     {
#       std::vector<Point3D<float> > oocPoints;
#       std::vector< std::vector< int > > polygons;
#       int polygonIndex;
#       int oocPointIndex;
# 
#       public:
#         CoredVectorMeshData ();
#         
#         virtual ~CoredVectorMeshData () {}
# 
#         void resetIterator (void);
# 
#         int addOutOfCorePoint (const Point3D<float>& p);
#         int addPolygon (const std::vector< CoredVertexIndex >& vertices);
# 
#         int nextOutOfCorePoint (Point3D<float>& p);
#         int nextPolygon (std::vector< CoredVertexIndex >& vertices);
# 
#         int outOfCorePointCount (void);
#         int polygonCount (void);
#     };
# 
#     class CoredFileMeshData : public CoredMeshData
#     {
#       FILE *oocPointFile ,  *polygonFile;
#       int oocPoints ,  polygons;
#       public:
#         CoredFileMeshData ();
#         virtual ~CoredFileMeshData ();
# 
#         void resetIterator (void);
# 
#         int addOutOfCorePoint (const Point3D<float>& p);
#         int addPolygon (const std::vector< CoredVertexIndex >& vertices);
# 
#         int nextOutOfCorePoint (Point3D<float>& p);
#         int nextPolygon (std::vector< CoredVertexIndex >& vertices);
# 
#         int outOfCorePointCount (void);
#         int polygonCount (void);
#     };
#   }
# 
###

# gp3.h
# namespace pcl
#   /** \brief Returns if a point X is visible from point R (or the origin)
#     * when taking into account the segment between the points S1 and S2
#     * \param X 2D coordinate of the point
#     * \param S1 2D coordinate of the segment's first point
#     * \param S2 2D coordinate of the segment's secont point
#     * \param R 2D coorddinate of the reference point (defaults to 0,0)
#     * \ingroup surface
#     */
#   inline bool 
#   isVisible (const Eigen::Vector2f &X, const Eigen::Vector2f &S1, const Eigen::Vector2f &S2, 
#              const Eigen::Vector2f &R = Eigen::Vector2f::Zero ())
# 
# /** \brief GreedyProjectionTriangulation is an implementation of a greedy triangulation algorithm for 3D points
#   * based on local 2D projections. It assumes locally smooth surfaces and relatively smooth transitions between
#   * areas with different point densities.
#   * \author Zoltan Csaba Marton
#   * \ingroup surface
#   */
# template <typename PointInT>
# class GreedyProjectionTriangulation : public MeshConstruction<PointInT>
cdef extern from "pcl/surface/gp3.h" namespace "pcl::poisson":
    cdef cppclass GreedyProjectionTriangulation[In](MeshConstruction[In]):
        GreedyProjectionTriangulation()
#       public:
#       using MeshConstruction<PointInT>::tree_;
#       using MeshConstruction<PointInT>::input_;
#       using MeshConstruction<PointInT>::indices_;
#       typedef typename pcl::KdTree<PointInT> KdTree;
#       typedef typename pcl::KdTree<PointInT>::Ptr KdTreePtr;
#       typedef pcl::PointCloud<PointInT> PointCloudIn;
#       typedef typename PointCloudIn::Ptr PointCloudInPtr;
#       typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
#       // FIXME this enum should have a type.  Not be anonymous. 
#       // Otherplaces where consts are used probably should be fixed.
#       enum 
#       { 
#         NONE = -1,    // not-defined
#         FREE = 0,    
#         FRINGE = 1,  
#         BOUNDARY = 2,
#         COMPLETED = 3
#       };
# 
#       /** \brief Set the multiplier of the nearest neighbor distance to obtain the final search radius for each point
#        *  (this will make the algorithm adapt to different point densities in the cloud).
#        * \param[in] mu the multiplier
#       inline void setMu (double mu)
#       /** \brief Get the nearest neighbor distance multiplier. */
#       inline double getMu ()
#       /** \brief Set the maximum number of nearest neighbors to be searched for.
#         * \param[in] nnn the maximum number of nearest neighbors
#       inline void setMaximumNearestNeighbors (int nnn)
#       /** \brief Get the maximum number of nearest neighbors to be searched for. */
#       inline int getMaximumNearestNeighbors ()
#       /** \brief Set the sphere radius that is to be used for determining the k-nearest neighbors used for triangulating.
#         * \param[in] radius the sphere radius that is to contain all k-nearest neighbors
#         * \note This distance limits the maximum edge length!
#       inline void setSearchRadius (double radius)
#       /** \brief Get the sphere radius used for determining the k-nearest neighbors. */
#       inline double getSearchRadius ()
#       /** \brief Set the minimum angle each triangle should have.
#         * \param[in] minimum_angle the minimum angle each triangle should have
#         * \note As this is a greedy approach, this will have to be violated from time to time
#       inline void setMinimumAngle (double minimum_angle)
#       /** \brief Get the parameter for distance based weighting of neighbors. */
#       inline double getMinimumAngle ()
#       /** \brief Set the maximum angle each triangle can have.
#         * \param[in] maximum_angle the maximum angle each triangle can have
#         * \note For best results, its value should be around 120 degrees
#       inline void setMaximumAngle (double maximum_angle)
#       /** \brief Get the parameter for distance based weighting of neighbors. */
#       inline double getMaximumAngle ()
#       /** \brief Don't consider points for triangulation if their normal deviates more than this value from the query point's normal.
#         * \param[in] eps_angle maximum surface angle
#         * \note As normal estimation methods usually give smooth transitions at sharp edges, this ensures correct triangulation
#         *       by avoiding connecting points from one side to points from the other through forcing the use of the edge points.
#       inline void setMaximumSurfaceAngle (double eps_angle)
#       /** \brief Get the maximum surface angle. */
#       inline double getMaximumSurfaceAngle ()
#       /** \brief Set the flag if the input normals are oriented consistently.
#         * \param[in] consistent set it to true if the normals are consistently oriented
#       inline void setNormalConsistency (bool consistent)
#       /** \brief Get the flag for consistently oriented normals. */
#       inline bool getNormalConsistency ()
#       /** \brief Set the flag to order the resulting triangle vertices consistently (positive direction around normal).
#         * @note Assumes consistently oriented normals (towards the viewpoint) -- see setNormalConsistency ()
#         * \param[in] consistent_ordering set it to true if triangle vertices should be ordered consistently
#       inline void setConsistentVertexOrdering (bool consistent_ordering)
#       /** \brief Get the flag signaling consistently ordered triangle vertices. */
#       inline bool getConsistentVertexOrdering ()
#       /** \brief Get the state of each point after reconstruction.
#         * \note Options are defined as constants: FREE, FRINGE, COMPLETED, BOUNDARY and NONE
#       inline std::vector<int> getPointStates ()
#       /** \brief Get the ID of each point after reconstruction.
#         * \note parts are numbered from 0, a -1 denotes unconnected points
#       inline std::vector<int> getPartIDs ()
#       /** \brief Get the sfn list. */
#       inline std::vector<int> getSFN ()
#       /** \brief Get the ffn list. */
#       inline std::vector<int> getFFN ()
#       protected:
#       /** \brief The nearest neighbor distance multiplier to obtain the final search radius. */
#       double mu_;
#       /** \brief The nearest neighbors search radius for each point and the maximum edge length. */
#       double search_radius_;
#       /** \brief The maximum number of nearest neighbors accepted by searching. */
#       int nnn_;
#       /** \brief The preferred minimum angle for the triangles. */
#       double minimum_angle_;
#       /** \brief The maximum angle for the triangles. */
#       double maximum_angle_;
#       /** \brief Maximum surface angle. */
#       double eps_angle_;
#       /** \brief Set this to true if the normals of the input are consistently oriented. */
#       bool consistent_;
#       /** \brief Set this to true if the output triangle vertices should be consistently oriented. */
#       bool consistent_ordering_;
###

# grid_projection.h
# namespace pcl
# {
#   /** \brief The 12 edges of a cell. */
#   const int I_SHIFT_EP[12][2] = {
#     {0, 4}, {1, 5}, {2, 6}, {3, 7}, 
#     {0, 1}, {1, 2}, {2, 3}, {3, 0},
#     {4, 5}, {5, 6}, {6, 7}, {7, 4}
#   };
# 
#   const int I_SHIFT_PT[4] = {
#     0, 4, 5, 7
#   };
# 
#   const int I_SHIFT_EDGE[3][2] = {
#     {0,1}, {1,3}, {1,2}
#   };
# 
# 
#   /** \brief Grid projection surface reconstruction method.
#     * \author Rosie Li
#     *
#     * \note If you use this code in any academic work, please cite:
#     *   - Ruosi Li, Lu Liu, Ly Phan, Sasakthi Abeysinghe, Cindy Grimm, Tao Ju.
#     *     Polygonizing extremal surfaces with manifold guarantees.
#     *     In Proceedings of the 14th ACM Symposium on Solid and Physical Modeling, 2010.
#      * \ingroup surface
#     */
#   template <typename PointNT>
#   class GridProjection : public SurfaceReconstruction<PointNT>
#   {
#     public:
#       using SurfaceReconstruction<PointNT>::input_;
#       using SurfaceReconstruction<PointNT>::tree_;
# 
#       typedef typename pcl::PointCloud<PointNT>::Ptr PointCloudPtr;
# 
#       typedef typename pcl::KdTree<PointNT> KdTree;
#       typedef typename pcl::KdTree<PointNT>::Ptr KdTreePtr;
# 
#       /** \brief Data leaf. */
#       struct Leaf
#       {
#         Leaf () : data_indices (), pt_on_surface (), vect_at_grid_pt () {}
# 
#         std::vector<int> data_indices;
#         Eigen::Vector4f pt_on_surface; 
#         Eigen::Vector3f vect_at_grid_pt;
#       };
# 
#       typedef boost::unordered_map<int, Leaf, boost::hash<int>, std::equal_to<int>, Eigen::aligned_allocator<int> > HashMap;
# 
#       /** \brief Constructor. */ 
#       GridProjection ();
# 
#       /** \brief Constructor. 
#         * \param in_resolution set the resolution of the grid
#         */ 
#       GridProjection (double in_resolution);
# 
#       /** \brief Destructor. */
#       ~GridProjection ();
# 
#       /** \brief Set the size of the grid cell
#         * \param resolution  the size of the grid cell
#         */
#       inline void 
#       setResolution (double resolution)
#       {
#         leaf_size_ = resolution;
#       }
# 
#       inline double 
#       getResolution () const
#       {
#         return (leaf_size_);
#       }
# 
#       /** \brief When averaging the vectors, we find the union of all the input data 
#         *  points within the padding area,and do a weighted average. Say if the padding
#         *  size is 1, when we process cell (x,y,z), we will find union of input data points
#         *  from (x-1) to (x+1), (y-1) to (y+1), (z-1) to (z+1)(in total, 27 cells). In this
#         *  way, even the cells itself doesnt contain any data points, we will stil process it
#         *  because there are data points in the padding area. This can help us fix holes which 
#         *  is smaller than the padding size.
#         * \param padding_size The num of padding cells we want to create 
#         */
#       inline void 
#       setPaddingSize (int padding_size)
#       {
#         padding_size_ = padding_size;
#       }
#       inline int 
#       getPaddingSize () const
#       {
#         return (padding_size_);
#       }
# 
#       /** \brief Set this only when using the k nearest neighbors search 
#         * instead of finding the point union
#         * \param k The number of nearest neighbors we are looking for
#         */
#       inline void 
#       setNearestNeighborNum (int k)
#       {
#         k_ = k;
#       }
#       inline int 
#       getNearestNeighborNum () const
#       {
#         return (k_);
#       }
# 
#       /** \brief Binary search is used in projection. given a point x, we find another point
#         *  which is 3*cell_size_ far away from x. Then we do a binary search between these 
#         *  two points to find where the projected point should be.
#         */
#       inline void 
#       setMaxBinarySearchLevel (int max_binary_search_level)
#       {
#         max_binary_search_level_ = max_binary_search_level;
#       }
#       inline int 
#       getMaxBinarySearchLevel () const
#       {
#         return (max_binary_search_level_);
#       }
# 
#       ///////////////////////////////////////////////////////////
#       inline const HashMap& 
#       getCellHashMap () const
#       {
#         return (cell_hash_map_);
#       }
# 
#       inline const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> >& 
#       getVectorAtDataPoint () const
#       {
#         return (vector_at_data_point_);
#       }
#       
#       inline const std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& 
#       getSurface () const
#       {
#         return (surface_);
#       }
# 
#     protected:
#       /** \brief Get the bounding box for the input data points, also calculating the
#         * cell size, and the gaussian scale factor
#         */
#       void 
#       getBoundingBox ();
# 
#       /** \brief The actual surface reconstruction method.
#         * \param[out] polygons the resultant polygons, as a set of vertices. The Vertices structure contains an array of point indices.
#         */
#       bool
#       reconstructPolygons (std::vector<pcl::Vertices> &polygons);
# 
#       /** \brief Create the surface. 
#         *
#         * The 1st step is filling the padding, so that all the cells in the padding
#         * area are in the hash map. The 2nd step is store the vector, and projected
#         * point. The 3rd step is finding all the edges intersects the surface, and
#         * creating surface.
#         *
#         * \param[out] output the resultant polygonal mesh
#         */
#       void 
#       performReconstruction (pcl::PolygonMesh &output);
# 
#       /** \brief Create the surface. 
#         *
#         * The 1st step is filling the padding, so that all the cells in the padding
#         * area are in the hash map. The 2nd step is store the vector, and projected
#         * point. The 3rd step is finding all the edges intersects the surface, and
#         * creating surface.
#         *
#         * \param[out] points the resultant points lying on the surface
#         * \param[out] polygons the resultant polygons, as a set of vertices. The Vertices structure contains an array of point indices.
#         */
#       void 
#       performReconstruction (pcl::PointCloud<PointNT> &points, 
#                              std::vector<pcl::Vertices> &polygons);
# 
#       /** \brief When the input data points don't fill into the 1*1*1 box, 
#         * scale them so that they can be filled in the unit box. Otherwise, 
#         * it will be some drawing problem when doing visulization
#         * \param scale_factor scale all the input data point by scale_factor
#         */
#       void 
#       scaleInputDataPoint (double scale_factor);
# 
#       /** \brief Get the 3d index (x,y,z) of the cell based on the location of
#         * the cell
#         * \param p the coordinate of the input point
#         * \param index the output 3d index
#         */
#       inline void 
#       getCellIndex (const Eigen::Vector4f &p, Eigen::Vector3i& index) const
#       {
#         for (int i = 0; i < 3; ++i)
#           index[i] = static_cast<int> ((p[i] - min_p_(i)) / leaf_size_);
#       }
# 
#       /** \brief Given the 3d index (x, y, z) of the cell, get the 
#         * coordinates of the cell center
#         * \param index the output 3d index
#         * \param center the resultant cell center
#         */
#       inline void
#       getCellCenterFromIndex (const Eigen::Vector3i &index, Eigen::Vector4f &center) const
#       {
#         for (int i = 0; i < 3; ++i)
#           center[i] = 
#             min_p_[i] + static_cast<float> (index[i]) * 
#             static_cast<float> (leaf_size_) + 
#             static_cast<float> (leaf_size_) / 2.0f;
#       }
# 
#       /** \brief Given cell center, caluate the coordinates of the eight vertices of the cell
#         * \param cell_center the coordinates of the cell center
#         * \param pts the coordinates of the 8 vertices
#         */
#       void 
#       getVertexFromCellCenter (const Eigen::Vector4f &cell_center, 
#                                std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > &pts) const;
# 
#       /** \brief Given an index (x, y, z) in 3d, translate it into the index 
#         * in 1d
#         * \param index the index of the cell in (x,y,z) 3d format
#         */
#       inline int 
#       getIndexIn1D (const Eigen::Vector3i &index) const
#       {
#         //assert(data_size_ > 0);
#         return (index[0] * data_size_ * data_size_ + 
#                 index[1] * data_size_ + index[2]);
#       }
# 
#       /** \brief Given an index in 1d, translate it into the index (x, y, z) 
#         * in 3d
#         * \param index_1d the input 1d index
#         * \param index_3d the output 3d index
#         */
#       inline void 
#       getIndexIn3D (int index_1d, Eigen::Vector3i& index_3d) const
#       {
#         //assert(data_size_ > 0);
#         index_3d[0] = index_1d / (data_size_ * data_size_);
#         index_1d -= index_3d[0] * data_size_ * data_size_;
#         index_3d[1] = index_1d / data_size_;
#         index_1d -= index_3d[1] * data_size_;
#         index_3d[2] = index_1d;
#       }
# 
#       /** \brief For a given 3d index of a cell, test whether the cells within its
#         * padding area exist in the hash table, if no, create an entry for that cell.
#         * \param index the index of the cell in (x,y,z) format
#         */
#       void 
#       fillPad (const Eigen::Vector3i &index);
# 
#       /** \brief Obtain the index of a cell and the pad size.
#         * \param index the input index
#         * \param pt_union_indices the union of input data points within the cell and padding cells
#         */
#       void 
#       getDataPtsUnion (const Eigen::Vector3i &index, std::vector <int> &pt_union_indices);
# 
#       /** \brief Given the index of a cell, exam it's up, left, front edges, and add
#         * the vectices to m_surface list.the up, left, front edges only share 4
#         * points, we first get the vectors at these 4 points and exam whether those
#         * three edges are intersected by the surface \param index the input index
#         * \param pt_union_indices the union of input data points within the cell and padding cells
#         */
#       void 
#       createSurfaceForCell (const Eigen::Vector3i &index, std::vector <int> &pt_union_indices);
# 
# 
#       /** \brief Given the coordinates of one point, project it onto the surface, 
#         * return the projected point. Do a binary search between p and p+projection_distance 
#         * to find the projected point
#         * \param p the coordinates of the input point
#         * \param pt_union_indices the union of input data points within the cell and padding cells
#         * \param projection the resultant point projected
#         */
#       void
#       getProjection (const Eigen::Vector4f &p, std::vector<int> &pt_union_indices, Eigen::Vector4f &projection);
# 
#       /** \brief Given the coordinates of one point, project it onto the surface,
#         * return the projected point. Find the plane which fits all the points in
#         *  pt_union_indices, projected p to the plane to get the projected point.
#         * \param p the coordinates of the input point
#         * \param pt_union_indices the union of input data points within the cell and padding cells
#         * \param projection the resultant point projected
#         */
#       void 
#       getProjectionWithPlaneFit (const Eigen::Vector4f &p, 
#                                  std::vector<int> &pt_union_indices, 
#                                  Eigen::Vector4f &projection);
# 
# 
#       /** \brief Given the location of a point, get it's vector
#         * \param p the coordinates of the input point
#         * \param pt_union_indices the union of input data points within the cell and padding cells
#         * \param vo the resultant vector
#         */
#       void
#       getVectorAtPoint (const Eigen::Vector4f &p, 
#                         std::vector <int> &pt_union_indices, Eigen::Vector3f &vo);
# 
#       /** \brief Given the location of a point, get it's vector
#         * \param p the coordinates of the input point
#         * \param k_indices the k nearest neighbors of the query point
#         * \param k_squared_distances the squared distances of the k nearest 
#         * neighbors to the query point
#         * \param vo the resultant vector
#         */
#       void
#       getVectorAtPointKNN (const Eigen::Vector4f &p, 
#                            std::vector<int> &k_indices, 
#                            std::vector<float> &k_squared_distances,
#                            Eigen::Vector3f &vo);
# 
#       /** \brief Get the magnitude of the vector by summing up the distance.
#         * \param p the coordinate of the input point
#         * \param pt_union_indices the union of input data points within the cell and padding cells
#         */
#       double 
#       getMagAtPoint (const Eigen::Vector4f &p, const std::vector <int> &pt_union_indices);
# 
#       /** \brief Get the 1st derivative
#         * \param p the coordinate of the input point
#         * \param vec the vector at point p
#         * \param pt_union_indices the union of input data points within the cell and padding cells
#         */
#       double 
#       getD1AtPoint (const Eigen::Vector4f &p, const Eigen::Vector3f &vec, 
#                     const std::vector <int> &pt_union_indices);
# 
#       /** \brief Get the 2nd derivative
#         * \param p the coordinate of the input point
#         * \param vec the vector at point p
#         * \param pt_union_indices the union of input data points within the cell and padding cells
#         */
#       double 
#       getD2AtPoint (const Eigen::Vector4f &p, const Eigen::Vector3f &vec, 
#                     const std::vector <int> &pt_union_indices);
# 
#       /** \brief Test whether the edge is intersected by the surface by 
#         * doing the dot product of the vector at two end points. Also test 
#         * whether the edge is intersected by the maximum surface by examing 
#         * the 2nd derivative of the intersection point 
#         * \param end_pts the two points of the edge
#         * \param vect_at_end_pts 
#         * \param pt_union_indices the union of input data points within the cell and padding cells
#         */
#       bool 
#       isIntersected (const std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > &end_pts, 
#                      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &vect_at_end_pts, 
#                      std::vector <int> &pt_union_indices);
# 
#       /** \brief Find point where the edge intersects the surface.
#         * \param level binary search level
#         * \param end_pts the two end points on the edge
#         * \param vect_at_end_pts the vectors at the two end points
#         * \param start_pt the starting point we use for binary search
#         * \param pt_union_indices the union of input data points within the cell and padding cells
#         * \param intersection the resultant intersection point
#         */
#       void
#       findIntersection (int level, 
#                         const std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > &end_pts, 
#                         const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &vect_at_end_pts, 
#                         const Eigen::Vector4f &start_pt, 
#                         std::vector<int> &pt_union_indices,
#                         Eigen::Vector4f &intersection);
# 
#       /** \brief Go through all the entries in the hash table and update the
#        * cellData. 
#        *
#        * When creating the hash table, the pt_on_surface field store the center
#        * point of the cell.After calling this function, the projection operator will
#        * project the center point onto the surface, and the pt_on_surface field will
#        * be updated using the projected point.Also the vect_at_grid_pt field will be
#        * updated using the vector at the upper left front vertex of the cell.
#        *
#        * \param index_1d the index of the cell after flatting it's 3d index into a 1d array
#        * \param index_3d the index of the cell in (x,y,z) 3d format
#        * \param pt_union_indices the union of input data points within the cell and pads
#        * \param cell_data information stored in the cell
#        */
#       void
#       storeVectAndSurfacePoint (int index_1d, const Eigen::Vector3i &index_3d, 
#                                 std::vector<int> &pt_union_indices, const Leaf &cell_data);
# 
#       /** \brief Go through all the entries in the hash table and update the cellData. 
#         * When creating the hash table, the pt_on_surface field store the center point
#         * of the cell.After calling this function, the projection operator will project the 
#         * center point onto the surface, and the pt_on_surface field will be updated 
#         * using the projected point.Also the vect_at_grid_pt field will be updated using 
#         * the vector at the upper left front vertex of the cell. When projecting the point 
#         * and calculating the vector, using K nearest neighbors instead of using the 
#         * union of input data point within the cell and pads.
#         *
#         * \param index_1d the index of the cell after flatting it's 3d index into a 1d array
#         * \param index_3d the index of the cell in (x,y,z) 3d format
#         * \param cell_data information stored in the cell
#         */
#       void 
#       storeVectAndSurfacePointKNN (int index_1d, const Eigen::Vector3i &index_3d, const Leaf &cell_data);
# 
#     private:
#       /** \brief Map containing the set of leaves. */
#       HashMap cell_hash_map_;
# 
#       /** \brief Min and max data points. */
#       Eigen::Vector4f min_p_, max_p_;
# 
#       /** \brief The size of a leaf. */
#       double leaf_size_;
# 
#       /** \brief Gaussian scale. */
#       double gaussian_scale_;
# 
#       /** \brief Data size. */
#       int data_size_;
# 
#       /** \brief Max binary search level. */
#       int max_binary_search_level_;
# 
#       /** \brief Number of neighbors (k) to use. */
#       int k_;
# 
#       /** \brief Padding size. */
#       int padding_size_;
# 
#       /** \brief The point cloud input (XYZ+Normals). */
#       PointCloudPtr data_;
# 
#       /** \brief Store the surface normal(vector) at the each input data point. */
#       std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > vector_at_data_point_;
#       
#       /** \brief An array of points which lay on the output surface. */
#       std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > surface_;
# 
#       /** \brief Bit map which tells if there is any input data point in the cell. */
#       boost::dynamic_bitset<> occupied_cell_list_;
# 
#       /** \brief Class get name method. */
#       std::string getClassName () const { return ("GridProjection"); }
# 
#     public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
###

# hash.h (1.6.0)
# pcl/surface/3rdparty/poisson4/hash.h (1.7.2)
###

# marching_cubes.h (1.6.0)
# pcl/surface/3rdparty/poisson4/marching_cubes_poisson.h (1.7.2)
# 
# namespace pcl
# {
#   /*
#    * Tables, and functions, derived from Paul Bourke's Marching Cubes implementation:
#    * http://paulbourke.net/geometry/polygonise/
#    * Cube vertex indices:
#    *   y_dir 4 ________ 5
#    *         /|       /|
#    *       /  |     /  |
#    *   7 /_______ /    |
#    *    |     |  |6    |
#    *    |    0|__|_____|1 x_dir
#    *    |    /   |    /
#    *    |  /     |  /
#    z_dir|/_______|/
#    *   3          2
#    */
#   const unsigned int edgeTable[256] = {
#     0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
#     0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
#     0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
#     0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
#     0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
#     0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
#     0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
#     0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
#     0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
#     0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
#     0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
#     0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
#     0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
#     0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
#     0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
#     0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
#     0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
#     0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
#     0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
#     0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
#     0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
#     0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
#     0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
#     0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
#     0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
#     0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
#     0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
#     0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
#     0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
#     0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
#     0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
#     0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
#   };
#   const int triTable[256][16] = {
#     {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
#     {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
#     {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
#     {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
#     {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
#     {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
#     {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
#     {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
#     {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
#     {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
#     {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
#     {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
#     {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
#     {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
#     {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
#     {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
#     {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
#     {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
#     {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
#     {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
#     {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
#     {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
#     {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
#     {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
#     {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
#     {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
#     {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
#     {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
#     {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
#     {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
#     {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
#     {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
#     {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
#     {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
#     {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
#     {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
#     {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
#     {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
#     {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
#     {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
#     {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
#     {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
#     {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
#     {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
#     {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
#     {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
#     {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
#     {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
#     {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
#     {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
#     {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
#     {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
#     {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
#     {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
#     {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
#     {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
#     {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
#     {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
#     {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
#     {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
#     {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
#     {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
#     {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
#     {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
#     {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
#     {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
#     {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
#     {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
#     {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
#     {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
#     {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
#     {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
#     {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
#     {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
#     {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
#     {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
#     {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
#     {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
#     {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
#     {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
#     {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
#     {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
#     {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
#     {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
#     {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
#     {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
#     {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
#     {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
#     {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
#     {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
#     {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
#     {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
#     {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
#     {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
#     {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
#     {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
#     {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
#     {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
#     {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
#     {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
#     {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
#     {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
#     {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
#     {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
#     {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
#     {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
#     {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
#     {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
#     {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
#     {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
#     {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
#     {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
#     {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
#     {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
#     {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
#     {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
#     {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
#     {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
#     {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
#     {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
#     {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
#     {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
#     {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
#     {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
#     {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
#     {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
#     {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
#     {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
#     {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
#     {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
#     {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
#     {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
#     {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
#     {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
#     {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
#     {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
#     {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
#     {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
#     {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
#     {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
#     {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
#     {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
#     {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
#     {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
#     {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
#     {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
#     {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
#     {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
#     {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
#     {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
#     {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
#     {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
#     {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
#     {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
#     {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
#     {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
#     {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
#     {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
#     {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
#     {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
#     {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
#     {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
#     {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
#     {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
#     {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
#     {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
#     {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
#     {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
#     {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
#     {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
#     {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
#     {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
#     {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
#     {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
#     {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
#     {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
#     {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
#     {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
#     {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
#     {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
#     {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
#     {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
#     {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
#     {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
#     {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
#   };
# 
# 
#   /** \brief The marching cubes surface reconstruction algorithm. This is an abstract class that takes a grid and
#     * extracts the isosurface as a mesh, based on the original marching cubes paper:
#     *
#     * Lorensen W.E., Cline H.E., "Marching cubes: A high resolution 3d surface construction algorithm",
#     * SIGGRAPH '87
#     *
#     * \author Alexandru E. Ichim
#     * \ingroup surface
#     */
#   template <typename PointNT>
#   class MarchingCubes : public SurfaceReconstruction<PointNT>
#   {
#     public:
#       using SurfaceReconstruction<PointNT>::input_;
#       using SurfaceReconstruction<PointNT>::tree_;
# 
#       typedef typename pcl::PointCloud<PointNT>::Ptr PointCloudPtr;
# 
#       typedef typename pcl::KdTree<PointNT> KdTree;
#       typedef typename pcl::KdTree<PointNT>::Ptr KdTreePtr;
# 
# 
#       /** \brief Constructor. */
#       MarchingCubes ();
# 
#       /** \brief Destructor. */
#       ~MarchingCubes ();
# 
# 
#       /** \brief Method that sets the iso level of the surface to be extracted.
#         * \param[in] iso_level the iso level.
#         */
#       inline void
#       setIsoLevel (float iso_level)
#       { iso_level_ = iso_level; }
# 
#       /** \brief Method that returns the iso level of the surface to be extracted. */
#       inline float
#       getIsoLevel ()
#       { return iso_level_; }
# 
#       /** \brief Method that sets the marching cubes grid resolution.
#         * \param[in] res_x the resolution of the grid along the x-axis
#         * \param[in] res_y the resolution of the grid along the y-axis
#         * \param[in] res_z the resolution of the grid along the z-axis
#         */
#       inline void
#       setGridResolution (int res_x, int res_y, int res_z)
#       { res_x_ = res_x; res_y_ = res_y; res_z_ = res_z; }
# 
# 
#       /** \brief Method to get the marching cubes grid resolution.
#         * \param[in] res_x the resolution of the grid along the x-axis
#         * \param[in] res_y the resolution of the grid along the y-axis
#         * \param[in] res_z the resolution of the grid along the z-axis
#         */
#       inline void
#       getGridResolution (int &res_x, int &res_y, int &res_z)
#       { res_x = res_x_; res_y = res_y_; res_z = res_z_; }
# 
#       /** \brief Method that sets the parameter that defines how much free space should be left inside the grid between
#         * the bounding box of the point cloud and the grid limits. Does not affect the resolution of the grid, it just
#         * changes the voxel size accordingly.
#         * \param[in] percentage the percentage of the bounding box that should be left empty between the bounding box and
#         * the grid limits.
#         */
#       inline void
#       setPercentageExtendGrid (float percentage)
#       { percentage_extend_grid_ = percentage; }
# 
#       /** \brief Method that gets the parameter that defines how much free space should be left inside the grid between
#         * the bounding box of the point cloud and the grid limits, as a percentage of the bounding box.
#         */
#       inline float
#       getPercentageExtendGrid ()
#       { return percentage_extend_grid_; }
# 
# protected:
#       /** \brief The data structure storing the 3D grid */
#       std::vector<float> grid_;
# 
#       /** \brief The grid resolution */
#       int res_x_, res_y_, res_z_;
# 
#       /** \brief Parameter that defines how much free space should be left inside the grid between
#         * the bounding box of the point cloud and the grid limits, as a percentage of the bounding box.*/
#       float percentage_extend_grid_;
# 
#       /** \brief Min and max data points. */
#       Eigen::Vector4f min_p_, max_p_;
# 
#       /** \brief The iso level to be extracted. */
#       float iso_level_;
# 
#       /** \brief Convert the point cloud into voxel data. */
#       virtual void
#       voxelizeData () = 0;
# 
#       /** \brief Interpolate along the voxel edge.
#         * \param[in] p1 The first point on the edge
#         * \param[in] p2 The second point on the edge
#         * \param[in] val_p1 The scalar value at p1
#         * \param[in] val_p2 The scalar value at p2
#         * \param[out] output The interpolated point along the edge
#         */
#       void
#       interpolateEdge (Eigen::Vector3f &p1, Eigen::Vector3f &p2, float val_p1, float val_p2, Eigen::Vector3f &output);
# 
# 
#       /** \brief Calculate out the corresponding polygons in the leaf node
#         * \param leaf_node the leaf node to be checked
#         * \param index_3d the 3d index of the leaf node to be checked
#         * \param cloud point cloud to store the vertices of the polygon
#        */
#       void
#       createSurface (std::vector<float> &leaf_node,
#                      Eigen::Vector3i &index_3d,
#                      pcl::PointCloud<PointNT> &cloud);
# 
#       /** \brief Get the bounding box for the input data points. */
#       void
#       getBoundingBox ();
# 
# 
#       /** \brief Method that returns the scalar value at the given grid position.
#         * \param[in] pos The 3D position in the grid
#         */
#       float
#       getGridValue (Eigen::Vector3i pos);
# 
#       /** \brief Method that returns the scalar values of the neighbors of a given 3D position in the grid.
#         * \param[in] index3d the point in the grid
#         * \param[out] leaf the set of values
#         */
#       void
#       getNeighborList1D (std::vector<float> &leaf,
#                          Eigen::Vector3i &index3d);
# 
#       /** \brief Class get name method. */
#       std::string getClassName () const { return ("MarchingCubes"); }
# 
#       /** \brief Extract the surface.
#         * \param[out] output the resultant polygonal mesh
#         */
#        void
#        performReconstruction (pcl::PolygonMesh &output);
# 
#        /** \brief Extract the surface.
#          * \param[out] points the points of the extracted mesh
#          * \param[out] polygons the connectivity between the point of the extracted mesh.
#          */
#        void
#        performReconstruction (pcl::PointCloud<PointNT> &points,
#                               std::vector<pcl::Vertices> &polygons);
# 
#     public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
###

# marching_cubes_hoppe.h (1.6.0)
# pcl/surface/3rdparty/poisson4/marching_cubes_poisson.h (1.7.2) ?
# namespace pcl
# {
#    /** \brief The marching cubes surface reconstruction algorithm, using a signed distance function based on the distance
#      * from tangent planes, proposed by Hoppe et. al. in:
#      * Hoppe H., DeRose T., Duchamp T., MC-Donald J., Stuetzle W., "Surface reconstruction from unorganized points",
#      * SIGGRAPH '92
#      * \author Alexandru E. Ichim
#      * \ingroup surface
#      */
#   template <typename PointNT>
#   class MarchingCubesHoppe : public MarchingCubes<PointNT>
#   {
#     public:
#       using SurfaceReconstruction<PointNT>::input_;
#       using SurfaceReconstruction<PointNT>::tree_;
#       using MarchingCubes<PointNT>::grid_;
#       using MarchingCubes<PointNT>::res_x_;
#       using MarchingCubes<PointNT>::res_y_;
#       using MarchingCubes<PointNT>::res_z_;
#       using MarchingCubes<PointNT>::min_p_;
#       using MarchingCubes<PointNT>::max_p_;
# 
#       typedef typename pcl::PointCloud<PointNT>::Ptr PointCloudPtr;
# 
#       typedef typename pcl::KdTree<PointNT> KdTree;
#       typedef typename pcl::KdTree<PointNT>::Ptr KdTreePtr;
# 
# 
#       /** \brief Constructor. */
#       MarchingCubesHoppe ();
# 
#       /** \brief Destructor. */
#       ~MarchingCubesHoppe ();
# 
#       /** \brief Convert the point cloud into voxel data. */
#       void
#       voxelizeData ();
# 
# 
#     public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
###

# marching_cubes_poisson.h (1.6.0)
# pcl/surface/3rdparty/poisson4/marching_cubes_poisson.h (1.7.2)
# namespace pcl {
#   namespace poisson {
# 
# 
#     class Square
#     {
#     public:
#       const static int CORNERS = 4, EDGES = 4, NEIGHBORS = 4;
#       static int  CornerIndex (const int& x, const int& y);
#       static void FactorCornerIndex (const int& idx, int& x, int& y);
#       static int  EdgeIndex (const int& orientation, const int& i);
#       static void FactorEdgeIndex (const int& idx, int& orientation, int& i);
# 
#       static int  ReflectCornerIndex (const int& idx, const int& edgeIndex);
#       static int  ReflectEdgeIndex (const int& idx, const int& edgeIndex);
# 
#       static void EdgeCorners (const int& idx, int& c1, int &c2);
#     };
# 
#     class Cube{
#     public:
#       const static int CORNERS = 8, EDGES = 12, NEIGHBORS = 6;
# 
#       static int CornerIndex (const int& x, const int& y, const int& z);
#       static void FactorCornerIndex (const int& idx, int& x, int& y, int& z);
#       static int EdgeIndex (const int& orientation, const int& i, const int& j);
#       static void FactorEdgeIndex (const int& idx, int& orientation, int& i, int &j);
#       static int FaceIndex (const int& dir, const int& offSet);
#       static int FaceIndex (const int& x, const int& y, const int& z);
#       static void FactorFaceIndex (const int& idx, int& x, int &y, int& z);
#       static void FactorFaceIndex (const int& idx, int& dir, int& offSet);
# 
#       static int AntipodalCornerIndex (const int& idx);
#       static int FaceReflectCornerIndex (const int& idx, const int& faceIndex);
#       static int FaceReflectEdgeIndex (const int& idx, const int& faceIndex);
#       static int FaceReflectFaceIndex (const int& idx, const int& faceIndex);
#       static int EdgeReflectCornerIndex   (const int& idx, const int& edgeIndex);
#       static int EdgeReflectEdgeIndex (const int& edgeIndex);
# 
#       static int FaceAdjacentToEdges (const int& eIndex1, const int& eIndex2);
#       static void FacesAdjacentToEdge (const int& eIndex, int& f1Index, int& f2Index);
# 
#       static void EdgeCorners (const int& idx, int& c1, int &c2);
#       static void FaceCorners (const int& idx, int& c1, int &c2, int& c3, int& c4);
#     };
# 
#     class MarchingSquares
#     {
#       static double Interpolate (const double& v1, const double& v2);
#       static void SetVertex (const int& e, const double values[Square::CORNERS], const double& iso);
#     public:
#       const static int MAX_EDGES = 2;
#       static const int edgeMask[1<<Square::CORNERS];
#       static const int edges[1<<Square::CORNERS][2*MAX_EDGES+1];
#       static double vertexList[Square::EDGES][2];
# 
#       static int GetIndex (const double values[Square::CORNERS], const double& iso);
#       static int IsAmbiguous (const double v[Square::CORNERS] ,const double& isoValue);
#       static int AddEdges (const double v[Square::CORNERS], const double& isoValue, Edge* edges);
#       static int AddEdgeIndices (const double v[Square::CORNERS], const double& isoValue, int* edges);
#     };
# 
#     class MarchingCubes
#     {
#       static double Interpolate (const double& v1, const double& v2);
#       static void SetVertex (const int& e, const double values[Cube::CORNERS], const double& iso);
#       static int GetFaceIndex (const double values[Cube::CORNERS], const double& iso, const int& faceIndex);
# 
#       static float Interpolate (const float& v1, const float& v2);
#       static void SetVertex (const int& e, const float values[Cube::CORNERS], const float& iso);
#       static int GetFaceIndex (const float values[Cube::CORNERS], const float& iso, const int& faceIndex);
# 
#       static int GetFaceIndex (const int& mcIndex, const int& faceIndex);
#     public:
#       const static int MAX_TRIANGLES=5;
#       static const int edgeMask[1<<Cube::CORNERS];
#       static const int triangles[1<<Cube::CORNERS][3*MAX_TRIANGLES+1];
#       static const int cornerMap[Cube::CORNERS];
#       static double vertexList[Cube::EDGES][3];
# 
#       static int AddTriangleIndices (const int& mcIndex, int* triangles);
# 
#       static int GetIndex (const double values[Cube::CORNERS],const double& iso);
#       static int IsAmbiguous (const double v[Cube::CORNERS],const double& isoValue,const int& faceIndex);
#       static int HasRoots (const double v[Cube::CORNERS],const double& isoValue);
#       static int HasRoots (const double v[Cube::CORNERS],const double& isoValue,const int& faceIndex);
#       static int AddTriangles (const double v[Cube::CORNERS],const double& isoValue,Triangle* triangles);
#       static int AddTriangleIndices (const double v[Cube::CORNERS],const double& isoValue,int* triangles);
# 
#       static int GetIndex (const float values[Cube::CORNERS], const float& iso);
#       static int IsAmbiguous (const float v[Cube::CORNERS], const float& isoValue, const int& faceIndex);
#       static int HasRoots (const float v[Cube::CORNERS], const float& isoValue);
#       static int HasRoots (const float v[Cube::CORNERS], const float& isoValue, const int& faceIndex);
#       static int AddTriangles (const float v[Cube::CORNERS], const float& isoValue, Triangle* triangles);
#       static int AddTriangleIndices (const float v[Cube::CORNERS], const float& isoValue, int* triangles);
# 
#       static int IsAmbiguous (const int& mcIndex, const int& faceIndex);
#       static int HasRoots (const int& mcIndex);
#       static int HasFaceRoots (const int& mcIndex, const int& faceIndex);
#       static int HasEdgeRoots (const int& mcIndex, const int& edgeIndex);
#     };
# 
# 
###

# marching_cubes_rbf.h (1.6.0)
# pcl/surface/3rdparty/poisson4/marching_cubes_poisson.h (1.7.2) ?
# namespace pcl
# {
#   /** \brief The marching cubes surface reconstruction algorithm, using a signed distance function based on radial
#     * basis functions. Partially based on:
#     * Carr J.C., Beatson R.K., Cherrie J.B., Mitchell T.J., Fright W.R., McCallum B.C. and Evans T.R.,
#     * "Reconstruction and representation of 3D objects with radial basis functions"
#     * SIGGRAPH '01
#     *
#     * \author Alexandru E. Ichim
#     * \ingroup surface
#     */
#   template <typename PointNT>
#   class MarchingCubesRBF : public MarchingCubes<PointNT>
#   {
#     public:
#       using SurfaceReconstruction<PointNT>::input_;
#       using SurfaceReconstruction<PointNT>::tree_;
#       using MarchingCubes<PointNT>::grid_;
#       using MarchingCubes<PointNT>::res_x_;
#       using MarchingCubes<PointNT>::res_y_;
#       using MarchingCubes<PointNT>::res_z_;
#       using MarchingCubes<PointNT>::min_p_;
#       using MarchingCubes<PointNT>::max_p_;
# 
#       typedef typename pcl::PointCloud<PointNT>::Ptr PointCloudPtr;
# 
#       typedef typename pcl::KdTree<PointNT> KdTree;
#       typedef typename pcl::KdTree<PointNT>::Ptr KdTreePtr;
# 
# 
#       /** \brief Constructor. */
#       MarchingCubesRBF ();
# 
#       /** \brief Destructor. */
#       ~MarchingCubesRBF ();
# 
#       /** \brief Convert the point cloud into voxel data. */
#       void
#       voxelizeData ();
# 
# 
#       /** \brief Set the off-surface points displacement value.
#         * \param[in] epsilon the value
#         */
#       inline void
#       setOffSurfaceDisplacement (float epsilon)
#       { off_surface_epsilon_ = epsilon; }
# 
#       /** \brief Get the off-surface points displacement value. */
#       inline float
#       getOffSurfaceDisplacement ()
#       { return off_surface_epsilon_; }
# 
# 
#     protected:
#       /** \brief the Radial Basis Function kernel. */
#       double
#       kernel (Eigen::Vector3d c, Eigen::Vector3d x);
# 
#       /** \brief The off-surface displacement value. */
#       float off_surface_epsilon_;
# 
#     public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
# }
###

# mls.h
cdef extern from "pcl/surface/mls.h" namespace "pcl":
    cdef cppclass MovingLeastSquares[I,O]:
        MovingLeastSquares()
        void setInputCloud (shared_ptr[cpp.PointCloud[I]])
        void setSearchRadius (double)
        void setComputeNormals (bool compute_normals)
        void setPolynomialOrder(bool)
        void setPolynomialFit(int)
        # void process(cpp.PointCloud[O] &) except +
        void process(cpp.PointCloud[O] &) except +
        
        # KdTree
        void setSearchMethod (const pclkdt.KdTreePtr_t &tree)
        pclkdt.KdTreePtr_t getSearchMethod ()

ctypedef MovingLeastSquares[cpp.PointXYZ, cpp.PointXYZ] MovingLeastSquares_t
ctypedef MovingLeastSquares[cpp.PointXYZI, cpp.PointXYZI] MovingLeastSquares_PointXYZI_t
ctypedef MovingLeastSquares[cpp.PointXYZRGB, cpp.PointXYZRGB] MovingLeastSquares_PointXYZRGB_t
ctypedef MovingLeastSquares[cpp.PointXYZRGBA, cpp.PointXYZRGBA] MovingLeastSquares_PointXYZRGBA_t
# NG
# ctypedef MovingLeastSquares[cpp.PointXYZ, cpp.PointNormal] MovingLeastSquares_t
# ctypedef MovingLeastSquares[cpp.PointXYZI, cpp.PointNormal] MovingLeastSquares_PointXYZI_t
# ctypedef MovingLeastSquares[cpp.PointXYZRGB, cpp.PointNormal] MovingLeastSquares_PointXYZRGB_t
# ctypedef MovingLeastSquares[cpp.PointXYZRGBA, cpp.PointNormal] MovingLeastSquares_PointXYZRGBA_t


# namespace pcl
# {
#   /** \brief MovingLeastSquares represent an implementation of the MLS (Moving Least Squares) algorithm 
#     * for data smoothing and improved normal estimation. It also contains methods for upsampling the 
#     * resulting cloud based on the parametric fit.
#     * Reference paper: "Computing and Rendering Point Set Surfaces" by Marc Alexa, Johannes Behr, 
#     * Daniel Cohen-Or, Shachar Fleishman, David Levin and Claudio T. Silva
#     * www.sci.utah.edu/~shachar/Publications/crpss.pdf
#     * \author Zoltan Csaba Marton, Radu B. Rusu, Alexandru E. Ichim, Suat Gedikli
#     * \ingroup surface
#     */
#   template <typename PointInT, typename PointOutT>
#   class MovingLeastSquares: public CloudSurfaceProcessing<PointInT, PointOutT>
#   {
#     public:
#       using PCLBase<PointInT>::input_;
#       using PCLBase<PointInT>::indices_;
#       using PCLBase<PointInT>::fake_indices_;
#       using PCLBase<PointInT>::initCompute;
#       using PCLBase<PointInT>::deinitCompute;
# 
#       typedef typename pcl::search::Search<PointInT> KdTree;
#       typedef typename pcl::search::Search<PointInT>::Ptr KdTreePtr;
#       typedef pcl::PointCloud<pcl::Normal> NormalCloud;
#       typedef pcl::PointCloud<pcl::Normal>::Ptr NormalCloudPtr;
# 
#       typedef pcl::PointCloud<PointOutT> PointCloudOut;
#       typedef typename PointCloudOut::Ptr PointCloudOutPtr;
#       typedef typename PointCloudOut::ConstPtr PointCloudOutConstPtr;
# 
#       typedef pcl::PointCloud<PointInT> PointCloudIn;
#       typedef typename PointCloudIn::Ptr PointCloudInPtr;
#       typedef typename PointCloudIn::ConstPtr PointCloudInConstPtr;
# 
#       typedef boost::function<int (int, double, std::vector<int> &, std::vector<float> &)> SearchMethod;
# 
#       enum UpsamplingMethod { NONE, SAMPLE_LOCAL_PLANE, RANDOM_UNIFORM_DENSITY, VOXEL_GRID_DILATION };
# 
#       /** \brief Empty constructor. */
#       MovingLeastSquares () : CloudSurfaceProcessing<PointInT, PointOutT> (),
#                               normals_ (),
#                               search_method_ (),
#                               tree_ (),
#                               order_ (2),
#                               polynomial_fit_ (true),
#                               search_radius_ (0.0),
#                               sqr_gauss_param_ (0.0),
#                               compute_normals_ (false),
#                               upsample_method_ (NONE),
#                               upsampling_radius_ (0.0),
#                               upsampling_step_ (0.0),
#                               rng_uniform_distribution_ (),
#                               desired_num_points_in_radius_ (0),
#                               mls_results_ (),
#                               voxel_size_ (1.0),
#                               dilation_iteration_num_ (0),
#                               nr_coeff_ ()
#                               {};
# 
# 
#       /** \brief Set whether the algorithm should also store the normals computed
#         * \note This is optional, but need a proper output cloud type
#         */
#       inline void
#       setComputeNormals (bool compute_normals) { compute_normals_ = compute_normals; }
# 
#       /** \brief Provide a pointer to the search object.
#         * \param[in] tree a pointer to the spatial search object.
#         */
#       inline void
#       setSearchMethod (const KdTreePtr &tree)
#       {
#         tree_ = tree;
#         // Declare the search locator definition
#         int (KdTree::*radiusSearch)(int index, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances, unsigned int max_nn) const = &KdTree::radiusSearch;
#         search_method_ = boost::bind (radiusSearch, boost::ref (tree_), _1, _2, _3, _4, 0);
#       }
# 
#       /** \brief Get a pointer to the search method used. */
#       inline KdTreePtr 
#       getSearchMethod () { return (tree_); }
# 
#       /** \brief Set the order of the polynomial to be fit.
#         * \param[in] order the order of the polynomial
#         */
#       inline void 
#       setPolynomialOrder (int order) { order_ = order; }
# 
#       /** \brief Get the order of the polynomial to be fit. */
#       inline int 
#       getPolynomialOrder () { return (order_); }
# 
#       /** \brief Sets whether the surface and normal are approximated using a polynomial, or only via tangent estimation.
#         * \param[in] polynomial_fit set to true for polynomial fit
#         */
#       inline void 
#       setPolynomialFit (bool polynomial_fit) { polynomial_fit_ = polynomial_fit; }
# 
#       /** \brief Get the polynomial_fit value (true if the surface and normal are approximated using a polynomial). */
#       inline bool 
#       getPolynomialFit () { return (polynomial_fit_); }
# 
#       /** \brief Set the sphere radius that is to be used for determining the k-nearest neighbors used for fitting.
#         * \param[in] radius the sphere radius that is to contain all k-nearest neighbors
#         * \note Calling this method resets the squared Gaussian parameter to radius * radius !
#         */
#       inline void 
#       setSearchRadius (double radius) { search_radius_ = radius; sqr_gauss_param_ = search_radius_ * search_radius_; }
# 
#       /** \brief Get the sphere radius used for determining the k-nearest neighbors. */
#       inline double 
#       getSearchRadius () { return (search_radius_); }
# 
#       /** \brief Set the parameter used for distance based weighting of neighbors (the square of the search radius works
#         * best in general).
#         * \param[in] sqr_gauss_param the squared Gaussian parameter
#         */
#       inline void 
#       setSqrGaussParam (double sqr_gauss_param) { sqr_gauss_param_ = sqr_gauss_param; }
# 
#       /** \brief Get the parameter for distance based weighting of neighbors. */
#       inline double 
#       getSqrGaussParam () const { return (sqr_gauss_param_); }
# 
#       /** \brief Set the upsampling method to be used
#         * \note Options are: * NONE - no upsampling will be done, only the input points will be projected to their own
#         *                             MLS surfaces
#         *                    * SAMPLE_LOCAL_PLANE - the local plane of each input point will be sampled in a circular
#         *                                           fashion using the \ref upsampling_radius_ and the \ref upsampling_step_
#         *                                           parameters
#         *                    * RANDOM_UNIFORM_DENSITY - the local plane of each input point will be sampled using an
#         *                                               uniform random distribution such that the density of points is
#         *                                               constant throughout the cloud - given by the \ref \ref desired_num_points_in_radius_
#         *                                               parameter
#         *                    * VOXEL_GRID_DILATION - the input cloud will be inserted into a voxel grid with voxels of
#         *                                            size \ref voxel_size_; this voxel grid will be dilated \ref dilation_iteration_num_
#         *                                            times and the resulting points will be projected to the MLS surface
#         *                                            of the closest point in the input cloud; the result is a point cloud
#         *                                            with filled holes and a constant point density
#         */
#       inline void
#       setUpsamplingMethod (UpsamplingMethod method) { upsample_method_ = method; }
# 
# 
#       /** \brief Set the radius of the circle in the local point plane that will be sampled
#         * \note Used only in the case of SAMPLE_LOCAL_PLANE upsampling
#         * \param[in] radius the radius of the circle
#         */
#       inline void
#       setUpsamplingRadius (double radius) { upsampling_radius_ = radius; }
# 
#       /** \brief Get the radius of the circle in the local point plane that will be sampled
#         * \note Used only in the case of SAMPLE_LOCAL_PLANE upsampling
#         */
#       inline double
#       getUpsamplingRadius () { return upsampling_radius_; }
# 
#       /** \brief Set the step size for the local plane sampling
#         * \note Used only in the case of SAMPLE_LOCAL_PLANE upsampling
#         * \param[in] step_size the step size
#         */
#       inline void
#       setUpsamplingStepSize (double step_size) { upsampling_step_ = step_size; }
# 
# 
#       /** \brief Get the step size for the local plane sampling
#         * \note Used only in the case of SAMPLE_LOCAL_PLANE upsampling
#         */
#       inline double
#       getUpsamplingStepSize () { return upsampling_step_; }
# 
#       /** \brief Set the parameter that specifies the desired number of points within the search radius
#         * \note Used only in the case of RANDOM_UNIFORM_DENSITY upsampling
#         * \param[in] desired_num_points_in_radius the desired number of points in the output cloud in a sphere of
#         * radius \ref search_radius_ around each point
#         */
#       inline void
#       setPointDensity (int desired_num_points_in_radius) { desired_num_points_in_radius_ = desired_num_points_in_radius; }
# 
# 
#       /** \brief Get the parameter that specifies the desired number of points within the search radius
#         * \note Used only in the case of RANDOM_UNIFORM_DENSITY upsampling
#         */
#       inline int
#       getPointDensity () { return desired_num_points_in_radius_; }
# 
#       /** \brief Set the voxel size for the voxel grid
#         * \note Used only in the VOXEL_GRID_DILATION upsampling method
#         * \param[in] voxel_size the edge length of a cubic voxel in the voxel grid
#         */
#       inline void
#       setDilationVoxelSize (float voxel_size) { voxel_size_ = voxel_size; }
# 
# 
#       /** \brief Get the voxel size for the voxel grid
#         * \note Used only in the VOXEL_GRID_DILATION upsampling method
#         */
#       inline float
#       getDilationVoxelSize () { return voxel_size_; }
# 
#       /** \brief Set the number of dilation steps of the voxel grid
#         * \note Used only in the VOXEL_GRID_DILATION upsampling method
#         * \param[in] iterations the number of dilation iterations
#         */
#       inline void
#       setDilationIterations (int iterations) { dilation_iteration_num_ = iterations; }
# 
#       /** \brief Get the number of dilation steps of the voxel grid
#         * \note Used only in the VOXEL_GRID_DILATION upsampling method
#         */
#       inline int
#       getDilationIterations () { return dilation_iteration_num_; }
# 
#       /** \brief Base method for surface reconstruction for all points given in <setInputCloud (), setIndices ()>
#         * \param[out] output the resultant reconstructed surface model
#         */
#       void 
#       process (PointCloudOut &output);
# 
#     protected:
#       /** \brief The point cloud that will hold the estimated normals, if set. */
#       NormalCloudPtr normals_;
# 
#       /** \brief The search method template for indices. */
#       SearchMethod search_method_;
# 
#       /** \brief A pointer to the spatial search object. */
#       KdTreePtr tree_;
# 
#       /** \brief The order of the polynomial to be fit. */
#       int order_;
# 
#       /** True if the surface and normal be approximated using a polynomial, false if tangent estimation is sufficient. */
#       bool polynomial_fit_;
# 
#       /** \brief The nearest neighbors search radius for each point. */
#       double search_radius_;
# 
#       /** \brief Parameter for distance based weighting of neighbors (search_radius_ * search_radius_ works fine) */
#       double sqr_gauss_param_;
# 
#       /** \brief Parameter that specifies whether the normals should be computed for the input cloud or not */
#       bool compute_normals_;
# 
#       /** \brief Parameter that specifies the upsampling method to be used */
#       UpsamplingMethod upsample_method_;
# 
#       /** \brief Radius of the circle in the local point plane that will be sampled
#         * \note Used only in the case of SAMPLE_LOCAL_PLANE upsampling
#         */
#       double upsampling_radius_;
# 
#       /** \brief Step size for the local plane sampling
#         * \note Used only in the case of SAMPLE_LOCAL_PLANE upsampling
#         */
#       double upsampling_step_;
# 
#       /** \brief Random number generator using an uniform distribution of floats
#         * \note Used only in the case of RANDOM_UNIFORM_DENSITY upsampling
#         */
#       boost::variate_generator<boost::mt19937, boost::uniform_real<float> > *rng_uniform_distribution_;
# 
#       /** \brief Parameter that specifies the desired number of points within the search radius
#         * \note Used only in the case of RANDOM_UNIFORM_DENSITY upsampling
#         */
#       int desired_num_points_in_radius_;
# 
#       
#       /** \brief Data structure used to store the results of the MLS fitting
#         * \note Used only in the case of VOXEL_GRID_DILATION upsampling
#         */
#       struct MLSResult
#       {
#         MLSResult () : plane_normal (), u (), v (), c_vec (), num_neighbors (), curvature (), valid (false) {}
# 
#         MLSResult (Eigen::Vector3d &a_plane_normal,
#                    Eigen::Vector3d &a_u,
#                    Eigen::Vector3d &a_v,
#                    Eigen::VectorXd a_c_vec,
#                    int a_num_neighbors,
#                    float &a_curvature);
# 
#         Eigen::Vector3d plane_normal, u, v;
#         Eigen::VectorXd c_vec;
#         int num_neighbors;
#         float curvature;
#         bool valid;
#       };
# 
#       /** \brief Stores the MLS result for each point in the input cloud
#         * \note Used only in the case of VOXEL_GRID_DILATION upsampling
#         */
#       std::vector<MLSResult> mls_results_;
# 
#       
#       /** \brief A minimalistic implementation of a voxel grid, necessary for the point cloud upsampling
#         * \note Used only in the case of VOXEL_GRID_DILATION upsampling
#         */
#       class MLSVoxelGrid
#       {
#         public:
#           struct Leaf { Leaf () : valid (true) {} bool valid; };
# 
#           MLSVoxelGrid (PointCloudInConstPtr& cloud,
#                         IndicesPtr &indices,
#                         float voxel_size);
# 
#           void
#           dilate ();
# 
#           inline void
#           getIndexIn1D (const Eigen::Vector3i &index, uint64_t &index_1d) const
#           {
#             index_1d = index[0] * data_size_ * data_size_ +
#                        index[1] * data_size_ + index[2];
#           }
# 
#           inline void
#           getIndexIn3D (uint64_t index_1d, Eigen::Vector3i& index_3d) const
#           {
#             index_3d[0] = static_cast<Eigen::Vector3i::Scalar> (index_1d / (data_size_ * data_size_));
#             index_1d -= index_3d[0] * data_size_ * data_size_;
#             index_3d[1] = static_cast<Eigen::Vector3i::Scalar> (index_1d / data_size_);
#             index_1d -= index_3d[1] * data_size_;
#             index_3d[2] = static_cast<Eigen::Vector3i::Scalar> (index_1d);
#           }
# 
#           inline void
#           getCellIndex (const Eigen::Vector3f &p, Eigen::Vector3i& index) const
#           {
#             for (int i = 0; i < 3; ++i)
#               index[i] = static_cast<Eigen::Vector3i::Scalar> ((p[i] - bounding_min_(i)) / voxel_size_);
#           }
# 
#           inline void
#           getPosition (const uint64_t &index_1d, Eigen::Vector3f &point) const
#           {
#             Eigen::Vector3i index_3d;
#             getIndexIn3D (index_1d, index_3d);
#             for (int i = 0; i < 3; ++i)
#               point[i] = static_cast<Eigen::Vector3f::Scalar> (index_3d[i]) * voxel_size_ + bounding_min_[i];
#           }
# 
#           typedef std::map<uint64_t, Leaf> HashMap;
#           HashMap voxel_grid_;
#           Eigen::Vector4f bounding_min_, bounding_max_;
#           uint64_t data_size_;
#           float voxel_size_;
#       };
# 
# 
#       /** \brief Voxel size for the VOXEL_GRID_DILATION upsampling method */
#       float voxel_size_;
# 
#       /** \brief Number of dilation steps for the VOXEL_GRID_DILATION upsampling method */
#       int dilation_iteration_num_; 
# 
#       /** \brief Number of coefficients, to be computed from the requested order.*/
#       int nr_coeff_;
# 
#       /** \brief Search for the closest nearest neighbors of a given point using a radius search
#         * \param[in] index the index of the query point
#         * \param[out] indices the resultant vector of indices representing the k-nearest neighbors
#         * \param[out] sqr_distances the resultant squared distances from the query point to the k-nearest neighbors
#         */
#       inline int
#       searchForNeighbors (int index, std::vector<int> &indices, std::vector<float> &sqr_distances)
#       {
#         return (search_method_ (index, search_radius_, indices, sqr_distances));
#       }
# 
#       /** \brief Smooth a given point and its neighborghood using Moving Least Squares.
#         * \param[in] index the inex of the query point in the \ref input cloud
#         * \param[in] input the input point cloud that \ref nn_indices refer to
#         * \param[in] nn_indices the set of nearest neighbors indices for \ref pt
#         * \param[in] nn_sqr_dists the set of nearest neighbors squared distances for \ref pt
#         * \param[out] projected_points the set of points projected points around the query point
#         * (in the case of upsampling method NONE, only the query point projected to its own fitted surface will be returned,
#         * in the case of the other upsampling methods, multiple points will be returned)
#         * \param[out] projected_points_normals the normals corresponding to the projected points
#         */
#       void
#       computeMLSPointNormal (int index,
#                              const PointCloudIn &input,
#                              const std::vector<int> &nn_indices,
#                              std::vector<float> &nn_sqr_dists,
#                              PointCloudOut &projected_points,
#                              NormalCloud &projected_points_normals);
# 
#       /** \brief Fits a point (sample point) given in the local plane coordinates of an input point (query point) to
#         * the MLS surface of the input point
#         * \param[in] u_disp the u coordinate of the sample point in the local plane of the query point
#         * \param[in] v_disp the v coordinate of the sample point in the local plane of the query point
#         * \param[in] u the axis corresponding to the u-coordinates of the local plane of the query point
#         * \param[in] v the axis corresponding to the v-coordinates of the local plane of the query point
#         * \param[in] plane_normal the normal to the local plane of the query point
#         * \param[in] curvature the curvature of the surface at the query point
#         * \param[in] query_point the absolute 3D position of the query point
#         * \param[in] c_vec the coefficients of the polynomial fit on the MLS surface of the query point
#         * \param[in] num_neighbors the number of neighbors of the query point in the input cloud
#         * \param[out] result_point the absolute 3D position of the resulting projected point
#         * \param[out] result_normal the normal of the resulting projected point
#         */
#       void
#       projectPointToMLSSurface (float &u_disp, float &v_disp,
#                                 Eigen::Vector3d &u, Eigen::Vector3d &v,
#                                 Eigen::Vector3d &plane_normal,
#                                 float &curvature,
#                                 Eigen::Vector3f &query_point,
#                                 Eigen::VectorXd &c_vec,
#                                 int num_neighbors,
#                                 PointOutT &result_point,
#                                 pcl::Normal &result_normal);
#     public:
#         EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
###

# mls_omp.h
# namespace pcl
# {
#   /** \brief MovingLeastSquaresOMP represent an OpenMP implementation of the MLS (Moving Least Squares) algorithm for 
#     * data smoothing and improved normal estimation.
#     * \author Radu B. Rusu
#     * \ingroup surface
#     */
#   template <typename PointInT, typename PointOutT>
#   class MovingLeastSquaresOMP : public MovingLeastSquares<PointInT, PointOutT>
#   {
#     using MovingLeastSquares<PointInT, PointOutT>::input_;
#     using MovingLeastSquares<PointInT, PointOutT>::indices_;
#     using MovingLeastSquares<PointInT, PointOutT>::fake_indices_;
#     using MovingLeastSquares<PointInT, PointOutT>::initCompute;
#     using MovingLeastSquares<PointInT, PointOutT>::deinitCompute;
#     using MovingLeastSquares<PointInT, PointOutT>::nr_coeff_;
#     using MovingLeastSquares<PointInT, PointOutT>::order_;
#     using MovingLeastSquares<PointInT, PointOutT>::normals_;
#     using MovingLeastSquares<PointInT, PointOutT>::upsample_method_;
#     using MovingLeastSquares<PointInT, PointOutT>::voxel_size_;
#     using MovingLeastSquares<PointInT, PointOutT>::dilation_iteration_num_;
#     using MovingLeastSquares<PointInT, PointOutT>::tree_;
#     using MovingLeastSquares<PointInT, PointOutT>::mls_results_;
#     using MovingLeastSquares<PointInT, PointOutT>::search_radius_;
#     using MovingLeastSquares<PointInT, PointOutT>::compute_normals_;
#     using MovingLeastSquares<PointInT, PointOutT>::searchForNeighbors;
#       
#     typedef typename MovingLeastSquares<PointInT, PointOutT>::PointCloudIn PointCloudIn;
#     typedef typename MovingLeastSquares<PointInT, PointOutT>::PointCloudOut PointCloudOut;
#     typedef typename MovingLeastSquares<PointInT, PointOutT>::NormalCloud NormalCloud;
#     typedef typename MovingLeastSquares<PointInT, PointOutT>::MLSVoxelGrid MLSVoxelGrid;
# 
#     public:
#       /** \brief Empty constructor. */
#       MovingLeastSquaresOMP () : threads_ (1)
#       {};
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       MovingLeastSquaresOMP (unsigned int nr_threads) : threads_ (0)
#       {
#         setNumberOfThreads (nr_threads);
#       }
# 
#       /** \brief Initialize the scheduler and set the number of threads to use.
#         * \param nr_threads the number of hardware threads to use (-1 sets the value back to automatic)
#         */
#       inline void 
#       setNumberOfThreads (unsigned int nr_threads)
#       { 
#         if (nr_threads == 0)
#           nr_threads = 1;
#         threads_ = nr_threads; 
#       }
# 
###

# multi_grid_octree_data.h
# pcl/surface/3rdparty/poisson4/multi_grid_octree_data.h (1.7.2)
# namespace pcl 
# {
#   namespace poisson 
#   {
#     typedef float Real;
#     typedef float FunctionDataReal;
#     typedef OctNode<class TreeNodeData,Real> TreeOctNode;
# 
#     class RootInfo
#     {
#       public:
#         const TreeOctNode* node;
#         int edgeIndex;
#         long long key;
#     };
# 
#     class VertexData
#     {
#       public:
#         static long long 
#         EdgeIndex (const TreeOctNode* node, const int& eIndex, const int& maxDepth, int index[DIMENSION]);
# 
#         static long long 
#         EdgeIndex (const TreeOctNode* node, const int& eIndex, const int& maxDepth);
# 
#         static long long 
#         FaceIndex (const TreeOctNode* node, const int& fIndex, const int& maxDepth, int index[DIMENSION]);
# 
#         static long long 
#         FaceIndex (const TreeOctNode* node, const int& fIndex, const int& maxDepth);
# 
#         static long long 
#         CornerIndex (const int& depth, const int offSet[DIMENSION] ,const int& cIndex, const int& maxDepth, int index[DIMENSION]);
# 
#         static long long 
#         CornerIndex (const TreeOctNode* node, const int& cIndex, const int& maxDepth, int index[DIMENSION]);
# 
#         static long long 
#         CornerIndex (const TreeOctNode* node, const int& cIndex, const int& maxDepth);
# 
#         static long long 
#         CenterIndex (const int& depth, const int offSet[DIMENSION], const int& maxDepth, int index[DIMENSION]);
# 
#         static long long 
#         CenterIndex (const TreeOctNode* node, const int& maxDepth, int index[DIMENSION]);
#         
#         static long long 
#         CenterIndex (const TreeOctNode* node, const int& maxDepth);
#     };
# 
#     class SortedTreeNodes
#     {
#       public:
#         TreeOctNode** treeNodes;
#         int *nodeCount;
#         int maxDepth;
#         SortedTreeNodes ();
#         ~SortedTreeNodes ();
#         void 
#         set (TreeOctNode& root,const int& setIndex);
#     };
# 
#     class TreeNodeData
#     {
#       public:
#         static int UseIndex;
#         union
#         {
#           int mcIndex;
#           struct
#           {
#             int nodeIndex;
#             Real centerWeightContribution;
#           };
#         };
#         Real value;
# 
#         TreeNodeData (void);
#         ~TreeNodeData (void);
#     };
# 
#     template<int Degree>
#     class Octree
#     {
#       TreeOctNode::NeighborKey neighborKey;
#       TreeOctNode::NeighborKey2 neighborKey2;
# 
#       Real radius;
#       int width;
# 
#       void 
#       setNodeIndices (TreeOctNode& tree,int& idx);
#       
#       Real 
#       GetDotProduct (const int index[DIMENSION]) const;
# 
#       Real 
#       GetLaplacian (const int index[DIMENSION]) const;
# 
#       Real 
#       GetDivergence (const int index[DIMENSION], const Point3D<Real>& normal) const;
# 
#       class DivergenceFunction
#       {
#         public:
#           Point3D<Real> normal;
#           Octree<Degree>* ot;
#           int index[DIMENSION],scratch[DIMENSION];
# 
#           void 
#           Function (TreeOctNode* node1, const TreeOctNode* node2);
#       };
# 
#       class LaplacianProjectionFunction
#       {
#         public:
#           double value;
#           Octree<Degree>* ot;
#           int index[DIMENSION],scratch[DIMENSION];
# 
#           void 
#           Function (TreeOctNode* node1, const TreeOctNode* node2);
#       };
# 
#       class LaplacianMatrixFunction
#       {
#         public:
#           int x2,y2,z2,d2;
#           Octree<Degree>* ot;
#           int index[DIMENSION],scratch[DIMENSION];
#           int elementCount,offset;
#           MatrixEntry<float>* rowElements;
# 
#           int 
#           Function (const TreeOctNode* node1, const TreeOctNode* node2);
#       };
# 
#       class RestrictedLaplacianMatrixFunction
#       {
#         public:
#           int depth,offset[3];
#           Octree<Degree>* ot;
#           Real radius;
#           int index[DIMENSION], scratch[DIMENSION];
#           int elementCount;
#           MatrixEntry<float>* rowElements;
# 
#           int 
#           Function (const TreeOctNode* node1, const TreeOctNode* node2);
#       };
# 
#       ///////////////////////////
#       // Evaluation Functions  //
#       ///////////////////////////
#       class PointIndexValueFunction
#       {
#         public:
#           int res2;
#           FunctionDataReal* valueTables;
#           int index[DIMENSION];
#           Real value;
# 
#           void 
#           Function (const TreeOctNode* node);
#       };
# 
#       class PointIndexValueAndNormalFunction
#       {
#         public:
#           int res2;
#           FunctionDataReal* valueTables;
#           FunctionDataReal* dValueTables;
#           Real value;
#           Point3D<Real> normal;
#           int index[DIMENSION];
# 
#           void 
#           Function (const TreeOctNode* node);
#       };
# 
#       class AdjacencyCountFunction
#       {
#         public:
#           int adjacencyCount;
# 
#           void 
#           Function (const TreeOctNode* node1, const TreeOctNode* node2);
#       };
# 
#       class AdjacencySetFunction
#       {
#         public:
#           int *adjacencies,adjacencyCount;
#           void 
#           Function (const TreeOctNode* node1, const TreeOctNode* node2);
#       };
# 
#       class RefineFunction
#       {
#         public:
#           int depth;
#           void 
#           Function (TreeOctNode* node1, const TreeOctNode* node2);
#       };
# 
#       class FaceEdgesFunction 
#       {
#         public:
#           int fIndex,maxDepth;
#           std::vector<std::pair<long long,long long> >* edges;
#           hash_map<long long, std::pair<RootInfo,int> >* vertexCount;
# 
#           void 
#           Function (const TreeOctNode* node1, const TreeOctNode* node2);
#       };
# 
#       int 
#       SolveFixedDepthMatrix (const int& depth, const SortedTreeNodes& sNodes);
#       
#       int 
#       SolveFixedDepthMatrix (const int& depth, const int& startingDepth, const SortedTreeNodes& sNodes);
# 
#       int 
#       GetFixedDepthLaplacian (SparseSymmetricMatrix<float>& matrix, const int& depth, const SortedTreeNodes& sNodes);
# 
#       int 
#       GetRestrictedFixedDepthLaplacian (SparseSymmetricMatrix<float>& matrix,
#                                         const int& depth,
#                                         const int* entries,
#                                         const int& entryCount,
#                                         const TreeOctNode* rNode,
#                                         const Real& radius,
#                                         const SortedTreeNodes& sNodes);
# 
#       void 
#       SetIsoSurfaceCorners (const Real& isoValue, const int& subdivisionDepth, const int& fullDepthIso);
# 
#       static int 
#       IsBoundaryFace (const TreeOctNode* node, const int& faceIndex, const int& subdivideDepth);
#       
#       static int 
#       IsBoundaryEdge (const TreeOctNode* node, const int& edgeIndex, const int& subdivideDepth);
#       
#       static int 
#       IsBoundaryEdge (const TreeOctNode* node, const int& dir, const int& x, const int& y, const int& subidivideDepth);
#       
#       void 
#       PreValidate (const Real& isoValue, const int& maxDepth, const int& subdivideDepth);
#       
#       void 
#       PreValidate (TreeOctNode* node, const Real& isoValue, const int& maxDepth, const int& subdivideDepth);
#       
#       void 
#       Validate (TreeOctNode* node,
#                 const Real& isoValue,
#                 const int& maxDepth,
#                 const int& fullDepthIso,
#                 const int& subdivideDepth);
# 
#       void 
#       Validate (TreeOctNode* node, const Real& isoValue, const int& maxDepth, const int& fullDepthIso);
# 
#       void 
#       Subdivide (TreeOctNode* node, const Real& isoValue, const int& maxDepth);
# 
#       int 
#       SetBoundaryMCRootPositions (const int& sDepth,const Real& isoValue,
#                                   hash_map<long long,int>& boundaryRoots,
#                                   hash_map<long long,
#                                   std::pair<Real,Point3D<Real> > >& boundaryNormalHash,
#                                   CoredMeshData* mesh,
#                                   const int& nonLinearFit);
# 
#       int 
#       SetMCRootPositions (TreeOctNode* node,
#                           const int& sDepth,
#                           const Real& isoValue,
#                           hash_map<long long, int>& boundaryRoots,
#                           hash_map<long long, int>* interiorRoots,
#                           hash_map<long long, std::pair<Real,Point3D<Real> > >& boundaryNormalHash,
#                           hash_map<long long, std::pair<Real,Point3D<Real> > >* interiorNormalHash,
#                           std::vector<Point3D<float> >* interiorPositions,
#                           CoredMeshData* mesh,
#                           const int& nonLinearFit);
# 
#       int 
#       GetMCIsoTriangles (TreeOctNode* node,
#                          CoredMeshData* mesh,
#                          hash_map<long long,int>& boundaryRoots,
#                          hash_map<long long,int>* interiorRoots,
#                          std::vector<Point3D<float> >* interiorPositions,
#                          const int& offSet,
#                          const int& sDepth, 
#                          bool addBarycenter, 
#                          bool polygonMesh);
# 
#       static int 
#       AddTriangles (CoredMeshData* mesh,
#                     std::vector<CoredPointIndex> edges[3],
#                     std::vector<Point3D<float> >* interiorPositions, 
#                     const int& offSet);
#       
#       static int 
#       AddTriangles (CoredMeshData* mesh, 
#                     std::vector<CoredPointIndex>& edges, std::vector<Point3D<float> >* interiorPositions, 
#                     const int& offSet, 
#                     bool addBarycenter, 
#                     bool polygonMesh);
# 
#       void 
#       GetMCIsoEdges (TreeOctNode* node,
#                      hash_map<long long,int>& boundaryRoots,
#                      hash_map<long long,int>* interiorRoots,
#                      const int& sDepth,
#                      std::vector<std::pair<long long,long long> >& edges);
# 
#       static int 
#       GetEdgeLoops (std::vector<std::pair<long long,long long> >& edges,
#                     std::vector<std::vector<std::pair<long long,long long> > >& loops);
# 
#       static int 
#       InteriorFaceRootCount (const TreeOctNode* node,const int &faceIndex,const int& maxDepth);
# 
#       static int 
#       EdgeRootCount (const TreeOctNode* node,const int& edgeIndex,const int& maxDepth);
# 
#       int 
#       GetRoot (const RootInfo& ri,
#                const Real& isoValue,
#                const int& maxDepth,Point3D<Real> & position,
#                hash_map<long long,std::pair<Real,Point3D<Real> > >& normalHash,
#                Point3D<Real>* normal,
#                const int& nonLinearFit);
# 
#       int 
#       GetRoot (const RootInfo& ri,
#                const Real& isoValue,
#                Point3D<Real> & position,
#                hash_map<long long,
#                std::pair<Real,Point3D<Real> > >& normalHash,
#                const int& nonLinearFit);
# 
#       static int 
#       GetRootIndex (const TreeOctNode* node,const int& edgeIndex,const int& maxDepth,RootInfo& ri);
# 
#       static int 
#       GetRootIndex (const TreeOctNode* node, 
#                     const int& edgeIndex,
#                     const int& maxDepth,
#                     const int& sDepth,
#                     RootInfo& ri);
#       
#       static int 
#       GetRootIndex (const long long& key,
#                     hash_map<long long,int>& boundaryRoots,
#                     hash_map<long long,int>* interiorRoots,
#                     CoredPointIndex& index);
#       
#       static int 
#       GetRootPair (const RootInfo& root,const int& maxDepth,RootInfo& pair);
# 
#       int 
#       NonLinearUpdateWeightContribution (TreeOctNode* node,
#                                          const Point3D<Real>& position,
#                                          const Real& weight = Real(1.0));
# 
#       Real 
#       NonLinearGetSampleWeight (TreeOctNode* node,
#                                 const Point3D<Real>& position);
#       
#       void 
#       NonLinearGetSampleDepthAndWeight (TreeOctNode* node,
#                                         const Point3D<Real>& position,
#                                         const Real& samplesPerNode,
#                                         Real& depth,
#                                         Real& weight);
# 
#       int 
#       NonLinearSplatOrientedPoint (TreeOctNode* node,
#                                    const Point3D<Real>& point,
#                                    const Point3D<Real>& normal);
#       
#       void 
#       NonLinearSplatOrientedPoint (const Point3D<Real>& point,
#                                    const Point3D<Real>& normal,
#                                    const int& kernelDepth,
#                                    const Real& samplesPerNode,
#                                    const int& minDepth,
#                                    const int& maxDepth);
# 
#       int 
#       HasNormals (TreeOctNode* node,const Real& epsilon);
# 
#       Real 
#       getCenterValue (const TreeOctNode* node);
# 
#       Real 
#       getCornerValue (const TreeOctNode* node,const int& corner);
# 
#       void 
#       getCornerValueAndNormal (const TreeOctNode* node,const int& corner,Real& value,Point3D<Real>& normal);
# 
#       public:
#         static double maxMemoryUsage;
#         static double 
#         MemoryUsage ();
# 
#         std::vector<Point3D<Real> >* normals;
#         Real postNormalSmooth;
#         TreeOctNode tree;
#         FunctionData<Degree,FunctionDataReal> fData;
#         Octree ();
# 
#         void 
#         setFunctionData (const PPolynomial<Degree>& ReconstructionFunction,
#                          const int& maxDepth,
#                          const int& normalize,
#                          const Real& normalSmooth = -1);
#         
#         void 
#         finalize1 (const int& refineNeighbors=-1);
#         
#         void 
#         finalize2 (const int& refineNeighbors=-1);
# 
#         //int setTree(char* fileName,const int& maxDepth,const int& binary,const int& kernelDepth,const Real& samplesPerNode,
#         //    const Real& scaleFactor,Point3D<Real>& center,Real& scale,const int& resetSampleDepths,const int& useConfidence);
# 
#         template<typename PointNT> int
#         setTree (boost::shared_ptr<const pcl::PointCloud<PointNT> > input_,
#                  const int& maxDepth,
#                  const int& kernelDepth,
#                  const Real& samplesPerNode,
#                  const Real& scaleFactor,
#                  Point3D<Real>& center,
#                  Real& scale,
#                  const int& resetSamples,
#                  const int& useConfidence);
# 
# 
#         void 
#         SetLaplacianWeights (void);
#         
#         void 
#         ClipTree (void);
# 
#         int 
#         LaplacianMatrixIteration (const int& subdivideDepth);
# 
#         Real 
#         GetIsoValue (void);
#         
#         void 
#         GetMCIsoTriangles (const Real& isoValue,
#                            CoredMeshData* mesh,
#                            const int& fullDepthIso = 0,
#                            const int& nonLinearFit = 1, 
#                            bool addBarycenter = false, 
#                            bool polygonMesh = false);
#         
#         void 
#         GetMCIsoTriangles (const Real& isoValue,
#                            const int& subdivideDepth,
#                            CoredMeshData* mesh,
#                            const int& fullDepthIso = 0,
#                            const int& nonLinearFit = 1, 
#                            bool addBarycenter = false, 
#                            bool polygonMesh = false );
#     };
#   }
# }
# 
###

# octree_poisson.h (1.6.0)
# pcl/surface/3rdparty/poisson4/octree_poisson.h (1.7.2)
# namespace pcl 
# {
#   namespace poisson 
#   {
# 
#     template<class NodeData,class Real=float>
#     class OctNode
#     {
#       private:
#         static int UseAlloc;
# 
#         class AdjacencyCountFunction
#         {
#           public:
#             int count;
#             void Function(const OctNode<NodeData,Real>* node1,const OctNode<NodeData,Real>* node2);
#         };
# 
#         template<class NodeAdjacencyFunction>
#         void __processNodeFaces (OctNode* node,
#                                  NodeAdjacencyFunction* F,
#                                  const int& cIndex1, const int& cIndex2, const int& cIndex3, const int& cIndex4);
#         template<class NodeAdjacencyFunction>
#         void __processNodeEdges (OctNode* node,
#                                  NodeAdjacencyFunction* F,
#                                  const int& cIndex1, const int& cIndex2);
#         template<class NodeAdjacencyFunction>
#         void __processNodeNodes (OctNode* node, NodeAdjacencyFunction* F);
#         template<class NodeAdjacencyFunction>
#         static void __ProcessNodeAdjacentNodes (const int& dx, const int& dy, const int& dz,
#                                                 OctNode* node1, const int& radius1,
#                                                 OctNode* node2, const int& radius2,
#                                                 const int& cWidth2,
#                                                 NodeAdjacencyFunction* F);
#         template<class TerminatingNodeAdjacencyFunction>
#         static void __ProcessTerminatingNodeAdjacentNodes(const int& dx, const int& dy, const int& dz,
#                                                           OctNode* node1, const int& radius1,
#                                                           OctNode* node2, const int& radius2,
#                                                           const int& cWidth2,
#                                                           TerminatingNodeAdjacencyFunction* F);
#         template<class PointAdjacencyFunction>
#         static void __ProcessPointAdjacentNodes (const int& dx, const int& dy, const int& dz,
#                                                  OctNode* node2, const int& radius2,
#                                                  const int& cWidth2,
#                                                  PointAdjacencyFunction* F);
#         template<class NodeAdjacencyFunction>
#         static void __ProcessFixedDepthNodeAdjacentNodes (const int& dx, const int& dy, const int& dz,
#                                                           OctNode* node1, const int& radius1,
#                                                           OctNode* node2, const int& radius2,
#                                                           const int& cWidth2,
#                                                           const int& depth,
#                                                           NodeAdjacencyFunction* F);
#         template<class NodeAdjacencyFunction>
#         static void __ProcessMaxDepthNodeAdjacentNodes (const int& dx, const int& dy, const int& dz,
#                                                         OctNode* node1, const int& radius1,
#                                                         OctNode* node2, const int& radius2,
#                                                         const int& cWidth2,
#                                                         const int& depth,
#                                                         NodeAdjacencyFunction* F);
# 
#         // This is made private because the division by two has been pulled out.
#         static inline int Overlap (const int& c1, const int& c2, const int& c3, const int& dWidth);
#         inline static int ChildOverlap (const int& dx, const int& dy, const int& dz, const int& d, const int& cRadius2);
# 
#         const OctNode* __faceNeighbor (const int& dir, const int& off) const;
#         const OctNode* __edgeNeighbor (const int& o, const int i[2], const int idx[2]) const;
#         OctNode* __faceNeighbor (const int& dir, const int& off, const int& forceChildren);
#         OctNode* __edgeNeighbor (const int& o, const int i[2], const int idx[2], const int& forceChildren);
#       public:
#         static const int DepthShift,OffsetShift,OffsetShift1,OffsetShift2,OffsetShift3;
#         static const int DepthMask,OffsetMask;
# 
#         static Allocator<OctNode> AllocatorOctNode;
#         static int UseAllocator (void);
#         static void SetAllocator (int blockSize);
# 
#         OctNode* parent;
#         OctNode* children;
#         short d,off[3];
#         NodeData nodeData;
# 
# 
#         OctNode (void);
#         ~OctNode (void);
#         int initChildren (void);
# 
#         void depthAndOffset (int& depth, int offset[3]) const;
#         int depth (void) const;
#         static inline void DepthAndOffset (const long long& index, int& depth, int offset[3]);
#         static inline void CenterAndWidth (const long long& index, Point3D<Real>& center, Real& width);
#         static inline int Depth (const long long& index);
#         static inline void Index (const int& depth, const int offset[3], short& d, short off[3]);
#         void centerAndWidth (Point3D<Real>& center, Real& width) const;
# 
#         int leaves (void) const;
#         int maxDepthLeaves (const int& maxDepth) const;
#         int nodes (void) const;
#         int maxDepth (void) const;
# 
#         const OctNode* root (void) const;
# 
#         const OctNode* nextLeaf (const OctNode* currentLeaf = NULL) const;
#         OctNode* nextLeaf (OctNode* currentLeaf = NULL);
#         const OctNode* nextNode (const OctNode* currentNode = NULL) const;
#         OctNode* nextNode (OctNode* currentNode = NULL);
#         const OctNode* nextBranch (const OctNode* current) const;
#         OctNode* nextBranch (OctNode* current);
# 
#         void setFullDepth (const int& maxDepth);
# 
#         void printLeaves (void) const;
#         void printRange (void) const;
# 
#         template<class NodeAdjacencyFunction>
#         void processNodeFaces (OctNode* node,NodeAdjacencyFunction* F, const int& fIndex, const int& processCurrent = 1);
#         template<class NodeAdjacencyFunction>
#         void processNodeEdges (OctNode* node, NodeAdjacencyFunction* F, const int& eIndex, const int& processCurrent = 1);
#         template<class NodeAdjacencyFunction>
#         void processNodeCorners (OctNode* node, NodeAdjacencyFunction* F, const int& cIndex, const int& processCurrent = 1);
#         template<class NodeAdjacencyFunction>
#         void processNodeNodes (OctNode* node, NodeAdjacencyFunction* F, const int& processCurrent=1);
# 
#         template<class NodeAdjacencyFunction>
#         static void ProcessNodeAdjacentNodes (const int& maxDepth,
#                                               OctNode* node1, const int& width1,
#                                               OctNode* node2, const int& width2,
#                                               NodeAdjacencyFunction* F,
#                                               const int& processCurrent=1);
#         template<class NodeAdjacencyFunction>
#         static void ProcessNodeAdjacentNodes (const int& dx, const int& dy, const int& dz,
#                                               OctNode* node1, const int& radius1,
#                                               OctNode* node2, const int& radius2,
#                                               const int& width2,
#                                               NodeAdjacencyFunction* F,
#                                               const int& processCurrent = 1);
#         template<class TerminatingNodeAdjacencyFunction>
#         static void ProcessTerminatingNodeAdjacentNodes (const int& maxDepth,
#                                                          OctNode* node1, const int& width1,
#                                                          OctNode* node2, const int& width2,
#                                                          TerminatingNodeAdjacencyFunction* F,
#                                                          const int& processCurrent = 1);
#         template<class TerminatingNodeAdjacencyFunction>
#         static void ProcessTerminatingNodeAdjacentNodes (const int& dx, const int& dy, const int& dz,
#                                                          OctNode* node1, const int& radius1,
#                                                          OctNode* node2, const int& radius2,
#                                                          const int& width2,
#                                                          TerminatingNodeAdjacencyFunction* F,
#                                                          const int& processCurrent = 1);
#         template<class PointAdjacencyFunction>
#         static void ProcessPointAdjacentNodes (const int& maxDepth,
#                                                const int center1[3],
#                                                OctNode* node2, const int& width2,
#                                                PointAdjacencyFunction* F,
#                                                const int& processCurrent = 1);
#         template<class PointAdjacencyFunction>
#         static void ProcessPointAdjacentNodes (const int& dx, const int& dy, const int& dz,
#                                                OctNode* node2, const int& radius2, const int& width2,
#                                                PointAdjacencyFunction* F,
#                                                const int& processCurrent = 1);
#         template<class NodeAdjacencyFunction>
#         static void ProcessFixedDepthNodeAdjacentNodes (const int& maxDepth,
#                                                         OctNode* node1, const int& width1,
#                                                         OctNode* node2, const int& width2,
#                                                         const int& depth,
#                                                         NodeAdjacencyFunction* F,
#                                                         const int& processCurrent = 1);
#         template<class NodeAdjacencyFunction>
#         static void ProcessFixedDepthNodeAdjacentNodes (const int& dx, const int& dy, const int& dz,
#                                                         OctNode* node1, const int& radius1,
#                                                         OctNode* node2, const int& radius2,
#                                                         const int& width2,
#                                                         const int& depth,
#                                                         NodeAdjacencyFunction* F,
#                                                         const int& processCurrent = 1);
#         template<class NodeAdjacencyFunction>
#         static void ProcessMaxDepthNodeAdjacentNodes (const int& maxDepth,
#                                                       OctNode* node1, const int& width1,
#                                                       OctNode* node2, const int& width2,
#                                                       const int& depth,
#                                                       NodeAdjacencyFunction* F,
#                                                       const int& processCurrent = 1);
#         template<class NodeAdjacencyFunction>
#         static void ProcessMaxDepthNodeAdjacentNodes (const int& dx, const int& dy, const int& dz,
#                                                       OctNode* node1, const int& radius1,
#                                                       OctNode* node2, const int& radius2,
#                                                       const int& width2,
#                                                       const int& depth,
#                                                       NodeAdjacencyFunction* F,
#                                                       const int& processCurrent = 1);
# 
#         static int CornerIndex (const Point3D<Real>& center, const Point3D<Real> &p);
# 
#         OctNode* faceNeighbor (const int& faceIndex, const int& forceChildren = 0);
#         const OctNode* faceNeighbor (const int& faceIndex) const;
#         OctNode* edgeNeighbor (const int& edgeIndex, const int& forceChildren = 0);
#         const OctNode* edgeNeighbor (const int& edgeIndex) const;
#         OctNode* cornerNeighbor (const int& cornerIndex, const int& forceChildren = 0);
#         const OctNode* cornerNeighbor (const int& cornerIndex) const;
# 
#         OctNode* getNearestLeaf (const Point3D<Real>& p);
#         const OctNode* getNearestLeaf (const Point3D<Real>& p) const;
# 
#         static int CommonEdge (const OctNode* node1, const int& eIndex1,
#                                const OctNode* node2, const int& eIndex2);
#         static int CompareForwardDepths (const void* v1, const void* v2);
#         static int CompareForwardPointerDepths (const void* v1, const void* v2);
#         static int CompareBackwardDepths (const void* v1, const void* v2);
#         static int CompareBackwardPointerDepths (const void* v1, const void* v2);
# 
# 
#         template<class NodeData2>
#         OctNode& operator = (const OctNode<NodeData2, Real>& node);
# 
#         static inline int Overlap2 (const int &depth1,
#                                     const int offSet1[DIMENSION],
#                                     const Real& multiplier1,
#                                     const int &depth2,
#                                     const int offSet2[DIMENSION],
#                                     const Real& multiplier2);
# 
# 
#         int write (const char* fileName) const;
#         int write (FILE* fp) const;
#         int read (const char* fileName);
#         int read (FILE* fp);
# 
#         class Neighbors{
#         public:
#           OctNode* neighbors[3][3][3];
#           Neighbors (void);
#           void clear (void);
#         };
#         class NeighborKey{
#         public:
#           Neighbors* neighbors;
# 
#           NeighborKey (void);
#           ~NeighborKey (void);
# 
#           void set (const int& depth);
#           Neighbors& setNeighbors (OctNode* node);
#           Neighbors& getNeighbors (OctNode* node);
#         };
#         class Neighbors2{
#         public:
#           const OctNode* neighbors[3][3][3];
#           Neighbors2 (void);
#           void clear (void);
#         };
#         class NeighborKey2{
#         public:
#           Neighbors2* neighbors;
# 
#           NeighborKey2 (void);
#           ~NeighborKey2 (void);
# 
#           void set (const int& depth);
#           Neighbors2& getNeighbors (const OctNode* node);
#         };
# 
#         void centerIndex (const int& maxDepth, int index[DIMENSION]) const;
#         int width (const int& maxDepth) const;
#     };
# 
#   }
# }
###

# organized_fast_mesh.h
# namespace pcl
# {
# 
#   /** \brief Simple triangulation/surface reconstruction for organized point
#     * clouds. Neighboring points (pixels in image space) are connected to
#     * construct a triangular mesh.
#     *
#     * \author Dirk Holz, Radu B. Rusu
#     * \ingroup surface
#     */
#   template <typename PointInT>
#   class OrganizedFastMesh : public MeshConstruction<PointInT>
#   {
#     public:
#       using MeshConstruction<PointInT>::input_;
#       using MeshConstruction<PointInT>::check_tree_;
# 
#       typedef typename pcl::PointCloud<PointInT>::Ptr PointCloudPtr;
# 
#       typedef std::vector<pcl::Vertices> Polygons;
# 
#       enum TriangulationType
#       {
#         TRIANGLE_RIGHT_CUT,     // _always_ "cuts" a quad from top left to bottom right
#         TRIANGLE_LEFT_CUT,      // _always_ "cuts" a quad from top right to bottom left
#         TRIANGLE_ADAPTIVE_CUT,  // "cuts" where possible and prefers larger differences in 'z' direction
#         QUAD_MESH               // create a simple quad mesh
#       };
# 
#       /** \brief Constructor. Triangulation type defaults to \a QUAD_MESH. */
#       OrganizedFastMesh ()
#       : max_edge_length_squared_ (0.025f)
#       , triangle_pixel_size_ (1)
#       , triangulation_type_ (QUAD_MESH)
#       , store_shadowed_faces_ (false)
#       , cos_angle_tolerance_ (fabsf (cosf (pcl::deg2rad (12.5f))))
#       {
#         check_tree_ = false;
#       };
# 
#       /** \brief Destructor. */
#       ~OrganizedFastMesh () {};
# 
#       /** \brief Set a maximum edge length. TODO: Implement!
#         * \param[in] max_edge_length the maximum edge length
#         */
#       inline void
#       setMaxEdgeLength (float max_edge_length)
#       {
#         max_edge_length_squared_ = max_edge_length * max_edge_length;
#       };
# 
#       /** \brief Set the edge length (in pixels) used for constructing the fixed mesh.
#         * \param[in] triangle_size edge length in pixels
#         * (Default: 1 = neighboring pixels are connected)
#         */
#       inline void
#       setTrianglePixelSize (int triangle_size)
#       {
#         triangle_pixel_size_ = std::max (1, (triangle_size - 1));
#       }
# 
#       /** \brief Set the triangulation type (see \a TriangulationType)
#         * \param[in] type quad mesh, triangle mesh with fixed left, right cut,
#         * or adaptive cut (splits a quad wrt. the depth (z) of the points)
#         */
#       inline void
#       setTriangulationType (TriangulationType type)
#       {
#         triangulation_type_ = type;
#       }
# 
#       /** \brief Store shadowed faces or not.
#         * \param[in] enable set to true to store shadowed faces
#         */
#       inline void
#       storeShadowedFaces (bool enable)
#       {
#         store_shadowed_faces_ = enable;
#       }
# 
#     protected:
#       /** \brief max (squared) length of edge */
#       float max_edge_length_squared_;
# 
#       /** \brief size of triangle endges (in pixels) */
#       int triangle_pixel_size_;
# 
#       /** \brief Type of meshin scheme (quads vs. triangles, left cut vs. right cut ... */
#       TriangulationType triangulation_type_;
# 
#       /** \brief Whether or not shadowed faces are stored, e.g., for exploration */
#       bool store_shadowed_faces_;
# 
#       float cos_angle_tolerance_;
# 
#       /** \brief Perform the actual polygonal reconstruction.
#         * \param[out] polygons the resultant polygons
#         */
#       void
#       reconstructPolygons (std::vector<pcl::Vertices>& polygons);
# 
#       /** \brief Create the surface.
#         * \param[out] polygons the resultant polygons, as a set of vertices. The Vertices structure contains an array of point indices.
#         */
#       virtual void
#       performReconstruction (std::vector<pcl::Vertices> &polygons);
# 
#       /** \brief Create the surface.
#         *
#         * Simply uses image indices to create an initial polygonal mesh for organized point clouds.
#         * \a indices_ are ignored!
#         *
#         * \param[out] output the resultant polygonal mesh
#         */
#       void
#       performReconstruction (pcl::PolygonMesh &output);
# 
#       /** \brief Add a new triangle to the current polygon mesh
#         * \param[in] a index of the first vertex
#         * \param[in] b index of the second vertex
#         * \param[in] c index of the third vertex
#         * \param[in] idx the index in the set of polygon vertices (assumes \a idx is valid in \a polygons)
#         * \param[out] polygons the polygon mesh to be updated
#         */
#       inline void
#       addTriangle (int a, int b, int c, int idx, std::vector<pcl::Vertices>& polygons)
#       {
#         assert (idx < static_cast<int> (polygons.size ()));
#         polygons[idx].vertices.resize (3);
#         polygons[idx].vertices[0] = a;
#         polygons[idx].vertices[1] = b;
#         polygons[idx].vertices[2] = c;
#       }
# 
#       /** \brief Add a new quad to the current polygon mesh
#         * \param[in] a index of the first vertex
#         * \param[in] b index of the second vertex
#         * \param[in] c index of the third vertex
#         * \param[in] d index of the fourth vertex
#         * \param[in] idx the index in the set of polygon vertices (assumes \a idx is valid in \a polygons)
#         * \param[out] polygons the polygon mesh to be updated
#         */
#       inline void
#       addQuad (int a, int b, int c, int d, int idx, std::vector<pcl::Vertices>& polygons)
#       {
#         assert (idx < static_cast<int> (polygons.size ()));
#         polygons[idx].vertices.resize (4);
#         polygons[idx].vertices[0] = a;
#         polygons[idx].vertices[1] = b;
#         polygons[idx].vertices[2] = c;
#         polygons[idx].vertices[3] = d;
#       }
# 
#       /** \brief Set (all) coordinates of a particular point to the specified value
#         * \param[in] point_index index of point
#         * \param[out] mesh to modify
#         * \param[in] value value to use when re-setting
#         * \param[in] field_x_idx the X coordinate of the point
#         * \param[in] field_y_idx the Y coordinate of the point
#         * \param[in] field_z_idx the Z coordinate of the point
#         */
#       inline void
#       resetPointData (const int &point_index, pcl::PolygonMesh &mesh, const float &value = 0.0f,
#                       int field_x_idx = 0, int field_y_idx = 1, int field_z_idx = 2)
#       {
#         float new_value = value;
#         memcpy (&mesh.cloud.data[point_index * mesh.cloud.point_step + mesh.cloud.fields[field_x_idx].offset], &new_value, sizeof (float));
#         memcpy (&mesh.cloud.data[point_index * mesh.cloud.point_step + mesh.cloud.fields[field_y_idx].offset], &new_value, sizeof (float));
#         memcpy (&mesh.cloud.data[point_index * mesh.cloud.point_step + mesh.cloud.fields[field_z_idx].offset], &new_value, sizeof (float));
#       }
# 
#       /** \brief Check if a point is shadowed by another point
#         * \param[in] point_a the first point
#         * \param[in] point_b the second point
#         */
#       inline bool
#       isShadowed (const PointInT& point_a, const PointInT& point_b)
#       {
#         Eigen::Vector3f viewpoint = Eigen::Vector3f::Zero (); // TODO: allow for passing viewpoint information
#         Eigen::Vector3f dir_a = viewpoint - point_a.getVector3fMap ();
#         Eigen::Vector3f dir_b = point_b.getVector3fMap () - point_a.getVector3fMap ();
#         float distance_to_points = dir_a.norm ();
#         float distance_between_points = dir_b.norm ();
#         float cos_angle = dir_a.dot (dir_b) / (distance_to_points*distance_between_points);
#         if (cos_angle != cos_angle)
#           cos_angle = 1.0f;
#         return (fabs (cos_angle) >= cos_angle_tolerance_);
#         // TODO: check for both: angle almost 0/180 _and_ distance between points larger than noise level
#       }
# 
#       /** \brief Check if a triangle is valid.
#         * \param[in] a index of the first vertex
#         * \param[in] b index of the second vertex
#         * \param[in] c index of the third vertex
#         */
#       inline bool
#       isValidTriangle (const int& a, const int& b, const int& c)
#       {
#         if (!pcl::isFinite (input_->points[a])) return (false);
#         if (!pcl::isFinite (input_->points[b])) return (false);
#         if (!pcl::isFinite (input_->points[c])) return (false);
#         return (true);
#       }
# 
#       /** \brief Check if a triangle is shadowed.
#         * \param[in] a index of the first vertex
#         * \param[in] b index of the second vertex
#         * \param[in] c index of the third vertex
#         */
#       inline bool
#       isShadowedTriangle (const int& a, const int& b, const int& c)
#       {
#         if (isShadowed (input_->points[a], input_->points[b])) return (true);
#         if (isShadowed (input_->points[b], input_->points[c])) return (true);
#         if (isShadowed (input_->points[c], input_->points[a])) return (true);
#         return (false);
#       }
# 
#       /** \brief Check if a quad is valid.
#         * \param[in] a index of the first vertex
#         * \param[in] b index of the second vertex
#         * \param[in] c index of the third vertex
#         * \param[in] d index of the fourth vertex
#         */
#       inline bool
#       isValidQuad (const int& a, const int& b, const int& c, const int& d)
#       {
#         if (!pcl::isFinite (input_->points[a])) return (false);
#         if (!pcl::isFinite (input_->points[b])) return (false);
#         if (!pcl::isFinite (input_->points[c])) return (false);
#         if (!pcl::isFinite (input_->points[d])) return (false);
#         return (true);
#       }
# 
#       /** \brief Check if a triangle is shadowed.
#         * \param[in] a index of the first vertex
#         * \param[in] b index of the second vertex
#         * \param[in] c index of the third vertex
#         * \param[in] d index of the fourth vertex
#         */
#       inline bool
#       isShadowedQuad (const int& a, const int& b, const int& c, const int& d)
#       {
#         if (isShadowed (input_->points[a], input_->points[b])) return (true);
#         if (isShadowed (input_->points[b], input_->points[c])) return (true);
#         if (isShadowed (input_->points[c], input_->points[d])) return (true);
#         if (isShadowed (input_->points[d], input_->points[a])) return (true);
#         return (false);
#       }
# 
#       /** \brief Create a quad mesh.
#         * \param[out] polygons the resultant mesh
#         */
#       void
#       makeQuadMesh (std::vector<pcl::Vertices>& polygons);
# 
#       /** \brief Create a right cut mesh.
#         * \param[out] polygons the resultant mesh
#         */
#       void
#       makeRightCutMesh (std::vector<pcl::Vertices>& polygons);
# 
#       /** \brief Create a left cut mesh.
#         * \param[out] polygons the resultant mesh
#         */
#       void
#       makeLeftCutMesh (std::vector<pcl::Vertices>& polygons);
# 
#       /** \brief Create an adaptive cut mesh.
#         * \param[out] polygons the resultant mesh
#         */
#       void
#       makeAdaptiveCutMesh (std::vector<pcl::Vertices>& polygons);
#   };
# 
###

# poisson.h
# namespace pcl
# {
#   /** \brief The Poisson surface reconstruction algorithm.
#     * \note Code adapted from Misha Kazhdan: http://www.cs.jhu.edu/~misha/Code/PoissonRecon/
#     * \note Based on the paper:
#     *       * Michael Kazhdan, Matthew Bolitho, Hugues Hoppe, "Poisson surface reconstruction",
#     *         SGP '06 Proceedings of the fourth Eurographics symposium on Geometry processing
#     * \author Alexandru-Eugen Ichim
#     * \ingroup surface
#     */
#   template<typename PointNT>
#   class Poisson : public SurfaceReconstruction<PointNT>
#   {
#     public:
#       using SurfaceReconstruction<PointNT>::input_;
#       using SurfaceReconstruction<PointNT>::tree_;
# 
#       typedef typename pcl::PointCloud<PointNT>::Ptr PointCloudPtr;
# 
#       typedef typename pcl::KdTree<PointNT> KdTree;
#       typedef typename pcl::KdTree<PointNT>::Ptr KdTreePtr;
# 
#       /** \brief Constructor that sets all the parameters to working default values. */
#       Poisson ();
# 
#       /** \brief Destructor. */
#       ~Poisson ();
# 
#       /** \brief Create the surface.
#         * \param[out] output the resultant polygonal mesh
#         */
#       void
#       performReconstruction (pcl::PolygonMesh &output);
# 
#       /** \brief Create the surface.
#         * \param[out] points the vertex positions of the resulting mesh
#         * \param[out] polygons the connectivity of the resulting mesh
#         */
#       void
#       performReconstruction (pcl::PointCloud<PointNT> &points,
#                              std::vector<pcl::Vertices> &polygons);
# 
#       /** \brief Set the confidence flag
#         * \note Enabling this flag tells the reconstructor to use the size of the normals as confidence information.
#         * When the flag is not enabled, all normals are normalized to have unit-length prior to reconstruction.
#         * \param[in] confidence the given flag
#         */
#       inline void
#       setConfidence (bool confidence) { confidence_ = confidence; }
# 
#       /** \brief Get the confidence flag */
#       inline bool
#       getConfidence () { return confidence_; }
# 
#       /** \brief Set the manifold flag.
#         * \note Enabling this flag tells the reconstructor to add the polygon barycenter when triangulating polygons
#         * with more than three vertices.
#         * \param[in] manifold the given flag
#         */
#       inline void
#       setManifold (bool manifold) { manifold_ = manifold; }
# 
#       /** \brief Get the manifold flag */
#       inline bool
#       getManifold () { return manifold_; }
# 
#       /** \brief Enabling this flag tells the reconstructor to output a polygon mesh (rather than triangulating the
#         * results of Marching Cubes).
#         * \param[in] output_polygons the given flag
#         */
#       inline void
#       setOutputPolygons (bool output_polygons) { output_polygons_ = output_polygons; }
# 
#       /** \brief Get whether the algorithm outputs a polygon mesh or a triangle mesh */
#       inline bool
#       getOutputPolygons () { return output_polygons_; }
# 
# 
#       /** \brief Set the maximum depth of the tree that will be used for surface reconstruction.
#         * \note Running at depth d corresponds to solving on a voxel grid whose resolution is no larger than
#         * 2^d x 2^d x 2^d. Note that since the reconstructor adapts the octree to the sampling density, the specified
#         * reconstruction depth is only an upper bound.
#         * \param[in] depth the depth parameter
#         */
#       inline void
#       setDepth (int depth) { depth_ = depth; }
# 
#       /** \brief Get the depth parameter */
#       inline int
#       getDepth () { return depth_; }
# 
#       /** \brief Set the the depth at which a block Gauss-Seidel solver is used to solve the Laplacian equation
#         * \note Using this parameter helps reduce the memory overhead at the cost of a small increase in
#         * reconstruction time. (In practice, we have found that for reconstructions of depth 9 or higher a subdivide
#         * depth of 7 or 8 can greatly reduce the memory usage.)
#         * \param[in] solver_divide the given parameter value
#         */
#       inline void
#       setSolverDivide (int solver_divide) { solver_divide_ = solver_divide; }
# 
#       /** \brief Get the the depth at which a block Gauss-Seidel solver is used to solve the Laplacian equation */
#       inline int
#       getSolverDivide () { return solver_divide_; }
# 
#       /** \brief Set the depth at which a block iso-surface extractor should be used to extract the iso-surface
#         * \note Using this parameter helps reduce the memory overhead at the cost of a small increase in extraction
#         * time. (In practice, we have found that for reconstructions of depth 9 or higher a subdivide depth of 7 or 8
#         * can greatly reduce the memory usage.)
#         * \param[in] iso_divide the given parameter value
#         */
#       inline void
#       setIsoDivide (int iso_divide) { iso_divide_ = iso_divide; }
# 
#       /** \brief Get the depth at which a block iso-surface extractor should be used to extract the iso-surface */
#       inline int
#       getIsoDivide () { return iso_divide_; }
# 
#       /** \brief Set the minimum number of sample points that should fall within an octree node as the octree
#         * construction is adapted to sampling density
#         * \note For noise-free samples, small values in the range [1.0 - 5.0] can be used. For more noisy samples,
#         * larger values in the range [15.0 - 20.0] may be needed to provide a smoother, noise-reduced, reconstruction.
#         * \param[in] samples_per_node the given parameter value
#         */
#       inline void
#       setSamplesPerNode (float samples_per_node) { samples_per_node_ = samples_per_node; }
# 
#       /** \brief Get the minimum number of sample points that should fall within an octree node as the octree
#         * construction is adapted to sampling density
#         */
#       inline float
#       getSamplesPerNode () { return samples_per_node_; }
# 
#       /** \brief Set the ratio between the diameter of the cube used for reconstruction and the diameter of the
#         * samples' bounding cube.
#         * \param[in] scale the given parameter value
#         */
#       inline void
#       setScale (float scale) { scale_ = scale; }
# 
#       /** Get the ratio between the diameter of the cube used for reconstruction and the diameter of the
#         * samples' bounding cube.
#         */
#       inline float
#       getScale () { return scale_; }
# 
#       /** \brief Set the degree parameter
#         * \param[in] degree the given degree
#         */
#       inline void
#       setDegree (int degree) { degree_ = degree; }
# 
#       /** \brief Get the degree parameter */
#       inline int
#       getDegree () { return degree_; }
# 
# 
#     protected:
#       /** \brief The point cloud input (XYZ+Normals). */
#       PointCloudPtr data_;
# 
#       /** \brief Class get name method. */
#       std::string
#       getClassName () const { return ("Poisson"); }
# 
#     private:
#       bool no_reset_samples_;
#       bool no_clip_tree_;
#       bool confidence_;
#       bool manifold_;
#       bool output_polygons_;
# 
#       int depth_;
#       int solver_divide_;
#       int iso_divide_;
#       int refine_;
#       int kernel_depth_;
#       int degree_;
# 
#       float samples_per_node_;
#       float scale_;
# 
#       template<int Degree> void
#       execute (poisson::CoredMeshData &mesh,
#                poisson::Point3D<float> &translate,
#                float &scale);
# 
#     public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
# 
###

# polynomial.h (1.6.0)
# pcl/surface/3rdparty/poisson4/polynomial.h (1.7.2)
# namespace pcl
# namespace poisson
# template<int Degree>
# class Polynomial
#       public:
#       double coefficients[Degree+1];
# 
#       Polynomial(void);
#       template<int Degree2>
#       Polynomial(const Polynomial<Degree2>& P);
#       double operator() (const double& t) const;
#       double integral (const double& tMin,const double& tMax) const;
# 
#       int operator == (const Polynomial& p) const;
#       int operator != (const Polynomial& p) const;
#       int isZero(void) const;
#       void setZero(void);
# 
#       template<int Degree2>
#       Polynomial& operator  = (const Polynomial<Degree2> &p);
#       Polynomial& operator += (const Polynomial& p);
#       Polynomial& operator -= (const Polynomial& p);
#       Polynomial  operator -  (void) const;
#       Polynomial  operator +  (const Polynomial& p) const;
#       Polynomial  operator -  (const Polynomial& p) const;
#       template<int Degree2>
#       Polynomial<Degree+Degree2>  operator *  (const Polynomial<Degree2>& p) const;
# 
#       Polynomial& operator += (const double& s);
#       Polynomial& operator -= (const double& s);
#       Polynomial& operator *= (const double& s);
#       Polynomial& operator /= (const double& s);
#       Polynomial  operator +  (const double& s) const;
#       Polynomial  operator -  (const double& s) const;
#       Polynomial  operator *  (const double& s) const;
#       Polynomial  operator /  (const double& s) const;
# 
#       Polynomial scale (const double& s) const;
#       Polynomial shift (const double& t) const;
# 
#       Polynomial<Degree-1> derivative (void) const;
#       Polynomial<Degree+1> integral (void) const;
# 
#       void printnl (void) const;
# 
#       Polynomial& addScaled (const Polynomial& p, const double& scale);
# 
#       static void Negate (const Polynomial& in, Polynomial& out);
#       static void Subtract (const Polynomial& p1, const Polynomial& p2, Polynomial& q);
#       static void Scale (const Polynomial& p, const double& w, Polynomial& q);
#       static void AddScaled (const Polynomial& p1, const double& w1, const Polynomial& p2, const double& w2, Polynomial& q);
#       static void AddScaled (const Polynomial& p1, const Polynomial& p2, const double& w2, Polynomial& q);
#       static void AddScaled (const Polynomial& p1, const double& w1, const Polynomial& p2, Polynomial& q);
# 
#       void getSolutions (const double& c, std::vector<double>& roots, const double& EPS) const;
#     };
#   }
# }
###

# ppolynomial.h (1.6.0)
# pcl/surface/3rdparty/poisson4/ppolynomial.h (1.7.2)
# namespace pcl
# {
#   namespace poisson
#   {
#     template <int Degree> 
#     class StartingPolynomial
#     {
#       public:
#         Polynomial<Degree> p;
#         double start;
# 
#         StartingPolynomial () : p (), start () {}
# 
#         template <int Degree2> StartingPolynomial<Degree+Degree2> operator* (const StartingPolynomial<Degree2>&p) const;
#         StartingPolynomial scale (const double&s) const;
#         StartingPolynomial shift (const double&t) const;
#         int operator < (const StartingPolynomial &sp) const;
#         static int Compare (const void *v1,const void *v2);
#     };
# 
#     template <int Degree> 
#     class PPolynomial
#     {
#       public:
#         size_t polyCount;
#         StartingPolynomial<Degree>*polys;
# 
#         PPolynomial (void);
#         PPolynomial (const PPolynomial<Degree>&p);
#         ~PPolynomial (void);
# 
#         PPolynomial& operator = (const PPolynomial&p);
# 
#         int size (void) const;
# 
#         void set (const size_t&size);
#         // Note: this method will sort the elements in sps
#         void set (StartingPolynomial<Degree>*sps,const int&count);
#         void reset (const size_t&newSize);
# 
# 
#         double operator() (const double &t) const;
#         double integral (const double &tMin,const double &tMax) const;
#         double Integral (void) const;
# 
#         template <int Degree2> PPolynomial<Degree>& operator = (const PPolynomial<Degree2>&p);
# 
#         PPolynomial operator + (const PPolynomial&p) const;
#         PPolynomial operator - (const PPolynomial &p) const;
# 
#         template <int Degree2> PPolynomial<Degree+Degree2> operator * (const Polynomial<Degree2> &p) const;
# 
#         template <int Degree2> PPolynomial<Degree+Degree2> operator* (const PPolynomial<Degree2>&p) const;
# 
# 
#         PPolynomial& operator += (const double&s);
#         PPolynomial& operator -= (const double&s);
#         PPolynomial& operator *= (const double&s);
#         PPolynomial& operator /= (const double&s);
#         PPolynomial operator +  (const double&s) const;
#         PPolynomial operator -  (const double&s) const;
#         PPolynomial operator*  (const double&s) const;
#         PPolynomial operator /  (const double &s) const;
# 
#         PPolynomial& addScaled (const PPolynomial &poly,const double &scale);
# 
#         PPolynomial scale (const double &s) const;
#         PPolynomial shift (const double &t) const;
# 
#         PPolynomial<Degree-1> derivative (void) const;
#         PPolynomial<Degree+1> integral (void) const;
# 
#         void getSolutions (const double &c,
#                            std::vector<double> &roots,
#                            const double &EPS,
#                            const double &min =- DBL_MAX,
#                            const double &max=DBL_MAX) const;
# 
#         void printnl (void) const;
# 
#         PPolynomial<Degree+1> MovingAverage (const double &radius);
# 
#         static PPolynomial ConstantFunction (const double &width=0.5);
#         static PPolynomial GaussianApproximation (const double &width=0.5);
#         void write (FILE *fp,
#                     const int &samples,
#                     const double &min,
#                     const double &max) const;
#     };
# 
# 
#   }
# }
###

# qhull.h
# 
# #if defined __GNUC__
# #  pragma GCC system_header 
# #endif
# 
# extern "C"
# {
# #ifdef HAVE_QHULL_2011
# #  include "libqhull/libqhull.h"
# #  include "libqhull/mem.h"
# #  include "libqhull/qset.h"
# #  include "libqhull/geom.h"
# #  include "libqhull/merge.h"
# #  include "libqhull/poly.h"
# #  include "libqhull/io.h"
# #  include "libqhull/stat.h"
# #else
# #  include "qhull/qhull.h"
# #  include "qhull/mem.h"
# #  include "qhull/qset.h"
# #  include "qhull/geom.h"
# #  include "qhull/merge.h"
# #  include "qhull/poly.h"
# #  include "qhull/io.h"
# #  include "qhull/stat.h"
# #endif
# }
# 
###

# simplification_remove_unused_vertices.h
# namespace pcl
# {
#   namespace surface
#   {
#     class PCL_EXPORTS SimplificationRemoveUnusedVertices
#     {
#       public:
#         /** \brief Constructor. */
#         SimplificationRemoveUnusedVertices () {};
#         /** \brief Destructor. */
#         ~SimplificationRemoveUnusedVertices () {};
# 
#         /** \brief Simply a polygonal mesh.
#           * \param[in] input the input mesh
#           * \param[out] output the output mesh
#           */
#         inline void
#         simplify (const pcl::PolygonMesh& input, pcl::PolygonMesh& output)
#         {
#           std::vector<int> indices;
#           simplify (input, output, indices);
#         }
# 
#         /** \brief Perform simplification (remove unused vertices).
#           * \param[in] input the input mesh
#           * \param[out] output the output mesh
#           * \param[out] indices the resultant vector of indices
#           */
#         void
#         simplify (const pcl::PolygonMesh& input, pcl::PolygonMesh& output, std::vector<int>& indices);
# 
#     };
#   }
###

# sparse_matrix.h
# pcl/surface/3rdparty/poisson4/sparse_matrix.h (1.7.2)
# 
# namespace pcl 
# namespace poisson 
# template <class T>
# struct MatrixEntry
#     {
#       MatrixEntry () : N (-1), Value (0) {}
#       MatrixEntry (int i) : N (i), Value (0) {}
#       int N;
#       T Value;
#     };
# 
#     template <class T,int Dim>
#     struct NMatrixEntry
#     {
#       NMatrixEntry () : N (-1), Value () { memset (Value, 0, sizeof (T) * Dim); }
#       NMatrixEntry (int i) : N (i), Value () { memset (Value, 0, sizeof (T) * Dim); }
#       int N;
#       T Value[Dim];
#     };
# 
#     template<class T> class SparseMatrix
#     {
#       private:
#         static int UseAlloc;
#       public:
#         static Allocator<MatrixEntry<T> > AllocatorMatrixEntry;
#         static int UseAllocator (void);
#         static void SetAllocator (const int& blockSize);
# 
#         int rows;
#         int* rowSizes;
#         MatrixEntry<T>** m_ppElements;
# 
#         SparseMatrix ();
#         SparseMatrix (int rows);
#         void Resize (int rows);
#         void SetRowSize (int row , int count);
#         int Entries (void);
# 
#         SparseMatrix (const SparseMatrix& M);
#         virtual ~SparseMatrix ();
# 
#         void SetZero ();
#         void SetIdentity ();
# 
#         SparseMatrix<T>& operator = (const SparseMatrix<T>& M);
# 
#         SparseMatrix<T> operator * (const T& V) const;
#         SparseMatrix<T>& operator *= (const T& V);
# 
# 
#         SparseMatrix<T> operator * (const SparseMatrix<T>& M) const;
#         SparseMatrix<T> Multiply (const SparseMatrix<T>& M) const;
#         SparseMatrix<T> MultiplyTranspose (const SparseMatrix<T>& Mt) const;
# 
#         template<class T2>
#         Vector<T2> operator * (const Vector<T2>& V) const;
#         template<class T2>
#         Vector<T2> Multiply (const Vector<T2>& V) const;
#         template<class T2>
#         void Multiply (const Vector<T2>& In, Vector<T2>& Out) const;
# 
# 
#         SparseMatrix<T> Transpose() const;
# 
#         static int Solve (const SparseMatrix<T>& M,
#                           const Vector<T>& b,
#                           const int& iters,
#                           Vector<T>& solution,
#                           const T eps = 1e-8);
# 
#         template<class T2>
#         static int SolveSymmetric (const SparseMatrix<T>& M,
#                                    const Vector<T2>& b,
#                                    const int& iters,
#                                    Vector<T2>& solution,
#                                    const T2 eps = 1e-8,
#                                    const int& reset=1);
# 
#     };
# 
#     template<class T,int Dim> class SparseNMatrix
#     {
#       private:
#         static int UseAlloc;
#       public:
#         static Allocator<NMatrixEntry<T,Dim> > AllocatorNMatrixEntry;
#         static int UseAllocator (void);
#         static void SetAllocator (const int& blockSize);
# 
#         int rows;
#         int* rowSizes;
#         NMatrixEntry<T,Dim>** m_ppElements;
# 
#         SparseNMatrix ();
#         SparseNMatrix (int rows);
#         void Resize (int rows);
#         void SetRowSize (int row, int count);
#         int Entries ();
# 
#         SparseNMatrix (const SparseNMatrix& M);
#         ~SparseNMatrix ();
# 
#         SparseNMatrix& operator = (const SparseNMatrix& M);
# 
#         SparseNMatrix  operator *  (const T& V) const;
#         SparseNMatrix& operator *= (const T& V);
# 
#         template<class T2>
#         NVector<T2,Dim> operator * (const Vector<T2>& V) const;
#         template<class T2>
#         Vector<T2> operator * (const NVector<T2,Dim>& V) const;
#     };
# 
#     template <class T>
#     class SparseSymmetricMatrix : public SparseMatrix<T>
#     {
#       public:
#         virtual ~SparseSymmetricMatrix () {}
# 
#         template<class T2>
#         Vector<T2> operator * (const Vector<T2>& V) const;
# 
#         template<class T2>
#         Vector<T2> Multiply (const Vector<T2>& V ) const;
# 
#         template<class T2> void 
#         Multiply (const Vector<T2>& In, Vector<T2>& Out ) const;
# 
#         template<class T2> static int 
#         Solve (const SparseSymmetricMatrix<T>& M,
#                const Vector<T2>& b,
#                const int& iters,
#                Vector<T2>& solution,
#                const T2 eps = 1e-8,
#                const int& reset=1);
# 
#         template<class T2> static int 
#         Solve (const SparseSymmetricMatrix<T>& M,
#                const Vector<T>& diagonal,
#                const Vector<T2>& b,
#                const int& iters,
#                Vector<T2>& solution,
#                const T2 eps = 1e-8,
#                const int& reset=1);
#     };
#   }
# 
###

# surfel_smoothing.h
# namespace pcl
# {
#   template <typename PointT, typename PointNT>
#   class SurfelSmoothing : public PCLBase<PointT>
#   {
#     using PCLBase<PointT>::input_;
#     using PCLBase<PointT>::initCompute;
# 
#     public:
#       typedef pcl::PointCloud<PointT> PointCloudIn;
#       typedef typename pcl::PointCloud<PointT>::Ptr PointCloudInPtr;
#       typedef pcl::PointCloud<PointNT> NormalCloud;
#       typedef typename pcl::PointCloud<PointNT>::Ptr NormalCloudPtr;
#       typedef pcl::search::Search<PointT> CloudKdTree;
#       typedef typename pcl::search::Search<PointT>::Ptr CloudKdTreePtr;
# 
#       SurfelSmoothing (float a_scale = 0.01)
#         : PCLBase<PointT> ()
#         , scale_ (a_scale)
#         , scale_squared_ (a_scale * a_scale)
#         , normals_ ()
#         , interm_cloud_ ()
#         , interm_normals_ ()
#         , tree_ ()
#       {
#       }
# 
#       void
#       setInputNormals (NormalCloudPtr &a_normals) { normals_ = a_normals; };
# 
#       void
#       setSearchMethod (const CloudKdTreePtr &a_tree) { tree_ = a_tree; };
# 
#       bool
#       initCompute ();
# 
#       float
#       smoothCloudIteration (PointCloudInPtr &output_positions,
#                             NormalCloudPtr &output_normals);
# 
#       void
#       computeSmoothedCloud (PointCloudInPtr &output_positions,
#                             NormalCloudPtr &output_normals);
# 
# 
#       void
#       smoothPoint (size_t &point_index,
#                    PointT &output_point,
#                    PointNT &output_normal);
# 
#       void
#       extractSalientFeaturesBetweenScales (PointCloudInPtr &cloud2,
#                                            NormalCloudPtr &cloud2_normals,
#                                            boost::shared_ptr<std::vector<int> > &output_features);
# 
#     private:
#       float scale_, scale_squared_;
#       NormalCloudPtr normals_;
# 
#       PointCloudInPtr interm_cloud_;
#       NormalCloudPtr interm_normals_;
# 
#       CloudKdTreePtr tree_;
# 
#   };
###

# texture_mapping.h
# namespace pcl
# {
#   namespace texture_mapping
#   {
#         
#     /** \brief Structure to store camera pose and focal length. */
#     struct Camera
#     {
#       Camera () : pose (), focal_length (), height (), width (), texture_file () {}
#       Eigen::Affine3f pose;
#       double focal_length;
#       double height;
#       double width;
#       std::string texture_file;
# 
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#     };
# 
#     /** \brief Structure that links a uv coordinate to its 3D point and face.
#       */
#     struct UvIndex
#     {
#       UvIndex () : idx_cloud (), idx_face () {}
#       int idx_cloud; // Index of the PointXYZ in the camera's cloud
#       int idx_face; // Face corresponding to that projection
#     };
#     
#     typedef std::vector<Camera, Eigen::aligned_allocator<Camera> > CameraVector;
#     
#   }
#   
#   /** \brief The texture mapping algorithm
#     * \author Khai Tran, Raphael Favier
#     * \ingroup surface
#     */
#   template<typename PointInT>
#   class TextureMapping
#   {
#     public:
#      
#       typedef boost::shared_ptr< PointInT > Ptr;
#       typedef boost::shared_ptr< const PointInT > ConstPtr;
# 
#       typedef pcl::PointCloud<PointInT> PointCloud;
#       typedef typename PointCloud::Ptr PointCloudPtr;
#       typedef typename PointCloud::ConstPtr PointCloudConstPtr;
# 
#       typedef pcl::octree::OctreePointCloudSearch<PointInT> Octree;
#       typedef typename Octree::Ptr OctreePtr;
#       typedef typename Octree::ConstPtr OctreeConstPtr;
#       
#       typedef pcl::texture_mapping::Camera Camera;
#       typedef pcl::texture_mapping::UvIndex UvIndex;
# 
#       /** \brief Constructor. */
#       TextureMapping () :
#         f_ (), vector_field_ (), tex_files_ (), tex_material_ ()
#       {
#       }
# 
#       /** \brief Destructor. */
#       ~TextureMapping ()
#       {
#       }
# 
#       /** \brief Set mesh scale control
#         * \param[in] f
#         */
#       inline void
#       setF (float f)
#       {
#         f_ = f;
#       }
# 
#       /** \brief Set vector field
#         * \param[in] x data point x
#         * \param[in] y data point y
#         * \param[in] z data point z
#         */
#       inline void
#       setVectorField (float x, float y, float z)
#       {
#         vector_field_ = Eigen::Vector3f (x, y, z);
#         // normalize vector field
#         vector_field_ = vector_field_ / std::sqrt (vector_field_.dot (vector_field_));
#       }
# 
#       /** \brief Set texture files
#         * \param[in] tex_files list of texture files
#         */
#       inline void
#       setTextureFiles (std::vector<std::string> tex_files)
#       {
#         tex_files_ = tex_files;
#       }
# 
#       /** \brief Set texture materials
#         * \param[in] tex_material texture material
#         */
#       inline void
#       setTextureMaterials (TexMaterial tex_material)
#       {
#         tex_material_ = tex_material;
#       }
# 
#       /** \brief Map texture to a mesh synthesis algorithm
#         * \param[in] tex_mesh texture mesh
#         */
#       void
#       mapTexture2Mesh (pcl::TextureMesh &tex_mesh);
# 
#       /** \brief map texture to a mesh UV mapping
#         * \param[in] tex_mesh texture mesh
#         */
#       void
#       mapTexture2MeshUV (pcl::TextureMesh &tex_mesh);
# 
#       /** \brief map textures aquired from a set of cameras onto a mesh.
#         * \details With UV mapping, the mesh must be divided into NbCamera + 1 sub-meshes.
#         * Each sub-mesh corresponding to the faces visible by one camera. The last submesh containing all non-visible faces
#         * \param[in] tex_mesh texture mesh
#         * \param[in] cams cameras used for UV mapping
#         */
#       void
#       mapMultipleTexturesToMeshUV (pcl::TextureMesh &tex_mesh, 
#                                    pcl::texture_mapping::CameraVector &cams);
# 
#       /** \brief computes UV coordinates of point, observed by one particular camera
#         * \param[in] pt XYZ point to project on camera plane
#         * \param[in] cam the camera used for projection
#         * \param[out] UV_coordinates the resulting uv coordinates. Set to (-1.0,-1.0) if the point is not visible by the camera
#         * \returns false if the point is not visible by the camera
#         */
#       inline bool
#       getPointUVCoordinates (const pcl::PointXYZ &pt, const Camera &cam, Eigen::Vector2f &UV_coordinates)
#       {
#         // if the point is in front of the camera
#         if (pt.z > 0)
#         {
#           // compute image center and dimension
#           double sizeX = cam.width;
#           double sizeY = cam.height;
#           double cx = (sizeX) / 2.0;
#           double cy = (sizeY) / 2.0;
# 
#           double focal_x = cam.focal_length;
#           double focal_y = cam.focal_length;
# 
#           // project point on image frame
#           UV_coordinates[0] = static_cast<float> ((focal_x * (pt.x / pt.z) + cx) / sizeX); //horizontal
#           UV_coordinates[1] = 1.0f - static_cast<float> (((focal_y * (pt.y / pt.z) + cy) / sizeY)); //vertical
# 
#           // point is visible!
#           if (UV_coordinates[0] >= 0.0 && UV_coordinates[0] <= 1.0 && UV_coordinates[1] >= 0.0 && UV_coordinates[1]
#                                                                                                                  <= 1.0)
#             return (true);
#         }
# 
#         // point is NOT visible by the camera
#         UV_coordinates[0] = -1.0;
#         UV_coordinates[1] = -1.0;
#         return (false);
#       }
# 
#       /** \brief Check if a point is occluded using raycasting on octree.
#         * \param[in] pt XYZ from which the ray will start (toward the camera)
#         * \param[in] octree the octree used for raycasting. It must be initialized with a cloud transformed into the camera's frame
#         * \returns true if the point is occluded.
#         */
#       inline bool
#       isPointOccluded (const pcl::PointXYZ &pt, const OctreePtr octree);
# 
#       /** \brief Remove occluded points from a point cloud
#         * \param[in] input_cloud the cloud on which to perform occlusion detection
#         * \param[out] filtered_cloud resulting cloud, containing only visible points
#         * \param[in] octree_voxel_size octree resolution (in meters)
#         * \param[out] visible_indices will contain indices of visible points
#         * \param[out] occluded_indices will contain indices of occluded points
#         */
#       void
#       removeOccludedPoints (const PointCloudPtr &input_cloud,
#                             PointCloudPtr &filtered_cloud, const double octree_voxel_size,
#                             std::vector<int> &visible_indices, std::vector<int> &occluded_indices);
# 
#       /** \brief Remove occluded points from a textureMesh
#         * \param[in] tex_mesh input mesh, on witch to perform occlusion detection
#         * \param[out] cleaned_mesh resulting mesh, containing only visible points
#         * \param[in] octree_voxel_size octree resolution (in meters)
#         */
#       void
#       removeOccludedPoints (const pcl::TextureMesh &tex_mesh, pcl::TextureMesh &cleaned_mesh, const double octree_voxel_size);
# 
# 
#       /** \brief Remove occluded points from a textureMesh
#         * \param[in] tex_mesh input mesh, on witch to perform occlusion detection
#         * \param[out] filtered_cloud resulting cloud, containing only visible points
#         * \param[in] octree_voxel_size octree resolution (in meters)
#         */
#       void
#       removeOccludedPoints (const pcl::TextureMesh &tex_mesh, PointCloudPtr &filtered_cloud, const double octree_voxel_size);
# 
# 
#       /** \brief Segment faces by camera visibility. Point-based segmentation.
#         * \details With N camera, faces will be arranged into N+1 groups: 1 for each camera, plus 1 for faces not visible from any camera.
#         * \param[in] tex_mesh input mesh that needs sorting. Must contain only 1 sub-mesh.
#         * \param[in] sorted_mesh resulting mesh, will contain nbCamera + 1 sub-mesh.
#         * \param[in] cameras vector containing the cameras used for texture mapping.
#         * \param[in] octree_voxel_size octree resolution (in meters)
#         * \param[out] visible_pts cloud containing only visible points
#         */
#       int
#       sortFacesByCamera (pcl::TextureMesh &tex_mesh, 
#                          pcl::TextureMesh &sorted_mesh, 
#                          const pcl::texture_mapping::CameraVector &cameras,
#                          const double octree_voxel_size, PointCloud &visible_pts);
# 
#       /** \brief Colors a point cloud, depending on its occlusions.
#         * \details If showNbOcclusions is set to True, each point is colored depending on the number of points occluding it.
#         * Else, each point is given a different a 0 value is not occluded, 1 if occluded.
#         * By default, the number of occlusions is bounded to 4.
#         * \param[in] input_cloud input cloud on which occlusions will be computed.
#         * \param[out] colored_cloud resulting colored cloud showing the number of occlusions per point.
#         * \param[in] octree_voxel_size octree resolution (in meters).
#         * \param[in] show_nb_occlusions If false, color information will only represent.
#         * \param[in] max_occlusions Limit the number of occlusions per point.
#         */
#       void
#       showOcclusions (const PointCloudPtr &input_cloud, 
#                       pcl::PointCloud<pcl::PointXYZI>::Ptr &colored_cloud,
#                       const double octree_voxel_size, 
#                       const bool show_nb_occlusions = true,
#                       const int max_occlusions = 4);
# 
#       /** \brief Colors the point cloud of a Mesh, depending on its occlusions.
#         * \details If showNbOcclusions is set to True, each point is colored depending on the number of points occluding it.
#         * Else, each point is given a different a 0 value is not occluded, 1 if occluded.
#         * By default, the number of occlusions is bounded to 4.
#         * \param[in] tex_mesh input mesh on which occlusions will be computed.
#         * \param[out] colored_cloud resulting colored cloud showing the number of occlusions per point.
#         * \param[in] octree_voxel_size octree resolution (in meters).
#         * \param[in] show_nb_occlusions If false, color information will only represent.
#         * \param[in] max_occlusions Limit the number of occlusions per point.
#         */
#       void
#       showOcclusions (pcl::TextureMesh &tex_mesh, 
#                       pcl::PointCloud<pcl::PointXYZI>::Ptr &colored_cloud,
#                       double octree_voxel_size, 
#                       bool show_nb_occlusions = true, 
#                       int max_occlusions = 4);
# 
#       /** \brief Segment and texture faces by camera visibility. Face-based segmentation.
#         * \details With N camera, faces will be arranged into N+1 groups: 1 for each camera, plus 1 for faces not visible from any camera.
#         * The mesh will also contain uv coordinates for each face
#         * \param[in/out] tex_mesh input mesh that needs sorting. Should contain only 1 sub-mesh.
#         * \param[in] cameras vector containing the cameras used for texture mapping.
#         */
#       void 
#       textureMeshwithMultipleCameras (pcl::TextureMesh &mesh, 
#                                       const pcl::texture_mapping::CameraVector &cameras);
# 
#     protected:
#       /** \brief mesh scale control. */
#       float f_;
# 
#       /** \brief vector field */
#       Eigen::Vector3f vector_field_;
# 
#       /** \brief list of texture files */
#       std::vector<std::string> tex_files_;
# 
#       /** \brief list of texture materials */
#       TexMaterial tex_material_;
# 
#       /** \brief Map texture to a face
#         * \param[in] p1 the first point
#         * \param[in] p2 the second point
#         * \param[in] p3 the third point
#         */
#       std::vector<Eigen::Vector2f>
#       mapTexture2Face (const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, const Eigen::Vector3f &p3);
# 
#       /** \brief Returns the circumcenter of a triangle and the circle's radius.
#         * \details see http://en.wikipedia.org/wiki/Circumcenter for formulas.
#         * \param[in] p1 first point of the triangle.
#         * \param[in] p2 second point of the triangle.
#         * \param[in] p3 third point of the triangle.
#         * \param[out] circumcenter resulting circumcenter
#         * \param[out] radius the radius of the circumscribed circle.
#         */
#       inline void
#       getTriangleCircumcenterAndSize (const pcl::PointXY &p1, const pcl::PointXY &p2, const pcl::PointXY &p3, pcl::PointXY &circomcenter, double &radius);
# 
#       /** \brief computes UV coordinates of point, observed by one particular camera
#         * \param[in] pt XYZ point to project on camera plane
#         * \param[in] cam the camera used for projection
#         * \param[out] UV_coordinates the resulting UV coordinates. Set to (-1.0,-1.0) if the point is not visible by the camera
#         * \returns false if the point is not visible by the camera
#         */
#       inline bool
#       getPointUVCoordinates (const pcl::PointXYZ &pt, const Camera &cam, pcl::PointXY &UV_coordinates);
# 
#       /** \brief Returns true if all the vertices of one face are projected on the camera's image plane.
#         * \param[in] camera camera on which to project the face.
#         * \param[in] p1 first point of the face.
#         * \param[in] p2 second point of the face.
#         * \param[in] p3 third point of the face.
#         * \param[out] proj1 UV coordinates corresponding to p1.
#         * \param[out] proj2 UV coordinates corresponding to p2.
#         * \param[out] proj3 UV coordinates corresponding to p3.
#         */
#       inline bool
#       isFaceProjected (const Camera &camera, 
#                        const pcl::PointXYZ &p1, const pcl::PointXYZ &p2, const pcl::PointXYZ &p3, 
#                        pcl::PointXY &proj1, pcl::PointXY &proj2, pcl::PointXY &proj3);
# 
#       /** \brief Returns True if a point lays within a triangle
#         * \details see http://www.blackpawn.com/texts/pointinpoly/default.html
#         * \param[in] p1 first point of the triangle.
#         * \param[in] p2 second point of the triangle.
#         * \param[in] p3 third point of the triangle.
#         * \param[in] pt the querry point.
#         */
#       inline bool
#       checkPointInsideTriangle (const pcl::PointXY &p1, const pcl::PointXY &p2, const pcl::PointXY &p3, const pcl::PointXY &pt);
# 
#       /** \brief Class get name method. */
#       std::string
#       getClassName () const
#       {
#         return ("TextureMapping");
#       }
# 
#     public:
#       EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#   };
# 
###

# vector.h (1.6.0)
# pcl/surface/3rdparty/poisson4/vector.h (1.7.2)
# namespace pcl {
# namespace poisson {
# 
#     template<class T>
#     class Vector
#     {
#     public:
#       Vector ();
#       Vector (const Vector<T>& V);
#       Vector (size_t N);
#       Vector (size_t N, T* pV);
#       ~Vector();
# 
#       const T& operator () (size_t i) const;
#       T& operator () (size_t i);
#       const T& operator [] (size_t i) const;
#       T& operator [] (size_t i);
# 
#       void SetZero();
# 
#       size_t Dimensions() const;
#       void Resize( size_t N );
# 
#       Vector operator * (const T& A) const;
#       Vector operator / (const T& A) const;
#       Vector operator - (const Vector& V) const;
#       Vector operator + (const Vector& V) const;
# 
#       Vector& operator *= (const T& A);
#       Vector& operator /= (const T& A);
#       Vector& operator += (const Vector& V);
#       Vector& operator -= (const Vector& V);
# 
#       Vector& AddScaled (const Vector& V,const T& scale);
#       Vector& SubtractScaled (const Vector& V,const T& scale);
#       static void Add (const Vector& V1,const T& scale1,const Vector& V2,const T& scale2,Vector& Out);
#       static void Add (const Vector& V1,const T& scale1,const Vector& V2,Vector& Out);
# 
#       Vector operator - () const;
# 
#       Vector& operator = (const Vector& V);
# 
#       T Dot (const Vector& V) const;
# 
#       T Length() const;
# 
#       T Norm (size_t Ln) const;
#       void Normalize();
# 
#       T* m_pV;
#     protected:
#       size_t m_N;
# 
#     };
# 
#     template<class T,int Dim>
#     class NVector
#     {
#     public:
#       NVector ();
#       NVector (const NVector& V);
#       NVector (size_t N);
#       NVector (size_t N, T* pV);
#       ~NVector ();
# 
#       const T* operator () (size_t i) const;
#       T* operator () (size_t i);
#       const T* operator [] (size_t i) const;
#       T* operator [] (size_t i);
# 
#       void SetZero();
# 
#       size_t Dimensions() const;
#       void Resize( size_t N );
# 
#       NVector operator * (const T& A) const;
#       NVector operator / (const T& A) const;
#       NVector operator - (const NVector& V) const;
#       NVector operator + (const NVector& V) const;
# 
#       NVector& operator *= (const T& A);
#       NVector& operator /= (const T& A);
#       NVector& operator += (const NVector& V);
#       NVector& operator -= (const NVector& V);
# 
#       NVector& AddScaled (const NVector& V,const T& scale);
#       NVector& SubtractScaled (const NVector& V,const T& scale);
#       static void Add (const NVector& V1,const T& scale1,const NVector& V2,const T& scale2,NVector& Out);
#       static void Add (const NVector& V1,const T& scale1,const NVector& V2, NVector& Out);
# 
#       NVector operator - () const;
# 
#       NVector& operator = (const NVector& V);
# 
#       T Dot (const NVector& V) const;
# 
#       T Length () const;
# 
#       T Norm (size_t Ln) const;
#       void Normalize ();
# 
#       T* m_pV;
#     protected:
#       size_t m_N;
# 
#     };
# 
#   }
# }
###

# vtk.h (1.6.0)
# pcl\surface\vtk_smoothing\vtk_smoothing.h (1.7.2)
# #include <vtkPolyData.h>
# #include <vtkSmartPointer.h>
###

# pcl\surface\vtk_smoothing\vtk_mesh_quadric_decimation.h (1.7.2)

# vtk_mesh_smoothing_laplacian.h (1.6.0)
# pcl\surface\vtk_smoothing\vtk_mesh_smoothing_laplacian.h (1.7.2)
# namespace pcl
# {
#   /** \brief PCL mesh smoothing based on the vtkSmoothPolyDataFilter algorithm from the VTK library.
#     * Please check out the original documentation for more details on the inner workings of the algorithm
#     * Warning: This wrapper does two fairly computationally expensive conversions from the PCL PolygonMesh
#     * data structure to the vtkPolyData data structure and back.
#     */
#   class PCL_EXPORTS MeshSmoothingLaplacianVTK : public MeshProcessing
#   {
#     public:
#       /** \brief Empty constructor that sets the values of the algorithm parameters to the VTK defaults */
#       MeshSmoothingLaplacianVTK ()
#         : MeshProcessing ()
#         , vtk_polygons_ ()
#         , num_iter_ (20)
#         , convergence_ (0.0f)
#         , relaxation_factor_ (0.01f)
#         , feature_edge_smoothing_ (false)
#         , feature_angle_ (45.f)
#         , edge_angle_ (15.f)
#         , boundary_smoothing_ (true)
#       {};
# 
#       /** \brief Set the number of iterations for the smoothing filter.
#         * \param[in] num_iter the number of iterations
#         */
#       inline void
#       setNumIter (int num_iter)
#       {
#         num_iter_ = num_iter;
#       };
# 
#       /** \brief Get the number of iterations. */
#       inline int
#       getNumIter ()
#       {
#         return num_iter_;
#       };
# 
#       /** \brief Specify a convergence criterion for the iteration process. Smaller numbers result in more smoothing iterations.
#        * \param[in] convergence convergence criterion for the Laplacian smoothing
#        */
#       inline void
#       setConvergence (float convergence)
#       {
#         convergence_ = convergence;
#       };
# 
#       /** \brief Get the convergence criterion. */
#       inline float
#       getConvergence ()
#       {
#         return convergence_;
#       };
# 
#       /** \brief Specify the relaxation factor for Laplacian smoothing. As in all iterative methods,
#        * the stability of the process is sensitive to this parameter.
#        * In general, small relaxation factors and large numbers of iterations are more stable than larger relaxation
#        * factors and smaller numbers of iterations.
#        * \param[in] relaxation_factor the relaxation factor of the Laplacian smoothing algorithm
#        */
#       inline void
#       setRelaxationFactor (float relaxation_factor)
#       {
#         relaxation_factor_ = relaxation_factor;
#       };
# 
#       /** \brief Get the relaxation factor of the Laplacian smoothing */
#       inline float
#       getRelaxationFactor ()
#       {
#         return relaxation_factor_;
#       };
# 
#       /** \brief Turn on/off smoothing along sharp interior edges.
#        * \param[in] status decision whether to enable/disable smoothing along sharp interior edges
#        */
#       inline void
#       setFeatureEdgeSmoothing (bool feature_edge_smoothing)
#       {
#         feature_edge_smoothing_ = feature_edge_smoothing;
#       };
# 
#       /** \brief Get the status of the feature edge smoothing */
#       inline bool
#       getFeatureEdgeSmoothing ()
#       {
#         return feature_edge_smoothing_;
#       };
# 
#       /** \brief Specify the feature angle for sharp edge identification.
#        * \param[in] feature_angle the angle threshold for considering an edge to be sharp
#        */
#       inline void
#       setFeatureAngle (float feature_angle)
#       {
#         feature_angle_ = feature_angle;
#       };
# 
#       /** \brief Get the angle threshold for considering an edge to be sharp */
#       inline float
#       getFeatureAngle ()
#       {
#         return feature_angle_;
#       };
# 
#       /** \brief Specify the edge angle to control smoothing along edges (either interior or boundary).
#        * \param[in] edge_angle the angle to control smoothing along edges
#        */
#       inline void
#       setEdgeAngle (float edge_angle)
#       {
#         edge_angle_ = edge_angle;
#       };
# 
#       /** \brief Get the edge angle to control smoothing along edges */
#       inline float
#       getEdgeAngle ()
#       {
#         return edge_angle_;
#       };
# 
#       /** \brief Turn on/off the smoothing of vertices on the boundary of the mesh.
#        * \param[in] boundary_smoothing decision whether boundary smoothing is on or off
#        */
#       inline void
#       setBoundarySmoothing (bool boundary_smoothing)
#       {
#         boundary_smoothing_ = boundary_smoothing;
#       };
# 
#       /** \brief Get the status of the boundary smoothing */
#       inline bool
#       getBoundarySmoothing ()
#       {
#         return boundary_smoothing_;
#       }
# 
#     protected:
#       void
#       performProcessing (pcl::PolygonMesh &output);
# 
#     private:
#       vtkSmartPointer<vtkPolyData> vtk_polygons_;
# 
#       /// Parameters
#       int num_iter_;
#       float convergence_;
#       float relaxation_factor_;
#       bool feature_edge_smoothing_;
#       float feature_angle_;
#       float edge_angle_;
#       bool boundary_smoothing_;
#   };
###

# vtk_mesh_smoothing_windowed_sinc.h (1.6.0)
# pcl\surface\vtk_smoothing\vtk_mesh_smoothing_windowed_sinc.h (1.7.2)
# namespace pcl
# /** \brief PCL mesh smoothing based on the vtkWindowedSincPolyDataFilter algorithm from the VTK library.
#   * Please check out the original documentation for more details on the inner workings of the algorithm
#   * Warning: This wrapper does two fairly computationally expensive conversions from the PCL PolygonMesh
#   * data structure to the vtkPolyData data structure and back.
#   */
# class PCL_EXPORTS MeshSmoothingWindowedSincVTK : public MeshProcessing
#       public:
#       /** \brief Empty constructor that sets the values of the algorithm parameters to the VTK defaults */
#       MeshSmoothingWindowedSincVTK ()
#         : MeshProcessing (),
#           num_iter_ (20),
#           pass_band_ (0.1f),
#           feature_edge_smoothing_ (false),
#           feature_angle_ (45.f),
#           edge_angle_ (15.f),
#           boundary_smoothing_ (true),
#           normalize_coordinates_ (false)
#       {};
# 
#       /** \brief Set the number of iterations for the smoothing filter.
#         * \param[in] num_iter the number of iterations
#       inline void setNumIter (int num_iter)
#       /** \brief Get the number of iterations. */
#       inline int getNumIter ()
#       /** \brief Set the pass band value for windowed sinc filtering.
#         * \param[in] pass_band value for the pass band.
#       inline void setPassBand (float pass_band)
#       /** \brief Get the pass band value. */
#       inline float getPassBand ()
#       /** \brief Turn on/off coordinate normalization. The positions can be translated and scaled such that they fit
#        * within a [-1, 1] prior to the smoothing computation. The default is off. The numerical stability of the
#        * solution can be improved by turning normalization on. If normalization is on, the coordinates will be rescaled
#        * to the original coordinate system after smoothing has completed.
#        * \param[in] normalize_coordinates decision whether to normalize coordinates or not
#       inline void setNormalizeCoordinates (bool normalize_coordinates)
#       /** \brief Get whether the coordinate normalization is active or not */
#       inline bool getNormalizeCoordinates ()
#       /** \brief Turn on/off smoothing along sharp interior edges.
#        * \param[in] status decision whether to enable/disable smoothing along sharp interior edges
#       inline void setFeatureEdgeSmoothing (bool feature_edge_smoothing)
#       /** \brief Get the status of the feature edge smoothing */
#       inline bool getFeatureEdgeSmoothing ()
#       /** \brief Specify the feature angle for sharp edge identification.
#        * \param[in] feature_angle the angle threshold for considering an edge to be sharp
#       inline void setFeatureAngle (float feature_angle)
#       /** \brief Get the angle threshold for considering an edge to be sharp */
#       inline float getFeatureAngle ()
#       /** \brief Specify the edge angle to control smoothing along edges (either interior or boundary).
#        * \param[in] edge_angle the angle to control smoothing along edges
#       inline void setEdgeAngle (float edge_angle)
#       /** \brief Get the edge angle to control smoothing along edges */
#       inline float getEdgeAngle ()
#       /** \brief Turn on/off the smoothing of vertices on the boundary of the mesh.
#        * \param[in] boundary_smoothing decision whether boundary smoothing is on or off
#       inline void setBoundarySmoothing (bool boundary_smoothing)
#       /** \brief Get the status of the boundary smoothing */
#       inline bool getBoundarySmoothing ()
#       protected:
#       void performProcessing (pcl::PolygonMesh &output);
###

# vtk_mesh_subdivision.h (1.6.0)
# pcl\surface\vtk_smoothing\vtk_mesh_subdivision.h (1.7.2)
# namespace pcl
# /** \brief PCL mesh smoothing based on the vtkLinearSubdivisionFilter, vtkLoopSubdivisionFilter, vtkButterflySubdivisionFilter
#   * depending on the selected MeshSubdivisionVTKFilterType algorithm from the VTK library.
#   * Please check out the original documentation for more details on the inner workings of the algorithm
#   * Warning: This wrapper does two fairly computationally expensive conversions from the PCL PolygonMesh
#   * data structure to the vtkPolyData data structure and back.
#   */
# class PCL_EXPORTS MeshSubdivisionVTK : public MeshProcessing
#       public:
#       /** \brief Empty constructor */
#       MeshSubdivisionVTK ();
#       enum MeshSubdivisionVTKFilterType
#       { LINEAR, LOOP, BUTTERFLY };
#       /** \brief Set the mesh subdivision filter type
#         * \param[in] type the filter type
#       inline void setFilterType (MeshSubdivisionVTKFilterType type)
#       /** \brief Get the mesh subdivision filter type */
#       inline MeshSubdivisionVTKFilterType getFilterType ()
#       protected:
#       void performProcessing (pcl::PolygonMesh &output);
###

# vtk_utils.h (1.6.0)
# pcl\surface\vtk_smoothing\vtk_utils.h (1.7.2)
# namespace pcl
# class PCL_EXPORTS VTKUtils
#       public:
#       /** \brief Convert a PCL PolygonMesh to a VTK vtkPolyData.
#         * \param[in] triangles PolygonMesh to be converted to vtkPolyData, stored in the object.
#         */
#       static int
#       convertToVTK (const pcl::PolygonMesh &triangles, vtkSmartPointer<vtkPolyData> &triangles_out_vtk);
#       /** \brief Convert the vtkPolyData object back to PolygonMesh.
#         * \param[out] triangles the PolygonMesh to store the vtkPolyData in.
#         */
#       static void
#       convertToPCL (vtkSmartPointer<vtkPolyData> &vtk_polygons, pcl::PolygonMesh &triangles);
#       /** \brief Convert vtkPolyData object to a PCL PolygonMesh
#         * \param[in] poly_data Pointer (vtkSmartPointer) to a vtkPolyData object
#         * \param[out] mesh PCL Polygon Mesh to fill
#         * \return Number of points in the point cloud of mesh.
#         */
#       static int
#       vtk2mesh (const vtkSmartPointer<vtkPolyData>& poly_data, pcl::PolygonMesh& mesh);
#       /** \brief Convert a PCL PolygonMesh to a vtkPolyData object
#         * \param[in] mesh Reference to PCL Polygon Mesh
#         * \param[out] poly_data Pointer (vtkSmartPointer) to a vtkPolyData object
#         * \return Number of points in the point cloud of mesh.
#         */
#       static int
#       mesh2vtk (const pcl::PolygonMesh& mesh, vtkSmartPointer<vtkPolyData> &poly_data);
###

###############################################################################
# Enum
###############################################################################

