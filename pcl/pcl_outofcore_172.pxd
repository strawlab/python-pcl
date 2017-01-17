from libc.stddef cimport size_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

# main
cimport pcl_defs as cpp

# boost
from boost_shared_ptr cimport shared_ptr


###############################################################################
# Types
###############################################################################

# # visualization
# # axis.h
# class Axes : public Object
# {
# public:
    # 
    # // Operators
    # Axes (std::string name, float size = 1.0) : Object (name)
    # 
    # // Accessors
    # inline vtkSmartPointer<vtkAxes> getAxes () const
    # 
    # vtkSmartPointer<vtkActor> getAxesActor () const


###

# # camera.h
# class Camera : public Object
# {
# public:
    # // Operators
    # Camera (std::string name);
    # Camera (std::string name, vtkSmartPointer<vtkCamera> camera);
    # 
    # public:
    # 
    # // Accessors
    # inline vtkSmartPointer<vtkCamera> getCamera () const
    # 
    # inline vtkSmartPointer<vtkCameraActor> getCameraActor () const
    # 
    # inline vtkSmartPointer<vtkActor> getHullActor () const
    # 
    # inline bool getDisplay () const
    # 
    # void setDisplay (bool display)
    # 
    # void getFrustum (double frustum[])
    # 
    # void setProjectionMatrix (const Eigen::Matrix4d &projection_matrix)
    # 
    # Eigen::Matrix4d getProjectionMatrix ()
    # 
    # void setModelViewMatrix (const Eigen::Matrix4d &model_view_matrix)
    # 
    # Eigen::Matrix4d getModelViewMatrix ()
    # 
    # Eigen::Matrix4d getViewProjectionMatrix ()
    # 
    # Eigen::Vector3d getPosition ()
    # 
    # inline void setClippingRange (float near_value = 0.0001f, float far_value = 100000.f)
    # 
    # virtual void render (vtkRenderer* renderer);
    # 
    # // Methods
    # void computeFrustum ();
    # void printFrustum ();


###

# common.h
###

# geometry.h
# class Geometry : public Object
# {
    # protected:
    # 
    # // Operators
    # // -----------------------------------------------------------------------------
    # Geometry (std::string name) : Object (name)
    # 
    # 
    # public:
    # virtual ~Geometry () { }
    # 
    # public:
    # // Accessors
    # virtual vtkSmartPointer<vtkActor> getActor () const


###

# grid.h
# //class Grid : public Geometry
# class Grid : public Object
# {
    # public:
    # 
    # // Operators
    # Grid (std::string name, int size = 10, double spacing = 1.0);
    # ~Grid () { }
    # 
    # // Accessors
    # inline vtkSmartPointer<vtkRectilinearGrid> getGrid () const
    # 
    # vtkSmartPointer<vtkActor> getGridActor () const


###


# # object.h
# class Object
# {
    # public:
    # 
    # // Operators
    # Object (std::string name);
    # 
    # virtual ~Object () { }
    # 
    # 
    # // Accessors
    # std::string getName () const;
    # 
    # void setName (std::string name);
    # 
    # virtual void render (vtkRenderer* renderer);
    # 
    # bool hasActor (vtkActor *actor);
    # 
    # void addActor (vtkActor *actor);
    # 
    # void removeActor (vtkActor *actor);
    # 
    # vtkSmartPointer<vtkActorCollection> getActors ();
    # 


###


# outofcore_cloud.h
# class OutofcoreCloud : public Object
# {
    # // Typedefs
    # // -----------------------------------------------------------------------------
    # typedef pcl::PointXYZ PointT;
    # //    typedef pcl::outofcore::OutofcoreOctreeBase<pcl::outofcore::OutofcoreOctreeDiskContainer<PointT>, PointT> octree_disk;
    # //    typedef pcl::outofcore::OutofcoreOctreeBaseNode<pcl::outofcore::OutofcoreOctreeDiskContainer<PointT>, PointT> octree_disk_node;
    # 
    # typedef pcl::outofcore::OutofcoreOctreeBase<> OctreeDisk;
    # typedef pcl::outofcore::OutofcoreOctreeBaseNode<> OctreeDiskNode;
    # //    typedef pcl::outofcore::OutofcoreBreadthFirstIterator<> OctreeBreadthFirstIterator;
    # 
    # typedef boost::shared_ptr<OctreeDisk> OctreeDiskPtr;
    # typedef Eigen::aligned_allocator<PointT> AlignedPointT;
    # 
    # typedef std::map<std::string, vtkSmartPointer<vtkActor> > CloudActorMap;
    # 
    # public:
    # 
    # //    typedef std::map<std::string, vtkSmartPointer<vtkPolyData> > CloudDataCache;
    # //    typedef std::map<std::string, vtkSmartPointer<vtkPolyData> >::iterator CloudDataCacheIterator;
    # 
    # static boost::shared_ptr<boost::thread> pcd_reader_thread;
    # //static MonitorQueue<std::string> pcd_queue;
    # 
    # struct PcdQueueItem
    # {
        # PcdQueueItem (std::string pcd_file, float coverage)
        # 
        # bool operator< (const PcdQueueItem& rhs) const
        # 
        # std::string pcd_file;
        # float coverage;
    # };
    # 
    # typedef std::priority_queue<PcdQueueItem> PcdQueue;
    # static PcdQueue pcd_queue;
    # static boost::mutex pcd_queue_mutex;
    # static boost::condition pcd_queue_ready;
    # 
    # 
    # 
    # class CloudDataCacheItem : public LRUCacheItem< vtkSmartPointer<vtkPolyData> >
    # {
        # public:
        # 
        # CloudDataCacheItem (std::string pcd_file, float coverage, vtkSmartPointer<vtkPolyData> cloud_data, size_t timestamp)
        # 
        # virtual size_t sizeOf() const
        # 
        # std::string pcd_file;
        # float coverage;
    # };
    # 
    # 
    # //    static CloudDataCache cloud_data_map;
    # //    static boost::mutex cloud_data_map_mutex;
    # typedef LRUCache<std::string, CloudDataCacheItem> CloudDataCache;
    # static CloudDataCache cloud_data_cache;
    # static boost::mutex cloud_data_cache_mutex;
    # 
    # static void pcdReaderThread();
    # 
    # // Operators
    # OutofcoreCloud (std::string name, boost::filesystem::path& tree_root);
    # 
    # // Methods
    # void updateVoxelData ();
    # 
    # // Accessors
    # OctreeDiskPtr getOctree ()
    # 
    # inline vtkSmartPointer<vtkActor> getVoxelActor () const
    # 
    # inline vtkSmartPointer<vtkActorCollection> getCloudActors () const
    # 
    # 
    # void setDisplayDepth (int displayDepth)
    # 
    # int getDisplayDepth ()
    # 
    # uint64_t getPointsLoaded ()
    # 
    # uint64_t getDataLoaded ()
    # 
    # Eigen::Vector3d getBoundingBoxMin ()
    # 
    # Eigen::Vector3d getBoundingBoxMax ()
    # 
    # void setDisplayVoxels (bool display_voxels)
    # bool getDisplayVoxels()
    # void setRenderCamera(Camera *render_camera)
    # int getLodPixelThreshold ()
    # 
    # void setLodPixelThreshold (int lod_pixel_threshold)
    # void increaseLodPixelThreshold ()
    # void decreaseLodPixelThreshold ()
    # 
    # virtual void render (vtkRenderer* renderer);


###


# scene.h
# class Scene
# {
    # public:
    # 
    # // Singleton
    # static Scene* instance ()
    # 
    # // Accessors - Cameras
    # void addCamera (Camera *camera);
    # 
    # std::vector<Camera*> getCameras ();
    # 
    # Camera* getCamera (vtkCamera *camera);
    # 
    # Camera* getCamera (std::string name);
    # 
    # // Accessors - Objects
    # void addObject (Object *object);
    # 
    # Object* getObjectByName (std::string name);
    # 
    # std::vector<Object*> getObjects ();
    # 
    # // Accessors - Viewports
    # 
    # void addViewport (Viewport *viewport);
    # 
    # std::vector<Viewport*> getViewports ();
    # 
    # 
    # void lock ()
    # void unlock ()


###

# viewport.h
# class Viewport
# {
    # public:
    # 
    # // Operators
    # Viewport (vtkSmartPointer<vtkRenderWindow> window, double xmin = 0.0, double ymin = 0.0, double xmax = 1.0, double ymax = 1.0);
    # 
    # // Accessors
    # inline vtkSmartPointer<vtkRenderer> getRenderer () const
    # 
    # void setCamera (Camera* camera)


###


# boost.h
###


# cJSON.h
# /* cJSON Types: */
# #define cJSON_False 0
# #define cJSON_True 1
# #define cJSON_NULL 2
# #define cJSON_Number 3
# #define cJSON_String 4
# #define cJSON_Array 5
# #define cJSON_Object 6
#   
# #define cJSON_IsReference 256
# 
# /* The cJSON structure: */
# typedef struct cJSON {
#   struct cJSON *next,*prev;   /* next/prev allow you to walk array/object chains. Alternatively, use GetArraySize/GetArrayItem/GetObjectItem */
#   struct cJSON *child;        /* An array or object item will have a child pointer pointing to a chain of the items in the array/object. */
# 
#   int type;                   /* The type of the item, as above. */
# 
#   char *valuestring;          /* The item's string, if type==cJSON_String */
#   int valueint;               /* The item's number, if type==cJSON_Number */
#   double valuedouble;         /* The item's number, if type==cJSON_Number */
# 
#   char *string;               /* The item's name string, if this item is the child of, or is in the list of subitems of an object. */
# } cJSON;
# 
# typedef struct cJSON_Hooks {
#       void *(*malloc_fn)(size_t sz);
#       void (*free_fn)(void *ptr);
# } cJSON_Hooks;
# 
# /* Supply malloc, realloc and free functions to cJSON */
# PCLAPI(void) cJSON_InitHooks(cJSON_Hooks* hooks);
# 
# 
# /* Supply a block of JSON, and this returns a cJSON object you can interrogate. Call cJSON_Delete when finished. */
# PCLAPI(cJSON *) cJSON_Parse(const char *value);
# /* Render a cJSON entity to text for transfer/storage. Free the char* when finished. */
# PCLAPI(char  *) cJSON_Print(cJSON *item);
# /* Render a cJSON entity to text for transfer/storage without any formatting. Free the char* when finished. */
# PCLAPI(char  *) cJSON_PrintUnformatted(cJSON *item);
# /* Delete a cJSON entity and all subentities. */
# PCLAPI(void)   cJSON_Delete(cJSON *c);
# /* Render a cJSON entity to text for transfer/storage. */
# PCLAPI(void) cJSON_PrintStr(cJSON *item, std::string& s);
# /* Render a cJSON entity to text for transfer/storage without any formatting. */
# PCLAPI(void) cJSON_PrintUnformattedStr(cJSON *item, std::string& s);
# 
# /* Returns the number of items in an array (or object). */
# PCLAPI(int)     cJSON_GetArraySize(cJSON *array);
# /* Retrieve item number "item" from array "array". Returns NULL if unsuccessful. */
# PCLAPI(cJSON *) cJSON_GetArrayItem(cJSON *array,int item);
# /* Get item "string" from object. Case insensitive. */
# PCLAPI(cJSON *) cJSON_GetObjectItem(cJSON *object,const char *string);
# 
# /* For analysing failed parses. This returns a pointer to the parse error. You'll probably need to look a few chars back to make sense of it. Defined when cJSON_Parse() returns 0. 0 when cJSON_Parse() succeeds. */
# PCLAPI(const char *) cJSON_GetErrorPtr();
#   
# /* These calls create a cJSON item of the appropriate type. */
# PCLAPI(cJSON *) cJSON_CreateNull();
# PCLAPI(cJSON *) cJSON_CreateTrue();
# PCLAPI(cJSON *) cJSON_CreateFalse();
# PCLAPI(cJSON *) cJSON_CreateBool(int b);
# PCLAPI(cJSON *) cJSON_CreateNumber(double num);
# PCLAPI(cJSON *) cJSON_CreateString(const char *string);
# PCLAPI(cJSON *) cJSON_CreateArray();
# PCLAPI(cJSON *) cJSON_CreateObject();
# 
# /* These utilities create an Array of count items. */
# PCLAPI(cJSON *) cJSON_CreateIntArray(int *numbers,int count);
# PCLAPI(cJSON *) cJSON_CreateFloatArray(float *numbers,int count);
# PCLAPI(cJSON *) cJSON_CreateDoubleArray(double *numbers,int count);
# PCLAPI(cJSON *) cJSON_CreateStringArray(const char **strings,int count);
# 
# /* Append item to the specified array/object. */
# PCLAPI(void) cJSON_AddItemToArray(cJSON *array, cJSON *item);
# PCLAPI(void) cJSON_AddItemToObject(cJSON *object,const char *string,cJSON *item);
# /* Append reference to item to the specified array/object. Use this when you want to add an existing cJSON to a new cJSON, but don't want to corrupt your existing cJSON. */
# PCLAPI(void) cJSON_AddItemReferenceToArray(cJSON *array, cJSON *item);
# PCLAPI(void) cJSON_AddItemReferenceToObject(cJSON *object,const char *string,cJSON *item);
# 
# /* Remove/Detatch items from Arrays/Objects. */
# PCLAPI(cJSON *) cJSON_DetachItemFromArray(cJSON *array,int which);
# PCLAPI(void)    cJSON_DeleteItemFromArray(cJSON *array,int which);
# PCLAPI(cJSON *) cJSON_DetachItemFromObject(cJSON *object,const char *string);
# PCLAPI(void)    cJSON_DeleteItemFromObject(cJSON *object,const char *string);
#   
# /* Update array items. */
# PCLAPI(void) cJSON_ReplaceItemInArray(cJSON *array,int which,cJSON *newitem);
# PCLAPI(void) cJSON_ReplaceItemInObject(cJSON *object,const char *string,cJSON *newitem);
# 
# #define cJSON_AddNullToObject(object,name)    cJSON_AddItemToObject(object, name, cJSON_CreateNull())
# #define cJSON_AddTrueToObject(object,name)    cJSON_AddItemToObject(object, name, cJSON_CreateTrue())
# #define cJSON_AddFalseToObject(object,name)       cJSON_AddItemToObject(object, name, cJSON_CreateFalse())
# #define cJSON_AddNumberToObject(object,name,n)    cJSON_AddItemToObject(object, name, cJSON_CreateNumber(n))
# #define cJSON_AddStringToObject(object,name,s)    cJSON_AddItemToObject(object, name, cJSON_CreateString(s))


###

# metadata.h
# namespace pcl
# namespace outofcore
#     
# /** \class AbstractMetadata
#  *
#  *  \brief Abstract interface for outofcore metadata file types
#  *
#  *  \ingroup outofcore
#  *  \author Stephen Fox (foxstephend@gmail.com)
#  */
# class PCL_EXPORTS OutofcoreAbstractMetadata
# {
    # public:
    # /** \brief Empty constructor */
    # OutofcoreAbstractMetadata ()
    # 
    # virtual ~OutofcoreAbstractMetadata ()
    # 
    # /** \brief Write the metadata in the on-disk format, e.g. JSON. */
    # virtual void serializeMetadataToDisk () = 0;
    # 
    # /** \brief Method which should read and parse metadata and store
    #  *  it in variables that have public getters and setters*/
    # virtual int loadMetadataFromDisk (const boost::filesystem::path& path_to_metadata) = 0;
    # 
    # /** \brief Should write the same ascii metadata that is saved on
    #  *   disk, or a human readable format of the metadata in case a binary format is being used */
    # friend std::ostream& operator<<(std::ostream& os, const OutofcoreAbstractMetadata& metadata_arg);


###

# octree_abstract_node_container.h
# namespace pcl
# namespace outofcore
    # template<typename PointT>
    # class OutofcoreAbstractNodeContainer 
        # public:
        # typedef std::vector<PointT, Eigen::aligned_allocator<PointT> > AlignedPointTVector;
        # 
        # OutofcoreAbstractNodeContainer () : container_ ()
        # 
        # OutofcoreAbstractNodeContainer (const boost::filesystem::path&) {}
        # 
        # virtual ~OutofcoreAbstractNodeContainer () {}        
        # virtual void insertRange (const PointT* start, const uint64_t count)=0;
        # virtual void insertRange (const PointT* const* start, const uint64_t count)=0;
        # virtual void readRange (const uint64_t start, const uint64_t count, AlignedPointTVector& v)=0;
        # virtual void readRangeSubSample (const uint64_t start, const uint64_t count, const double percent, AlignedPointTVector& v) =0;
        # virtual bool empty () const=0;
        # virtual uint64_t size () const =0;
        # virtual void clear ()=0;
        # virtual void convertToXYZ (const boost::filesystem::path& path)=0;
        # virtual PointT operator[] (uint64_t idx) const=0;


###

# octree_base.h
# namespace pcl
# namespace outofcore
    # struct OutofcoreParams
    # {
    #   std::string node_index_basename_;
    #   std::string node_container_basename_;
    #   std::string node_index_extension_;
    #   std::string node_container_extension_;
    #   double sample_percent;
    # };
    # 
    # /** \class OutofcoreOctreeBase 
    #  *  \brief This code defines the octree used for point storage at Urban Robotics. 
    #  * 
    #  *  \note Code was adapted from the Urban Robotics out of core octree implementation. 
    #  *  Contact Jacob Schloss <jacob.schloss@urbanrobotics.net> with any questions. 
    #  *  http://www.urbanrobotics.net/. This code was integrated for the Urban Robotics 
    #  *  Code Sprint (URCS) by Stephen Fox (foxstephend@gmail.com). Additional development notes can be found at
    #  *  http://www.pointclouds.org/blog/urcs/.
    #  *
    #  *  The primary purpose of this class is an interface to the
    #  *  recursive traversal (recursion handled by \ref pcl::outofcore::OutofcoreOctreeBaseNode) of the
    #  *  in-memory/top-level octree structure. The metadata in each node
    #  *  can be loaded entirely into main memory, from which the tree can be traversed
    #  *  recursively in this state. This class provides an the interface
    #  *  for: 
    #  *               -# Point/Region insertion methods 
    #  *               -# Frustrum/box/region queries
    #  *               -# Parameterization of resolution, container type, etc...
    #  *
    #  *  For lower-level node access, there is a Depth-First iterator
    #  *  for traversing the trees with direct access to the nodes. This
    #  *  can be used for implementing other algorithms, and other
    #  *  iterators can be written in a similar fashion.
    #  *
    #  *  The format of the octree is stored on disk in a hierarchical
    #  *  octree structure, where .oct_idx are the JSON-based node
    #  *  metadata files managed by \ref pcl::outofcore::OutofcoreOctreeNodeMetadata,
    #  *  and .octree is the JSON-based octree metadata file managed by
    #  *  \ref pcl::outofcore::OutofcoreOctreeBaseMetadata. Children of each node live
    #  *  in up to eight subdirectories named from 0 to 7, where a
    #  *  metadata and optionally a pcd file will exist. The PCD files
    #  *  are stored in compressed binary PCD format, containing all of
    #  *  the fields existing in the PCLPointCloud2 objects originally
    #  *  inserted into the out of core object.
    #  *  
    #  *  A brief outline of the out of core octree can be seen
    #  *  below. The files in [brackets] exist only when the LOD are
    #  *  built.
    #  *
    #  *  At this point in time, there is not support for multiple trees
    #  *  existing in a single directory hierarchy.
    #  *
    #  *  \verbatim
    #  tree_name/
    #       tree_name.oct_idx
    #       tree_name.octree
    #       [tree_name-uuid.pcd]
    #       0/
    #            tree_name.oct_idx
    #            [tree_name-uuid.pcd]
    #            0/
    #               ...
    #            1/
    #                ...
    #                  ...
    #                      0/
    #                          tree_name.oct_idx
    #                          tree_name.pcd
    #       1/
    #       ...
    #       7/
    #  \endverbatim
    #  *
    #  *  \ingroup outofcore
    #  *  \author Jacob Schloss (jacob.schloss@urbanrobotics.net)
    #  *  \author Stephen Fox, Urban Robotics Code Sprint (foxstephend@gmail.com)
    #  *
    #  */
    # template<typename ContainerT = OutofcoreOctreeDiskContainer<pcl::PointXYZ>, typename PointT = pcl::PointXYZ>
    # class OutofcoreOctreeBase
    # {
        friend class OutofcoreOctreeBaseNode<ContainerT, PointT>;
        friend class pcl::outofcore::OutofcoreIteratorBase<PointT, ContainerT>;
        
        public:
        
        // public typedefs
        typedef OutofcoreOctreeBase<OutofcoreOctreeDiskContainer<PointT>, PointT > octree_disk;
        typedef OutofcoreOctreeBaseNode<OutofcoreOctreeDiskContainer<PointT>, PointT > octree_disk_node;
        
        typedef OutofcoreOctreeBase<OutofcoreOctreeRamContainer<PointT>, PointT> octree_ram;
        typedef OutofcoreOctreeBaseNode<OutofcoreOctreeRamContainer<PointT>, PointT> octree_ram_node;
        
        typedef OutofcoreOctreeBaseNode<ContainerT, PointT> OutofcoreNodeType;
        
        typedef OutofcoreOctreeBaseNode<ContainerT, PointT> BranchNode;
        typedef OutofcoreOctreeBaseNode<ContainerT, PointT> LeafNode;
        
        typedef OutofcoreDepthFirstIterator<PointT, ContainerT> Iterator;
        typedef const OutofcoreDepthFirstIterator<PointT, ContainerT> ConstIterator;
        
        typedef OutofcoreBreadthFirstIterator<PointT, ContainerT> BreadthFirstIterator;
        typedef const OutofcoreBreadthFirstIterator<PointT, ContainerT> BreadthFirstConstIterator;
        
        typedef OutofcoreDepthFirstIterator<PointT, ContainerT> DepthFirstIterator;
        typedef const OutofcoreDepthFirstIterator<PointT, ContainerT> DepthFirstConstIterator;
        
        typedef boost::shared_ptr<OutofcoreOctreeBase<ContainerT, PointT> > Ptr;
        typedef boost::shared_ptr<const OutofcoreOctreeBase<ContainerT, PointT> > ConstPtr;
        
        typedef pcl::PointCloud<PointT> PointCloud;
        
        typedef boost::shared_ptr<std::vector<int> > IndicesPtr;
        typedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;
        
        typedef boost::shared_ptr<PointCloud> PointCloudPtr;
        typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
        
        typedef std::vector<PointT, Eigen::aligned_allocator<PointT> > AlignedPointTVector;
        
        // Constructors
        // -----------------------------------------------------------------------
        /** \brief Load an existing tree
         *
         * If load_all is set, the BB and point count for every node is loaded,
         * otherwise only the root node is actually created, and the rest will be
         * generated on insertion or query.
         *
         * \param root_node_name Path to the top-level tree/tree.oct_idx metadata file
         * \param load_all Load entire tree metadata (does not load any points from disk)
         * \throws PCLException for bad extension (root node metadata must be .oct_idx extension)
         */
        OutofcoreOctreeBase (const boost::filesystem::path &root_node_name, const bool load_all);
        
        /** \brief Create a new tree
         *
         * Create a new tree rootname with specified bounding box; will remove and overwrite existing tree with the same name
         *
         * Computes the depth of the tree based on desired leaf , then calls the other constructor.
         *
         * \param min Bounding box min
         * \param max Bounding box max
         * \param resolution_arg Node dimension in meters (assuming your point data is in meters)
         * \param root_node_name must end in ".oct_idx" 
         * \param coord_sys Coordinate system which is stored in the JSON metadata
         * \throws PCLException if root file extension does not match \ref pcl::outofcore::OutofcoreOctreeBaseNode::node_index_extension
         */
        OutofcoreOctreeBase (const Eigen::Vector3d& min, const Eigen::Vector3d& max, const double resolution_arg, const boost::filesystem::path &root_node_name, const std::string &coord_sys);
        
        /** \brief Create a new tree; will not overwrite existing tree of same name
         *
         * Create a new tree rootname with specified bounding box; will not overwrite an existing tree
         *
         * \param max_depth Specifies a fixed number of LODs to generate, which is the depth of the tree
         * \param min Bounding box min
         * \param max Bounding box max
         * \note Bounding box of the tree must be set before inserting any points. The tree \b cannot be resized at this time.
         * \param root_node_name must end in ".oct_idx" 
         * \param coord_sys Coordinate system which is stored in the JSON metadata
         * \throws PCLException if the parent directory has existing children (detects an existing tree)
         * \throws PCLException if file extension is not ".oct_idx"
         */
        OutofcoreOctreeBase (const boost::uint64_t max_depth, const Eigen::Vector3d &min, const Eigen::Vector3d &max, const boost::filesystem::path &root_node_name, const std::string &coord_sys);
        
        virtual ~OutofcoreOctreeBase ();
        
        // Point/Region INSERTION methods
        // --------------------------------------------------------------------------------
        /** \brief Recursively add points to the tree 
         *  \note shared read_write_mutex lock occurs
         */
        boost::uint64_t addDataToLeaf (const AlignedPointTVector &p);
        
        /** \brief Copies the points from the point_cloud falling within the bounding box of the octree to the
         *   out-of-core octree; this is an interface to addDataToLeaf and can be used multiple times.
         *  \param point_cloud Pointer to the point cloud data to copy to the outofcore octree; Assumes templated
         *   PointT matches for each.
         *  \return Number of points successfully copied from the point cloud to the octree.
         */
        boost::uint64_t addPointCloud (PointCloudConstPtr point_cloud);
        
        /** \brief Recursively copies points from input_cloud into the leaf nodes of the out-of-core octree, and stores them to disk.
         *
         * \param[in] input_cloud The cloud of points to be inserted into the out-of-core octree. Note if multiple PCLPointCloud2 objects are added to the tree, this assumes that they all have exactly the same fields.
         * \param[in] skip_bb_check (default=false) whether to skip the bounding box check on insertion. Note the bounding box check is never skipped in the current implementation.
         * \return Number of poitns successfully copied from the point cloud to the octree
         */
        boost::uint64_t addPointCloud (pcl::PCLPointCloud2::Ptr &input_cloud, const bool skip_bb_check = false);
        
        /** \brief Recursively add points to the tree. 
         *
         * Recursively add points to the tree. 1/8 of the remaining
         * points at each LOD are stored at each internal node of the
         * octree until either (a) runs out of points, in which case
         * the leaf is not at the maximum depth of the tree, or (b)
         * a larger set of points falls in the leaf at the maximum depth.
         * Note unlike the old implementation, multiple
         * copies of the same point will \b not be added at multiple
         * LODs as it walks the tree. Once the point is added to the
         * octree, it is no longer propagated further down the tree.
         *
         *\param[in] input_cloud The input cloud of points which will
         * be copied into the sorted nodes of the out-of-core octree
         * \return The total number of points added to the out-of-core
         * octree.
         */
        boost::uint64_t addPointCloud_and_genLOD (pcl::PCLPointCloud2::Ptr &input_cloud);
        
        boost::uint64_t addPointCloud (pcl::PCLPointCloud2::Ptr &input_cloud);
        
        boost::uint64_t addPointCloud_and_genLOD (PointCloudConstPtr point_cloud);
        
        /** \brief Recursively add points to the tree subsampling LODs on the way.
         * shared read_write_mutex lock occurs
         */
        boost::uint64_t addDataToLeaf_and_genLOD (AlignedPointTVector &p);
        
        // Frustrum/Box/Region REQUESTS/QUERIES: DB Accessors
        // -----------------------------------------------------------------------
        void queryFrustum (const double *planes, std::list<std::string>& file_names) const;
        
        void queryFrustum (const double *planes, std::list<std::string>& file_names, const boost::uint32_t query_depth) const;
        
        void queryFrustum (const double *planes, const Eigen::Vector3d &eye, const Eigen::Matrix4d &view_projection_matrix,
                            std::list<std::string>& file_names, const boost::uint32_t query_depth) const;
        
        // templated PointT methods
        
        /** \brief Get a list of file paths at query_depth that intersect with your bounding box specified by \c min and \c max.
         *  When querying with this method, you may be stuck with extra data (some outside of your query bounds) that reside in the files.
         *
         * \param[in] min The minimum corner of the bounding box
         * \param[in] max The maximum corner of the bounding box
         * \param[in] query_depth 0 is root, (this->depth) is full
         * \param[out] bin_name List of paths to point data files (PCD currently) which satisfy the query
         */
        void queryBBIntersects (const Eigen::Vector3d &min, const Eigen::Vector3d &max, const boost::uint32_t query_depth, std::list<std::string> &bin_name) const;
        
        /** \brief Get Points in BB, only points inside BB. The query
         * processes the data at each node, filtering points that fall
         * out of the query bounds, and returns a single, concatenated
         * point cloud.
         *
         * \param[in] min The minimum corner of the bounding box for querying
         * \param[in] max The maximum corner of the bounding box for querying
         * \param[in] query_depth The depth from which point data will be taken
         *   \note If the LODs of the tree have not been built, you must specify the maximum depth in order to retrieve any data
         * \param[out] dst The destination vector of points
         */
        void queryBBIncludes (const Eigen::Vector3d &min, const Eigen::Vector3d &max, const boost::uint64_t query_depth, AlignedPointTVector &dst) const;
        
        /** \brief Query all points falling within the input bounding box at \c query_depth and return a PCLPointCloud2 object in \c dst_blob.
         *
         * \param[in] min The minimum corner of the input bounding box.
         * \param[in] max The maximum corner of the input bounding box.
         * \param[in] query_depth The query depth at which to search for points; only points at this depth are returned
         * \param[out] dst_blob Storage location for the points satisfying the query.
         **/
        void queryBBIncludes (const Eigen::Vector3d &min, const Eigen::Vector3d &max, const boost::uint64_t query_depth, const pcl::PCLPointCloud2::Ptr &dst_blob) const;
        
        /** \brief Returns a random subsample of points within the given bounding box at \c query_depth.
         *
         * \param[in] min The minimum corner of the boudning box to query.
         * \param[out] max The maximum corner of the bounding box to query.
         * \param[in] query_depth The depth in the tree at which to look for the points. Only returns points within the given bounding box at the specified \c query_depth.
         * \param[out] dst The destination in which to return the points.
         * 
         */
        void queryBBIncludes_subsample (const Eigen::Vector3d &min, const Eigen::Vector3d &max, uint64_t query_depth, const double percent, AlignedPointTVector &dst) const;
        
        // PCLPointCloud2 methods
        
        /** \brief Query all points falling within the input bounding box at \c query_depth and return a PCLPointCloud2 object in \c dst_blob.
         *   If the optional argument for filter is given, points are processed by that filter before returning.
         *  \param[in] min The minimum corner of the input bounding box.
         *  \param[in] max The maximum corner of the input bounding box.
         *  \param[in] query_depth The depth of tree at which to query; only points at this depth are returned
         *  \param[out] dst_blob The destination in which points within the bounding box are stored.
         *  \param[in] percent optional sampling percentage which is applied after each time data are read from disk
         */
        virtual void queryBoundingBox (const Eigen::Vector3d &min, const Eigen::Vector3d &max, const int query_depth, const pcl::PCLPointCloud2::Ptr &dst_blob, double percent = 1.0);
        
        /** \brief Returns list of pcd files from nodes whose bounding boxes intersect with the input bounding box.
         * \param[in] min The minimum corner of the input bounding box.
         * \param[in] max The maximum corner of the input bounding box.
         * \param query_depth
         * \param[out] filenames The list of paths to the PCD files which can be loaded and processed.
         */
        inline virtual void queryBoundingBox (const Eigen::Vector3d &min, const Eigen::Vector3d &max, const int query_depth, std::list<std::string> &filenames) const
        
        // Parameterization: getters and setters
        
        /** \brief Get the overall bounding box of the outofcore
         *  octree; this is the same as the bounding box of the \c root_node_ node
         *  \param min
         *  \param max
         */
        bool getBoundingBox (Eigen::Vector3d &min, Eigen::Vector3d &max) const;
        
        /** \brief Get number of points at specified LOD 
         * \param[in] depth_index the level of detail at which we want the number of points (0 is root, 1, 2,...)
         * \return number of points in the tree at \b depth
         */
        inline boost::uint64_t getNumPointsAtDepth (const boost::uint64_t& depth_index) const
        
        /** \brief Queries the number of points in a bounding box 
         *  \param[in] min The minimum corner of the input bounding box
         *  \param[out] max The maximum corner of the input bounding box
         *  \param[in] query_depth The depth of the nodes to restrict the search to (only this depth is searched)
         *  \param[in] load_from_disk (default true) Whether to load PCD files to count exactly the number of points within the bounding box; setting this to false will return an upper bound by just reading the number of points from the PCD header, even if there may be some points in that node do not fall within the query bounding box.
         *  \return Number of points in the bounding box at depth \b query_depth
         **/
        boost::uint64_t queryBoundingBoxNumPoints (const Eigen::Vector3d& min, const Eigen::Vector3d& max, const int query_depth, bool load_from_disk = true);
        
        /** \brief Get number of points at each LOD 
         * \return vector of number of points in each LOD indexed by each level of depth, 0 to the depth of the tree.
         */
        inline const std::vector<boost::uint64_t>& getNumPointsVector () const
        
        /** \brief Get number of LODs, which is the height of the tree
         */
        inline boost::uint64_t getDepth () const
        inline boost::uint64_t getTreeDepth () const
        /** \brief Computes the expected voxel dimensions at the leaves 
         */
        bool getBinDimension (double &x, double &y) const;
        
        /** \brief gets the side length of an (assumed) perfect cubic voxel.
         *  \note If the initial bounding box specified in constructing the octree is not square, then this method does not return a sensible value 
         *  \return the side length of the cubic voxel size at the specified depth
         */
        double getVoxelSideLength (const boost::uint64_t& depth) const;
        
        /** \brief Gets the smallest (assumed) cubic voxel side lengths. The smallest voxels are located at the max depth of the tree.
         * \return The side length of a the cubic voxel located at the leaves
         */
        double getVoxelSideLength () const
        
        /** \brief Get coordinate system tag from the JSON metadata file
         */
        const std::string& getCoordSystem () const
        
        // Mutators
        
        /** \brief Generate multi-resolution LODs for the tree, which are a uniform random sampling all child leafs below the node.
         */
        void buildLOD ();
        
        /** \brief Prints size of BBox to stdout
         */ 
        void printBoundingBox (const size_t query_depth) const;
        
        /** \brief Prints the coordinates of the bounding box of the node to stdout */
        void printBoundingBox (OutofcoreNodeType& node) const;
        
        /** \brief Prints size of the bounding boxes to stdou
         */
        inline void printBoundingBox() const
        
        /** \brief Returns the voxel centers of all existing voxels at \c query_depth
            \param[out] voxel_centers Vector of PointXYZ voxel centers for nodes that exist at that depth
            \param[in] query_depth the depth of the tree at which to retrieve occupied/existing voxels
        */
        void getOccupiedVoxelCenters(AlignedPointTVector &voxel_centers, size_t query_depth) const;
        
        /** \brief Returns the voxel centers of all existing voxels at \c query_depth
            \param[out] voxel_centers Vector of PointXYZ voxel centers for nodes that exist at that depth
            \param[in] query_depth the depth of the tree at which to retrieve occupied/existing voxels
        */
        void getOccupiedVoxelCenters(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &voxel_centers, size_t query_depth) const;
        
        /** \brief Gets the voxel centers of all occupied/existing leaves of the tree */
        void getOccupiedVoxelCenters(AlignedPointTVector &voxel_centers) const
        
        /** \brief Returns the voxel centers of all occupied/existing leaves of the tree 
         *  \param[out] voxel_centers std::vector of the centers of all occupied leaves of the octree
         */
        void getOccupiedVoxelCenters(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &voxel_centers) const
        
        // Serializers
        
        /** \brief Save each .bin file as an XYZ file */
        void convertToXYZ ();
        
        /** \brief Write a python script using the vpython module containing all
         * the bounding boxes */
        void writeVPythonVisual (const boost::filesystem::path filename);
        
        OutofcoreNodeType* getBranchChildPtr (const BranchNode& branch_arg, unsigned char childIdx_arg) const;
        
        pcl::Filter<pcl::PCLPointCloud2>::Ptr getLODFilter ();
        const pcl::Filter<pcl::PCLPointCloud2>::ConstPtr getLODFilter () const;
        
        /** \brief Sets the filter to use when building the levels of depth. Recommended filters are pcl::RandomSample<pcl::PCLPointCloud2> or pcl::VoxelGrid */
        void setLODFilter (const pcl::Filter<pcl::PCLPointCloud2>::Ptr& filter_arg);
        
        /** \brief Returns the sample_percent_ used when constructing the LOD. */
        double getSamplePercent () const
        
        /** \brief Sets the sampling percent for constructing LODs. Each LOD gets sample_percent^d points. 
         * \param[in] sample_percent_arg Percentage between 0 and 1. */
        inline void setSamplePercent (const double sample_percent_arg)


###

# octree_base_node.h
# namespace pcl
# namespace outofcore
    # // Forward Declarations
    # template<typename ContainerT, typename PointT>
    # class OutofcoreOctreeBaseNode;
    # 
    # template<typename ContainerT, typename PointT>
    # class OutofcoreOctreeBase;
    # 
    # /** \brief Non-class function which creates a single child leaf; used with \ref queryBBIntersects_noload to avoid loading the data from disk */
    # template<typename ContainerT, typename PointT> OutofcoreOctreeBaseNode<ContainerT, PointT>*
    # makenode_norec (const boost::filesystem::path &path, OutofcoreOctreeBaseNode<ContainerT, PointT>* super);
    # 
    # /** \brief Non-class method which performs a bounding box query without loading any of the point cloud data from disk */
    # template<typename ContainerT, typename PointT> void
    # queryBBIntersects_noload (const boost::filesystem::path &root_node, const Eigen::Vector3d &min, const Eigen::Vector3d &max, const boost::uint32_t query_depth, std::list<std::string> &bin_name);
    # 
    # /** \brief Non-class method overload */
    # template<typename ContainerT, typename PointT> void
    # queryBBIntersects_noload (OutofcoreOctreeBaseNode<ContainerT, PointT>* current, const Eigen::Vector3d&, const Eigen::Vector3d &max, const boost::uint32_t query_depth, std::list<std::string> &bin_name);
    # 
    # /** \class OutofcoreOctreeBaseNode 
    #  *
    #  *  \note Code was adapted from the Urban Robotics out of core octree implementation. 
    #  *  Contact Jacob Schloss <jacob.schloss@urbanrobotics.net> with any questions. 
    #  *  http://www.urbanrobotics.net/
    #  *
    #  *  \brief OutofcoreOctreeBaseNode Class internally representing nodes of an
    #  *  outofcore octree, with accessors to its data via the \ref
    #  *  pcl::outofcore::OutofcoreOctreeDiskContainer class or \ref pcl::outofcore::OutofcoreOctreeRamContainer class,
    #  *  whichever it is templated against.  
    #  * 
    #  *  \ingroup outofcore
    #  *  \author Jacob Schloss (jacob.schloss@urbanrobotics.net)
    #  *
    #  */
    # template<typename ContainerT = OutofcoreOctreeDiskContainer<pcl::PointXYZ>, typename PointT = pcl::PointXYZ>
    # class OutofcoreOctreeBaseNode : public pcl::octree::OctreeNode
    # {
        # friend class OutofcoreOctreeBase<ContainerT, PointT> ;
        # 
        # // these methods can be rewritten with the iterators. 
        # friend OutofcoreOctreeBaseNode<ContainerT, PointT>*
        # makenode_norec<ContainerT, PointT> (const boost::filesystem::path &path, OutofcoreOctreeBaseNode<ContainerT, PointT>* super);
        # 
        # friend void queryBBIntersects_noload<ContainerT, PointT> (const boost::filesystem::path &rootnode, const Eigen::Vector3d &min, const Eigen::Vector3d &max, const boost::uint32_t query_depth, std::list<std::string> &bin_name);
        # 
        # friend void queryBBIntersects_noload<ContainerT, PointT> (OutofcoreOctreeBaseNode<ContainerT, PointT>* current, const Eigen::Vector3d &min, const Eigen::Vector3d &max, const boost::uint32_t query_depth, std::list<std::string> &bin_name);
        # 
        # public:
        # typedef OutofcoreOctreeBase<OutofcoreOctreeDiskContainer < PointT > , PointT > octree_disk;
        # typedef OutofcoreOctreeBaseNode<OutofcoreOctreeDiskContainer < PointT > , PointT > octree_disk_node;
        # typedef std::vector<PointT, Eigen::aligned_allocator<PointT> > AlignedPointTVector;
        # typedef pcl::octree::node_type_t node_type_t;
        # 
        # const static std::string node_index_basename;
        # const static std::string node_container_basename;
        # const static std::string node_index_extension;
        # const static std::string node_container_extension;
        # const static double sample_percent_;
        # 
        # /** \brief Empty constructor; sets pointers for children and for bounding boxes to 0
        #  */
        # OutofcoreOctreeBaseNode ();
        # 
        # /** \brief Create root node and directory */
        # OutofcoreOctreeBaseNode (const Eigen::Vector3d &bb_min, const Eigen::Vector3d &bb_max, OutofcoreOctreeBase<ContainerT, PointT> * const tree, const boost::filesystem::path &root_name);
        # 
        # /** \brief Will recursively delete all children calling recFreeChildrein */
        # virtual ~OutofcoreOctreeBaseNode ();
        # 
        # //query
        # /** \brief gets the minimum and maximum corner of the bounding box represented by this node
        #  * \param[out] min_bb returns the minimum corner of the bounding box indexed by 0-->X, 1-->Y, 2-->Z 
        #  * \param[out] max_bb returns the maximum corner of the bounding box indexed by 0-->X, 1-->Y, 2-->Z 
        #  */
        # virtual inline void getBoundingBox (Eigen::Vector3d &min_bb, Eigen::Vector3d &max_bb) const
        # 
        # const boost::filesystem::path& getPCDFilename () const
        # const boost::filesystem::path& getMetadataFilename () const
        # 
        # void queryFrustum (const double planes[24], std::list<std::string>& file_names);
        # void queryFrustum (const double planes[24], std::list<std::string>& file_names, const boost::uint32_t query_depth, const bool skip_vfc_check = false);
        # void queryFrustum (const double planes[24], const Eigen::Vector3d &eye, const Eigen::Matrix4d &view_projection_matrix, std::list<std::string>& file_names, const boost::uint32_t query_depth, const bool skip_vfc_check = false);
        # 
        # //point extraction
        # /** \brief Recursively add points that fall into the queried bounding box up to the \b query_depth 
        #  *
        #  *  \param[in] min_bb the minimum corner of the bounding box, indexed by X,Y,Z coordinates
        #  *  \param[in] max_bb the maximum corner of the bounding box, indexed by X,Y,Z coordinates
        #  *  \param[in] query_depth the maximum depth to query in the octree for points within the bounding box
        #  *  \param[out] dst destion of points returned by the queries
        #  */
        # virtual void queryBBIncludes (const Eigen::Vector3d &min_bb, const Eigen::Vector3d &max_bb, size_t query_depth, AlignedPointTVector &dst);
        # 
        # /** \brief Recursively add points that fall into the queried bounding box up to the \b query_depth
        #  *
        #  *  \param[in] min_bb the minimum corner of the bounding box, indexed by X,Y,Z coordinates
        #  *  \param[in] max_bb the maximum corner of the bounding box, indexed by X,Y,Z coordinates
        #  *  \param[in] query_depth the maximum depth to query in the octree for points within the bounding box
        #  *  \param[out] dst_blob destion of points returned by the queries
        #  */
        # virtual void queryBBIncludes (const Eigen::Vector3d &min_bb, const Eigen::Vector3d &max_bb, size_t query_depth, const pcl::PCLPointCloud2::Ptr &dst_blob);
        # 
        # /** \brief Recursively add points that fall into the queried bounding box up to the \b query_depth 
        #  *
        #  *  \param[in] min_bb the minimum corner of the bounding box, indexed by X,Y,Z coordinates
        #  *  \param[in] max_bb the maximum corner of the bounding box, indexed by X,Y,Z coordinates
        #  *  \param[in] query_depth
        #  *  \param percent
        #  *  \param[out] v std::list of points returned by the query
        #  */
        # virtual void queryBBIncludes_subsample (const Eigen::Vector3d &min_bb, const Eigen::Vector3d &max_bb, boost::uint64_t query_depth, const double percent, AlignedPointTVector &v);
        # virtual void queryBBIncludes_subsample (const Eigen::Vector3d &min_bb, const Eigen::Vector3d &max_bb, boost::uint64_t query_depth, const pcl::PCLPointCloud2::Ptr& dst_blob, double percent = 1.0);
        # 
        # /** \brief Recursive acquires PCD paths to any node with which the queried bounding box intersects (at query_depth only).
        #  */
        # virtual void queryBBIntersects (const Eigen::Vector3d &min_bb, const Eigen::Vector3d &max_bb, const boost::uint32_t query_depth, std::list<std::string> &file_names);
        # 
        # /** \brief Write the voxel size to stdout at \c query_depth 
        #  * \param[in] query_depth The depth at which to print the size of the voxel/bounding boxes
        #  */
        # virtual void printBoundingBox (const size_t query_depth) const;
        # 
        # /** \brief add point to this node if we are a leaf, or find the leaf below us that is supposed to take the point 
        #  *  \param[in] p vector of points to add to the leaf
        #  *  \param[in] skip_bb_check whether to check if the point's coordinates fall within the bounding box
        #  */
        # virtual boost::uint64_t addDataToLeaf (const AlignedPointTVector &p, const bool skip_bb_check = true);
        # 
        # virtual boost::uint64_t addDataToLeaf (const std::vector<const PointT*> &p, const bool skip_bb_check = true);
        # 
        # /** \brief Add a single PCLPointCloud2 object into the octree.
        #  * \param[in] input_cloud
        #  * \param[in] skip_bb_check (default = false)
        #  */
        # virtual boost::uint64_t addPointCloud (const pcl::PCLPointCloud2::Ptr &input_cloud, const bool skip_bb_check = false);
        # 
        # /** \brief Add a single PCLPointCloud2 into the octree and build the subsampled LOD during construction; this method of LOD construction is <b>not</b> multiresolution. Rather, there are no redundant data. */
        # virtual boost::uint64_t addPointCloud_and_genLOD (const pcl::PCLPointCloud2::Ptr input_cloud); //, const bool skip_bb_check);
        # 
        # /** \brief Recursively add points to the leaf and children subsampling LODs
        #  * on the way down.
        #  * \note rng_mutex_ lock occurs
        #  */
        # virtual boost::uint64_t addDataToLeaf_and_genLOD (const AlignedPointTVector &p, const bool skip_bb_check);
        # 
        # /** \brief Write a python visual script to @b file
        #  * \param[in] file output file stream to write the python visual script
        #  */
        # void writeVPythonVisual (std::ofstream &file);
        # 
        # virtual int read (pcl::PCLPointCloud2::Ptr &output_cloud);
        # virtual inline node_type_t getNodeType () const
        # 
        # virtual OutofcoreOctreeBaseNode*  deepCopy () const
        # 
        # virtual inline size_t getDepth () const
        # 
        # /** \brief Returns the total number of children on disk */
        # virtual size_t getNumChildren () const 
        # 
        # /** \brief Count loaded chilren */
        # virtual size_t getNumLoadedChildren ()  const
        # 
        # /** \brief Returns a pointer to the child in octant index_arg */
        # virtual OutofcoreOctreeBaseNode* getChildPtr (size_t index_arg) const;
        # 
        # /** \brief Gets the number of points available in the PCD file */
        # virtual boost::uint64_t getDataSize () const;
        # 
        # inline virtual void clearData ()


###

# octree_disk_container.h
# namespace pcl
# namespace outofcore
    # /** \class OutofcoreOctreeDiskContainer
    # *  \note Code was adapted from the Urban Robotics out of core octree implementation. 
    # *  Contact Jacob Schloss <jacob.schloss@urbanrobotics.net> with any questions. 
    # *  http://www.urbanrobotics.net/
    # *
    # *  \brief Class responsible for serialization and deserialization of out of core point data
    # *  \ingroup outofcore
    # *  \author Jacob Schloss (jacob.schloss@urbanrobotics.net)
    # */
    # template<typename PointT = pcl::PointXYZ>
    # class OutofcoreOctreeDiskContainer : public OutofcoreAbstractNodeContainer<PointT>
    # {
        # public:
        # typedef typename OutofcoreAbstractNodeContainer<PointT>::AlignedPointTVector AlignedPointTVector;
        # 
        # /** \brief Empty constructor creates disk container and sets filename from random uuid string*/
        # OutofcoreOctreeDiskContainer ();
        # 
        # /** \brief Creates uuid named file or loads existing file
        #  * If \b dir is a directory, this constructor will create a new
        #  * uuid named file; if \b dir is an existing file, it will load the
        #  * file metadata for accessing the tree.
        #  * \param[in] dir Path to the tree. If it is a directory, it
        #  * will create the metadata. If it is a file, it will load the metadata into memory.
        #  */
        # OutofcoreOctreeDiskContainer (const boost::filesystem::path &dir);
        # 
        # /** \brief flushes write buffer, then frees memory */
        # ~OutofcoreOctreeDiskContainer ();
        # 
        # /** \brief provides random access to points based on a linear index
        #  */
        # inline PointT operator[] (uint64_t idx) const;
        # 
        # /** \brief Adds a single point to the buffer to be written to disk when the buffer grows sufficiently large, the object is destroyed, or the write buffer is manually flushed */
        # inline void push_back (const PointT& p);
        # 
        # /** \brief Inserts a vector of points into the disk data structure */
        # void insertRange (const AlignedPointTVector& src);
        # 
        # /** \brief Inserts a PCLPointCloud2 object directly into the disk container */
        # void insertRange (const pcl::PCLPointCloud2::Ptr &input_cloud);
        # 
        # void insertRange (const PointT* const * start, const uint64_t count);
        # 
        # /** \brief This is the primary method for serialization of
        #  * blocks of point data. This is called by the outofcore
        #  * octree interface, opens the binary file for appending data,
        #  * and writes it to disk.
        #  * \param[in] start address of the first point to insert
        #  * \param[in] count offset from start of the last point to insert
        #  */
        # void insertRange (const PointT* start, const uint64_t count);
        # 
        # /** \brief Reads \b count points into memory from the disk container
        #  * Reads \b count points into memory from the disk container, reading at most 2 million elements at a time
        #  * \param[in] start index of first point to read from disk
        #  * \param[in] count offset of last point to read from disk
        #  * \param[out] dst std::vector as destination for points read from disk into memory
        #  */
        # void readRange (const uint64_t start, const uint64_t count, AlignedPointTVector &dst);
        # 
        # void readRange (const uint64_t, const uint64_t, pcl::PCLPointCloud2::Ptr &dst);
        # 
        # /** \brief Reads the entire point contents from disk into \c output_cloud
        #  *  \param[out] output_cloud
        #  */
        # int read (pcl::PCLPointCloud2::Ptr &output_cloud);
        # 
        # /** \brief  grab percent*count random points. points are \b not guaranteed to be
        #  * unique (could have multiple identical points!)
        #  * \param[in] start The starting index of points to select
        #  * \param[in] count The length of the range of points from which to randomly sample 
        #  *  (i.e. from start to start+count)
        #  * \param[in] percent The percentage of count that is enough points to make up this random sample
        #  * \param[out] dst std::vector as destination for randomly sampled points; size will 
        #  * be percentage*count
        #  */
        # void readRangeSubSample (const uint64_t start, const uint64_t count, const double percent, AlignedPointTVector &dst);
        # 
        # /** \brief Use bernoulli trials to select points. All points selected will be unique.
        #  * \param[in] start The starting index of points to select
        #  * \param[in] count The length of the range of points from which to randomly sample 
        #  *  (i.e. from start to start+count)
        #  * \param[in] percent The percentage of count that is enough points to make up this random sample
        #  * \param[out] dst std::vector as destination for randomly sampled points; size will 
        #  * be percentage*count
        #  */
        # void readRangeSubSample_bernoulli (const uint64_t start, const uint64_t count, const double percent, AlignedPointTVector& dst);
        # 
        # /** \brief Returns the total number of points for which this container is responsible, \c filelen_ + points in \c writebuff_ that have not yet been flushed to the disk
        #  */
        # uint64_t size () const
        # 
        # /** \brief STL-like empty test
        #  * \return true if container has no data on disk or waiting to be written in \c writebuff_ */
        # inline bool empty () const
        # 
        # /** \brief Exposed functionality for manually flushing the write buffer during tree creation */
        # void flush (const bool force_cache_dealloc)
        # 
        # /** \brief Returns this objects path name */
        # inline std::string& path ()
        # 
        # inline void clear ()
        # 
        # /** \brief write points to disk as ascii
        #  * \param[in] path
        #  */
        # void convertToXYZ (const boost::filesystem::path &path)
        # 
        # /** \brief Generate a universally unique identifier (UUID)
        #  * A mutex lock happens to ensure uniquness
        #  */
        # static void getRandomUUIDString (std::string &s);
        # 
        # /** \brief Returns the number of points in the PCD file by reading the PCD header. */
        # boost::uint64_t getDataSize () const;


###

# octree_ram_container.h
# namespace pcl
# namespace outofcore
    # /** \class OutofcoreOctreeRamContainer
    #  *  \brief Storage container class which the outofcore octree base is templated against
    #  *  \note Code was adapted from the Urban Robotics out of core octree implementation. 
    #  *  Contact Jacob Schloss <jacob.schloss@urbanrobotics.net> with any questions. 
    #  *  http://www.urbanrobotics.net/
    #  * 
    #  *  \ingroup outofcore
    #  *  \author Jacob Schloss (jacob.scloss@urbanrobotics.net)
    #  */
    # template<typename PointT>
    # class OutofcoreOctreeRamContainer : public OutofcoreAbstractNodeContainer<PointT>
    # {
        # public:
        # typedef typename OutofcoreAbstractNodeContainer<PointT>::AlignedPointTVector AlignedPointTVector;
        # 
        # /** \brief empty contructor (with a path parameter?)
        #   */
        # OutofcoreOctreeRamContainer (const boost::filesystem::path&) : container_ () { }
        # 
        # /** \brief inserts count number of points into container; uses the container_ type's insert function
        #   * \param[in] start - address of first point in array
        #   * \param[in] count - the maximum offset from start of points inserted 
        #   */
        # void insertRange (const PointT* start, const uint64_t count);
        # 
        # /** \brief inserts count points into container 
        #   * \param[in] start - address of first point in array
        #   * \param[in] count - the maximum offset from start of points inserted 
        #   */
        # void insertRange (const PointT* const * start, const uint64_t count);
        # 
        # void insertRange (AlignedPointTVector& /*p*/)
        # 
        # void insertRange (const AlignedPointTVector& /*p*/)
        # 
        # /** \brief 
        #   * \param[in] start Index of first point to return from container
        #   * \param[in] count Offset (start + count) of the last point to return from container
        #   * \param[out] v Array of points read from the input range
        #   */
        # void readRange (const uint64_t start, const uint64_t count, AlignedPointTVector &v);
        # 
        # /** \brief grab percent*count random points. points are NOT
        #   *   guaranteed to be unique (could have multiple identical points!)
        #   * \param[in] start Index of first point in range to subsample
        #   * \param[in] count Offset (start+count) of last point in range to subsample
        #   * \param[in] percent Percentage of range to return
        #   * \param[out] v Vector with percent*count uniformly random sampled 
        #   * points from given input rangerange
        #   */
        # void readRangeSubSample (const uint64_t start, const uint64_t count, const double percent, AlignedPointTVector &v);
        # 
        # /** \brief returns the size of the vector of points stored in this class */
        # inline uint64_t size () const
        # 
        # inline bool empty () const
        # 
        # 
        # /** \brief clears the vector of points in this class */
        # inline void clear ()
        # 
        # /** \brief Writes ascii x,y,z point data to path.string().c_str()
        #   *  \param path The path/filename destination of the ascii xyz data
        #   */
        # void convertToXYZ (const boost::filesystem::path &path);
        # 
        # inline PointT operator[] (uint64_t index) const


###

# outofcore.h
###


# outofcore_base_data.h
# namespace pcl
# namespace outofcore
    # /** \class OutofcoreOctreeBaseMetadata 
    #  *  \brief Encapsulated class to read JSON metadata into memory,
    #  *  and write the JSON metadata associated with the octree root
    #  *  node. This is global information that is not the same as the
    #  *  metadata for the root node. Inherits OutofcoreAbstractMetadata
    #  *  interface for metadata in \b pcl_outofcore.
    #  *  This class encapsulates the outofcore base metadata
    #  *  serialization/deserialization. At the time it was written,
    #  *  this depended on cJSON to write JSON objects to disk. This
    #  *  class can be extended to have arbitrary JSON ascii metadata
    #  *  fields saved to the metadata object file on disk. The class
    #  *  has been encapuslated to abstract the detailso of the on-disk
    #  *  format from the outofcore implementation. For example, the
    #  *  format could be changed to XML/YAML, or any dynamic format at
    #  *  some point.
    #  *  The JSON file is formatted in the following way:
    #  *  \verbatim
    #  {
    #    "name": "nameoftree",
    #    "version": 3,
    #    "pointtype": "urp",               #(needs to be changed*)
    #    "lod": 3,                         #(depth of the tree
    #    "numpts":  [X0, X1, X2, ..., XD], #total number of points at each LOD
    #    "coord_system": "ECEF"            #the tree is not affected by this value
    #  }
    #  \endverbatim
    #  *
    #  *  Any properties not stored in the metadata file are computed
    #  *  when the file is loaded. By convention, and for historical
    #  *  reasons from the original Urban Robotics implementation, the
    #  *  JSON file representing the overall tree is a JSON file named
    #  *  with the ".octree" extension.
    #  *
    #  *  \ingroup outofcore
    #  *  \author Stephen Fox (foxstephend@gmail.com)
    #  */
    # class PCL_EXPORTS OutofcoreOctreeBaseMetadata : public OutofcoreAbstractMetadata
        # public:
        # /** \brief Empty constructor */
        # OutofcoreOctreeBaseMetadata ();
        # /** \brief Load metadata from disk 
        #  *
        #  *  \param[in] path_arg Location of JSON metadata file to load from disk
        #  */
        # OutofcoreOctreeBaseMetadata (const boost::filesystem::path& path_arg);
        # /** \brief Default destructor*/
        # ~OutofcoreOctreeBaseMetadata ();
        # 
        # /** \brief Copy constructor */
        # OutofcoreOctreeBaseMetadata (const OutofcoreOctreeBaseMetadata& orig);
        # 
        # /** \brief et the outofcore version read from the "version" field of the JSON object */
        # int getOutofcoreVersion () const;
        # /** \brief Set the outofcore version stored in the "version" field of the JSON object */
        # void setOutofcoreVersion (const int version);
        # 
        # /** \brief Gets the name of the JSON file */
        # boost::filesystem::path getMetadataFilename () const;
        # /** \brief Sets the name of the JSON file */
        # void setMetadataFilename (const boost::filesystem::path& path_to_metadata);
        #  
        # /** \brief Writes the data to a JSON file located at \ref metadata_filename_ */
        # virtual void serializeMetadataToDisk ();
        # 
        # /** \brief Loads the data from a JSON file located at \ref metadata_filename_ */
        # virtual int loadMetadataFromDisk ();
        # /** \brief Loads the data from a JSON file located at \ref metadata_filename_ */
        # 
        # virtual int loadMetadataFromDisk (const boost::filesystem::path& path_to_metadata);
        # 
        # /** \brief Returns the name of the tree; this is not the same as the filename */
        # virtual std::string getOctreeName ();
        # /** \brief Sets the name of the tree */
        # virtual void setOctreeName (const std::string& name_arg);
        # 
        # virtual std::string getPointType ();
        # /** \brief Sets a single string identifying the point type of this tree */
        # virtual void setPointType (const std::string& point_type_arg);
        # 
        # virtual std::vector<boost::uint64_t>& getLODPoints ();
        # virtual std::vector<boost::uint64_t> getLODPoints () const;
        # /** \brief Get the number of points at the given depth */
        # virtual boost::uint64_t getLODPoints (const boost::uint64_t& depth_index) const;
        # 
        # /** \brief Initialize the LOD vector with points all 0 */
        # virtual void setLODPoints (const boost::uint64_t& depth);
        # /** \brief Copy a vector of LOD points into this metadata (dangerous!)*/
        # virtual void setLODPoints (std::vector<boost::uint64_t>& lod_points_arg);
        # 
        # /** \brief Set the number of points at lod_index_arg manually 
        #  *  \param[in] lod_index_arg the depth at which this increments the number of LOD points
        #  *  \param[in] num_points_arg The number of points to store at that LOD
        #  *  \param[in] increment If true, increments the number of points at the LOD rather than overwriting the number of points
        #  */
        # virtual void setLODPoints (const boost::uint64_t& lod_index_arg, const boost::uint64_t& num_points_arg, const bool increment=true);
        # 
        # /** \brief Set information about the coordinate system */
        # virtual void setCoordinateSystem (const std::string& coordinate_system);
        # /** \brief Get metadata information about the coordinate system */
        # virtual std::string getCoordinateSystem () const;
        # 
        # /** \brief Set the depth of the tree corresponding to JSON "lod:number". This should always be equal to LOD_num_points_.size()-1 */
        # virtual void setDepth (const boost::uint64_t& depth_arg);
        # virtual boost::uint64_t getDepth () const;
        # 
        # /** \brief Provide operator overload to stream ascii file data*/
        # friend std::ostream& operator<<(std::ostream& os, const OutofcoreOctreeBaseMetadata& metadata_arg);


###


# outofcore_breadth_first_iterator.h
# namespace pcl
# namespace outofcore
    # /** \class OutofcoreBreadthFirstIterator
    #  *
    #  *  \ingroup outofcore
    #  *  \author Justin Rosen (jmylesrosen@gmail.com)
    #  *  \note Code adapted from \ref octree_iterator.h in Module \ref pcl::octree written by Julius Kammerl
    #  */
    # template<typename PointT=pcl::PointXYZ, typename ContainerT=OutofcoreOctreeDiskContainer<pcl::PointXYZ> >
    # class OutofcoreBreadthFirstIterator : public OutofcoreIteratorBase<PointT, ContainerT>
        # public:
        # typedef typename pcl::outofcore::OutofcoreOctreeBase<ContainerT, PointT> OctreeDisk;
        # typedef typename pcl::outofcore::OutofcoreOctreeBaseNode<ContainerT, PointT> OctreeDiskNode;
        # 
        # typedef typename pcl::outofcore::OutofcoreOctreeBaseNode<ContainerT, PointT> LeafNode;
        # typedef typename pcl::outofcore::OutofcoreOctreeBaseNode<ContainerT, PointT> BranchNode;
        # 
        # explicit OutofcoreBreadthFirstIterator (OctreeDisk& octree_arg);
        # virtual ~OutofcoreBreadthFirstIterator ();
        # 
        # OutofcoreBreadthFirstIterator& operator++ ();
        # inline OutofcoreBreadthFirstIterator operator++ (int)
        # 
        # virtual inline void reset ()
        # 
        # void skipChildVoxels ()


###


# outofcore_depth_first_iterator.h
# namespace pcl
# namespace outofcore
    # /** \class OutofcoreDepthFirstIterator
    #  *
    #  *  \ingroup outofcore
    #  *  \author Stephen Fox (foxstephend@gmail.com)
    #  *  \note Code adapted from \ref octree_iterator.h in Module \ref pcl::octree written by Julius Kammerl
    #  */
    # template<typename PointT=pcl::PointXYZ, typename ContainerT=OutofcoreOctreeDiskContainer<pcl::PointXYZ> >
    # class OutofcoreDepthFirstIterator : public OutofcoreIteratorBase<PointT, ContainerT>
        # public:
        # typedef typename pcl::outofcore::OutofcoreOctreeBase<ContainerT, PointT> OctreeDisk;
        # typedef typename pcl::outofcore::OutofcoreOctreeBaseNode<ContainerT, PointT> OctreeDiskNode;
        # 
        # typedef typename pcl::outofcore::OutofcoreOctreeBaseNode<ContainerT, PointT> LeafNode;
        # typedef typename pcl::outofcore::OutofcoreOctreeBaseNode<ContainerT, PointT> BranchNode;
        # 
        # explicit OutofcoreDepthFirstIterator (OctreeDisk& octree_arg);
        # 
        # virtual ~OutofcoreDepthFirstIterator ();
        
        OutofcoreDepthFirstIterator& operator++ ();
        
        inline OutofcoreDepthFirstIterator operator++ (int)
        
        void skipChildVoxels ();


###

# outofcore_impl.h
###


# outofcore_iterator_base.h
# namespace pcl
# namespace outofcore
    # /** \brief Abstract octree iterator class
    #  *  \note This class is based on the octree_iterator written by Julius Kammerl adapted to the outofcore octree. The interface is very similar, but it does \b not inherit the \ref pcl::octree iterator base.
    #  *  \ingroup outofcore
    #  *  \author Stephen Fox (foxstephend@gmail.com)
    #  */
    # template<typename PointT, typename ContainerT>
    # class OutofcoreIteratorBase : public std::iterator<std::forward_iterator_tag,     /*iterator type*/
    #                                                   const OutofcoreOctreeBaseNode<ContainerT, PointT>,
    #                                                    void,  /*no defined distance between iterator*/
    #                                                    const OutofcoreOctreeBaseNode<ContainerT, PointT>*,/*Pointer type*/
    #                                                    const OutofcoreOctreeBaseNode<ContainerT, PointT>&>/*Reference type*/
        # public:
        # typedef typename pcl::outofcore::OutofcoreOctreeBase<ContainerT, PointT> OctreeDisk;
        # typedef typename pcl::outofcore::OutofcoreOctreeBaseNode<ContainerT, PointT> OctreeDiskNode;
        # 
        # typedef typename pcl::outofcore::OutofcoreOctreeBase<ContainerT, PointT>::BranchNode BranchNode;
        # typedef typename pcl::outofcore::OutofcoreOctreeBase<ContainerT, PointT>::LeafNode LeafNode;
        # 
        # typedef typename OctreeDisk::OutofcoreNodeType OutofcoreNodeType;
        # 
        # explicit OutofcoreIteratorBase (OctreeDisk& octree_arg) : octree_ (octree_arg), currentNode_ (NULL)
        # virtual ~OutofcoreIteratorBase ()
        # 
        # OutofcoreIteratorBase (const OutofcoreIteratorBase& src) : octree_ (src.octree_), currentNode_ (src.currentNode_)
        # 
        # inline OutofcoreIteratorBase& operator = (const OutofcoreIteratorBase& src)
        # 
        # inline OutofcoreNodeType* operator* () const
        # virtual inline OutofcoreNodeType* getCurrentOctreeNode () const
        # 
        # virtual inline void reset ()
        # 
        # inline void setMaxDepth (unsigned int max_depth)


###

# outofcore_node_data.h
# namespace pcl
# namespace outofcore
    # /** \class OutofcoreOctreeNodeMetadata 
    #  *
    #  *  \brief Encapsulated class to read JSON metadata into memory, and write the JSON metadata for each
    #  *  node. 
    #  *
    #  *  This class encapsulates the outofcore node metadata
    #  *  serialization/deserialization. At the time it was written,
    #  *  this depended on cJSON to write JSON objects to disk. This
    #  *  class can be extended to have arbitrary ascii metadata fields
    #  *  saved to the metadata object file on disk.
    #  *
    #  *  The JSON file is formatted in the following way:
    #  *  \verbatim
    #  {
    #    "version": 3,
    #    "bb_min":  [xxx,yyy,zzz],
    #    "bb_max":  [xxx,yyy,zzz],
    #    "bin":     "path_to_data.pcd"
    #  }
    #  \endverbatim
    #  *
    #  *  Any properties not stored in the metadata file are computed
    #  *  when the file is loaded (e.g. \ref midpoint_xyz_). By
    #  *  convention, the JSON files are stored on disk with .oct_idx
    #  *  extension.
    #  *
    #  *  \ingroup outofcore
    #  *  \author Stephen Fox (foxstephend@gmail.com)
    #  */
    # class PCL_EXPORTS OutofcoreOctreeNodeMetadata
        # public:
        # //public typedefs
        # typedef boost::shared_ptr<OutofcoreOctreeNodeMetadata> Ptr;
        # typedef boost::shared_ptr<const OutofcoreOctreeNodeMetadata> ConstPtr;
        # 
        # /** \brief Empty constructor */
        # OutofcoreOctreeNodeMetadata ();
        # ~OutofcoreOctreeNodeMetadata ();
        # 
        # /** \brief Copy constructor */
        # OutofcoreOctreeNodeMetadata (const OutofcoreOctreeNodeMetadata& orig);
        # 
        # /** \brief Get the lower bounding box corner */
        # const Eigen::Vector3d& getBoundingBoxMin () const;
        # /** \brief Set the lower bounding box corner */
        # void setBoundingBoxMin (const Eigen::Vector3d& min_bb);
        # /** \brief Get the upper bounding box corner */
        # const Eigen::Vector3d& getBoundingBoxMax () const;
        # /** \brief Set the upper bounding box corner */
        # void setBoundingBoxMax (const Eigen::Vector3d& max_bb);
        # 
        # /** \brief Get the lower and upper corners of the bounding box enclosing this node */
        # void getBoundingBox (Eigen::Vector3d &min_bb, Eigen::Vector3d &max_bb) const;
        # /** \brief Set the lower and upper corners of the bounding box */
        # void setBoundingBox (const Eigen::Vector3d& min_bb, const Eigen::Vector3d& max_bb);
        # 
        # /** \brief Get the directory path name; this is the parent_path of  */
        # const boost::filesystem::path& getDirectoryPathname () const;
        # /** \brief Set the directory path name */
        # void setDirectoryPathname (const boost::filesystem::path& directory_pathname);
        # 
        # /** \brief Get the path to the PCD file */
        # const boost::filesystem::path& getPCDFilename () const;
        # /** \brief Set the point filename; extension .pcd */
        # void setPCDFilename (const boost::filesystem::path& point_filename);
        # 
        # /** \brief et the outofcore version read from the "version" field of the JSON object */
        # int getOutofcoreVersion () const;
        # /** \brief Set the outofcore version stored in the "version" field of the JSON object */
        # void setOutofcoreVersion (const int version);
        # 
        # /** \brief Sets the name of the JSON file */
        # const boost::filesystem::path& getMetadataFilename () const;
        # /** \brief Gets the name of the JSON file */
        # void setMetadataFilename (const boost::filesystem::path& path_to_metadata);
        # 
        # /** \brief Get the midpoint of this node's bounding box */
        # const Eigen::Vector3d& getVoxelCenter () const;
        # 
        # /** \brief Writes the data to a JSON file located at \ref metadata_filename_ */
        # void serializeMetadataToDisk ();
        # 
        # /** \brief Loads the data from a JSON file located at \ref metadata_filename_ */
        # int loadMetadataFromDisk ();
        # /** \brief Loads the data from a JSON file located at \ref metadata_filename_ */
        # int loadMetadataFromDisk (const boost::filesystem::path& path_to_metadata);
        # 
        # friend std::ostream& operator<<(std::ostream& os, const OutofcoreOctreeNodeMetadata& metadata_arg);
        # 


###

