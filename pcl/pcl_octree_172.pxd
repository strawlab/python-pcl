# -*- coding: utf-8 -*-
from libcpp cimport bool
from libcpp.vector cimport vector

# main
cimport pcl_defs as cpp
from boost_shared_ptr cimport shared_ptr

cimport eigen as eig
from vector cimport vector as vector2

###############################################################################
# Types
###############################################################################

### base class ###

# octree_container.h
# namespace pcl
# namespace octree
# \brief @b Octree container class that can serve as a base to construct own leaf node container classes.
# \author Julius Kammerl (julius@kammerl.de)
# class OctreeContainerBase
cdef extern from "pcl/octree/octree_container.h" namespace "pcl::octree":
    cdef cppclass OctreeContainerBase:
        OctreeContainerBase ()
        # \brief Empty constructor.
        
        # \brief Empty constructor.
        # OctreeContainerBase (const OctreeContainerBase&)
        
        # \brief Empty deconstructor.
        # virtual ~OctreeContainerBase ()
        
        # \brief Equal comparison operator
        # virtual bool operator== (const OctreeContainerBase&) const
        
        # \brief Inequal comparison operator
        # \param[in] other OctreeContainerBase to compare with
        # bool operator!= (const OctreeContainerBase& other) const
        
        # \brief Pure abstract method to get size of container (number of indices)
        # \return number of points/indices stored in leaf node container.
        # virtual size_t getSize () const
        
        # \brief Pure abstract reset leaf node implementation. */
        # virtual void reset () = 0;
        
        # \brief Empty addPointIndex implementation. This leaf node does not store any point indices.
        void addPointIndex (const int&)
        
        # \brief Empty getPointIndex implementation as this leaf node does not store any point indices.
        # void getPointIndex (int&) const
        
        # \brief Empty getPointIndices implementation as this leaf node does not store any data.
        # void getPointIndices (std::vector<int>&) const


##

# octree_container.h
# namespace pcl
# namespace octree
# class OctreeContainerEmpty : public OctreeContainerBase
cdef extern from "pcl/octree/octree_container.h" namespace "pcl::octree":
    cdef cppclass OctreeContainerEmpty(OctreeContainerBase):
        OctreeContainerEmpty()
        # OctreeContainerEmpty (const OctreeContainerEmpty&)
        # public:
        # /** \brief Octree deep copy method */
        # virtual OctreeContainerEmpty *deepCopy () const
        # /** \brief Octree deep copy method */
        # virtual OctreeContainerEmpty *deepCopy () const
        # 
        # /** \brief Abstract get size of container (number of DataT objects)
        # * \return number of DataT elements in leaf node container.
        # */
        # virtual size_t getSize () const
        # 
        # /** \brief Abstract reset leaf node implementation. */
        # virtual void reset ()
        # 
        # /** \brief Empty addPointIndex implementation. This leaf node does not store any point indices.
        # */
        # void addPointIndex (int)
        # 
        # /** \brief Empty getPointIndex implementation as this leaf node does not store any point indices.
        # */
        # int getPointIndex () const
        # 
        # /** \brief Empty getPointIndices implementation as this leaf node does not store any data. \
        # */
        # void getPointIndices (std::vector<int>&) const


ctypedef OctreeContainerEmpty OctreeContainerEmpty_t
ctypedef shared_ptr[OctreeContainerEmpty] OctreeContainerEmptyPtr_t
###


# octree_container.h
# namespace pcl
# namespace octree
# \brief @b Octree container class that does store a vector of point indices.
# \note Enables the octree to store multiple DataT elements within its leaf nodes.
# \author Julius Kammerl (julius@kammerl.de)
# class OctreeContainerPointIndices : public OctreeContainerBase
cdef extern from "pcl/octree/octree_container.h" namespace "pcl::octree":
    cdef cppclass OctreeContainerPointIndices(OctreeContainerBase):
        OctreeContainerPointIndices()
        #   /** \brief Empty constructor. */
        #   OctreeContainerPointIndices () :
        #     OctreeContainerBase (), leafDataTVector_ ()
        # 
        #   /** \brief Empty constructor. */
        #   OctreeContainerPointIndices (const OctreeContainerPointIndices& source) :
        #       OctreeContainerBase (), leafDataTVector_ (source.leafDataTVector_)
        # 
        #   /** \brief Empty deconstructor. */
        #   virtual
        #   ~OctreeContainerPointIndices ()
        # 
        #   /** \brief Octree deep copy method */
        #   virtual OctreeContainerPointIndices *deepCopy () const
        # 
        #   /** \brief Equal comparison operator
        #    * \param[in] other OctreeContainerDataTVector to compare with
        #    */
        #   virtual bool
        #   operator== (const OctreeContainerBase& other) const
        # 
        #   /** \brief Add point index to container memory. This container stores a vector of point indices.
        #    * \param[in] data_arg index to be stored within leaf node.
        #    */
        #   void addPointIndex (int data_arg)
        # 
        #   /** \brief Retrieve point index from container. This container stores a vector of point indices.
        #    * \return index stored within container.
        #    */
        #   int getPointIndex ( ) const
        # 
        #   /** \brief Retrieve point indices from container. This container stores a vector of point indices.
        #    * \param[out] data_vector_arg vector of point indices to be stored within data vector
        #    */
        #   void getPointIndices (std::vector<int>& data_vector_arg) const
        # 
        #   /** \brief Retrieve reference to point indices vector. This container stores a vector of point indices.
        #    * \return reference to vector of point indices to be stored within data vector
        #    */
        #   std::vector<int>& getPointIndicesVector ()
        # 
        #   /** \brief Get size of container (number of indices)
        #    * \return number of point indices in container.
        #    */
        #   size_t getSize () const
        # 
        #   /** \brief Reset leaf node. Clear DataT vector.*/
        #   virtual void reset ()


ctypedef OctreeContainerPointIndices OctreeContainerPointIndices_t
# ctypedef shared_ptr[OctreeContainerPointIndices] OctreeContainerPointIndicesPtr_t
###

# octree_pointcloud_density.h
# namespace pcl
# namespace octree
# class OctreePointCloudDensityContainer : public OctreeContainerBase
cdef extern from "pcl/octree/octree_pointcloud_density.h" namespace "pcl::octree":
    cdef cppclass OctreePointCloudDensityContainer(OctreeContainerBase):
        OctreePointCloudDensityContainer ()
        # /** \brief deep copy function */
        # virtual OctreePointCloudDensityContainer * deepCopy () const
        
        # /** \brief Equal comparison operator
        #  * \param[in] other OctreePointCloudDensityContainer to compare with
        #  */
        # virtual bool operator==(const OctreeContainerBase& other) const
        
        # /** \brief Read input data. Only an internal counter is increased.
        #   */
        # void addPointIndex (int)
        
        # /** \brief Return point counter.
        #   * \return Amount of points
        #   */
        # unsigned int getPointCounter ()
        
        # /** \brief Reset leaf node. */
        # virtual void reset ()


ctypedef OctreePointCloudDensityContainer OctreePointCloudDensityContainer_t
ctypedef shared_ptr[OctreePointCloudDensityContainer] OctreePointCloudDensityContainerPtr_t
###

# octree_base.h
# namespace pcl
# namespace octree
# pcl 1.7.2
# template<typename LeafContainerT = int, typename BranchContainerT = OctreeContainerEmpty >
# class OctreeBase
cdef extern from "pcl/octree/octree_base.h" namespace "pcl::octree":
     # cdef cppclass OctreeBase:
     cdef cppclass OctreeBase[LeafContainerT, BranchContainerT]:
        OctreeBase()
        # OctreeBase (const OctreeBase& source) :
        # inline OctreeBase& operator = (const OctreeBase &source)
        # public:
        # typedef OctreeBase<LeafContainerT, BranchContainerT> OctreeT;
        # typedef OctreeBranchNode<BranchContainerT> BranchNode;
        # typedef OctreeLeafNode<LeafContainerT> LeafNode;
        # typedef BranchContainerT BranchContainer;
        # typedef LeafContainerT LeafContainer;
        # // iterators are friends
        # friend class OctreeIteratorBase<OctreeT> ;
        # friend class OctreeDepthFirstIterator<OctreeT> ;
        # friend class OctreeBreadthFirstIterator<OctreeT> ;
        # friend class OctreeLeafNodeIterator<OctreeT> ;
        # 
        # // Octree default iterators
        # typedef OctreeDepthFirstIterator<OctreeT> Iterator;
        # typedef const OctreeDepthFirstIterator<OctreeT> ConstIterator;
        # Iterator begin(unsigned int max_depth_arg = 0) {return Iterator(this, max_depth_arg);};
        # const Iterator end() {return Iterator();};
        # 
        # // Octree leaf node iterators
        # typedef OctreeLeafNodeIterator<OctreeT> LeafNodeIterator;
        # typedef const OctreeLeafNodeIterator<OctreeT> ConstLeafNodeIterator;
        # LeafNodeIterator leaf_begin(unsigned int max_depth_arg = 0) {return LeafNodeIterator(this, max_depth_arg);};
        # const LeafNodeIterator leaf_end() {return LeafNodeIterator();};
        # 
        # // Octree depth-first iterators
        # typedef OctreeDepthFirstIterator<OctreeT> DepthFirstIterator;
        # typedef const OctreeDepthFirstIterator<OctreeT> ConstDepthFirstIterator;
        # DepthFirstIterator depth_begin(unsigned int max_depth_arg = 0) {return DepthFirstIterator(this, max_depth_arg);};
        # const DepthFirstIterator depth_end() {return DepthFirstIterator();};
        # 
        # // Octree breadth-first iterators
        # typedef OctreeBreadthFirstIterator<OctreeT> BreadthFirstIterator;
        # typedef const OctreeBreadthFirstIterator<OctreeT> ConstBreadthFirstIterator;
        # BreadthFirstIterator breadth_begin(unsigned int max_depth_arg = 0) {return BreadthFirstIterator(this, max_depth_arg);};
        # const BreadthFirstIterator breadth_end() {return BreadthFirstIterator();};
        # 
        # /** \brief Empty constructor. */
        # OctreeBase ();
        # 
        # /** \brief Empty deconstructor. */
        # virtual ~OctreeBase ();
        # 
        # /** \brief Copy constructor. */
        # OctreeBase (const OctreeBase& source)
        # 
        # /** \brief Copy operator. */
        # OctreeBase& operator = (const OctreeBase &source)
        # 
        # /** \brief Set the maximum amount of voxels per dimension.
        #   * \param[in] max_voxel_index_arg maximum amount of voxels per dimension
        #   */
        # void setMaxVoxelIndex (unsigned int max_voxel_index_arg);
        void setMaxVoxelIndex (unsigned int maxVoxelIndex_arg)
        
        # \brief Set the maximum depth of the octree.
        # \param max_depth_arg: maximum depth of octree
        # void setTreeDepth (unsigned int max_depth_arg);
        void setTreeDepth (unsigned int max_depth_arg)
        
        # \brief Get the maximum depth of the octree.
        # \return depth_arg: maximum depth of octree
        # unsigned int getTreeDepth () const
        unsigned int getTreeDepth ()

        # /** \brief Create new leaf node at (idx_x_arg, idx_y_arg, idx_z_arg).
        #  *  \note If leaf node already exist, this method returns the existing node
        #  *  \param idx_x_arg: index of leaf node in the X axis.
        #  *  \param idx_y_arg: index of leaf node in the Y axis.
        #  *  \param idx_z_arg: index of leaf node in the Z axis.
        #  *  \return pointer to new leaf node container.
        #  * */
        # LeafContainerT* createLeaf (unsigned int idx_x_arg, unsigned int idx_y_arg, unsigned int idx_z_arg);
        # 
        # /** \brief Find leaf node at (idx_x_arg, idx_y_arg, idx_z_arg).
        #  *  \note If leaf node already exist, this method returns the existing node
        #  *  \param idx_x_arg: index of leaf node in the X axis.
        #  *  \param idx_y_arg: index of leaf node in the Y axis.
        #  *  \param idx_z_arg: index of leaf node in the Z axis.
        #  *  \return pointer to leaf node container if found, null pointer otherwise.
        #  * */
        # LeafContainerT* findLeaf (unsigned int idx_x_arg, unsigned int idx_y_arg, unsigned int idx_z_arg);
        # 
        # /** \brief idx_x_arg for the existence of leaf node at (idx_x_arg, idx_y_arg, idx_z_arg).
        #  *  \param idx_x_arg: index of leaf node in the X axis.
        #  *  \param idx_y_arg: index of leaf node in the Y axis.
        #  *  \param idx_z_arg: index of leaf node in the Z axis.
        #  *  \return "true" if leaf node search is successful, otherwise it returns "false".
        #  * */
        # bool existLeaf (unsigned int idx_x_arg, unsigned int idx_y_arg, unsigned int idx_z_arg) const ;
        # 
        # /** \brief Remove leaf node at (idx_x_arg, idx_y_arg, idx_z_arg).
        #  *  \param idx_x_arg: index of leaf node in the X axis.
        #  *  \param idx_y_arg: index of leaf node in the Y axis.
        #  *  \param idx_z_arg: index of leaf node in the Z axis.
        #  * */
        # void removeLeaf (unsigned int idx_x_arg, unsigned int idx_y_arg, unsigned int idx_z_arg);
        # 
        # \brief Return the amount of existing leafs in the octree.
        # \return amount of registered leaf nodes.
        # std::size_t getLeafCount () const
        size_t getLeafCount ()
        
        # \brief Return the amount of existing branches in the octree.
        # \return amount of branch nodes.
        # std::size_t getBranchCount () const
        size_t getBranchCount ()
        
        # /** \brief Delete the octree structure and its leaf nodes.
        #  * */
        # void deleteTree ( );
        void deleteTree ( )
        
        # \brief Serialize octree into a binary output vector describing its branch node structure.
        # \param binary_tree_out_arg: reference to output vector for writing binary tree structure.
        # void serializeTree (vector[char]& binary_tree_out_arg)
        void serializeTree (vector[char]& binary_tree_out_arg)
        
        # 
        # /** \brief Serialize octree into a binary output vector describing its branch node structure and push all LeafContainerT elements stored in the octree to a vector.
        #  * \param binary_tree_out_arg: reference to output vector for writing binary tree structure.
        #  * \param leaf_container_vector_arg: pointer to all LeafContainerT objects in the octree
        #  * */
        # void serializeTree (std::vector<char>& binary_tree_out_arg, std::vector<LeafContainerT*>& leaf_container_vector_arg);
        # void serializeTree (vector[char]& binary_tree_out_arg, vector[LeafContainerT*]& leaf_container_vector_arg)
        # 
        # /** \brief Outputs a vector of all LeafContainerT elements that are stored within the octree leaf nodes.
        #  *  \param leaf_container_vector_arg: pointers to LeafContainerT vector that receives a copy of all LeafContainerT objects in the octree.
        #  * */
        # void serializeLeafs (std::vector<LeafContainerT*>& leaf_container_vector_arg);
        # void serializeLeafs (vector[LeafContainerT*]& leaf_container_vector_arg)
        # 
        # /** \brief Deserialize a binary octree description vector and create a corresponding octree structure. Leaf nodes are initialized with getDataTByKey(..).
        #  *  \param binary_tree_input_arg: reference to input vector for reading binary tree structure.
        #  * */
        # void deserializeTree (std::vector<char>& binary_tree_input_arg);
        # void deserializeTree (vector[char]& binary_tree_input_arg)
        # 
        # /** \brief Deserialize a binary octree description and create a corresponding octree structure. Leaf nodes are initialized with LeafContainerT elements from the dataVector.
        #  *  \param binary_tree_input_arg: reference to input vector for reading binary tree structure.
        #  *  \param leaf_container_vector_arg: pointer to container vector.
        #  * */
        # void deserializeTree (std::vector<char>& binary_tree_input_arg, std::vector<LeafContainerT*>& leaf_container_vector_arg);
        # void deserializeTree (vector[char]& binary_tree_input_arg, vector[LeafContainerT*]& leaf_container_vector_arg)
        # 
        # typedef OctreeBranchNode<BranchT> BranchNode;
        # typedef OctreeLeafNode<LeafT> LeafNode;
        # // Octree iterators


ctypedef OctreeBase[int, OctreeContainerEmpty_t] OctreeBase_t
ctypedef shared_ptr[OctreeBase[int, OctreeContainerEmpty_t]] OctreeBasePtr_t
# use OctreePointCloud
ctypedef OctreeBase[OctreeContainerPointIndices_t, OctreeContainerEmpty_t] OctreeBase_OctreeContainerPointIndices_t
ctypedef shared_ptr[OctreeBase[OctreeContainerPointIndices_t, OctreeContainerEmpty_t]] OctreeBase_OctreeContainerPointIndicesPtr_t
# use OctreePointCloudDensity
ctypedef OctreeBase[OctreePointCloudDensityContainer_t, OctreeContainerEmpty_t] OctreeBase_OctreePointCloudDensity_t
ctypedef shared_ptr[OctreeBase[OctreePointCloudDensityContainer_t, OctreeContainerEmpty_t]] OctreeBase_OctreePointCloudDensityPtr_t
###

### Inheritance class ###

# octree.h
# header include
###

# Version 1.7.2
# octree2buf_base.h
# namespace pcl
# namespace octree
    # template<typename ContainerT>
    # class BufferedBranchNode : public OctreeNode, ContainerT
    # {
        # using ContainerT::getSize;
        # using ContainerT::getData;
        # using ContainerT::setData;
        # 
        # public:
        # /** \brief Empty constructor. */
        # BufferedBranchNode () : OctreeNode(), ContainerT(),  preBuf(0xFFFFFF), postBuf(0xFFFFFF)
        # /** \brief Copy constructor. */
        # BufferedBranchNode (const BufferedBranchNode& source) : ContainerT(source)
        # /** \brief Copy operator. */
        # inline BufferedBranchNode& operator = (const BufferedBranchNode &source_arg)
        # /** \brief Empty constructor. */
        # virtual ~BufferedBranchNode ()
        # 
        # /** \brief Method to perform a deep copy of the octree */
        # virtual BufferedBranchNode* deepCopy () const
        # 
        # /** \brief Get child pointer in current branch node
        #  *  \param buffer_arg: buffer selector
        #  *  \param index_arg: index of child in node
        #  *  \return pointer to child node
        #  * */
        # inline OctreeNode* getChildPtr (unsigned char buffer_arg, unsigned char index_arg) const
        # 
        # /** \brief Set child pointer in current branch node
        #  *  \param buffer_arg: buffer selector
        #  *  \param index_arg: index of child in node
        #  *  \param newNode_arg: pointer to new child node
        #  * */
        # inline void setChildPtr (unsigned char buffer_arg, unsigned char index_arg, OctreeNode* newNode_arg)
        # 
        # /** \brief Check if branch is pointing to a particular child node
        #  *  \param buffer_arg: buffer selector
        #  *  \param index_arg: index of child in node
        #  *  \return "true" if pointer to child node exists; "false" otherwise
        #  * */
        # inline bool hasChild (unsigned char buffer_arg, unsigned char index_arg) const
        # 
        # /** \brief Get the type of octree node. Returns LEAVE_NODE type */
        # virtual node_type_t getNodeType () const
        # 
        # /** \brief Reset branch node container for every branch buffer. */
        # inline void reset ()


###

# namespace pcl
# namespace octree
# /** \brief @b Octree double buffer class
#  * \note This octree implementation keeps two separate octree structures
#  * in memory. This enables to create octree structures at high rate due to
#  * an advanced memory management.
#  * \note Furthermore, it allows for detecting and differentially compare the adjacent octree structures.
#  * \note The tree depth defines the maximum amount of octree voxels / leaf nodes (should be initially defined).
#  * \note All leaf nodes are addressed by integer indices.
#  * \note Note: The tree depth equates to the bit length of the voxel indices.
#  * \ingroup octree
#  * \author Julius Kammerl (julius@kammerl.de)
#  */
# template<typename LeafContainerT = int,
# typename BranchContainerT = OctreeContainerEmpty >
# class Octree2BufBase
cdef extern from "pcl/octree/octree2buf_base.h" namespace "pcl::octree":
    cdef cppclass Octree2BufBase[LeafContainerT, BranchContainerT]:
        Octree2BufBase()
        # public:
        # typedef Octree2BufBase<DataT, LeafT, BranchT> OctreeT;
        # // iterators are friends
        # friend class OctreeIteratorBase<DataT, OctreeT> ;
        # friend class OctreeDepthFirstIterator<DataT, OctreeT> ;
        # friend class OctreeBreadthFirstIterator<DataT, OctreeT> ;
        # friend class OctreeLeafNodeIterator<DataT, OctreeT> ;
        # typedef BufferedBranchNode<BranchT> BranchNode;
        # typedef OctreeLeafNode<LeafT> LeafNode;
        # 
        # // Octree iterators
        # typedef OctreeDepthFirstIterator<DataT, OctreeT> Iterator;
        # typedef const OctreeDepthFirstIterator<DataT, OctreeT> ConstIterator;
        # typedef OctreeLeafNodeIterator<DataT, OctreeT> LeafNodeIterator;
        # typedef const OctreeLeafNodeIterator<DataT, OctreeT> ConstLeafNodeIterator;
        # typedef OctreeDepthFirstIterator<DataT, OctreeT> DepthFirstIterator;
        # typedef const OctreeDepthFirstIterator<DataT, OctreeT> ConstDepthFirstIterator;
        # typedef OctreeBreadthFirstIterator<DataT, OctreeT> BreadthFirstIterator;
        # typedef const OctreeBreadthFirstIterator<DataT, OctreeT> ConstBreadthFirstIterator;
        # 
        # /** \brief Empty constructor. */
        # Octree2BufBase ();
        # 
        # /** \brief Empty deconstructor. */
        # virtual ~Octree2BufBase ();
        # 
        # /** \brief Copy constructor. */
        # Octree2BufBase (const Octree2BufBase& source) :
        #     leafCount_ (source.leafCount_), branchCount_ (source.branchCount_), objectCount_ (
        #     source.objectCount_), rootNode_ (
        #       new (BranchNode) (* (source.rootNode_))), depthMask_ (
        #       source.depthMask_), maxKey_ (source.maxKey_), branchNodePool_ (), leafNodePool_ (), bufferSelector_ (
        #         source.bufferSelector_), treeDirtyFlag_ (source.treeDirtyFlag_), octreeDepth_ (
        #         source.octreeDepth_)
        # 
        # /** \brief Copy constructor. */
        # inline Octree2BufBase& operator = (const Octree2BufBase& source)
        # 
        # /** \brief Set the maximum amount of voxels per dimension.
        #   * \param[in] max_voxel_index_arg maximum amount of voxels per dimension
        #   */
        # void setMaxVoxelIndex (unsigned int max_voxel_index_arg);
        void setMaxVoxelIndex (unsigned int maxVoxelIndex_arg)
        
        # \brief Set the maximum depth of the octree.
        # \param max_depth_arg: maximum depth of octree
        # void setTreeDepth (unsigned int max_depth_arg);
        void setTreeDepth (unsigned int max_depth_arg)
        
        # \brief Get the maximum depth of the octree.
        # \return depth_arg: maximum depth of octree
        # unsigned int getTreeDepth () const
        unsigned int getTreeDepth ()

        # /** \brief Create new leaf node at (idx_x_arg, idx_y_arg, idx_z_arg).
        #  *  \note If leaf node already exist, this method returns the existing node
        #  *  \param idx_x_arg: index of leaf node in the X axis.
        #  *  \param idx_y_arg: index of leaf node in the Y axis.
        #  *  \param idx_z_arg: index of leaf node in the Z axis.
        #  *  \return pointer to new leaf node container.
        #  * */
        # LeafContainerT* createLeaf (unsigned int idx_x_arg, unsigned int idx_y_arg, unsigned int idx_z_arg);
        # 
        # /** \brief Find leaf node at (idx_x_arg, idx_y_arg, idx_z_arg).
        #  *  \note If leaf node already exist, this method returns the existing node
        #  *  \param idx_x_arg: index of leaf node in the X axis.
        #  *  \param idx_y_arg: index of leaf node in the Y axis.
        #  *  \param idx_z_arg: index of leaf node in the Z axis.
        #  *  \return pointer to leaf node container if found, null pointer otherwise.
        #  * */
        # LeafContainerT* findLeaf (unsigned int idx_x_arg, unsigned int idx_y_arg, unsigned int idx_z_arg);
        # 
        # /** \brief idx_x_arg for the existence of leaf node at (idx_x_arg, idx_y_arg, idx_z_arg).
        #  *  \param idx_x_arg: index of leaf node in the X axis.
        #  *  \param idx_y_arg: index of leaf node in the Y axis.
        #  *  \param idx_z_arg: index of leaf node in the Z axis.
        #  *  \return "true" if leaf node search is successful, otherwise it returns "false".
        #  * */
        # bool existLeaf (unsigned int idx_x_arg, unsigned int idx_y_arg, unsigned int idx_z_arg) const ;
        # 
        # /** \brief Remove leaf node at (idx_x_arg, idx_y_arg, idx_z_arg).
        #  *  \param idx_x_arg: index of leaf node in the X axis.
        #  *  \param idx_y_arg: index of leaf node in the Y axis.
        #  *  \param idx_z_arg: index of leaf node in the Z axis.
        #  * */
        # void removeLeaf (unsigned int idx_x_arg, unsigned int idx_y_arg, unsigned int idx_z_arg);
        # 
        # \brief Return the amount of existing leafs in the octree.
        # \return amount of registered leaf nodes.
        # std::size_t getLeafCount () const
        size_t getLeafCount ()
        
        # \brief Return the amount of existing branches in the octree.
        # \return amount of branch nodes.
        # std::size_t getBranchCount () const
        size_t getBranchCount ()
        
        # /** \brief Delete the octree structure and its leaf nodes.
        #  * */
        # void deleteTree ( );
        void deleteTree ( )
        
        # /** \brief Delete octree structure of previous buffer. */
        # inline void deletePreviousBuffer ()
        
        # /** \brief Delete the octree structure in the current buffer. */
        # inline void deleteCurrentBuffer ()
        
        # /** \brief Switch buffers and reset current octree structure. */
        void switchBuffers ()
        
        # \brief Serialize octree into a binary output vector describing its branch node structure.
        # \param binary_tree_out_arg: reference to output vector for writing binary tree structure.
        # void serializeTree (vector[char]& binary_tree_out_arg)
        # void serializeTree (vector[char]& binary_tree_out_arg)
        # 
        # /** \brief Serialize octree into a binary output vector describing its branch node structure.
        #  *  \param binary_tree_out_arg: reference to output vector for writing binary tree structure.
        #  *  \param do_XOR_encoding_arg: select if binary tree structure should be generated based on current octree (false) of based on a XOR comparison between current and previous octree
        #  * */
        # void serializeTree (std::vector<char>& binary_tree_out_arg, bool do_XOR_encoding_arg = false);
        # 
        # /** \brief Serialize octree into a binary output vector describing its branch node structure and and push all DataT elements stored in the octree to a vector.
        #  * \param binary_tree_out_arg: reference to output vector for writing binary tree structure.
        #  * \param leaf_container_vector_arg: pointer to all LeafContainerT objects in the octree
        #  * \param do_XOR_encoding_arg: select if binary tree structure should be generated based on current octree (false) of based on a XOR comparison between current and previous octree
        #  * */
        # void serializeTree (std::vector<char>& binary_tree_out_arg,
        #                std::vector<LeafContainerT*>& leaf_container_vector_arg,
        #                bool do_XOR_encoding_arg = false);
        # 
        # /** \brief Outputs a vector of all DataT elements that are stored within the octree leaf nodes.
        #  *  \param leaf_container_vector_arg: vector of pointers to all LeafContainerT objects in the octree
        #  * */
        # void serializeLeafs (std::vector<LeafContainerT*>& leaf_container_vector_arg);
        # 
        # /** \brief Outputs a vector of all DataT elements from leaf nodes, that do not exist in the previous octree buffer.
        #  *  \param leaf_container_vector_arg: vector of pointers to all LeafContainerT objects in the octree
        #  * */
        # void serializeNewLeafs (std::vector<LeafContainerT*>& leaf_container_vector_arg);
        # 
        # /** \brief Deserialize a binary octree description vector and create a corresponding octree structure. Leaf nodes are initialized with getDataTByKey(..).
        #  *  \param binary_tree_in_arg: reference to input vector for reading binary tree structure.
        #  *  \param do_XOR_decoding_arg: select if binary tree structure is based on current octree (false) of based on a XOR comparison between current and previous octree
        #  * */
        # void deserializeTree (std::vector<char>& binary_tree_in_arg, bool do_XOR_decoding_arg = false);
        # 
        # /** \brief Deserialize a binary octree description and create a corresponding octree structure. Leaf nodes are initialized with DataT elements from the dataVector.
        #  *  \param binary_tree_in_arg: reference to inpvectoream for reading binary tree structure.
        #  *  \param leaf_container_vector_arg: vector of pointers to all LeafContainerT objects in the octree
        #  *  \param do_XOR_decoding_arg: select if binary tree structure is based on current octree (false) of based on a XOR comparison between current and previous octree
        #  * */
        # void deserializeTree (std::vector<char>& binary_tree_in_arg,
        #                  std::vector<LeafContainerT*>& leaf_container_vector_arg,
        #                  bool do_XOR_decoding_arg = false);


ctypedef Octree2BufBase[int, OctreeContainerEmpty_t] Octree2BufBase_t
ctypedef shared_ptr[Octree2BufBase[int, OctreeContainerEmpty_t]] Octree2BufBasePtr_t
ctypedef Octree2BufBase[OctreeContainerPointIndices_t, OctreeContainerEmpty_t] Octree2BufBase_OctreeContainerPointIndices_t
ctypedef shared_ptr[Octree2BufBase[OctreeContainerPointIndices_t, OctreeContainerEmpty_t]] Octree2BufBasePtr_OctreeContainerPointIndicesPtr_t
###

# nothing use PCL 1.7.2/1.8.0
# template<typename DataT>
# class OctreeContainerDataT
# cdef extern from "pcl/octree/octree_container.h" namespace "pcl::octree":
#     cdef cppclass OctreeContainerDataT[DataT]:
#         OctreeContainerDataT()
#         # OctreeContainerDataT (const OctreeContainerDataT& source) :
#         # public:
#         # /** \brief Octree deep copy method */
#         # virtual OctreeContainerDataT* deepCopy () const
#         # /** \brief Copies a DataT element to leaf node memorye.
#         #  * \param[in] data_arg reference to DataT element to be stored within leaf node.
#         # void setData (const DataT& data_arg)
#         # /** \brief Adds leaf node DataT element to dataVector vector of type DataT.
#         #  * \param[in] dataVector_arg: reference to DataT type to obtain the most recently added leaf node DataT element.
#         # void getData (DataT& dataVector_arg) const
#         # /** \brief Adds leaf node DataT element to dataVector vector of type DataT.
#         #  * \param[in] dataVector_arg: reference to DataT vector that is to be extended with leaf node DataT elements.
#         # void getData (vector<DataT>& dataVector_arg) const
#         # /** \brief Get size of container (number of DataT objects)
#         #  * \return number of DataT elements in leaf node container.
#         # size_t getSize () const
#         # /** \brief Reset leaf node memory to zero. */
#         # void reset ()
#         # protected:
#         # /** \brief Leaf node DataT storage. */
#         # DataT data_;
#         # /** \brief Bool indicating if leaf node is empty or not. */
#         # bool isEmpty_;
# 
# 
# ctypedef OctreeContainerDataT[int] OctreeContainerDataT_t
# # ctypedef shared_ptr[OctreeContainerDataT[int]] OctreeContainerDataTPtr_t
###

# \brief @b Octree container class that does store a single point index.
# \note Enables the octree to store a single DataT element within its leaf nodes.
# \author Julius Kammerl (julius@kammerl.de)
# class OctreeContainerPointIndex : public OctreeContainerBase
cdef extern from "pcl/octree/octree_container.h" namespace "pcl::octree":
    cdef cppclass OctreeContainerPointIndex(OctreeContainerBase):
        OctreeContainerPointIndex()
        # \brief Empty constructor.
        # OctreeContainerPointIndex (const OctreeContainerPointIndex& source)
        # 
        # /** \brief Empty deconstructor. */
        # virtual ~OctreeContainerPointIndex ()
        # 
        # /** \brief Octree deep copy method */
        # virtual OctreeContainerPointIndex* deepCopy () const
        # 
        # /** \brief Equal comparison operator
        #  * \param[in] other OctreeContainerBase to compare with
        # virtual bool operator== (const OctreeContainerBase& other) const
        # 
        # /** \brief Add point index to container memory. This container stores a only a single point index.
        #  * \param[in] data_arg index to be stored within leaf node.
        # void addPointIndex (int data_arg)
        # 
        # /** \brief Retrieve point index from container. This container stores a only a single point index
        #  * \return index stored within container.
        # int getPointIndex () const
        # 
        # /** \brief Retrieve point indices from container. This container stores only a single point index
        #  * \param[out] data_vector_arg vector of point indices to be stored within data vector
        # void getPointIndices (std::vector<int>& data_vector_arg) const
        # 
        # \brief Get size of container (number of DataT objects)
        # \return number of DataT elements in leaf node container.
        # size_t getSize () const
        # 
        # /** \brief Reset leaf node memory to zero. */
        # virtual void reset ()


###

# no use pcl 1.8.0
# replace : OctreeContainerPointIndices
# template<typename DataT>
# class OctreeContainerDataTVector
# cdef extern from "pcl/octree/octree_container.h" namespace "pcl::octree":
#     cdef cppclass OctreeContainerDataTVector[DataT]:
#         OctreeContainerDataTVector()
#         # OctreeContainerDataTVector (const OctreeContainerDataTVector& source) :
#         # public:
#         # /** \brief Octree deep copy method */
#         # virtual OctreeContainerDataTVector *deepCopy () const
#         # /** \brief Pushes a DataT element to internal DataT vector.
#         #  * \param[in] data_arg reference to DataT element to be stored within leaf node.
#         #  */
#         # void setData (const DataT& data_arg)
#         # /** \brief Receive the most recent DataT element that was pushed to the internal DataT vector.
#         #  * \param[in] data_arg reference to DataT type to obtain the most recently added leaf node DataT element.
#         #  */
#         # void getData (DataT& data_arg) const
#         # /** \brief Concatenate the internal DataT vector to vector argument dataVector_arg.
#         #  * \param[in] dataVector_arg: reference to DataT vector that is to be extended with leaf node DataT elements.
#         #  */
#         # void getData (vector[DataT]& dataVector_arg) const
#         # /** \brief Return const reference to internal DataT vector
#         #  * \return  const reference to internal DataT vector
#         # const vector[DataT]& getDataTVector () const
#         # /** \brief Get size of container (number of DataT objects)
#         #  * \return number of DataT elements in leaf node container.
#         # size_t getSize () const
#         # /** \brief Reset leaf node. Clear DataT vector.*/
#         void reset ()
# 
# 
# ctypedef OctreeContainerDataTVector[int] OctreeContainerDataTVector_t
# # ctypedef shared_ptr[OctreeContainerDataTVector[int]] OctreeContainerDataTVectorPtr_t
###

# octree_impl.h
# impl header include
###


# octree_iterator.h
# namespace pcl
# namespace octree
#   template<typename DataT, typename OctreeT>
#       class OctreeIteratorBase : public std::iterator<std::forward_iterator_tag, const OctreeNode, void, const OctreeNode*, const OctreeNode&>
# pcl 1.8.0
# template<typename OctreeT>
# class OctreeIteratorBase : public std::iterator<std::forward_iterator_tag, const OctreeNode, void,
# const OctreeNode*, const OctreeNode&>
cdef extern from "pcl/octree/octree_iterator.h" namespace "pcl::octree":
    cdef cppclass OctreeIteratorBase[OctreeT]:
        OctreeIteratorBase()
        # explicit OctreeIteratorBase (OctreeT& octree_arg)
        # OctreeIteratorBase (const OctreeIteratorBase& src) :
        # inline OctreeIteratorBase& operator = (const OctreeIteratorBase& src)
        # public:
        # typedef typename OctreeT::LeafNode LeafNode;
        # typedef typename OctreeT::BranchNode BranchNode;
        # /** \brief initialize iterator globals */
        # inline void reset ()
        # /** \brief Get octree key for the current iterator octree node
        #  * \return octree key of current node
        # inline const OctreeKey& getCurrentOctreeKey () const
        # /** \brief Get the current depth level of octree
        #  * \return depth level
        # inline unsigned int getCurrentOctreeDepth () const
        # /** \brief Get the current octree node
        #  * \return pointer to current octree node
        # inline OctreeNode* getCurrentOctreeNode () const
        # /** \brief *operator.
        #  * \return pointer to the current octree node
        # inline OctreeNode* operator* () const
        # /** \brief check if current node is a branch node
        #  * \return true if current node is a branch node, false otherwise
        # inline bool isBranchNode () const
        # /** \brief check if current node is a branch node
        #  * \return true if current node is a branch node, false otherwise
        # inline bool isLeafNode () const
        # /** \brief Get bit pattern of children configuration of current node
        #  * \return bit pattern (byte) describing the existence of 8 children of the current node
        # inline char getNodeConfiguration () const
        # /** \brief Method for retrieving a single DataT element from the octree leaf node
        #  * \param[in] data_arg reference to return pointer of leaf node DataT element.
        # virtual void getData (DataT& data_arg) const
        # /** \brief Method for retrieving a vector of DataT elements from the octree laef node
        #  * \param[in] dataVector_arg reference to DataT vector that is extended with leaf node DataT elements.
        # virtual void getData (std::vector<DataT>& dataVector_arg) const
        # /** \brief Method for retrieving the size of the DataT vector from the octree laef node
        # virtual std::size_t getSize () const
        # /** \brief get a integer identifier for current node (note: identifier depends on tree depth).
        #  * \return node id.
        # virtual unsigned long getNodeID () const


###

# pcl 1.7.2
# template<typename DataT, typename OctreeT>
# class OctreeDepthFirstIterator : public OctreeIteratorBase<DataT, OctreeT>
cdef extern from "pcl/octree/octree_iterator.h" namespace "pcl::octree":
    cdef cppclass OctreeDepthFirstIterator[OctreeT](OctreeIteratorBase[OctreeT]):
        OctreeDepthFirstIterator()
        # explicit OctreeDepthFirstIterator (OctreeT& octree_arg)
        # public:
        # typedef typename OctreeIteratorBase<DataT, OctreeT>::LeafNode LeafNode;
        # typedef typename OctreeIteratorBase<DataT, OctreeT>::BranchNode BranchNode;
        # /** \brief Reset the iterator to the root node of the octree
        # virtual void reset ();
        # /** \brief Preincrement operator.
        #  * \note recursively step to next octree node
        # OctreeDepthFirstIterator& operator++ ();
        # /** \brief postincrement operator.
        #  * \note recursively step to next octree node
        # inline OctreeDepthFirstIterator operator++ (int)
        # /** \brief Skip all child voxels of current node and return to parent node.
        # void skipChildVoxels ();
        # protected:
        # /** Child index at current octree node. */
        # unsigned char currentChildIdx_;
        # /** Stack structure. */
        # std::vector<std::pair<OctreeNode*, unsigned char> > stack_;


###

# pcl 1.7.2
# template<typename DataT, typename OctreeT>
# class OctreeBreadthFirstIterator : public OctreeIteratorBase<DataT, OctreeT>
cdef extern from "pcl/octree/octree_iterator.h" namespace "pcl::octree":
    cdef cppclass OctreeBreadthFirstIterator[OctreeT](OctreeIteratorBase[OctreeT]):
        OctreeDepthFirstIterator()
        # explicit OctreeBreadthFirstIterator (OctreeT& octree_arg);
        # // public typedefs
        # typedef typename OctreeIteratorBase<DataT, OctreeT>::BranchNode BranchNode;
        # typedef typename OctreeIteratorBase<DataT, OctreeT>::LeafNode LeafNode;
        # struct FIFOElement
        # {
        #   OctreeNode* node;
        #   OctreeKey key;
        #   unsigned int depth;
        # };
        # public:
        # /** \brief Reset the iterator to the root node of the octree
        # void reset ();
        # /** \brief Preincrement operator.
        #  * \note step to next octree node
        # OctreeBreadthFirstIterator& operator++ ();
        # /** \brief postincrement operator.
        #  * \note step to next octree node
        # inline OctreeBreadthFirstIterator operator++ (int)
        # protected:
        # /** \brief Add children of node to FIFO
        #  * \param[in] node: node with children to be added to FIFO
        # void addChildNodesToFIFO (const OctreeNode* node);
        # /** FIFO list */
        # std::deque<FIFOElement> FIFO_;


###

# template<typename DataT, typename OctreeT>
# class OctreeLeafNodeIterator : public OctreeDepthFirstIterator<DataT, OctreeT>
cdef extern from "pcl/octree/octree_iterator.h" namespace "pcl::octree":
    cdef cppclass OctreeLeafNodeIterator[OctreeT](OctreeDepthFirstIterator[OctreeT]):
        OctreeLeafNodeIterator()
        # explicit OctreeLeafNodeIterator (OctreeT& octree_arg)
        # typedef typename OctreeDepthFirstIterator<DataT, OctreeT>::BranchNode BranchNode;
        # typedef typename OctreeDepthFirstIterator<DataT, OctreeT>::LeafNode LeafNode;
        # public:
        # /** \brief Constructor.
        #  * \param[in] octree_arg Octree to be iterated. Initially the iterator is set to its root node.
        # /** \brief Reset the iterator to the root node of the octree
        # inline void reset ()
        # /** \brief Preincrement operator.
        #  * \note recursively step to next octree leaf node
        # inline OctreeLeafNodeIterator& operator++ ()
        # /** \brief postincrement operator.
        #  * \note step to next octree node
        # inline OctreeLeafNodeIterator operator++ (int)
        # /** \brief *operator.
        #  * \return pointer to the current octree leaf node
        # OctreeNode* operator* () const


###

# octree_key.h
# namespace pcl
# namespace octree
# class OctreeKey
cdef extern from "pcl/octree/octree_key.h" namespace "pcl::octree":
    cdef cppclass OctreeKey:
        OctreeKey()
        # OctreeKey (unsigned int keyX, unsigned int keyY, unsigned int keyZ) :
        # OctreeKey (const OctreeKey& source) :
        # public:
        # /** \brief Operator== for comparing octree keys with each other.
        # *  \return "true" if leaf node indices are identical; "false" otherwise.
        # bool operator == (const OctreeKey& b) const
        # /** \brief Operator<= for comparing octree keys with each other.
        # *  \return "true" if key indices are not greater than the key indices of b  ; "false" otherwise.
        # bool operator <= (const OctreeKey& b) const
        # /** \brief Operator>= for comparing octree keys with each other.
        # *  \return "true" if key indices are not smaller than the key indices of b  ; "false" otherwise.
        # bool operator >= (const OctreeKey& b) const
        # /** \brief push a child node to the octree key
        # *  \param[in] childIndex index of child node to be added (0-7)
        # */
        # inline void pushBranch (unsigned char childIndex)
        # /** \brief pop child node from octree key
        # inline void popBranch ()
        # /** \brief get child node index using depthMask
        # *  \param[in] depthMask bit mask with single bit set at query depth
        # *  \return child node index
        # * */
        # inline unsigned char getChildIdxWithDepthMask (unsigned int depthMask) const
        # // Indices addressing a voxel at (X, Y, Z)
        # unsigned int x;
        # unsigned int y;
        # unsigned int z;
###

# pcl 1.7.2/1.8.0 nothing
# octree_node_pool.h
# namespace pcl
# namespace octree
# template<typename NodeT>
# class OctreeNodePool
# cdef extern from "pcl/octree/octree_node_pool.h" namespace "pcl::octree":
#     cdef cppclass OctreeNodePool[NodeT]:
#         OctreeNodePool()
#         # public:
#         # /** \brief Push node to pool
#         # *  \param childIdx_arg: pointer of noe
#         # inline void pushNode (NodeT* node_arg)
#         # /** \brief Pop node from pool - Allocates new nodes if pool is empty
#         # *  \return Pointer to octree node
#         # inline NodeT* popNode ()
#         # /** \brief Delete all nodes in pool
#         # */
#         # void deletePool ()
#         # protected:
#         # vector<NodeT*> nodePool_;
###

# NG
# octree_nodes.h
# namespace pcl
# namespace octree
#     // enum of node types within the octree
#     enum node_type_t
#     {
#       BRANCH_NODE, LEAF_NODE
#     };
##
# namespace pcl
# namespace octree
# class PCL_EXPORTS OctreeNode
#       public:
#       OctreeNode ()
#       /** \brief Pure virtual method for receiving the type of octree node (branch or leaf)  */
#       virtual node_type_t getNodeType () const = 0;
#       /** \brief Pure virtual method to perform a deep copy of the octree */
#       virtual OctreeNode* deepCopy () const = 0;
##
# template<typename ContainerT>
# class OctreeLeafNode : public OctreeNode, public ContainerT
# cdef cppclass OctreeLeafNode[ContainerT](OctreeNode)(ContainerT):
# cdef extern from "pcl/octree/octree_nodes.h" namespace "pcl::octree":
#     cdef cppclass OctreeLeafNode[ContainerT]:
#         OctreeLeafNode()
#         # OctreeLeafNode (const OctreeLeafNode& source) :
#         # public:
#         # using ContainerT::getSize;
#         # using ContainerT::getData;
#         # using ContainerT::setData;
#         # /** \brief Method to perform a deep copy of the octree */
#         # virtual OctreeLeafNode<ContainerT>* deepCopy () const
#         # /** \brief Get the type of octree node. Returns LEAVE_NODE type */
#         # virtual node_type_t getNodeType () const
#         # /** \brief Reset node */
#         # inline void reset ()
###
# # template<typename ContainerT>
# # class OctreeBranchNode : public OctreeNode, ContainerT
# # cdef extern from "pcl/octree/octree_nodes.h" namespace "pcl::octree":
# #     cdef cppclass OctreeBranchNode[ContainerT]:
# #         OctreeBranchNode()
#         # OctreeBranchNode (const OctreeBranchNode& source)
#         # inline OctreeBranchNode& operator = (const OctreeBranchNode &source)
#         # public:
#         # using ContainerT::getSize;
#         # using ContainerT::getData;
#         # using ContainerT::setData;
#         # /** \brief Octree deep copy method */
#         # virtual OctreeBranchNode* deepCopy () const
#         # inline void reset ()
#         # /** \brief Access operator.
#         #  *  \param childIdx_arg: index to child node
#         #  *  \return OctreeNode pointer
#         #  * */
#         # inline OctreeNode*& operator[] (unsigned char childIdx_arg)
#         # /** \brief Get pointer to child
#         #  *  \param childIdx_arg: index to child node
#         #  *  \return OctreeNode pointer
#         #  * */
#         # inline OctreeNode* getChildPtr (unsigned char childIdx_arg) const
#         # /** \brief Get pointer to child
#         #  *  \return OctreeNode pointer
#         #  * */
#         # inline void setChildPtr (OctreeNode* child, unsigned char index)
#         # /** \brief Check if branch is pointing to a particular child node
#         #  *  \param childIdx_arg: index to child node
#         #  *  \return "true" if pointer to child node exists; "false" otherwise
#         #  * */
#         # inline bool hasChild (unsigned char childIdx_arg) const
#         # /** \brief Get the type of octree node. Returns LEAVE_NODE type */
#         # virtual node_type_t getNodeType () const
#         # protected:
#         # OctreeNode* childNodeArray_[8];
###

# octree_pointcloud.h
# namespace pcl
# namespace octree
# template<typename PointT, typename LeafT = OctreeContainerPointIndices,
#       typename BranchT = OctreeContainerEmpty,
#       typename OctreeT = OctreeBase<LeafT, BranchT> >
# class OctreePointCloud : public OctreeT
cdef extern from "pcl/octree/octree_pointcloud.h" namespace "pcl::octree":
    cdef cppclass OctreePointCloud[PointT, LeafT, BranchT, OctreeT]:
        OctreePointCloud(const double resolution_arg)
        # OctreePointCloud(double resolution_arg)
        
        # // iterators are friends
        # friend class OctreeIteratorBase<int, OctreeT> ;
        # friend class OctreeDepthFirstIterator<int, OctreeT> ;
        # friend class OctreeBreadthFirstIterator<int, OctreeT> ;
        # friend class OctreeLeafNodeIterator<int, OctreeT> ;
        # public:
        # typedef OctreeT Base;
        # typedef typename OctreeT::LeafNode LeafNode;
        # typedef typename OctreeT::BranchNode BranchNode;
        # // Octree iterators
        # typedef OctreeDepthFirstIterator<int, OctreeT> Iterator;
        # typedef const OctreeDepthFirstIterator<int, OctreeT> ConstIterator;
        # typedef OctreeLeafNodeIterator<int, OctreeT> LeafNodeIterator;
        # typedef const OctreeLeafNodeIterator<int, OctreeT> ConstLeafNodeIterator;
        # typedef OctreeDepthFirstIterator<int, OctreeT> DepthFirstIterator;
        # typedef const OctreeDepthFirstIterator<int, OctreeT> ConstDepthFirstIterator;
        # typedef OctreeBreadthFirstIterator<int, OctreeT> BreadthFirstIterator;
        # typedef const OctreeBreadthFirstIterator<int, OctreeT> ConstBreadthFirstIterator;
        # /** \brief Octree pointcloud constructor.
        #  * \param[in] resolution_arg octree resolution at lowest octree level
        # // public typedefs
        # typedef boost::shared_ptr<std::vector<int> > IndicesPtr;
        # typedef boost::shared_ptr<const std::vector<int> > IndicesConstPtr;
        # typedef pcl::PointCloud<PointT> PointCloud;
        # typedef boost::shared_ptr<PointCloud> PointCloudPtr;
        # typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
        # // public typedefs for single/double buffering
        # typedef OctreePointCloud<PointT, LeafT, OctreeBase<int, LeafT> > SingleBuffer;
        # typedef OctreePointCloud<PointT, LeafT, Octree2BufBase<int, LeafT> > DoubleBuffer;
        # // Boost shared pointers
        # typedef boost::shared_ptr<OctreePointCloud<PointT, LeafT, OctreeT> > Ptr;
        # typedef boost::shared_ptr<const OctreePointCloud<PointT, LeafT, OctreeT> > ConstPtr;
        # // Eigen aligned allocator
        # typedef std::vector<PointT, Eigen::aligned_allocator<PointT> > AlignedPointTVector;
        # 
        # /** \brief Provide a pointer to the input data set.
        #  * \param[in] cloud_arg the const boost shared pointer to a PointCloud message
        #  * \param[in] indices_arg the point indices subset that is to be used from \a cloud - if 0 the whole point cloud is used
        #  */
        # inline void setInputCloud (const PointCloudConstPtr &cloud_arg, const IndicesConstPtr &indices_arg = IndicesConstPtr ())
        void setInputCloud (shared_ptr[cpp.PointCloud[PointT]])
        # void setInputCloud (const shared_ptr[cpp.PointCloud] &cloud_arg, const shared_ptr[const vector[int]] &indices_ar)
        
        # /** \brief Get a pointer to the vector of indices used.
        #  * \return pointer to vector of indices used.
        #  */
        # inline IndicesConstPtr const getIndices () const
        const shared_ptr[const vector[int]] getIndices ()
        
        # /** \brief Get a pointer to the input point cloud dataset.
        #  * \return pointer to pointcloud input class.
        #  */
        # inline PointCloudConstPtr getInputCloud () const
        # PointCloudConstPtr getInputCloud () const
        shared_ptr[const cpp.PointCloud[PointT]] getInputCloud ()
        
        # /** \brief Set the search epsilon precision (error bound) for nearest neighbors searches.
        #  * \param[in] eps precision (error bound) for nearest neighbors searches
        #  */
        # inline void setEpsilon (double eps)
        void setEpsilon (double eps)
        
        # /** \brief Get the search epsilon precision (error bound) for nearest neighbors searches. */
        # inline double getEpsilon () const
        double getEpsilon () const
        
        # /** \brief Set/change the octree voxel resolution
        #  * \param[in] resolution_arg side length of voxels at lowest tree level
        #  */
        # inline void setResolution (double resolution_arg)
        void setResolution (double resolution_arg)
        
        # /** \brief Get octree voxel resolution
        #  * \return voxel resolution at lowest tree level
        #  */
        # inline double getResolution () const
        double getResolution () const
        
        # \brief Get the maximum depth of the octree.
        # \return depth_arg: maximum depth of octree
        # inline unsigned int getTreeDepth () const
        unsigned int getTreeDepth ()
        
        # brief Add points from input point cloud to octree.
        # void addPointsFromInputCloud ();
        void addPointsFromInputCloud ()
        
        # \brief Add point at given index from input point cloud to octree. Index will be also added to indices vector.
        # \param[in] pointIdx_arg index of point to be added
        # \param[in] indices_arg pointer to indices vector of the dataset (given by \a setInputCloud)
        # void addPointFromCloud (const int pointIdx_arg, IndicesPtr indices_arg);
        void addPointFromCloud (const int pointIdx_arg, shared_ptr[vector[int]] indices_arg)
        
        # \brief Add point simultaneously to octree and input point cloud.
        # \param[in] point_arg point to be added
        # \param[in] cloud_arg pointer to input point cloud dataset (given by \a setInputCloud)
        # void addPointToCloud (const PointT& point_arg, PointCloudPtr cloud_arg);
        void addPointToCloud (const PointT& point_arg, shared_ptr[cpp.PointCloud[PointT]] cloud_arg)
        
        # \brief Add point simultaneously to octree and input point cloud. A corresponding index will be added to the indices vector.
        # \param[in] point_arg point to be added
        # \param[in] cloud_arg pointer to input point cloud dataset (given by \a setInputCloud)
        # \param[in] indices_arg pointer to indices vector of the dataset (given by \a setInputCloud)
        # void addPointToCloud (const PointT& point_arg, PointCloudPtr cloud_arg, IndicesPtr indices_arg);
        void addPointToCloud (const PointT& point_arg, shared_ptr[cpp.PointCloud[PointT]] cloud_arg, shared_ptr[vector[int]] indices_arg)
        
        # \brief Check if voxel at given point exist.
        # \param[in] point_arg point to be checked
        # \return "true" if voxel exist; "false" otherwise
        # bool isVoxelOccupiedAtPoint (const PointT& point_arg) const;
        # bool isVoxelOccupiedAtPoint (const PointT& point_arg)
        
        # \brief Delete the octree structure and its leaf nodes.
        # \param freeMemory_arg: if "true", allocated octree nodes are deleted, otherwise they are pushed to the octree node pool
        # void deleteTree (bool freeMemory_arg = false)
        void deleteTree()
        # void deleteTree (bool freeMemory_arg)
        
        # \brief Check if voxel at given point coordinates exist.
        # \param[in] pointX_arg X coordinate of point to be checked
        # \param[in] pointY_arg Y coordinate of point to be checked
        # \param[in] pointZ_arg Z coordinate of point to be checked
        # \return "true" if voxel exist; "false" otherwise
        # bool isVoxelOccupiedAtPoint (const double pointX_arg, const double pointY_arg, const double pointZ_arg) const;
        # bool isVoxelOccupiedAtPoint(double, double, double)
        bool isVoxelOccupiedAtPoint (const double pointX_arg, const double pointY_arg, const double pointZ_arg)
        
        # \brief Check if voxel at given point from input cloud exist.
        # \param[in] pointIdx_arg point to be checked
        # \return "true" if voxel exist; "false" otherwise
        # bool isVoxelOccupiedAtPoint (const int& pointIdx_arg) const;
        # bool isVoxelOccupiedAtPoint (const int& pointIdx_arg)
        
        # \brief Get a T vector of centers of all occupied voxels.
        # \param[out] voxelCenterList_arg results are written to this vector of T elements
        # \return number of occupied voxels
        # int getOccupiedVoxelCenters (vector2[PointT, eig.aligned_allocator[PointT]] &voxelCenterList_arg) const;
        # int getOccupiedVoxelCenters(vector2[PointT, eig.aligned_allocator[PointT]])
        int getOccupiedVoxelCenters (vector2[PointT, eig.aligned_allocator[PointT]] &voxelCenterList_arg)
        
        # \brief Get a T vector of centers of voxels intersected by a line segment.
        #  This returns a approximation of the actual intersected voxels by walking
        #  along the line with small steps. Voxels are ordered, from closest to
        #  furthest w.r.t. the origin.
        # \param[in] origin origin of the line segment
        # \param[in] end end of the line segment
        # \param[out] voxel_center_list results are written to this vector of T elements
        # \param[in] precision determines the size of the steps: step_size = octree_resolution x precision
        # \return number of intersected voxels
        # int getApproxIntersectedVoxelCentersBySegment (const Eigen::Vector3f& origin, const Eigen::Vector3f& end, AlignedPointTVector &voxel_center_list, float precision = 0.2);
        int getApproxIntersectedVoxelCentersBySegment (const eig.Vector3f& origin, const eig.Vector3f& end, vector2[PointT, eig.aligned_allocator[PointT]] &voxel_center_list, float precision)
        
        # \brief Delete leaf node / voxel at given point
        # \param[in] point_arg point addressing the voxel to be deleted.
        # void deleteVoxelAtPoint(const PointT& point_arg);
        # void deleteVoxelAtPoint(PointT point)
        void deleteVoxelAtPoint (const PointT& point_arg)
        
        # \brief Delete leaf node / voxel at given point from input cloud
        # \param[in] pointIdx_arg index of point addressing the voxel to be deleted.
        # void deleteVoxelAtPoint (const int& pointIdx_arg);
        void deleteVoxelAtPoint (const int& pointIdx_arg)
        
        # Bounding box methods
        # \brief Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. */
        # void defineBoundingBox ();
        void defineBoundingBox ()
        
        # \brief Define bounding box for octree
        # \note Bounding box cannot be changed once the octree contains elements.
        # \param[in] minX_arg X coordinate of lower bounding box corner
        # \param[in] minY_arg Y coordinate of lower bounding box corner
        # \param[in] minZ_arg Z coordinate of lower bounding box corner
        # \param[in] maxX_arg X coordinate of upper bounding box corner
        # \param[in] maxY_arg Y coordinate of upper bounding box corner
        # \param[in] maxZ_arg Z coordinate of upper bounding box corner
        # void defineBoundingBox (const double minX_arg, const double minY_arg, const double minZ_arg, const double maxX_arg, const double maxY_arg, const double maxZ_arg);
        # void defineBoundingBox(double, double, double, double, double, double)
        void defineBoundingBox (const double minX_arg, const double minY_arg, const double minZ_arg, const double maxX_arg, const double maxY_arg, const double maxZ_arg)
        
        # \brief Define bounding box for octree
        # \note Lower bounding box point is set to (0, 0, 0)
        # \note Bounding box cannot be changed once the octree contains elements.
        # \param[in] maxX_arg X coordinate of upper bounding box corner
        # \param[in] maxY_arg Y coordinate of upper bounding box corner
        # \param[in] maxZ_arg Z coordinate of upper bounding box corner
        # void defineBoundingBox (const double maxX_arg, const double maxY_arg, const double maxZ_arg);
        # void defineBoundingBox (const double maxX_arg, const double maxY_arg, const double maxZ_arg)
        
        # \brief Define bounding box cube for octree
        # \note Lower bounding box corner is set to (0, 0, 0)
        # \note Bounding box cannot be changed once the octree contains elements.
        # \param[in] cubeLen_arg side length of bounding box cube.
        # void defineBoundingBox (const double cubeLen_arg);
        # void defineBoundingBox (const double cubeLen_arg)
        
        # \brief Get bounding box for octree
        # \note Bounding box cannot be changed once the octree contains elements.
        # \param[in] minX_arg X coordinate of lower bounding box corner
        # \param[in] minY_arg Y coordinate of lower bounding box corner
        # \param[in] minZ_arg Z coordinate of lower bounding box corner
        # \param[in] maxX_arg X coordinate of upper bounding box corner
        # \param[in] maxY_arg Y coordinate of upper bounding box corner
        # \param[in] maxZ_arg Z coordinate of upper bounding box corner
        # void getBoundingBox (double& minX_arg, double& minY_arg, double& minZ_arg, double& maxX_arg, double& maxY_arg, double& maxZ_arg) const;
        void getBoundingBox (double& minX_arg, double& minY_arg, double& minZ_arg, double& maxX_arg, double& maxY_arg, double& maxZ_arg)
        
        # \brief Calculates the squared diameter of a voxel at given tree depth
        # \param[in] treeDepth_arg depth/level in octree
        # \return squared diameter
        # double getVoxelSquaredDiameter (unsigned int treeDepth_arg) const;
        double getVoxelSquaredDiameter (unsigned int treeDepth_arg)
        
        # \brief Calculates the squared diameter of a voxel at leaf depth
        # \return squared diameter
        # inline double getVoxelSquaredDiameter () const
        double getVoxelSquaredDiameter ()
        
        # \brief Calculates the squared voxel cube side length at given tree depth
        # \param[in] treeDepth_arg depth/level in octree
        # \return squared voxel cube side length
        # double getVoxelSquaredSideLen (unsigned int treeDepth_arg) const;
        double getVoxelSquaredSideLen (unsigned int treeDepth_arg)
        
        # \brief Calculates the squared voxel cube side length at leaf level
        # \return squared voxel cube side length
        # inline double getVoxelSquaredSideLen () const
        double getVoxelSquaredSideLen ()
        
        # \brief Generate bounds of the current voxel of an octree iterator
        # \param[in] iterator: octree iterator
        # \param[out] min_pt lower bound of voxel
        # \param[out] max_pt upper bound of voxel
        # inline void getVoxelBounds (OctreeIteratorBase<int, OctreeT>& iterator, Eigen::Vector3f &min_pt, Eigen::Vector3f &max_pt)
        void getVoxelBounds (OctreeIteratorBase[OctreeT]& iterator, eig.Vector3f &min_pt, eig.Vector3f &max_pt)


ctypedef OctreePointCloud[cpp.PointXYZ, OctreeContainerPointIndices_t, OctreeContainerEmpty_t, OctreeBase_OctreeContainerPointIndices_t] OctreePointCloud_t
ctypedef OctreePointCloud[cpp.PointXYZI, OctreeContainerPointIndices_t, OctreeContainerEmpty_t, OctreeBase_OctreeContainerPointIndices_t] OctreePointCloud_PointXYZI_t
ctypedef OctreePointCloud[cpp.PointXYZRGB, OctreeContainerPointIndices_t, OctreeContainerEmpty_t, OctreeBase_OctreeContainerPointIndices_t] OctreePointCloud_PointXYZRGB_t
ctypedef OctreePointCloud[cpp.PointXYZRGBA, OctreeContainerPointIndices_t, OctreeContainerEmpty_t, OctreeBase_OctreeContainerPointIndices_t] OctreePointCloud_PointXYZRGBA_t
ctypedef OctreePointCloud[cpp.PointXYZ, OctreeContainerPointIndices_t, OctreeContainerEmpty_t, Octree2BufBase_OctreeContainerPointIndices_t] OctreePointCloud2Buf_t
ctypedef OctreePointCloud[cpp.PointXYZI, OctreeContainerPointIndices_t, OctreeContainerEmpty_t, Octree2BufBase_OctreeContainerPointIndices_t] OctreePointCloud2Buf_PointXYZI_t
ctypedef OctreePointCloud[cpp.PointXYZRGB, OctreeContainerPointIndices_t, OctreeContainerEmpty_t, Octree2BufBase_OctreeContainerPointIndices_t] OctreePointCloud2Buf_PointXYZRGB_t
ctypedef OctreePointCloud[cpp.PointXYZRGBA, OctreeContainerPointIndices_t, OctreeContainerEmpty_t, Octree2BufBase_OctreeContainerPointIndices_t] OctreePointCloud2Buf_PointXYZRGBA_t
###

# namespace pcl
# namespace octree
# pcl_1.7.2 use octree_pointcloud_changedetector.h
# pcl 1.7.2 Template Changed(OctreeContainerDataTVector to OctreeContainerPointIndices)
# template<typename PointT,
#         typename LeafContainerT = OctreeContainerPointIndices,
#         typename BranchContainerT = OctreeContainerEmpty >
cdef extern from "pcl/octree/octree_pointcloud_changedetector.h" namespace "pcl::octree":
    # pcl version 1.7.2
    cdef cppclass OctreePointCloudChangeDetector[PointT](OctreePointCloud[PointT, OctreeContainerPointIndices_t, OctreeContainerEmpty_t, Octree2BufBase_OctreeContainerPointIndices_t]):
        OctreePointCloudChangeDetector (const double resolution_arg)
        # public:
        # \brief Get a indices from all leaf nodes that did not exist in previous buffer.
        # \param indicesVector_arg: results are written to this vector of int indices
        # \param minPointsPerLeaf_arg: minimum amount of points required within leaf node to become serialized.
        # \return number of point indices
        # int getPointIndicesFromNewVoxels (std::vector<int> &indicesVector_arg, const int minPointsPerLeaf_arg = 0)
        int getPointIndicesFromNewVoxels (vector[int] &indicesVector_arg, const int minPointsPerLeaf_arg)


ctypedef OctreePointCloudChangeDetector[cpp.PointXYZ] OctreePointCloudChangeDetector_t
ctypedef OctreePointCloudChangeDetector[cpp.PointXYZI] OctreePointCloudChangeDetector_PointXYZI_t
ctypedef OctreePointCloudChangeDetector[cpp.PointXYZRGB] OctreePointCloudChangeDetector_PointXYZRGB_t
ctypedef OctreePointCloudChangeDetector[cpp.PointXYZRGBA] OctreePointCloudChangeDetector_PointXYZRGBA_t
ctypedef shared_ptr[OctreePointCloudChangeDetector[cpp.PointXYZ]] OctreePointCloudChangeDetectorPtr_t
ctypedef shared_ptr[OctreePointCloudChangeDetector[cpp.PointXYZI]] OctreePointCloudChangeDetector_PointXYZI_Ptr_t
ctypedef shared_ptr[OctreePointCloudChangeDetector[cpp.PointXYZRGB]] OctreePointCloudChangeDetector_PointXYZRGB_Ptr_t
ctypedef shared_ptr[OctreePointCloudChangeDetector[cpp.PointXYZRGBA]] OctreePointCloudChangeDetector_PointXYZRGBA_Ptr_t
###

# octree_pointcloud_density.h
# namespace pcl
# namespace octree
# template<typename PointT, typename LeafContainerT = OctreePointCloudDensityContainer, typename BranchContainerT = OctreeContainerEmpty >
# class OctreePointCloudDensity : public OctreePointCloud<PointT, LeafContainerT, BranchContainerT>
cdef extern from "pcl/octree/octree_pointcloud_density.h" namespace "pcl::octree":
    cdef cppclass OctreePointCloudDensity[PointT](OctreePointCloud[PointT, OctreePointCloudDensityContainer_t, OctreeContainerEmpty_t, OctreeBase_OctreePointCloudDensity_t]):
        OctreePointCloudDensity (const double resolution_arg)
        # /** \brief Get the amount of points within a leaf node voxel which is addressed by a point
        #   * \param[in] point_arg: a point addressing a voxel
        #   * \return amount of points that fall within leaf node voxel
        #   */
        # unsigned int getVoxelDensityAtPoint (const PointT& point_arg) const


ctypedef OctreePointCloudDensity[cpp.PointXYZ] OctreePointCloudDensity_t
ctypedef OctreePointCloudDensity[cpp.PointXYZI] OctreePointCloudDensity_PointXYZI_t
ctypedef OctreePointCloudDensity[cpp.PointXYZRGB] OctreePointCloudDensity_PointXYZRGB_t
ctypedef OctreePointCloudDensity[cpp.PointXYZRGBA] OctreePointCloudDensity_PointXYZRGBA_t
ctypedef shared_ptr[OctreePointCloudDensity[cpp.PointXYZ]] OctreePointCloudDensityPtr_t
ctypedef shared_ptr[OctreePointCloudDensity[cpp.PointXYZI]] OctreePointCloudDensity_PointXYZI_Ptr_t
ctypedef shared_ptr[OctreePointCloudDensity[cpp.PointXYZRGB]] OctreePointCloudDensity_PointXYZRGB_Ptr_t
ctypedef shared_ptr[OctreePointCloudDensity[cpp.PointXYZRGBA]] OctreePointCloudDensity_PointXYZRGBA_Ptr_t
###

# octree_pointcloud_occupancy.h
###

# octree_pointcloud_pointvector.h
###

# octree_pointcloud_singlepoint.h
###

# octree_pointcloud_voxelcentroid.h
###

# octree_search.h
cdef extern from "pcl/octree/octree_search.h" namespace "pcl::octree":
    cdef cppclass OctreePointCloudSearch[PointT](OctreePointCloud[PointT, OctreeContainerPointIndices_t, OctreeContainerEmpty_t, OctreeBase_OctreeContainerPointIndices_t]):
        OctreePointCloudSearch (const double resolution_arg)
        # int radiusSearch (cpp.PointXYZ, double, vector[int], vector[float], unsigned int)
        # int radiusSearch (cpp.PointXYZI, double, vector[int], vector[float], unsigned int)
        # int radiusSearch (cpp.PointXYZRGB, double, vector[int], vector[float], unsigned int)
        # int radiusSearch (cpp.PointXYZRGBA, double, vector[int], vector[float], unsigned int)
        # PointT
        int radiusSearch (PointT, double, vector[int], vector[float], unsigned int)
        
        # Add index(inline?)
        int radiusSearch (cpp.PointCloud[PointT], int, double, vector[int], vector[float], unsigned int)
        
        # inline define
        # int nearestKSearch (cpp.PointCloud[PointT], int, int, vector[int], vector[float])
        int nearestKSearch (cpp.PointCloud[PointT], int, int, vector[int], vector[float])
        
        # int nearestKSearch (const PointT &point, int k, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const;
        # int nearestKSearch (const PointT &point, int k, vector[int] &k_indices, vector[float] &k_sqr_distances)
        
        ## Functions
        # brief Search for neighbors within a voxel at given point
        # param[in] point point addressing a leaf node voxel
        # param[out] point_idx_data the resultant indices of the neighboring voxel points
        # return "true" if leaf node exist; "false" otherwise
        # bool voxelSearch (const PointT& point, std::vector<int>& point_idx_data);
        bool voxelSearch (const PointT& point, vector[int] point_idx_data)
        
        # brief Search for neighbors within a voxel at given point referenced by a point index
        # param[in] index the index in input cloud defining the query point
        # param[out] point_idx_data the resultant indices of the neighboring voxel points
        # return "true" if leaf node exist; "false" otherwise
        # bool voxelSearch (const int index, std::vector<int>& point_idx_data);
        bool voxelSearch (const int index, vector[int] point_idx_data)
        
        # brief Search for approx. nearest neighbor at the query point.
        # param[in] cloud the point cloud data
        # param[in] query_index the index in \a cloud representing the query point
        # param[out] result_index the resultant index of the neighbor point
        # param[out] sqr_distance the resultant squared distance to the neighboring point
        # return number of neighbors found
        # 
        # inline void approxNearestSearch (const PointCloud &cloud, int query_index, int &result_index, float &sqr_distance)
        approxNearestSearch (const cpp.PointCloud[PointT] &cloud, int query_index, int &result_index, float &sqr_distance)
        
        # /** \brief Search for approx. nearest neighbor at the query point.
        #   * \param[in] p_q the given query point
        #   * \param[out] result_index the resultant index of the neighbor point
        #   * \param[out] sqr_distance the resultant squared distance to the neighboring point
        #   */
        # void approxNearestSearch (const PointT &p_q, int &result_index, float &sqr_distance);
        void approxNearestSearch (const PointT &p_q, int &result_index, float &sqr_distance)
        
        # /** \brief Search for approx. nearest neighbor at the query point.
        #   * \param[in] query_index index representing the query point in the dataset given by \a setInputCloud.
        #   *        If indices were given in setInputCloud, index will be the position in the indices vector.
        #   * \param[out] result_index the resultant index of the neighbor point
        #   * \param[out] sqr_distance the resultant squared distance to the neighboring point
        #   * \return number of neighbors found
        #   */
        # void approxNearestSearch (int query_index, int &result_index, float &sqr_distance);
        void approxNearestSearch (int query_index, int &result_index, float &sqr_distance)
        
        # /** \brief Get a PointT vector of centers of all voxels that intersected by a ray (origin, direction).
        #   * \param[in] origin ray origin
        #   * \param[in] direction ray direction vector
        #   * \param[out] voxel_center_list results are written to this vector of PointT elements
        #   * \param[in] max_voxel_count stop raycasting when this many voxels intersected (0: disable)
        #   * \return number of intersected voxels
        #  */
        # int getIntersectedVoxelCenters (Eigen::Vector3f origin, Eigen::Vector3f direction, AlignedPointTVector &voxel_center_list, int max_voxel_count = 0) const;
        # int getIntersectedVoxelCenters (eig.Vector3f origin, eig.Vector3f direction, AlignedPointTVector &voxel_center_list, int max_voxel_count = 0) const;
        
        # /** \brief Get indices of all voxels that are intersected by a ray (origin, direction).
        #   * \param[in] origin ray origin
        #   * \param[in] direction ray direction vector
        #   * \param[out] k_indices resulting point indices from intersected voxels
        #   * \param[in] max_voxel_count stop raycasting when this many voxels intersected (0: disable)
        #  * \return number of intersected voxels
        #  */
        # int getIntersectedVoxelIndices (Eigen::Vector3f origin, Eigen::Vector3f direction, std::vector<int> &k_indices, int max_voxel_count = 0) const; 
        int getIntersectedVoxelIndices (eig.Vector3f origin, eig.Vector3f direction, vector[int] &k_indices, int max_voxel_count)
        
        # /** \brief Search for points within rectangular search area
        #  * \param[in] min_pt lower corner of search area
        #  * \param[in] max_pt upper corner of search area
        #  * \param[out] k_indices the resultant point indices
        #  * \return number of points found within search area
        #  */
        # int boxSearch (const Eigen::Vector3f &min_pt, const Eigen::Vector3f &max_pt, std::vector<int> &k_indices) const;
        int boxSearch (const eig.Vector3f &min_pt, const eig.Vector3f &max_pt, vector[int] &k_indices)


ctypedef OctreePointCloudSearch[cpp.PointXYZ] OctreePointCloudSearch_t
ctypedef OctreePointCloudSearch[cpp.PointXYZI] OctreePointCloudSearch_PointXYZI_t
ctypedef OctreePointCloudSearch[cpp.PointXYZRGB] OctreePointCloudSearch_PointXYZRGB_t
ctypedef OctreePointCloudSearch[cpp.PointXYZRGBA] OctreePointCloudSearch_PointXYZRGBA_t
ctypedef shared_ptr[OctreePointCloudSearch[cpp.PointXYZ]] OctreePointCloudSearchPtr_t
ctypedef shared_ptr[OctreePointCloudSearch[cpp.PointXYZI]] OctreePointCloudSearch_PointXYZI_Ptr_t
ctypedef shared_ptr[OctreePointCloudSearch[cpp.PointXYZRGB]] OctreePointCloudSearch_PointXYZRGB_Ptr_t
ctypedef shared_ptr[OctreePointCloudSearch[cpp.PointXYZRGBA]] OctreePointCloudSearch_PointXYZRGBA_Ptr_t
###


# 1.7.2 Add headers
# octree2buf_base.h
# octree_pointcloud_adjacency.h
# octree_pointcloud_adjacency_container.h
# octree_pointcloud_changedetector.h
###


###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################


