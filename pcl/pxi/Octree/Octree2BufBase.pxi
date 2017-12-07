# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_octree as pcloct

cimport eigen as eig

include "../PointXYZtoPointXYZ.pxi"

cdef class Octree2BufBase:
    """
    Octree2BufBase
    """
    cdef pcloct.Octree2BufBase_t *me

    # def __cinit__(self, double resolution):
    #     self.me = NULL

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me = new pcloct.Octree2BufBase_t(resolution)

    def __dealloc__(self):
        del self.me
        self.me = NULL      # just to be sure

    def set_MaxVoxelIndex (self, unsigned int maxVoxelIndex_arg):
        self.me.setMaxVoxelIndex (maxVoxelIndex_arg)

    def set_TreeDepth (self, unsigned int depth_arg):
        self.me.setTreeDepth (depth_arg)

    def get_TreeDepth (self)
        self.me.getTreeDepth ()


    def AddData (self, unsigned int idxX_arg, unsigned int idxY_arg, unsigned int idxZ_arg, const DataT& data_arg):
        self.me.addData (idxX_arg, idxY_arg, idxZ_arg, const DataT& data_arg)

    # return bool
    def get_Data (self, unsigned int idxX_arg, unsigned int idxY_arg, unsigned int idxZ_arg, const DataT& data_arg):
        return self.me.getData (idxX_arg, idxY_arg, idxZ_arg, DataT& data_arg)

    # return bool
    def existLeaf (self, unsigned int idxX_arg, unsigned int idxY_arg, unsigned int idxZ_arg):
        return self.me.existLeaf (unsigned int idxX_arg, unsigned int idxY_arg, unsigned int idxZ_arg) const

    def removeLeaf (self, unsigned int idxX_arg, unsigned int idxY_arg, unsigned int idxZ_arg):
        self.me.removeLeaf (idxX_arg, idxY_arg, idxZ_arg)

    def getLeafCount (self):
        self.me.getLeafCount()

    def getBranchCount (self):
        self.me.getBranchCount()

    def deleteTree (bool freeMemory_arg)
        self.me.deleteTree (freeMemory_arg)

    def deletePreviousBuffer ()
        self.me.deletePreviousBuffer ()

    def deleteCurrentBuffer ()
        self.me.deleteCurrentBuffer ()

    def switchBuffers ()
        self.me.switchBuffers ()

#     def serializeTree (vector[char]& binaryTreeOut_arg, bool doXOREncoding_arg)
#         self.me.serializeTree (binaryTreeOut_arg, doXOREncoding_arg)
# 
#     def serializeTree (vector[char]& binaryTreeOut_arg, vector[DataT]& dataVector_arg, bool doXOREncoding_arg)
#         self.me.serializeTree (binaryTreeOut_arg, dataVector_arg, doXOREncoding_arg)
# 
#     def serializeLeafs (vector[DataT]& dataVector_arg)
#         self.me.serializeLeafs (dataVector_arg)
# 
#     def serializeNewLeafs (vector[DataT]& dataVector_arg, const int minPointsPerLeaf_arg)
#         self.me.serializeNewLeafs (dataVector_arg, minPointsPerLeaf_arg)
# 
#     def deserializeTree (vector[char]& binaryTreeIn_arg, bool doXORDecoding_arg)
#         self.me.deserializeTree (binaryTreeIn_arg, doXORDecoding_arg)
# 
#     def deserializeTree (vector[char]& binaryTreeIn_arg, vector[DataT]& dataVector_arg, bool doXORDecoding_arg)
#         self.me.deserializeTree (binaryTreeIn_arg, dataVector_arg, doXORDecoding_arg)


