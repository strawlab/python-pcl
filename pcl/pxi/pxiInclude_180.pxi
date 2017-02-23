# -*- coding: utf-8 -*-

### include ###
# common?
include "PointXYZtoPointXYZ.pxi"
# Segmentation
include "Segmentation/Segmentation.pxi"
include "Segmentation/SegmentationNormal.pxi"
include "Segmentation/EuclideanClusterExtraction.pxi"
# Filters
include "Filters/StatisticalOutlierRemovalFilter.pxi"
include "Filters/VoxelGridFilter.pxi"
include "Filters/PassThroughFilter.pxi"
include "Filters/ApproximateVoxelGrid.pxi"
include "Surface/MovingLeastSquares.pxi"
# include "KdTree/KdTree.pxi"
include "KdTree/KdTree_FLANN.pxi"
# Octree
include "Octree/OctreePointCloud_180.pxi"
include "Octree/OctreePointCloud2Buf_180.pxi"
include "Octree/OctreePointCloudSearch_180.pxi"
include "Octree/OctreePointCloudChangeDetector_180.pxi"

include "Vertices.pxi"
include "Filters/CropHull_172.pxi"
include "Filters/CropBox_172.pxi"
include "Filters/ProjectInliers.pxi"
include "Filters/RadiusOutlierRemoval_172.pxi"
include "Filters/ConditionAnd.pxi"
include "Filters/ConditionalRemoval.pxi"
include "Surface/ConcaveHull.pxi"
include "Common/RangeImage/RangeImages_172.pxi"
# include "Visualization/PointCloudColorHandlerCustoms.pxi"

# Features
include "Features/NormalEstimation_172.pxi"
include "Features/VFHEstimation_172.pxi"
include "Features/IntegralImageNormalEstimation_172.pxi"
include "Features/MomentOfInertiaEstimation_172.pxi"

# keyPoint
include "KeyPoint/HarrisKeypoint3D_172.pxi"
# execute NG?
# include "KeyPoint/UniformSampling_172.pxi"

# Registration
include "registration/NormalDistributionsTransform.pxi"
###



