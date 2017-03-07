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
include "Octree/OctreePointCloud.pxi"
include "Octree/OctreePointCloud2Buf.pxi"
include "Octree/OctreePointCloudSearch.pxi"
include "Octree/OctreePointCloudChangeDetector.pxi"
include "Vertices.pxi"
include "Filters/CropHull.pxi"
include "Filters/CropBox.pxi"
include "Filters/ProjectInliers.pxi"
include "Filters/RadiusOutlierRemoval.pxi"
include "Filters/ConditionAnd.pxi"
include "Filters/ConditionalRemoval.pxi"
include "Surface/ConcaveHull.pxi"
include "Common/RangeImage/RangeImages.pxi"
# Registration
include "registration/GeneralizedIterativeClosestPoint.pxi"
include "registration/IterativeClosestPoint.pxi"
include "registration/IterativeClosestPointNonLinear.pxi"
# pcl 1.7.2?
# include "registration/NormalDistributionsTransform.pxi"

# include "Visualization/PointCloudColorHandlerCustoms.pxi"

# Features
include "Features/NormalEstimation.pxi"
include "Features/VFHEstimation.pxi"
include "Features/IntegralImageNormalEstimation.pxi"

# keyPoint
include "KeyPoint/HarrisKeypoint3D.pxi"
include "KeyPoint/UniformSampling.pxi"

