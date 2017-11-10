# -*- coding: utf-8 -*-

### include ###
# common?
include "Vertices.pxi"
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
# Kdtree
# same 1.6 ～ 1.8
# include "KdTree/KdTree.pxi"
include "KdTree/KdTree_FLANN.pxi"
# Octree
include "Octree/OctreePointCloud.pxi"
include "Octree/OctreePointCloud2Buf.pxi"
include "Octree/OctreePointCloudSearch.pxi"
include "Octree/OctreePointCloudChangeDetector.pxi"
# Filters
include "Filters/CropHull.pxi"
include "Filters/CropBox.pxi"
include "Filters/ProjectInliers.pxi"
include "Filters/RadiusOutlierRemoval.pxi"
include "Filters/ConditionAnd.pxi"
include "Filters/ConditionalRemoval.pxi"
# Surface
include "Surface/ConcaveHull.pxi"
include "Surface/MovingLeastSquares.pxi"
# RangeImage
include "Common/RangeImage/RangeImages.pxi"

# Registration
include "registration/GeneralizedIterativeClosestPoint.pxi"
include "registration/IterativeClosestPoint.pxi"
include "registration/IterativeClosestPointNonLinear.pxi"
# SampleConsensus
# same 1.6 ～ 1.8
include "SampleConsensus/RandomSampleConsensus.pxi"
include "SampleConsensus/SampleConsensusModelPlane.pxi"
include "SampleConsensus/SampleConsensusModelSphere.pxi"
include "SampleConsensus/SampleConsensusModelCylinder.pxi"
include "SampleConsensus/SampleConsensusModelLine.pxi"
include "SampleConsensus/SampleConsensusModelRegistration.pxi"
include "SampleConsensus/SampleConsensusModelStick.pxi"
# Features
include "Features/NormalEstimation.pxi"
include "Features/VFHEstimation.pxi"
include "Features/IntegralImageNormalEstimation.pxi"

# keyPoint
include "KeyPoint/HarrisKeypoint3D.pxi"
include "KeyPoint/UniformSampling.pxi"
# pcl 1.7.2?
# include "registration/NormalDistributionsTransform.pxi"
# visual
# include "Visualization/PointCloudColorHandlerCustoms.pxi"
###


