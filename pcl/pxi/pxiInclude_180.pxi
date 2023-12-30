# -*- coding: utf-8 -*-

### include ###
# common?
include "Vertices.pxi"
include "PointXYZtoPointXYZ.pxi"
# Segmentation
include "Segmentation/Segmentation_180.pxi"
include "Segmentation/SegmentationNormal_180.pxi"
include "Segmentation/EuclideanClusterExtraction_180.pxi"
include "Segmentation/RegionGrowing_180.pxi"
# Filters
include "Filters/StatisticalOutlierRemovalFilter_180.pxi"
include "Filters/VoxelGridFilter_180.pxi"
include "Filters/PassThroughFilter_180.pxi"
include "Filters/ApproximateVoxelGrid_180.pxi"
# Kdtree
# same 1.6 to 1.8
# include "KdTree/KdTree.pxi"
include "KdTree/KdTree_FLANN.pxi"
# Octree
include "Octree/OctreePointCloud_180.pxi"
include "Octree/OctreePointCloud2Buf_180.pxi"
include "Octree/OctreePointCloudSearch_180.pxi"
include "Octree/OctreePointCloudChangeDetector_180.pxi"
# Filters
include "Filters/CropHull_180.pxi"
include "Filters/CropBox_180.pxi"
include "Filters/ProjectInliers_180.pxi"
include "Filters/RadiusOutlierRemoval_180.pxi"
include "Filters/ConditionAnd_180.pxi"
include "Filters/ConditionalRemoval_180.pxi"
# Surface
include "Surface/ConcaveHull_180.pxi"
include "Surface/MovingLeastSquares_180.pxi"
# RangeImage
include "Common/RangeImage/RangeImages_180.pxi"


# Registration
include "registration/GeneralizedIterativeClosestPoint_180.pxi"
include "registration/IterativeClosestPoint_180.pxi"
include "registration/IterativeClosestPointNonLinear_180.pxi"
# SampleConsensus
# same 1.6 to 1.8
include "SampleConsensus/RandomSampleConsensus.pxi"
include "SampleConsensus/SampleConsensusModelPlane.pxi"
include "SampleConsensus/SampleConsensusModelSphere.pxi"
include "SampleConsensus/SampleConsensusModelCylinder.pxi"
include "SampleConsensus/SampleConsensusModelLine.pxi"
include "SampleConsensus/SampleConsensusModelRegistration.pxi"
include "SampleConsensus/SampleConsensusModelStick.pxi"
# Features
include "Features/NormalEstimation_180.pxi"
include "Features/VFHEstimation_180.pxi"
include "Features/IntegralImageNormalEstimation_180.pxi"
include "Features/MomentOfInertiaEstimation_180.pxi"

# keyPoint
include "KeyPoint/HarrisKeypoint3D_180.pxi"
# execute NG?
# include "KeyPoint/UniformSampling_180.pxi"

# Registration
include "registration/NormalDistributionsTransform_180.pxi"
# pcl 1.7.2?
# include "registration/NormalDistributionsTransform.pxi"
# visual
# include "Visualization/PointCloudColorHandlerCustoms.pxi"
###


