.. raw:: html

   <a href="https://github.com/strawlab/python-pcl">
   <img style="position: absolute; top: 0; right: 0; border: 0;" src="https://s3.amazonaws.com/github/ribbons/forkme_right_darkblue_121621.png" alt="Fork me on GitHub"></a>

Introduction
============

This is a small python binding to the `pointcloud <http://pointclouds.org/>`_ library.
Currently, the following parts of the API are wrapped (all methods operate on PointXYZ)
point types

 * I/O and integration; saving and loading PCD files
 * segmentation
 * SAC
 * smoothing
 * filtering

The code tries to follow the Point Cloud API, and also provides helper function
for interacting with numpy. For example (from tests/test.py)

.. code-block:: python

    import pcl
    p = pcl.PointCloud()
    p.from_array(np.array([[1,2,3],[3,4,5]], dtype=np.float32)))
    seg = self.p.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    indices, model = seg.segment()

or, for smoothing

.. code-block:: python

    import pcl
    p = pcl.PointCloud()
    p.from_file("C/table_scene_lms400.pcd")
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k (50)
    fil.set_std_dev_mul_thresh (1.0)
    fil.filter().to_file("inliers.pcd")

More samples can be found in the `examples directory <https://github.com/strawlab/python-pcl/tree/master/examples>`_,
and in the `unit tests <https://github.com/strawlab/python-pcl/blob/master/tests/test.py>`_.

This work was supported by `Strawlab <http://strawlab.org/>`_.

Requirements
------------

This release has been tested with

 * pcl 1.5.1
 * Cython 0.16

A note about types
------------------

Point Cloud is a heavily templated API, and consequently mapping this into python
using Cython is challenging. 

It is written in Cython, and implements enough hard bits of the API
(from Cythons perspective, i.e the template/smart_ptr bits)  to
provide a foundation for someone wishing to carry on.


API Documentation
=================

.. autosummary::
   pcl.PointCloud
   pcl.Segmentation
   pcl.SegmentationNormal
   pcl.StatisticalOutlierRemovalFilter
   pcl.MovingLeastSquares
   pcl.PassThroughFilter
   pcl.VoxelGridFilter

For deficiencies in this documentation, please consule the
`PCL API docs <http://docs.pointclouds.org/trunk/index.html>`_, and the
`PCL tutorials <http://pointclouds.org/documentation/tutorials/>`_.

.. automodule:: pcl
   :members:
   :undoc-members:

