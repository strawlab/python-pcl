.. raw:: html

    <a href="https://github.com/strawlab/python-pcl">
    <img style="position: absolute; top: 0; right: 0; border: 0;" src="https://s3.amazonaws.com/github/ribbons/forkme_right_darkblue_121621.png" alt="Fork me on GitHub"></a>

    <script type="text/javascript">
      var _gaq = _gaq || [];
      _gaq.push(['_setAccount', 'UA-3707626-7']);
      _gaq.push(['_trackPageview']);
      (function() {
        var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
        ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
        var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(ga, s);
      })();
    </script>

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
 * registration (ICP, GICP, ICP_NL)

The code tries to follow the Point Cloud API, and also provides helper function
for interacting with NumPy. For example (from tests/test.py)

.. code-block:: python

    import pcl
    import numpy as np
    p = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32))
    seg = p.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    indices, model = seg.segment()

or, for smoothing

.. code-block:: python

    import pcl
    p = pcl.load("C/table_scene_lms400.pcd")
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k (50)
    fil.set_std_dev_mul_thresh (1.0)
    fil.filter().to_file("inliers.pcd")

Point clouds can be viewed as NumPy arrays, so modifying them is possible
using all the familiar NumPy functionality:

.. code-block:: python

    import numpy as np
    import pcl
    p = pcl.PointCloud(10)  # "empty" point cloud
    a = np.asarray(p)       # NumPy view on the cloud
    a[:] = 0                # fill with zeros
    print(p[3])             # prints (0.0, 0.0, 0.0)
    a[:, 0] = 1             # set x coordinates to 1
    print(p[3])             # prints (1.0, 0.0, 0.0)

More samples can be found in the `examples directory <https://github.com/strawlab/python-pcl/tree/master/examples>`_,
and in the `unit tests <https://github.com/strawlab/python-pcl/blob/master/tests/test.py>`_.

This work was supported by `Strawlab <http://strawlab.org/>`_.

Requirements
------------

This release has been tested on Linux Ubuntu 14.04 with

 * Python 2.7.6, 3.4.0, 3.5.2
 * pcl 1.7.2
 * Cython 0.25.2

and MacOS with
 * Python 2.7.6, 3.4.0, 3.5.2
 * pcl 1.8.0
 * Cython 0.25.2

and Windows with
 * (Miniconda/Anaconda) - Python 3.4
 * pcl 1.6.0(VS2010)
 * Cython 0.25.2
 * Gtk+

and Windows with
 * (Miniconda/Anaconda) - Python 3.5
 * pcl 1.7.2(VS2015)
 * Cython 0.25.2
 * Gtk+

and Windows with
 * (Miniconda/Anaconda) - Python 3.5
 * pcl 1.8.0(VS2015)
 * Cython 0.25.2
 * Gtk+

`Visual Studio 2015 C++ Compiler Tools <http://landinghub.visualstudio.com/visual-cpp-build-tools>`_ 

`Python Version use VisualStudio Compiler <https://wiki.python.org/moin/WindowsCompilers>`_ 

`Windows Gtk+ Download <http://win32builder.gnome.org/>`_ 

Copy bin Folder to pkg-config Folder

set Environment variable

1. PCL_ROOT
    $(PCL Install FolderPath)

2. PATH
    (pcl 1.6.0, 1.7.2?)
    $(PCL_ROOT)/bin/;$(OPEN_NI_ROOT)/Tools;$(VTK_ROOT)/bin;

    (pcl 1.8.0)
    $(PCL_ROOT)/bin/;$(OPEN_NI2_ROOT)/Tools;$(VTK_ROOT)/bin;

* use Cython 0.25.2 reason
* `override method support version <http://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html/>`_

A note about types
------------------

Point Cloud is a heavily templated API, and consequently mapping this into
Python using Cython is challenging. 

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

For deficiencies in this documentation, please consult the
`PCL API docs <http://docs.pointclouds.org/trunk/index.html>`_, and the
`PCL tutorials <http://pointclouds.org/documentation/tutorials/>`_.

.. automodule:: pcl
   :members:
   :undoc-members:


