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

    <div align="center"><img src="docs/image/pcl_logo_958x309.png" width="309"/></div>

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
============

This release has been tested on Linux Ubuntu 14.04 with

 * Python 2.7.6, 3.4.0, 3.5.2
 * pcl 1.7.0
 * Cython <= 0.25.2

and MacOS with

 * Python 2.7.6, 3.4.0, 3.5.2
 * pcl 1.8.1(use homebrew)
 * Cython <= 0.25.2

and Windows with

 * (Miniconda/Anaconda) - Python 3.4
 * pcl 1.6.0(VS2010)
 * Cython <= 0.25.2
 * Gtk+

and Windows with

 * (Miniconda/Anaconda) - Python 3.5
 * pcl 1.8.1(VS2015)
 * Cython <= 0.25.2
 * Gtk+

Installation
============

Linux(Ubuntu)
-------------

before Install module
^^^^^^^^^^^^^^^^^^^^^

    PCL 1.7.0 and Ubuntu14.04 (use apt-get)

        1. Install PCL Module.

        .. code-block:: none

            sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl -y

            sudo apt-get update -y

            sudo apt-get install libpcl-all -y


    PCL 1.7.2 and Ubuntu16.04 (use Debian package)

        1. Install PCL Module.?

        .. code-block:: none

            sudo apt-get update -y

            sudo apt-get install build-essential devscripts

            dget -u https://launchpad.net/ubuntu/+archive/primary/+files/pcl_1.7.2-14ubuntu1.16.04.1.dsc

            cd pcl-1.7.2

            sudo dpkg-buildpackage -r -uc -b

            sudo dpkg -i pcl_*.deb

            * current add ppa 
              (sudo add-apt-repository -remove ppa:v-launchpad-jochen-sprickerhof-de/pcl -y)

            Reference `here <https://launchpad.net/ubuntu/xenial/+package/pcl-tools>`_.


    PCL 1.8.0 and Ubuntu16.04(build module)([CI Test Timeout])

        1. Build Module

            Reference `here <https://askubuntu.com/questions/916260/how-to-install-point-cloud-library-v1-8-pcl-1-8-0-on-ubuntu-16-04-2-lts-for>`_.

MacOSX
------

before Install module
^^^^^^^^^^^^^^^^^^^^^

        Case1. use homebrew(PCL 1.8.1 - 2017/11/13 current)

        1. Install PCL Module.

            .. code-block:: none

            brew tap homebrew/science

            brew install pcl

Warning:
   
   Current Installer (2017/10/02) Not generated pcl-2d-1.8.pc file.(Issue #119)
   
   Reference PointCloudLibrary Issue.
   
       `Pull request 1679 <https://github.com/PointCloudLibrary/pcl/pull/1679>`_.
   
       `Issue 1978 <https://github.com/PointCloudLibrary/pcl/issues/1978>`_.

circumvent:

    copy travis/pcl-2d-1.8.pc file to /usr/local/lib/pkgconfig folder.

Windows
-------

Using pip with a precompiled wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	This is the simpliest method on windows. The wheel contains the PCL binaries and thus you do not need to install the original PCL library.
	
	1. Go in the history on the `appveyor page <https://ci.appveyor.com/project/Sirokujira/python-pcl-iju42/history>`_
	2. Click on the last successful revision (green) and click on the job corresponding to your python version 
	3. Go in the artfacts section for that job and download the wheel (the file with extension whl)
	4. In the command line, move to your download folder and run the following command (replacing XXX by the right string)	
	
.. code-block:: none

			pip install python_pcl-XXX.whl
	
		
Compiling the binding from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^		

	If the method using the procompiled wheel does not work you can compile the binding from the source.
	
before Install module
~~~~~~~~~~~~~~~~~~~~~

        Case1. use PCL 1.6.0 

            `Windows SDK 7.1 <http://www.microsoft.com/download/en/details.aspx?id=8279>`_

            `PCL All-In-One Installer <http://pointclouds.org/downloads/windows.html>`_
                
                `32 bit <http://sourceforge.net/projects/pointclouds/files/1.6.0/PCL-1.6.0-AllInOne-msvc2010-win32.exe/download>`_
                
                `64 bit <http://sourceforge.net/projects/pointclouds/files/1.6.0/PCL-1.6.0-AllInOne-msvc2010-win64.exe/download>`_

            OpenNI2[(PCL Install FolderPath)\\3rdParty\\OpenNI\\OpenNI-(win32/x64)-1.3.2-Dev.msi]

        Case2. use 1.8.1

            `Visual Studio 2015 C++ Compiler Tools <https://www.visualstudio.com/vs/older-downloads/>`_ 

            `PCL All-In-One Installer <https://github.com/PointCloudLibrary/pcl/releases/>`_
                
                `32 bit <https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.8.1/PCL-1.8.1-AllInOne-msvc2015-win32.exe>`_
                
                `64 bit <https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.8.1/PCL-1.8.1-AllInOne-msvc2015-win64.exe>`_

            OpenNI2[(PCL Install FolderPath)\\3rdParty\\OpenNI2\\OpenNI-Windows-(win32/x64)-2.2.msi]

        Common setting            

            `Windows Gtk+ Download <http://win32builder.gnome.org/>`_                   

                Download file unzip. Copy bin Folder to pkg-config Folder                  
                 
                or execute powershell file [Install-GTKPlus.ps1].

`Python Version use VisualStudio Compiler <https://wiki.python.org/moin/WindowsCompilers>`_ 

set before Environment variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    1. PCL_ROOT

        set PCL_ROOT=$(PCL Install FolderPath)

    2. PATH

        (pcl 1.6.0)
        set PATH=$(PCL_ROOT)/bin/;$(OPEN_NI_ROOT)/Tools;$(VTK_ROOT)/bin;%PATH%

        (pcl 1.8.1)
        set PATH=$(PCL_ROOT)/bin/;$(OPEN_NI2_ROOT)/Tools;$(VTK_ROOT)/bin;%PATH%

Common setting
--------------

1. pip module install.

.. code-block:: none

    pip install --upgrade pip
    
    pip install cython==0.25.2
    
    pip install numpy

2. instal python module

.. code-block:: none

    python setup.py build_ext -i
    
    python setup.py install


Build & Test Status
===================

windows(1.6.0/1.8.1)

    .. image:: https://ci.appveyor.com/api/projects/status/w52fee7j22q211cm/branch/master?svg=true
        :target: https://ci.appveyor.com/project/Sirokujira/python-pcl-iju42

Mac OSX(1.8.1)/Ubuntu14.04(1.7.0)

    .. image:: https://travis-ci.org/strawlab/python-pcl.svg?branch=master
        :target: https://travis-ci.org/strawlab/python-pcl


A note about types
------------------

Point Cloud is a heavily templated API, and consequently mapping this into
Python using Cython is challenging. 

It is written in Cython, and implements enough hard bits of the API
(from Cythons perspective, i.e the template/smart_ptr bits)  to
provide a foundation for someone wishing to carry on.


API Documentation
=================


For deficiencies in this documentation, please consult the
`PCL API docs <http://docs.pointclouds.org/trunk/index.html>`_, and the
`PCL tutorials <http://pointclouds.org/documentation/tutorials/>`_.



