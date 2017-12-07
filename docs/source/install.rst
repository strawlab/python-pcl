Installation Guide
==================

.. contents:: :local:

Recommended Environments
------------------------

We recommend Windows and these Linux distributions.

* `Ubuntu <http://www.ubuntu.com/>`_ 14.04 64bit
* `MacOS <https://www.apple.com/macos/>`_ 10.10/10.11/10.12
* `Windows <https://www.microsoft.com/>`_ 7/8.1/10 64bit

The following versions of Python can be used: 2.7.6+, 3.5.1+, and 3.6.0+.

python-pcl is supported on Python 2.7.6+, 3.4.0, 3.5.0+, 3.6.0+.
python-pcl uses C++ compiler such as g++.
You need to install it before installing python-pcl.
This is typical installation method for each platform::

  Linux(Ubuntu)

  PCL 1.7.0(use apt-get)

  1.Install PCL Module.
  sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl -y

  sudo apt-get update -y

  sudo apt-get install libpcl-all -y

 
  PCL 1.8.0 (build module)([CI Test Timeout])


  1.Build Module


  Reference here.


  MacOSX

  use homebrew

  1.Install PCL Module.
  brew tap homebrew/science

  brew install pcl


  Warning:


  Current Installer (2017/10/02) Not generated pcl-2d-1.8.pc file.(Issue #119)

  Reference PointCloudLibrary Issue.


  Pull qequests 1679.

  Issue 1978.

  circumvent:

  copy travis/pcl-2d-1.8.pc file to /usr/local/lib/pkgconfig folder.
 
  Windows
 
  before Install module


  Case1. use PCL 1.6.0

  Windows SDK 7.1

              PCL All-In-One Installer

              32 bit



  64 bit

  OpenNI2[(PCL Install FolderPath)\3rdParty\OpenNI\OpenNI-(win32/x64)-1.3.2-Dev.msi]

  Case2. use 1.8.1

              Visual Studio 2015 C++ Compiler Tools

              PCL All-In-One Installer

              32 bit

              64 bit

  OpenNI2[(PCL Install FolderPath)\3rdParty\OpenNI2\OpenNI-Windows-(win32/x64)-2.2.msi]

          Common setting


  Windows Gtk+ Download


  Download file unzip. Copy bin Folder to pkg-config Folder

  or execute powershell file [Install-GTKPlus.ps1].

  Python Version use VisualStudio Compiler
 
  set before Environment variable


  1.PCL_ROOT


  set PCL_ROOT=$(PCL Install FolderPath)


  2.PATH


  (pcl 1.6.0) set PATH=$(PCL_ROOT)/bin/;$(OPEN_NI_ROOT)/Tools;$(VTK_ROOT)/bin;%PATH%

  (pcl 1.8.1) set PATH=$(PCL_ROOT)/bin/;$(OPEN_NI2_ROOT)/Tools;$(VTK_ROOT)/bin;%PATH%


If you use old ``setuptools``, upgrade it::

  $ pip install -U setuptools


Dependencies
------------

Before installing python-pcl, we recommend to upgrade ``setuptools`` if you are using an old one::

  $ pip install -U setuptools

The following Python packages are required to install python-pcl.
The latest version of each package will automatically be installed if missing.

* `PointCloudLibrary <http://pointclouds.org/>`_ 1.6.x 1.7.x 1.8.x
* `NumPy <http://www.numpy.org/>`_ 1.9, 1.10, 1.11, 1.12, 1.13
* `Cython <http://cython.readthedocs.io/en/latest/index.html>`_ >=0.25.2

Install python-pcl
------------------

Install python-pcl via pip
~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend to install python-pcl via pip::

  $ pip install python-pcl

.. note::

   All optional PointCloudLibrary related libraries, need to be installed before installing python-pcl.
   After you update these libraries, please reinstall python-pcl because you need to compile and link to the newer version of them.


Install python-pcl from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tarball of the source tree is available via ``pip download python-pcl`` or from `the release notes page <https://github.com/strawlab/python-pcl/releases>`_.
You can use ``setup.py`` to install python-pcl from the tarball::

  $ tar zxf python-pcl-x.x.x.tar.gz
  $ cd python-pcl-x.x.x
  $ python setup.py install

You can also install the development version of python-pcl from a cloned Git repository::

  $ git clone https://github.com/strawlab/python-pcl.git
  $ cd pcl/Python
  $ python setup.py install


.. _install_error:

When an error occurs...
~~~~~~~~~~~~~~~~~~~~~~~

Use ``-vvvv`` option with ``pip`` command.
That shows all logs of installation.
It may help you::

  $ pip install python-pcl -vvvv


.. _install_PointCloudLibrary:


Install python-pcl for developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

python-pcl uses Cython (>=0.25.2).
Developers need to use Cython to regenerate C++ sources from ``pyx`` files.
We recommend to use ``pip`` with ``-e`` option for editable mode::

  $ pip install -U cython
  $ cd /path/to/python-pcl/source
  $ pip install -e .

Users need not to install Cython as a distribution package of python-pcl only contains generated sources.


Uninstall python-pcl
--------------------

Use pip to uninstall python-pcl::

  $ pip uninstall python-pcl

.. note::

   When you upgrade python-pcl, ``pip`` sometimes install the new version without removing the old one in ``site-packages``.
   In this case, ``pip uninstall`` only removes the latest one.
   To ensure that python-pcl is completely removed, run the above command repeatedly until ``pip`` returns an error.


Upgrade python-pcl
------------------

Just use ``pip`` with ``-U`` option::

  $ pip install -U python-pcl


Reinstall python-pcl
--------------------

If you want to reinstall python-pcl, please uninstall python-pcl and then install it.
We recommend to use ``--no-cache-dir`` option as ``pip`` sometimes uses cache::

  $ pip uninstall python-pcl
  $ pip install python-pcl --no-cache-dir

When you install python-pcl without PointCloudLibrary, and after that you want to use PointCloudLibrary, please reinstall python-pcl.
You need to reinstall python-pcl when you want to upgrade PointCloudLibrary.


FAQ
---

