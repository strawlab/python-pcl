Installation Guide
==================

.. contents:: :local:

Recommended Environments
------------------------

We recommend these Linux distributions.

* `Ubuntu <http://www.ubuntu.com/>`_ 14.04/16.04 LTS 64bit
* `MacOS <https://www.apple.com/macos/>`_ 10.10/10.11/10.12
* `Windows <https://www.microsoft.com/>`_ 7/8.1/10 64bit

The following versions of Python can be used: 2.7.6+, 3.5.1+, and 3.6.0+.

.. note::

   We are testing python-pcl automatically with Jenkins, where all the above *recommended* environments are tested.

python-pcl is supported on Python 2.7.6+, 3.5.1+, 3.6.0+.
python-pcl uses C++ compiler such as g++.
You need to install it before installing python-pcl.
This is typical installation method for each platform::

  # Ubuntu 14.04
  $ apt-get install g++

  # MacOS
  brew update >/dev/null
  brew install homebrew/boneyard/pyenv-pip-rehash
  brew tap homebrew/science
  brew search versions/pcl
  brew install pcl --without-qt

  # Windows
  `Gtk+ <http://win32builder.gnome.org/>`_
  `Visual Studio 2015 C++ Compiler Tools <http://landinghub.visualstudio.com/visual-cpp-build-tools>`_

  set Environment variable
  1.PCL_ROOT
  $(PCL Install FolderPath)

  2.PATH
  (pcl 1.6.0) $(PCL_ROOT)/bin/;$(OPEN_NI_ROOT)/Tools;$(VTK_ROOT)/bin;
  (pcl 1.8.1) $(PCL_ROOT)/bin/;$(OPEN_NI2_ROOT)/Tools;$(VTK_ROOT)/bin;

If you use old ``setuptools``, upgrade it::

  $ pip install -U setuptools


Dependencies
------------

Before installing python-pcl, we recommend to upgrade ``setuptools`` if you are using an old one::

  $ pip install -U setuptools

The following Python packages are required to install python-pcl.
The latest version of each package will automatically be installed if missing.

* `PointCloudLibrary <http://www.numpy.org/>`_ 1.6 1.7 1.8
* `NumPy <http://www.numpy.org/>`_ 1.9, 1.10, 1.11, 1.12, 1.13
* `Cython <http://www.numpy.org/>`_ 0.25.2

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
~~~~~~~~~~~~~~~~~~~~~~~~

The tarball of the source tree is available via ``pip download python-pcl`` or from `the release notes page <https://github.com/pfnet/python-pcl/releases>`_.
You can use ``setup.py`` to install python-pcl from the tarball::

  $ tar zxf python-pcl-x.x.x.tar.gz
  $ cd python-pcl-x.x.x
  $ python setup.py install

You can also install the development version of python-pcl from a cloned Git repository::

  $ git clone https://github.com/google/python-pcl.git
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

Install python-pcl with pcl.dll
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Install python-pcl for developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

python-pcl uses Cython (>=0.25).
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

