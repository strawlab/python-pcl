# -*- coding: utf-8 -*-
from __future__ import print_function
from collections import defaultdict
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
# from Cython.Build import cythonize    # MacOS NG
from setuptools import setup, find_packages, Extension

import subprocess
import numpy
import sys
import platform
import os
import time

import shutil
from ctypes.util import find_library
setup_requires = []
install_requires = [
    'filelock',
    'mock',
    'nose',
    # RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility
    # https://github.com/scikit-image/scikit-image/issues/3655
    # 'numpy>=1.15.1,!=1.50.0',
    # numpy.ufunc size changed, may indicate binary incompatibility. 
    'numpy>=1.16.1,!=1.16.2',
    'Cython>=0.26.0',
]


def pkgconfig(flag):
    # Equivalent in Python 2.7 (but not 2.6):
    # subprocess.check_output(['pkg-config', flag] + pcl_libs).split()
    p = subprocess.Popen(['pkg-config', flag] +
                         pcl_libs, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    # Assume no evil spaces in filenames; unsure how pkg-config would
    # handle those, anyway.
    # decode() is required in Python 3. TODO how do know the encoding?
    return stdout.decode().split()


def pkgconfig_win(flag, cut):
    # Equivalent in Python 2.7 (but not 2.6):
    # subprocess.check_output(['pkg-config', flag] + pcl_libs).split()
    p = subprocess.Popen(['.\\pkg-config\\pkg-config.exe', flag] +
                         pcl_libs, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    # Assume no evil spaces in filenames; unsure how pkg-config would
    # handle those, anyway.
    # decode() is required in Python 3. TODO how do know the encoding?
    # return stdout.decode().split()
    # Windows
    return stdout.decode().replace('\r\n', '').replace('\ ', ' ').replace('/', '\\').split(cut)


if platform.system() == "Windows":
    # Check 32bit or 64bit
    is_64bits = sys.maxsize > 2**32
    # if is_64bits == True

    # environment Value
    for k, v in os.environ.items():
        # print("{key} : {value}".format(key=k, value=v))
        if k == "PCL_ROOT":
            pcl_root = v
            # print(pcl_root)
            # print("%s: find environment PCL_ROOT" % pcl_root)
            break
    else:
        print("cannot find environment PCL_ROOT", file=sys.stderr)
        sys.exit(1)

    # Add environment Value
    for k, v in os.environ.items():
        # print("{key} : {value}".format(key=k, value=v))
        if k == "PKG_CONFIG_PATH":
            pkgconfigstr = v
            break
    else:
        # print("cannot find environment PKG_CONFIG_PATH", file=sys.stderr)
        print("cannot find environment PKG_CONFIG_PATH")
        pkgconfigstr = pcl_root + '\\lib\\pkgconfig;' + pcl_root + \
            '\\3rdParty\\FLANN\\lib\\pkgconfig;' + \
            pcl_root + '\\3rdParty\\Eigen\\lib\\pkgconfig;'
        os.environ["PKG_CONFIG_PATH"] = pcl_root + '\\lib\\pkgconfig;' + pcl_root + \
            '\\3rdParty\\FLANN\\lib\\pkgconfig;' + \
            pcl_root + '\\3rdParty\\Eigen\\lib\\pkgconfig;'

    print("set environment PKG_CONFIG_PATH=%s" % pkgconfigstr)

    # other package(common)
    # BOOST_ROOT
    for k, v in os.environ.items():
        # print("{key} : {value}".format(key=k, value=v))
        if k == "BOOST_ROOT":
            boost_root = v
            break
    else:
        boost_root = pcl_root + '\\3rdParty\\Boost'

    # EIGEN_ROOT
    for k, v in os.environ.items():
        # print("{key} : {value}".format(key=k, value=v))
        if k == "EIGEN_ROOT":
            eigen_root = v
            break
    else:
        eigen_root = pcl_root + '\\3rdParty\\Eigen'

    # FLANN_ROOT
    for k, v in os.environ.items():
        # print("{key} : {value}".format(key=k, value=v))
        if k == "FLANN_ROOT":
            flann_root = v
            break
    else:
        flann_root = pcl_root + '\\3rdParty\\FLANN'

    # QHULL_ROOT
    for k, v in os.environ.items():
        # print("{key} : {value}".format(key=k, value=v))
        if k == "QHULL_ROOT":
            qhull_root = v
            break
    else:
        qhull_root = pcl_root + '\\3rdParty\\Qhull'

    # VTK_DIR
    for k, v in os.environ.items():
        # print("{key} : {value}".format(key=k, value=v))
        if k == "VTK_DIR":
            vtk_root = v
            break
    else:
        vtk_root = pcl_root + '\\3rdParty\\VTK'

    # custom(CUDA)
    # custom(WinPcap)

    # get pkg-config.exe filePath
    pkgconfigPath = os.getcwd() + '\\pkg-config\\pkg-config.exe'
    print(pkgconfigPath)

    # AppVeyor Check
    for k, v in os.environ.items():
        # print("{key} : {value}".format(key=k, value=v))
        if k == "PCL_VERSION":
            pcl_version = '-' + v
            break
    else:
        # Try to find PCL. XXX we should only do this when trying to build or install.
        # in order of preference
        PCL_SUPPORTED = ["-1.9", "-1.8", "-1.7", "-1.6", ""]

        for pcl_version in PCL_SUPPORTED:
            if subprocess.call(['.\\pkg-config\\pkg-config.exe', 'pcl_common%s' % pcl_version]) == 0:
                # if subprocess.call([pkgconfigPath, 'pcl_common%s' % pcl_version]) == 0:
                break
        else:
            print("%s: error: cannot find PCL, tried" %
                  sys.argv[0], file=sys.stderr)
            for version in PCL_SUPPORTED:
                print('    pkg-config pcl_common%s' % version, file=sys.stderr)
            sys.exit(1)

    print(pcl_version)
    # pcl_version = '-1.6'

    # Python Version Check
    info = sys.version_info

    if pcl_version == '-1.6':
        # PCL 1.6.0 python Version == 3.4(>= 3.4?, 2.7 -> NG)
        # Visual Studio 2010
        if info.major == 3 and info.minor == 4:
            boost_version = '1_49'
            vtk_version = '5.8'
            pcl_libs = ["common", "features", "filters", "kdtree", "octree",
                        "registration", "sample_consensus", "search", "segmentation",
                        "surface", "tracking", "visualization"]
            pass
        else:
            print('no building Python Version')
            sys.exit(1)
    elif pcl_version == '-1.7':
        # PCL 1.7.2 python Version >= 3.5
        # Visual Studio 2015
        if info.major == 3 and info.minor >= 5:
            boost_version = '1_57'
            vtk_version = '6.2'
            pass
            # pcl-1.7?
            pcl_libs = ["2d", "common", "features", "filters", "geometry",
                        "io", "kdtree", "keypoints", "ml", "octree", "outofcore", "people",
                        "recognition", "registration", "sample_consensus", "search",
                        "segmentation", "surface", "tracking", "visualization"]
        else:
            print('no building Python Version')
            sys.exit(1)
    elif pcl_version == '-1.8':
        # PCL 1.8.0 python Version >= 3.5
        # Visual Studio 2015/2017
        if info.major == 3 and info.minor >= 5:
            # PCL 1.8.1
            boost_version = '1_64'
            vtk_version = '8.0'
            # pcl-1.8
            # 1.8.1 use 2d required features
            pcl_libs = ["2d", "common", "features", "filters", "geometry",
                        "io", "kdtree", "keypoints", "ml", "octree", "outofcore", "people",
                        "recognition", "registration", "sample_consensus", "search",
                        "segmentation", "stereo", "surface", "tracking", "visualization"]
            pass
        else:
            # if info.major == 2 and info.minor == 7:
            #     import _msvccompiler
            #     import distutils.msvc9compiler
            # 
            #     def find_vcvarsall(version):
            #         # use vc2017 set vcvarsall.bat path
            #         # return "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvarsall.bat"
            #         # return "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvarsall.bat"
            #         vcvarsall, vcruntime = _msvccompiler._find_vcvarsall('x64')
            #         if vcvarsall is not None:
            #             print('set msvc2017/2015 compiler')
            #             print(vcvarsall)
            #             return vcvarsall
            #         else:
            #             print('no set msvc2017/2015 compiler')
            #             return None
            # 
            #     distutils.msvc9compiler.find_vcvarsall = find_vcvarsall
            # 
            #     boost_version = '1_64'
            #     vtk_version = '8.0'
            #     # pcl-1.8
            #     # 1.8.1 use 2d required features
            #     pcl_libs = ["2d", "common", "features", "filters", "geometry",
            #                 "io", "kdtree", "keypoints", "ml", "octree", "outofcore", "people",
            #                 "recognition", "registration", "sample_consensus", "search",
            #                 "segmentation", "stereo", "surface", "tracking", "visualization"]
            # else:
            #     print('no building Python Version')
            #     sys.exit(1)
            print('no building Python Version')
            sys.exit(1)
    elif pcl_version == '-1.9':
        # PCL 1.9.1 python Version >= 3.5
        # Visual Studio 2015/2017
        if info.major == 3 and info.minor >= 5:
            # PCL 1.9.1
            boost_version = '1_68'
            vtk_version = '8.1'
            # pcl-1.9
            # 1.9.1 use 2d required features
            pcl_libs = ["2d", "common", "features", "filters", "geometry",
                        "io", "kdtree", "keypoints", "ml", "octree", "outofcore", "people",
                        "recognition", "registration", "sample_consensus", "search",
                        "segmentation", "stereo", "surface", "tracking", "visualization"]
            pass
        else:
            # if info.major == 2 and info.minor == 7:
            print('no building Python Version')
            sys.exit(1)
    else:
        print('pcl_version Unknown')
        sys.exit(1)

    # Find build/link options for PCL using pkg-config.

    pcl_libs = ["pcl_%s%s" % (lib, pcl_version) for lib in pcl_libs]
    # pcl_libs += ['Eigen3']
    # print(pcl_libs)

    ext_args = defaultdict(list)
    # set include path
    ext_args['include_dirs'].append(numpy.get_include())

    # no use pkg-config
    if pcl_version == '-1.6':
        # 1.6.0
        # boost 1.5.5
        # vtk 5.8
        # + add VTK
        inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version,
                    pcl_root + '\\3rdParty\\Eigen\\include',
                    pcl_root + '\\3rdParty\\Boost\\include',
                    flann_root + '\\include',
                    qhull_root + '\\include',
                    vtk_root + '\\include\\vtk-' + vtk_version]
    elif pcl_version == '-1.7':
        # 1.7.2
        # boost 1.5.7
        # vtk 6.2
        inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version,
                    eigen_root + '\\eigen3', 
                    boost_root + '\\include\\boost-' + boost_version,
                    flann_root + '\\include',
                    qhull_root + '\\include',
                    vtk_root + '\\include\\vtk-' + vtk_version]
    elif pcl_version == '-1.8':
        # 1.8.0
        # boost 1.6.1
        # vtk 7.0
        # 1.8.1/vtk 8.0
        inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version,
                    eigen_root + '\\eigen3', 
                    boost_root + '\\include\\boost-' + boost_version,
                    flann_root + '\\include',
                    qhull_root + '\\include',
                    vtk_root + '\\include\\vtk-' + vtk_version]
    elif pcl_version == '-1.9':
        # 1.9.1
        # boost 1.6.8
        # vtk 8.1?
        # not path set libqhull/libqhull_r(conflict io.h)
        inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version,
                    eigen_root + '\\eigen3', 
                    boost_root + '\\include\\boost-' + boost_version,
                    flann_root + '\\include',
                    qhull_root + '\\include',
                    vtk_root + '\\include\\vtk-' + vtk_version]
    else:
        inc_dirs = []

    for inc_dir in inc_dirs:
        ext_args['include_dirs'].append(inc_dir)

    # for flag in pkgconfig_win('--libs-only-L', '-L'):
    #     print(flag.lstrip().rstrip())
    #     ext_args['library_dirs'].append(flag[2:])

    # for flag in pkgconfig_win('--libs-only-other', '-l'):
    #     print(flag.lstrip().rstrip())
    #     ext_args['extra_link_args'].append(flag)
    # end

    # set library path
    if pcl_version == '-1.6':
        # 3rdParty(+Boost, +VTK)
        lib_dirs = [pcl_root + '\\lib',
                    boost_root + '\\lib',
                    flann_root + '\\lib',
                    qhull_root + '\\lib',
                    vtk_root + '\\lib']
    elif pcl_version == '-1.7':
        # 1.7.2
        # 3rdParty(+Boost, +VTK)
        lib_dirs = [pcl_root + '\\lib',
                    boost_root + '\\lib',
                    flann_root + '\\lib',
                    qhull_root + '\\lib',
                    vtk_root + '\\lib']
    elif pcl_version == '-1.8':
        # 1.8.0
        # 3rdParty(+Boost, +VTK)
        lib_dirs = [pcl_root + '\\lib',
                    boost_root + '\\lib',
                    flann_root + '\\lib',
                    qhull_root + '\\lib',
                    vtk_root + '\\lib']
    elif pcl_version == '-1.9':
        # 1.9.1
        # 3rdParty(+Boost, +VTK)
        lib_dirs = [pcl_root + '\\lib',
                    boost_root + '\\lib',
                    flann_root + '\\lib',
                    qhull_root + '\\lib',
                    vtk_root + '\\lib']
    else:
        lib_dir = []

    for lib_dir in lib_dirs:
        ext_args['library_dirs'].append(lib_dir)

    # OpenNI2?
    # %OPENNI2_REDIST64% %OPENNI2_REDIST%

    if pcl_version == '-1.6':
        # release
        # libreleases = ['pcl_apps_release', 'pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'qhull', 'qhull_p', 'qhull_r', 'qhullcpp']
        # release + vtk5.3?
        libreleases = ['pcl_apps_release', 'pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s']
    elif pcl_version == '-1.7':
        # release
        # libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'qhull', 'qhull_p', 'qhull_r', 'qhullcpp']
        # release + vtk6.2?/6.3?
        libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_outofcore_release', 'pcl_people_release', 'pcl_recognition_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_search_release', 'pcl_segmentation_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'qhull', 'qhull_p', 'qhull_r', 'qhullcpp']
    elif pcl_version == '-1.8':
        # release
        # libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'qhull', 'qhull_p', 'qhull_r', 'qhullcpp']
        # release + vtk7.0
        libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_ml_release', 'pcl_octree_release', 'pcl_outofcore_release', 'pcl_people_release', 'pcl_recognition_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_search_release', 'pcl_segmentation_release', 'pcl_stereo_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'qhull', 'qhull_p', 'qhull_r', 'qhullcpp']
    elif pcl_version == '-1.9':
        # release
        # libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'qhull', 'qhull_p', 'qhull_r', 'qhullcpp']
        # release + vtk8.1?
        libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_ml_release', 'pcl_octree_release', 'pcl_outofcore_release', 'pcl_people_release', 'pcl_recognition_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_search_release', 'pcl_segmentation_release', 'pcl_stereo_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'qhull', 'qhull_p', 'qhull_r', 'qhullcpp']
    else:
        libreleases = []

    for librelease in libreleases:
        ext_args['libraries'].append(librelease)

    # vtk 5.8
    # vtk 6.2/6.3
    # vtk 7.0/8.0
    # vtk 8.1
    if vtk_version == '5.8':
        # pcl1.6 3rdParty
        # vtklibreleases = ['vtkInfovis', 'MapReduceMPI', 'vtkNetCDF', 'QVTK', 'vtkNetCDF_cxx', 'vtkRendering', 'vtkViews', 'vtkVolumeRendering', 'vtkWidgets', 'mpistubs', 'vtkalglib', 'vtkCharts', 'vtkexoIIc', 'vtkexpat', 'vtkCommon', 'vtkfreetype', 'vtkDICOMParser', 'vtkftgl', 'vtkFiltering', 'vtkhdf5', 'vtkjpeg', 'vtkGenericFiltering', 'vtklibxml2', 'vtkGeovis', 'vtkmetaio', 'vtkpng', 'vtkGraphics', 'vtkproj4', 'vtkHybrid', 'vtksqlite', 'vtksys', 'vtkIO', 'vtktiff', 'vtkImaging', 'vtkverdict', 'vtkzlib']
        vtklibreleases = []
    elif vtk_version == '6.3':
        # pcl1.7.2 3rdParty
        # vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLIC-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksys-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version]
        vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkexpat-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkfreetype-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkhdf5-' + vtk_version, 'vtkhdf5_hl-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtkjpeg-' + vtk_version, 'vtkjsoncpp-' + vtk_version, 'vtklibxml2-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkNetCDF-' + vtk_version, 'vtkNetCDF_cxx-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkpng-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLIC-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtktiff-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkzlib-' + vtk_version]
    elif vtk_version == '7.0':
        # pcl_version 1.8.0
        # pcl1.6 3rdParty
        vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkexpat-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkfreetype-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkhdf5-' + vtk_version, 'vtkhdf5_hl-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtkjpeg-' + vtk_version, 'vtkjsoncpp-' + vtk_version, 'vtklibxml2-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkNetCDF-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkpng-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLIC-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtktiff-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkzlib-' + vtk_version]
    elif vtk_version == '8.0':
        # pcl_version 1.8.1
        # vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkfreetype-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkhdf5-' + vtk_version, 'vtkhdf5_hl-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtkjpeg-' + vtk_version, 'vtkjsoncpp-' + vtk_version, 'vtklibxml2-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkNetCDF-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkpng-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLIC-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtktiff-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkzlib-' + vtk_version]
        # vtk8.0
        # all-in-one-package(OpenGL)
        vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkhdf5-' + vtk_version, 'vtkhdf5_hl-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtkjsoncpp-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkNetCDF-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtktiff-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkzlib-' + vtk_version]
        # conda?(OpenGL2)
        # vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkDomainsChemistryOpenGL2-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersPoints-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersPython-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersTopology-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkglew-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOExportOpenGL2-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOTecplotTable-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtklibharu-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkPythonInterpreter-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL2-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PSOpenGL2-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingMatplotlib-' + vtk_version, 'vtkRenderingOpenGL2-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL2-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkWrappingTools-' + vtk_version, 'vtkCommonCorePython35D-8.0', 'vtkWrappingPython35Core-8.0']
    elif vtk_version == '8.1':
        # pcl_version 1.9.0/1.9.1
        # all-in-one-package(OpenGL)
        vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersPoints-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersTopology-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOTecplotTable-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtklibharu-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtknetcdfcpp-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version,  'vtkzlib-' + vtk_version]
        # conda?(OpenGL2)
        # vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkDomainsChemistryOpenGL2-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersPoints-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersPython-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersTopology-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkglew-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOExportOpenGL2-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOTecplotTable-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtklibharu-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtknetcdfcpp-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkPythonInterpreter-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL2-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PSOpenGL2-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingMatplotlib-' + vtk_version, 'vtkRenderingOpenGL2-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL2-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkWrappingTools-' + vtk_version]
    else:
        vtklibreleases = []

    for librelease in vtklibreleases:
        ext_args['libraries'].append(librelease)

    # Note :
    # vtk Version setting

    # use vtk need library(Windows base library)
    # http://public.kitware.com/pipermail/vtkusers/2008-July/047291.html
    win_libreleases = ['kernel32', 'user32', 'gdi32', 'winspool', 'comdlg32',
                       'advapi32', 'shell32', 'ole32', 'oleaut32', 'uuid', 'odbc32', 'odbccp32']
    for win_librelease in win_libreleases:
        ext_args['libraries'].append(win_librelease)

    # http://www.pcl-users.org/error-in-use-PCLVisualizer-td3719235.html
    # Download MSSDKs
    # http://msdn.microsoft.com/en-us/windows/bb980924.aspx
    #
    # http://stackoverflow.com/questions/1236670/how-to-make-opengl-apps-in-64-bits-windows
    # C:\Program Files (x86)\Microsoft SDKs\Windows\7.0\Lib\x64\OpenGL32.lib
    # C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\Lib\x64\OpenGL32.lib

    # Add OpenGL32 .h/.lib
    win_kit_incs = []
    win_kit_libdirs = []
    # using _open, _close, _chsize functions (pcl/io/low_level_io.h)
    # win_kit_libreleases = ['ucrt', 'libucrt']
    # for win_kit_librelease in win_kit_libreleases:
    #    ext_args['libraries'].append(win_kit_librelease)

    if pcl_version == '-1.6':
        if is_64bits == True:
            # win_opengl_libdirs = ['C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\Lib\\x64']
            # AppVeyor
            win_kit_libdirs = [
                'C:\\Program Files\\Microsoft SDKs\\Windows\\v7.1\\Lib\\x64']
        else:
            # win_opengl_libdirs = ['C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\Lib\\win32']
            # AppVeyor
            win_kit_libdirs = [
                'C:\\Program Files\\Microsoft SDKs\\Windows\\v7.1\\Lib\\win32']
    elif pcl_version == '-1.7':
        if is_64bits == True:
            win_kit_libdirs = [
                'C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v8.0A\\Lib\\x64']
        else:
            win_kit_libdirs = [
                'C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v8.0A\\Lib\\win32']
    elif pcl_version == '-1.8':
        if is_64bits == True:
            # already set path
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v8.1A\\Lib\\x64']
            # Windows OS 7?
            # win_kit_incs = ['C:\\Program Files (x86)\\Windows Kits\\8.1\\Include\\shared', 'C:\\Program Files (x86)\\Windows Kits\\8.1\\Include\\um']
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Windows Kits\\8.1\\Lib\\winv6.3\\um\\x64']
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.10240.0\\ucrt\\x64']
            # Windows OS 8/8.1/10?
            # win_kit_10_version = '10.0.10240.0'
            # win_kit_incs = ['C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt', 'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\um']
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt', 'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\um']
            pass
        else:
            # already set path
            # Windows OS 7
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v8.1A\\Lib\\win32']
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Windows Kits\\8.1\\Lib\\winv6.3\\um\\x86']
            # win_kit_incs = ['C:\\Program Files (x86)\\Windows Kits\\8.1\\Include\\shared', 'C:\\Program Files (x86)\\Windows Kits\\8.1\\Include\\um']
            pass
    elif pcl_version == '-1.9':
        if is_64bits == True:
            # win_kit_10_version = '10.0.15063.0'
            # win_kit_incs = ['C:\\Program Files (x86)\\Windows Kits\\10\\Include\\' + win_kit_10_version+ '\\ucrt', 'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\' + win_kit_10_version + '\\um']
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Windows Kits\\10\\Include\\' + win_kit_10_version + '\\ucrt\\x64', 'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\' + win_kit_10_version + '\\um\\x64']
            pass
        else:
            pass
    else:
        pass

    for inc_dir in win_kit_incs:
        ext_args['include_dirs'].append(inc_dir)

    for lib_dir in win_kit_libdirs:
        ext_args['library_dirs'].append(lib_dir)

    win_opengl_libreleases = ['OpenGL32']
    for opengl_librelease in win_opengl_libreleases:
        ext_args['libraries'].append(opengl_librelease)

    # use OpenNI
    # use OpenNI2
    # add environment PATH : pcl/bin, OpenNI2/Tools

    # use CUDA?
    # CUDA_PATH
    # CUDA_PATH_V7_5
    # CUDA_PATH_V8_0
    for k, v in os.environ.items():
        # print("{key} : {value}".format(key=k, value=v))
        if k == "CUDA_PATH":
            cuda_root = v
            break
    else:
        print('No use cuda.')
        pass

    # ext_args['define_macros'].append(('EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET', '1'))
    # define_macros=[('BOOST_NO_EXCEPTIONS', 'None')],
    # debugs = [('EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET', '1'), ('BOOST_NO_EXCEPTIONS', 'None')]
    # _CRT_SECURE_NO_WARNINGS : windows cutr warning no view
    defines = [('EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET',
                '1'), ('_CRT_SECURE_NO_WARNINGS', '1')]
    for define in defines:
        ext_args['define_macros'].append(define)

    # ext_args['extra_compile_args'].append('/DWIN32')
    # ext_args['extra_compile_args'].append('/D_WINDOWS')
    # ext_args['extra_compile_args'].append('/W3')
    # ext_args['extra_compile_args'].append('/GR')
    ext_args['extra_compile_args'].append('/EHsc')
    # FW: Link time errors in RangeImage (with /clr)
    # http://www.pcl-users.org/FW-Link-time-errors-in-RangeImage-with-clr-td3581422.html
    # ext_args['extra_compile_args'].append('/clr:nostdlib')
    # OpenNI2?(+Python3)
    # https://ci.appveyor.com/project/KazuakiM/vim-ms-translator/branch/master
    # ext_args['extra_compile_args'].append('/DDYNAMIC_MSVCRT_DLL=\"msvcr100.dll\"')
    # ext_args['extra_compile_args'].append('/DDYNAMIC_MSVCRT_DLL=\"msvcr100.dll\"')

    # NG
    # ext_args['extra_compile_args'].append('/NODEFAULTLIB:msvcrtd')
    # https://blogs.msdn.microsoft.com/vcblog/2015/03/03/introducing-the-universal-crt/
    # runtime libraries args(default MT?)
    # use all-in-one package on vtk libraries.(use Dynamic?)
    ext_args['extra_compile_args'].append('/MD')
    # ext_args['extra_compile_args'].append('/MDd')
    # custom build module(static build)
    # ext_args['extra_compile_args'].append('/MTd')
    # ext_args['extra_compile_args'].append('/MT')
    # use OpenMP
    # https://stackoverflow.com/questions/7844830/cython-openmp-compiler-flag
    # ext_args['extra_compile_args'].append('/openmp')
    # ext_args['extra_link_args'].append('/openmp')

    # Debug View
    # print(ext_args)

    if pcl_version == '-1.6':
        module = [Extension("pcl._pcl", ["pcl/_pcl.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  # Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_visualization", ["pcl/pcl_visualization_160.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    elif pcl_version == '-1.7':
        module = [Extension("pcl._pcl", ["pcl/_pcl_172.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    elif pcl_version == '-1.8':
        module = [Extension("pcl._pcl", ["pcl/_pcl_180.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # conda
                  # Extension("pcl.pcl_visualization", [
                  #     "pcl/pcl_visualization.pyx", "pcl/vtkInteracterWrapper.cpp"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    elif pcl_version == '-1.9':
        module = [Extension("pcl._pcl", ["pcl/_pcl_190.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # conda
                  # Extension("pcl.pcl_visualization", [
                  #     "pcl/pcl_visualization.pyx", "pcl/vtkInteracterWrapper.cpp"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    else:
        print('no pcl install or pkg-config missed.')
        sys.exit(1)

    # copy the pcl dll to local subfolder so that it can be added to the package through the data_files option
    listDlls = []
    if not os.path.isdir('./dlls'):
        os.mkdir('./dlls')
    for dll in libreleases:
        pathDll = find_library(dll)
        if not pathDll is None:
            shutil.copy2(pathDll, './dlls')
            listDlls.append(os.path.join('.\\dlls', dll+'.dll'))
    # the path is relative to the python root folder
    data_files = [('Lib/site-packages/pcl', listDlls)]

else:
    # Not 'Windows'
    if sys.platform == 'darwin':
        os.environ['ARCHFLAGS'] = ''

    # Try to find PCL. XXX we should only do this when trying to build or install.
    PCL_SUPPORTED = ["-1.9", "-1.8", "-1.7", "-1.6", ""]    # in order of preference

    for pcl_version in PCL_SUPPORTED:
        if subprocess.call(['pkg-config', 'pcl_common%s' % pcl_version]) == 0:
            break
    else:
        print("%s: error: cannot find PCL, tried" %
              sys.argv[0], file=sys.stderr)
        for version in PCL_SUPPORTED:
            print('    pkg-config pcl_common%s' % version, file=sys.stderr)
        sys.exit(1)

    # Find build/link options for PCL using pkg-config.
    # version 1.6
    # pcl_libs = ["common", "features", "filters", "io", "kdtree", "octree",
    #             "registration", "sample_consensus", "search", "segmentation",
    #             "surface", "tracking", "visualization"]
    # version 1.7
    if pcl_version == '-1.7':
        pcl_libs = ["common", "features", "filters", "geometry",
                    "io", "kdtree", "keypoints", "octree", "outofcore", "people",
                    "recognition", "registration", "sample_consensus", "search",
                    "segmentation", "surface", "tracking", "visualization"]
    else:
        # version 1.8
        pcl_libs = ["2d", "common", "features", "filters", "geometry",
                    "io", "kdtree", "keypoints", "ml", "octree", "outofcore", "people",
                    "recognition", "registration", "sample_consensus", "search",
                    "segmentation", "stereo", "surface", "tracking", "visualization"]
    pcl_libs = ["pcl_%s%s" % (lib, pcl_version) for lib in pcl_libs]

    ext_args = defaultdict(list)
    ext_args['include_dirs'].append(numpy.get_include())

    for flag in pkgconfig('--cflags-only-I'):
        ext_args['include_dirs'].append(flag[2:])

    # OpenNI?
    # "-I/usr/include/openni"
    # "-I/usr/include/openni"
    # /usr/include/ni
    ext_args['include_dirs'].append('/usr/include/ni')
    # ext_args['library_dirs'].append()
    # ext_args['libraries'].append()
    # OpenNI2
    ext_args['include_dirs'].append('/usr/include/openni2')

    # VTK use
    if sys.platform == 'darwin':
        # pcl 1.8.1(MacOSX)
        # if pcl_version == '-1.8':
        #     vtk_version = '8.0'
        #     ext_args['include_dirs'].append('/usr/local/include/vtk-' + vtk_version)
        #     ext_args['library_dirs'].append('/usr/local/lib')
        #     ext_args['include_dirs'].append('/usr/local/Cellar/vtk/8.0.1/include')
        #     ext_args['library_dirs'].append('/usr/local/Cellar/vtk/8.0.1/lib')
        if pcl_version == '-1.9':
            # pcl 1.9.1
            # build install?
            # vtk_version = '8.1'
            # vtk_include_dir = os.path.join('/usr/local' ,'include/vtk-8.1')
            # vtk_library_dir = os.path.join('/usr/local', 'lib')
            # homebrew(MacOSX homebrew)
            # (pcl 1.9.1_3)
            # vtk_version = '8.1.2_3'
            # vtk_include_dir = os.path.join('/usr/local/Cellar/vtk', vtk_version ,'include/vtk-8.2')
            # 2019/05/08 check(pcl 1.9.1_4)
            vtk_version = '8.2.0'
            vtk_include_dir = os.path.join('/usr/local/Cellar/vtk', vtk_version ,'include/vtk-8.2')
            vtk_library_dir = os.path.join('/usr/local/Cellar/vtk', vtk_version, 'lib')
        pass
    else:
        # pcl 1.7.0?(Ubuntu 14.04)
        # vtk_version = '5.8'
        # ext_args['include_dirs'].append('/usr/include/vtk-' + vtk_version)
        # ext_args['library_dirs'].append('/usr/lib')
        # pcl 1.7.2(Ubuntu 16.04)(xenial)
        if pcl_version == '-1.7':
            vtk_version = '6.2'
            vtk_include_dir = os.path.join('/usr/include/vtk-' + vtk_version)
            vtk_library_dir = os.path.join('/usr/lib')
        elif pcl_version == '-1.8':
            # pcl 1.8.0/1?(Ubuntu 18.04)(melodic)
            vtk_version = '6.3'
            # pcl 1.8.1?
            # vtk_version = '8.0'
            vtk_include_dir = os.path.join('/usr/include/vtk-' + vtk_version)
            vtk_library_dir = os.path.join('/usr/lib')
        elif pcl_version == '-1.9':
            # pcl 1.9.1
            # build install?
            vtk_version = '8.1'
            vtk_include_dir = os.path.join('/usr/include/vtk-' + vtk_version)
            vtk_library_dir = os.path.join('/usr/lib')
        else:
            pass

    # other
    # pcl 1.9.1(Conda)
    # vtk_version = '8.1'
    # vtk_include_dir = os.path.join(os.environ["PREFIX"] ,'include/vtk-8.1')
    # vtk_library_dir = os.path.join(os.environ["PREFIX"], 'lib')

    ext_args['include_dirs'].append(vtk_include_dir)
    ext_args['library_dirs'].append(vtk_library_dir)

    if vtk_version == '5.8':
        vtklibreleases = ['vtkInfovis', 'MapReduceMPI', 'vtkNetCDF', 'QVTK', 'vtkNetCDF_cxx', 'vtkRendering', 'vtkViews', 'vtkVolumeRendering', 'vtkWidgets', 'mpistubs', 'vtkalglib', 'vtkCharts', 'vtkexoIIc', 'vtkexpat', 'vtkCommon', 'vtkfreetype', 'vtkDICOMParser', 'vtkftgl', 'vtkFiltering', 'vtkhdf5', 'vtkjpeg', 'vtkGenericFiltering', 'vtklibxml2', 'vtkGeovis', 'vtkmetaio', 'vtkpng', 'vtkGraphics', 'vtkproj4', 'vtkHybrid', 'vtksqlite', 'vtksys', 'vtkIO', 'vtktiff', 'vtkImaging', 'vtkverdict', 'vtkzlib']
    elif vtk_version == '6.3':
        vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkexpat-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkfreetype-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkhdf5-' + vtk_version, 'vtkhdf5_hl-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtkjpeg-' + vtk_version, 'vtkjsoncpp-' + vtk_version, 'vtklibxml2-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkNetCDF-' + vtk_version, 'vtkNetCDF_cxx-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkpng-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLIC-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtktiff-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkzlib-' + vtk_version]
    elif vtk_version == '7.0':
        # apt package?(vtk use OpenGL?)
        vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkexpat-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkfreetype-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkhdf5-' + vtk_version, 'vtkhdf5_hl-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtkjpeg-' + vtk_version, 'vtkjsoncpp-' + vtk_version, 'vtklibxml2-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkNetCDF-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkpng-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLIC-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtktiff-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkzlib-' + vtk_version]
    elif vtk_version == '8.0':
        # vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkhdf5-' + vtk_version, 'vtkhdf5_hl-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtkjsoncpp-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkNetCDF-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PSOpenGL2-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtktiff-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkzlib-' + vtk_version]
        # apt package?(vtk use OpenGL?)
        vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkDomainsChemistryOpenGL2-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersPoints-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersPython-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersTopology-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkglew-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOExportOpenGL2-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOTecplotTable-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtklibharu-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkPythonInterpreter-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL2-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingMatplotlib-' + vtk_version, 'vtkRenderingOpenGL2-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL2-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkWrappingTools-' + vtk_version]
    elif vtk_version == '8.1':
        # pcl_version 1.9.1
        # conda or build module, MacOS X
        vtklibreleases = ['vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkDomainsChemistryOpenGL2-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersPoints-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersPython-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersTopology-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkglew-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOExportOpenGL2-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOTecplotTable-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtklibharu-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtknetcdfcpp-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkPythonInterpreter-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL2-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PSOpenGL2-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingMatplotlib-' + vtk_version, 'vtkRenderingOpenGL2-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL2-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkWrappingTools-' + vtk_version]
    else:
        vtklibreleases = []

    for librelease in vtklibreleases:
        ext_args['libraries'].append(librelease)

    for flag in pkgconfig('--cflags-only-other'):
        if flag.startswith('-D'):
            macro, value = flag[2:].split('=', 1)
            ext_args['define_macros'].append((macro, value))
        else:
            ext_args['extra_compile_args'].append(flag)

    # clang?
    # https://github.com/strawlab/python-pcl/issues/129
    # gcc base libc++, clang base libstdc++
    # ext_args['extra_compile_args'].append("-stdlib=libstdc++")
    # ext_args['extra_compile_args'].append("-stdlib=libc++")
    if sys.platform == 'darwin':
        # not use gcc?
        # ext_args['extra_compile_args'].append("-stdlib=libstdc++")
        # clang(min : 10.7?/10.9?)
        # minimum deployment target of OS X 10.9
        ext_args['extra_compile_args'].append("-stdlib=libc++")
        ext_args['extra_compile_args'].append("-mmacosx-version-min=10.9")
        ext_args['extra_link_args'].append("-stdlib=libc++")
        ext_args['extra_link_args'].append("-mmacosx-version-min=10.9")
        # vtk error : not set override function error.
        ext_args['extra_compile_args'].append("-std=c++11")
        # mac os using openmp
        # https://iscinumpy.gitlab.io/post/omp-on-high-sierra/
        # before setting.
        # $ brew install libomp
        # ext_args['extra_compile_args'].append('-fopenmp -Xpreprocessor')
        # ext_args['extra_link_args'].append('-fopenmp -Xpreprocessor -lomp')
        pass
    else:
        ext_args['extra_compile_args'].append("-std=c++11")
        ext_args['library_dirs'].append("/usr/lib/x86_64-linux-gnu/")
        # gcc? use standard library
        # ext_args['extra_compile_args'].append("-stdlib=libstdc++")
        # ext_args['extra_link_args'].append("-stdlib=libstdc++")
        # clang use standard library
        # ext_args['extra_compile_args'].append("-stdlib=libc++")
        # ext_args['extra_link_args'].append("-stdlib=libc++")
        # using openmp
        # ext_args['extra_compile_args'].append('-fopenmp')
        # ext_args['extra_link_args'].append('-fopenmp')
        pass

    for flag in pkgconfig('--libs-only-l'):
        if flag == "-lflann_cpp-gd":
            print(
                "skipping -lflann_cpp-gd (see https://github.com/strawlab/python-pcl/issues/29")
            continue
        ext_args['libraries'].append(flag[2:])

    for flag in pkgconfig('--libs-only-L'):
        ext_args['library_dirs'].append(flag[2:])

    for flag in pkgconfig('--libs-only-other'):
        ext_args['extra_link_args'].append(flag)

    # grabber?
    # -lboost_system
    # ext_args['extra_link_args'].append('-lboost_system')
    # MacOSX?
    # ext_args['extra_link_args'].append('-lboost_system_mt')
    # ext_args['extra_link_args'].append('-lboost_bind')

    # Fix compile error on Ubuntu 12.04 (e.g., Travis-CI).
    ext_args['define_macros'].append(
        ("EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET", "1"))

    if pcl_version == '-1.6':
        module = [Extension("pcl._pcl", ["pcl/_pcl.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  # Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  Extension("pcl.pcl_visualization", ["pcl/pcl_visualization_160.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    elif pcl_version == '-1.7':
        module = [Extension("pcl._pcl", ["pcl/_pcl_172.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    elif pcl_version == '-1.8':
        module = [Extension("pcl._pcl", ["pcl/_pcl_180.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    elif pcl_version == '-1.9':
        module = [Extension("pcl._pcl", ["pcl/_pcl_190.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    else:
        print('no pcl install or pkg-config missed.')
        sys.exit(1)

    listDlls = []
    data_files = None


setup(name='python-pcl',
      description='Python bindings for the Point Cloud Library (PCL). using Cython.',
      url='http://github.com/strawlab/python-pcl',
      version='0.3.0rc1',
      author='John Stowers',
      author_email='john.stowers@gmail.com',
      maintainer='Tooru Oonuma',
      maintainer_email='t753github@gmail.com',
      license='BSD',
      packages=[
          "pcl",
          # "pcl.pcl_visualization",
      ],
      zip_safe=False,
      setup_requires=setup_requires,
      install_requires=install_requires,
      classifiers=[
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      tests_require=['mock', 'nose'],
      ext_modules=module,
      cmdclass={'build_ext': build_ext},
      data_files=data_files
      )
