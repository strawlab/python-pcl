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
    'nose',
    'numpy',
    'Cython>=0.25.2',
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
    # return stdout.decode().replace('\r\n', '').replace('\ ', ' ').replace('/', '\\').split(cut)
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
        PCL_SUPPORTED = ["-1.8", "-1.7", "-1.6", ""]    # in order of preference

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
            # pcl-1.8
            # 1.8.1 use 2d required features
            pcl_libs = ["2d", "common", "features", "filters", "geometry",
                        "io", "kdtree", "keypoints", "ml", "octree", "outofcore", "people",
                    "recognition", "registration", "sample_consensus", "search",
                    "segmentation", "stereo", "surface", "tracking", "visualization"]
        else:
            print('no building Python Version')
            sys.exit(1)
    elif pcl_version == '-1.8':
        # PCL 1.8.0 python Version >= 3.5
        # Visual Studio 2015
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

    # Get setting pkg-config
    # add include headers
    # use pkg-config
    # print('test')
    # for flag in pkgconfig_win('--cflags-only-I', '-I'):
    #     print(flag.lstrip().rstrip())
    # print('test')
    #     ext_args['include_dirs'].append(flag.lstrip().rstrip())

    # no use pkg-config
    if pcl_version == '-1.6':
        # 1.6.0
        # boost 1.5.5
        # vtk 5.8
        # ext_args['include_dirs'].append([pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\Boost\\include', pcl_root + '\\3rdParty\\FLANN\include'])
        # inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\Boost\\include', pcl_root + '\\3rdParty\\FLANN\include']
        # NG
        # inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\FLANN\\include']
        # 3rdParty
        # inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\Boost\\include', pcl_root + '\\3rdParty\\FLANN\\include']
        # + add VTK
        inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\Boost\\include', pcl_root + '\\3rdParty\\FLANN\\include', pcl_root + '\\3rdParty\\VTK\\include\\vtk-' + vtk_version]
    elif pcl_version == '-1.7':
        # 1.7.2
        # boost 1.5.7
        # vtk 6.2
        inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\\Eigen\\eigen3', pcl_root + '\\3rdParty\\Boost\\include\\boost-' + boost_version, pcl_root + '\\3rdParty\\FLANN\\include', pcl_root + '\\3rdParty\\VTK\\include\\vtk-' + vtk_version]
    elif pcl_version == '-1.8':
        # 1.8.0
        # boost 1.6.1
        # vtk 7.0
        inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\\Eigen\\eigen3', pcl_root + '\\3rdParty\\Boost\\include\\boost-' + boost_version, pcl_root + '\\3rdParty\\FLANN\\include', pcl_root + '\\3rdParty\\VTK\\include\\vtk-' + vtk_version]
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
        lib_dirs = [pcl_root + '\\lib', pcl_root + '\\3rdParty\\Boost\\lib', pcl_root + '\\3rdParty\\FLANN\\lib', pcl_root + '\\3rdParty\\VTK\lib\\vtk-5.8']
        # extern -> NG?
    elif pcl_version == '-1.7':
        # 1.7.2
        # 3rdParty(+Boost, +VTK)
        lib_dirs = [pcl_root + '\\lib', pcl_root + '\\3rdParty\\Boost\\lib', pcl_root + '\\3rdParty\\FLANN\\lib', pcl_root + '\\3rdParty\\VTK\lib']
    elif pcl_version == '-1.8':
        # 1.8.0
        # 3rdParty(+Boost, +VTK)
        lib_dirs = [pcl_root + '\\lib', pcl_root + '\\3rdParty\\Boost\\lib', pcl_root + '\\3rdParty\\FLANN\\lib', pcl_root + '\\3rdParty\\VTK\lib']
    else:
        lib_dir = []

    for lib_dir in lib_dirs:
        ext_args['library_dirs'].append(lib_dir)

    # OpenNI2?
    # %OPENNI2_REDIST64% %OPENNI2_REDIST%

    # set compiler flags
    # for flag in pkgconfig_win('--cflags-only-other'):
    #     if flag.startswith('-D'):
    #         macro, value = flag[2:].split('=', 1)
    #         ext_args['define_macros'].append((macro, value))
    #     else:
    #         ext_args['extra_compile_args'].append(flag)
    # 
    # for flag in pkgconfig_win('--libs-only-l', '-l'):
    #     if flag == "-lflann_cpp-gd":
    #         print("skipping -lflann_cpp-gd (see https://github.com/strawlab/python-pcl/issues/29")
    #         continue
    #     ext_args['libraries'].append(flag.lstrip().rstrip())

    if pcl_version == '-1.6':
        # release
        # libreleases = ['pcl_apps_release', 'pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s']
        # release + vtk
        libreleases = ['pcl_apps_release', 'pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'vtkInfovis', 'MapReduceMPI', 'vtkNetCDF', 'QVTK', 'vtkNetCDF_cxx', 'vtkRendering', 'vtkViews', 'vtkVolumeRendering', 'vtkWidgets', 'mpistubs', 'vtkalglib', 'vtkCharts', 'vtkexoIIc', 'vtkexpat', 'vtkCommon', 'vtkfreetype', 'vtkDICOMParser', 'vtkftgl', 'vtkFiltering', 'vtkhdf5', 'vtkjpeg', 'vtkGenericFiltering', 'vtklibxml2', 'vtkGeovis', 'vtkmetaio', 'vtkpng', 'vtkGraphics', 'vtkproj4', 'vtkHybrid', 'vtksqlite', 'vtksys', 'vtkIO', 'vtktiff', 'vtkImaging', 'vtkverdict', 'vtkzlib']

        # add boost
        # dynamic lib
        # libreleases = ['pcl_apps_release', 'pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'boost_date_time-vc100-mt-1_47', 'boost_filesystem-vc100-mt-1_49', 'boost_graph-vc100-mt-1_49', 'boost_graph_parallel-vc100-mt-1_49', 'boost_iostreams-vc100-mt-1_49', 'boost_locale-vc100-mt-1_49', 'boost_math_c99-vc100-mt-1_49', 'boost_math_c99f-vc100-mt-1_49', 'boost_math_tr1-vc100-mt-1_49', 'boost_math_tr1f-vc100-mt-1_49', 'boost_mpi-vc100-mt-1_49', 'boost_prg_exec_monitor-vc100-mt-1_49', 'boost_program_options-vc100-mt-1_49', 'boost_random-vc100-mt-1_49', 'boost_regex-vc100-mt-1_49', 'boost_serialization-vc100-mt-1_49', 'boost_signals-vc100-mt-1_49', 'boost_system-vc100-mt-1_49', 'boost_thread-vc100-mt-1_49', 'boost_timer-vc100-mt-1_49', 'boost_unit_test_framework-vc100-mt-1_49', 'boost_wave-vc100-mt-1_49', 'boost_wserialization-vc100-mt-1_49']
        # static lib
        # boost_chrono-vc100-mt-1_49 -> NG(1.47/1.49)
        # boost_date_time-vc100-mt-1_49.lib -> NG
        # libreleases = ['pcl_apps_release', 'pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'libboost_date_time-vc100-mt-1_49', 'libboost_filesystem-vc100-mt-1_49', 'libboost_graph_parallel-vc100-mt-1_49', 'libboost_iostreams-vc100-mt-1_49', 'libboost_locale-vc100-mt-1_49', 'libboost_math_c99-vc100-mt-1_49', 'libboost_math_c99f-vc100-mt-1_49', 'libboost_math_tr1-vc100-mt-1_49', 'libboost_math_tr1f-vc100-mt-1_49', 'libboost_mpi-vc100-mt-1_49', 'libboost_prg_exec_monitor-vc100-mt-1_49', 'libboost_program_options-vc100-mt-1_49', 'libboost_random-vc100-mt-1_49', 'libboost_regex-vc100-mt-1_49', 'libboost_serialization-vc100-mt-1_49', 'libboost_signals-vc100-mt-1_49', 'libboost_system-vc100-mt-1_49', 'libboost_test_exec_monitor-vc100-mt-1_49', 'libboost_thread-vc100-mt-1_49', 'libboost_timer-vc100-mt-1_49', 'libboost_unit_test_framework-vc100-mt-1_49', 'libboost_wave-vc100-mt-1_49', 'libboost_wserialization-vc100-mt-1_49']
        # 'MapReduceMPI-gd.lib', 'vtkNetCDF-gd.lib', 'QVTK-gd.lib', 'vtkNetCDF_cxx-gd.lib', 'vtkRendering-gd.lib', 'vtkViews-gd.lib', 'vtkVolumeRendering-gd.lib', 'vtkWidgets-gd.lib', 'mpistubs-gd.lib', 'vtkalglib-gd.lib', 'vtkCharts-gd.lib', 'vtkexoIIc-gd.lib', 'vtkexpat-gd.lib', 'vtkCommon-gd.lib', 'vtkfreetype-gd.lib', 'vtkDICOMParser-gd.lib', 'vtkftgl-gd.lib', 'vtkFiltering-gd.lib', 'vtkhdf5-gd.lib', 'vtkjpeg-gd.lib', 'vtkGenericFiltering-gd.lib', 'vtklibxml2-gd.lib', 'vtkGeovis-gd.lib', 'vtkmetaio-gd.lib', 'vtkpng-gd.lib', 'vtkGraphics-gd.lib', 'vtkproj4-gd.lib', 'vtkHybrid-gd.lib', 'vtksqlite-gd.lib', 'vtksys-gd.lib', 'vtkIO-gd.lib', 'vtktiff-gd.lib', 'vtkImaging-gd.lib', 'vtkverdict-gd.lib', 'vtkzlib-gd.lib', 'vtkInfovis-gd.lib', 
        # add boost
        # dynamic lib
        # libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'boost_date_time-vc100-mt-1_47', 'boost_filesystem-vc100-mt-1_49', 'boost_graph-vc100-mt-1_49', 'boost_graph_parallel-vc100-mt-1_49', 'boost_iostreams-vc100-mt-1_49', 'boost_locale-vc100-mt-1_49', 'boost_math_c99-vc100-mt-1_49', 'boost_math_c99f-vc100-mt-1_49', 'boost_math_tr1-vc100-mt-1_49', 'boost_math_tr1f-vc100-mt-1_49', 'boost_mpi-vc100-mt-1_49', 'boost_prg_exec_monitor-vc100-mt-1_49', 'boost_program_options-vc100-mt-1_49', 'boost_random-vc100-mt-1_49', 'boost_regex-vc100-mt-1_49', 'boost_serialization-vc100-mt-1_49', 'boost_signals-vc100-mt-1_49', 'boost_system-vc100-mt-1_49', 'boost_thread-vc100-mt-1_49', 'boost_timer-vc100-mt-1_49', 'boost_unit_test_framework-vc100-mt-1_49', 'boost_wave-vc100-mt-1_49', 'boost_wserialization-vc100-mt-1_49']
        # static lib
        # libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'libboost_date_time-vc100-mt-1_49', 'libboost_filesystem-vc100-mt-1_49', 'libboost_graph_parallel-vc100-mt-1_49', 'libboost_iostreams-vc100-mt-1_49', 'libboost_locale-vc100-mt-1_49', 'libboost_math_c99-vc100-mt-1_49', 'libboost_math_c99f-vc100-mt-1_49', 'libboost_math_tr1-vc100-mt-1_49', 'libboost_math_tr1f-vc100-mt-1_49', 'libboost_mpi-vc100-mt-1_49', 'libboost_prg_exec_monitor-vc100-mt-1_49', 'libboost_program_options-vc100-mt-1_49', 'libboost_random-vc100-mt-1_49', 'libboost_regex-vc100-mt-1_49', 'libboost_serialization-vc100-mt-1_49', 'libboost_signals-vc100-mt-1_49', 'libboost_system-vc100-mt-1_49', 'libboost_test_exec_monitor-vc100-mt-1_49', 'libboost_thread-vc100-mt-1_49', 'libboost_timer-vc100-mt-1_49', 'libboost_unit_test_framework-vc100-mt-1_49', 'libboost_wave-vc100-mt-1_49', 'libboost_wserialization-vc100-mt-1_49']
    elif pcl_version == '-1.7':
        # release
        # libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s']
        # release + vtk
        libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_outofcore_release', 'pcl_people_release', 'pcl_recognition_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_search_release', 'pcl_segmentation_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkexpat-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkfreetype-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkhdf5-' + vtk_version, 'vtkhdf5_hl-' + vtk_version,
                       'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtkjpeg-' + vtk_version, 'vtkjsoncpp-' + vtk_version, 'vtklibxml2-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkNetCDF-' + vtk_version, 'vtkNetCDF_cxx-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkpng-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLIC-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtktiff-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkzlib-' + vtk_version]
    elif pcl_version == '-1.8':
        # release
        # libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s']
        # libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'vtkInfovis', 'MapReduceMPI', 'vtkNetCDF', 'QVTK', 'vtkNetCDF_cxx', 'vtkRendering', 'vtkViews', 'vtkVolumeRendering', 'vtkWidgets', 'mpistubs', 'vtkalglib', 'vtkCharts', 'vtkexoIIc', 'vtkexpat', 'vtkCommon', 'vtkfreetype', 'vtkDICOMParser', 'vtkftgl', 'vtkFiltering', 'vtkhdf5', 'vtkjpeg', 'vtkGenericFiltering', 'vtklibxml2', 'vtkGeovis', 'vtkmetaio', 'vtkpng', 'vtkGraphics', 'vtkproj4', 'vtkHybrid', 'vtksqlite', 'vtksys', 'vtkIO', 'vtktiff', 'vtkImaging', 'vtkverdict', 'vtkzlib']
        # release + vtk7.0
        libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_ml_release', 'pcl_octree_release', 'pcl_outofcore_release', 'pcl_people_release', 'pcl_recognition_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_search_release', 'pcl_segmentation_release', 'pcl_stereo_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'vtkalglib-' + vtk_version, 'vtkChartsCore-' + vtk_version, 'vtkCommonColor-' + vtk_version, 'vtkCommonComputationalGeometry-' + vtk_version, 'vtkCommonCore-' + vtk_version, 'vtkCommonDataModel-' + vtk_version, 'vtkCommonExecutionModel-' + vtk_version, 'vtkCommonMath-' + vtk_version, 'vtkCommonMisc-' + vtk_version, 'vtkCommonSystem-' + vtk_version, 'vtkCommonTransforms-' + vtk_version, 'vtkDICOMParser-' + vtk_version, 'vtkDomainsChemistry-' + vtk_version, 'vtkexoIIc-' + vtk_version, 'vtkexpat-' + vtk_version, 'vtkFiltersAMR-' + vtk_version, 'vtkFiltersCore-' + vtk_version, 'vtkFiltersExtraction-' + vtk_version, 'vtkFiltersFlowPaths-' + vtk_version, 'vtkFiltersGeneral-' + vtk_version, 'vtkFiltersGeneric-' + vtk_version, 'vtkFiltersGeometry-' + vtk_version, 'vtkFiltersHybrid-' + vtk_version, 'vtkFiltersHyperTree-' + vtk_version, 'vtkFiltersImaging-' + vtk_version, 'vtkFiltersModeling-' + vtk_version, 'vtkFiltersParallel-' + vtk_version, 'vtkFiltersParallelImaging-' + vtk_version, 'vtkFiltersProgrammable-' + vtk_version, 'vtkFiltersSelection-' + vtk_version, 'vtkFiltersSMP-' + vtk_version, 'vtkFiltersSources-' + vtk_version, 'vtkFiltersStatistics-' + vtk_version, 'vtkFiltersTexture-' + vtk_version, 'vtkFiltersVerdict-' + vtk_version, 'vtkfreetype-' + vtk_version, 'vtkGeovisCore-' + vtk_version, 'vtkgl2ps-' + vtk_version, 'vtkhdf5-' + vtk_version,
                       'vtkhdf5_hl-' + vtk_version, 'vtkImagingColor-' + vtk_version, 'vtkImagingCore-' + vtk_version, 'vtkImagingFourier-' + vtk_version, 'vtkImagingGeneral-' + vtk_version, 'vtkImagingHybrid-' + vtk_version, 'vtkImagingMath-' + vtk_version, 'vtkImagingMorphological-' + vtk_version, 'vtkImagingSources-' + vtk_version, 'vtkImagingStatistics-' + vtk_version, 'vtkImagingStencil-' + vtk_version, 'vtkInfovisCore-' + vtk_version, 'vtkInfovisLayout-' + vtk_version, 'vtkInteractionImage-' + vtk_version, 'vtkInteractionStyle-' + vtk_version, 'vtkInteractionWidgets-' + vtk_version, 'vtkIOAMR-' + vtk_version, 'vtkIOCore-' + vtk_version, 'vtkIOEnSight-' + vtk_version, 'vtkIOExodus-' + vtk_version, 'vtkIOExport-' + vtk_version, 'vtkIOGeometry-' + vtk_version, 'vtkIOImage-' + vtk_version, 'vtkIOImport-' + vtk_version, 'vtkIOInfovis-' + vtk_version, 'vtkIOLegacy-' + vtk_version, 'vtkIOLSDyna-' + vtk_version, 'vtkIOMINC-' + vtk_version, 'vtkIOMovie-' + vtk_version, 'vtkIONetCDF-' + vtk_version, 'vtkIOParallel-' + vtk_version, 'vtkIOParallelXML-' + vtk_version, 'vtkIOPLY-' + vtk_version, 'vtkIOSQL-' + vtk_version, 'vtkIOVideo-' + vtk_version, 'vtkIOXML-' + vtk_version, 'vtkIOXMLParser-' + vtk_version, 'vtkjpeg-' + vtk_version, 'vtkjsoncpp-' + vtk_version, 'vtklibxml2-' + vtk_version, 'vtkmetaio-' + vtk_version, 'vtkNetCDF-' + vtk_version, 'vtkoggtheora-' + vtk_version, 'vtkParallelCore-' + vtk_version, 'vtkpng-' + vtk_version, 'vtkproj4-' + vtk_version, 'vtkRenderingAnnotation-' + vtk_version, 'vtkRenderingContext2D-' + vtk_version, 'vtkRenderingContextOpenGL-' + vtk_version, 'vtkRenderingCore-' + vtk_version, 'vtkRenderingFreeType-' + vtk_version, 'vtkRenderingGL2PS-' + vtk_version, 'vtkRenderingImage-' + vtk_version, 'vtkRenderingLabel-' + vtk_version, 'vtkRenderingLIC-' + vtk_version, 'vtkRenderingLOD-' + vtk_version, 'vtkRenderingOpenGL-' + vtk_version, 'vtkRenderingVolume-' + vtk_version, 'vtkRenderingVolumeOpenGL-' + vtk_version, 'vtksqlite-' + vtk_version, 'vtksys-' + vtk_version, 'vtktiff-' + vtk_version, 'vtkverdict-' + vtk_version, 'vtkViewsContext2D-' + vtk_version, 'vtkViewsCore-' + vtk_version, 'vtkViewsInfovis-' + vtk_version, 'vtkzlib-' + vtk_version]

    else:
        libreleases = []

    for librelease in libreleases:
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
    if pcl_version == '-1.6':
        if is_64bits == True:
            # win_opengl_libdirs = ['C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\Lib\\x64']
            # AppVeyor
            win_kit_libdirs = ['C:\\Program Files\\Microsoft SDKs\\Windows\\v7.1\\Lib\\x64']
        else:
            # win_opengl_libdirs = ['C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\Lib\\win32']
            # AppVeyor
            win_kit_libdirs = ['C:\\Program Files\\Microsoft SDKs\\Windows\\v7.1\\Lib\\win32']
    elif pcl_version == '-1.7':
        if is_64bits == True:
            win_kit_libdirs = ['C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v8.0A\\Lib\\x64']
        else:
            win_kit_libdirs = ['C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v8.0A\\Lib\\win32']
    elif pcl_version == '-1.8':
        if is_64bits == True:
            # already set path
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v8.1A\\Lib\\x64']
            # Windows OS 7?
            # win_kit_incs = ['C:\\Program Files (x86)\\Windows Kits\\8.1\\Include\\shared', 'C:\\Program Files (x86)\\Windows Kits\\8.1\\Include\\um']
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Windows Kits\\8.1\\Lib\\winv6.3\\um\\x64']
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.10240.0\\ucrt\\x64']

            # Windows OS 8/8.1/10?
            # win_kit_incs = ['C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt']
            # win_kit_incs = ['C:\\Program Files (x86)\\Windows Kits\\10\\Include\\shared']
            pass
        else:
            # already set path
            # Windows OS 7
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v8.1A\\Lib\\win32']
            # win_kit_libdirs = ['C:\\Program Files (x86)\\Windows Kits\\8.1\\Lib\\winv6.3\\um\\x86']
            # win_kit_incs = ['C:\\Program Files (x86)\\Windows Kits\\8.1\\Include\\shared', 'C:\\Program Files (x86)\\Windows Kits\\8.1\\Include\\um']
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
    # default args
    # ext_args['extra_compile_args'].append('/MD')
    # custom
    # ext_args['extra_compile_args'].append('/MDd')
    # ext_args['extra_compile_args'].append('/MTd')
    # ext_args['extra_compile_args'].append('/MT')
    # use OpenMP
    # https://stackoverflow.com/questions/7844830/cython-openmp-compiler-flag
    # ext_args['extra_compile_args'].append('/openmp')

    # include_dirs=[pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\Boost\\include', pcl_root + '\\3rdParty\\FLANN\include', 'C:\\Anaconda2\\envs\\my_env\\Lib\\site-packages\\numpy\\core\\include'],
    # library_dirs=[pcl_root + '\\lib', pcl_root + '\\3rdParty\\Boost\\lib', pcl_root + '\\3rdParty\\FLANN\\lib'],
    # libraries=["pcl_apps_debug", "pcl_common_debug", "pcl_features_debug", "pcl_filters_debug", "pcl_io_debug", "pcl_io_ply_debug", "pcl_kdtree_debug", "pcl_keypoints_debug", "pcl_octree_debug", "pcl_registration_debug", "pcl_sample_consensus_debug", "pcl_segmentation_debug", "pcl_search_debug", "pcl_surface_debug", "pcl_tracking_debug", "pcl_visualization_debug", "flann-gd", "flann_s-gd", "boost_chrono-vc100-mt-1_49", "boost_date_time-vc100-mt-1_49", "boost_filesystem-vc100-mt-1_49", "boost_graph-vc100-mt-1_49", "boost_graph_parallel-vc100-mt-1_49", "boost_iostreams-vc100-mt-1_49", "boost_locale-vc100-mt-1_49", "boost_math_c99-vc100-mt-1_49", "boost_math_c99f-vc100-mt-1_49", "boost_math_tr1-vc100-mt-1_49", "boost_math_tr1f-vc100-mt-1_49", "boost_mpi-vc100-mt-1_49", "boost_prg_exec_monitor-vc100-mt-1_49", "boost_program_options-vc100-mt-1_49", "boost_random-vc100-mt-1_49", "boost_regex-vc100-mt-1_49", "boost_serialization-vc100-mt-1_49", "boost_signals-vc100-mt-1_49", "boost_system-vc100-mt-1_49", "boost_thread-vc100-mt-1_49", "boost_timer-vc100-mt-1_49", "boost_unit_test_framework-vc100-mt-1_49", "boost_wave-vc100-mt-1_49", "boost_wserialization-vc100-mt-1_49", "libboost_chrono-vc100-mt-1_49", "libboost_date_time-vc100-mt-1_49", "libboost_filesystem-vc100-mt-1_49", "libboost_graph_parallel-vc100-mt-1_49", "libboost_iostreams-vc100-mt-1_49", "libboost_locale-vc100-mt-1_49", "libboost_math_c99-vc100-mt-1_49", "libboost_math_c99f-vc100-mt-1_49", "libboost_math_tr1-vc100-mt-1_49", "libboost_math_tr1f-vc100-mt-1_49", "libboost_mpi-vc100-mt-1_49", "libboost_prg_exec_monitor-vc100-mt-1_49", "libboost_program_options-vc100-mt-1_49", "libboost_random-vc100-mt-1_49", "libboost_regex-vc100-mt-1_49", "libboost_serialization-vc100-mt-1_49", "libboost_signals-vc100-mt-1_49", "libboost_system-vc100-mt-1_49", "libboost_test_exec_monitor-vc100-mt-1_49", "libboost_thread-vc100-mt-1_49", "libboost_timer-vc100-mt-1_49", "libboost_unit_test_framework-vc100-mt-1_49", "libboost_wave-vc100-mt-1_49", "libboost_wserialization-vc100-mt-1_49"],
    ## define_macros=[('BOOST_NO_EXCEPTIONS', 'None')],
    # define_macros=[('EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET', '1')],
    # extra_compile_args=["/EHsc"],

    print(ext_args)

    if pcl_version == '-1.6':
        module = [Extension("pcl._pcl", ["pcl/_pcl.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  Extension("pcl.pcl_visualization", [
                      "pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    elif pcl_version == '-1.7':
        module = [Extension("pcl._pcl", ["pcl/_pcl_172.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  Extension("pcl.pcl_visualization", [
                      "pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    elif pcl_version == '-1.8':
        module = [Extension("pcl._pcl", ["pcl/_pcl_180.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  Extension("pcl.pcl_visualization", [
                      "pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    else:
        print('no pcl install or pkg-config missed.')
        sys.exit(1)

    # copy the pcl dll to local subfolder so that it can be added to the package through the data_files option
    listDlls=[]
    if not os.path.isdir('./dlls'):
        os.mkdir('./dlls')
    for dll in libreleases:
        pathDll=find_library(dll)
        if not pathDll is None:
            shutil.copy2(pathDll, './dlls' )
            listDlls.append(os.path.join('.\\dlls',dll+'.dll'))
    data_files=[('Lib/site-packages/pcl',listDlls)]# the path is relative to the python root folder 

else:
    # Not 'Windows'
    if platform.system() == "Darwin":
        os.environ['ARCHFLAGS'] = ''

    # Try to find PCL. XXX we should only do this when trying to build or install.
    PCL_SUPPORTED = ["-1.8", "-1.7", "-1.6", ""]    # in order of preference

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
    pcl_libs = ["common", "features", "filters", "io", "kdtree", "octree",
                "registration", "sample_consensus", "search", "segmentation",
                "surface", "tracking", "visualization"]
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

    # VTK use?
    # ext_args['include_dirs'].append('/usr/include/vtk')
    # ext_args['include_dirs'].append('/usr/local/include/vtk')
    # pcl 1.7(Ubuntu)
    # ext_args['include_dirs'].append('/usr/include/vtk-5.8')
    # ext_args['library_dirs'].append('/usr/lib')
    # ext_args['libraries'].append('libvtk*.so')
    # pcl 1.8.1(MacOSX)
    # ext_args['include_dirs'].append('/usr/local/include/vtk-8.0')
    # ext_args['library_dirs'].append('/usr/local/lib')
    # ext_args['include_dirs'].append('/usr/local/Cellar/vtk/8.0.1/include')
    # ext_args['library_dirs'].append('/usr/local/Cellar/vtk/8.0.1/lib')
    # ext_args['libraries'].append('libvtk*.dylib')

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
    if platform.system() == "Darwin":
        # or gcc5?
        # ext_args['extra_compile_args'].append("-stdlib=libstdc++")
        # ext_args['extra_compile_args'].append("-mmacosx-version-min=10.6")
        # ext_args['extra_compile_args'].append('-openmp')
        pass
    else:
        # gcc4?
        # ext_args['extra_compile_args'].append("-stdlib=libc++")
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
    ext_args['extra_link_args'].append('-lboost_system')
    # ext_args['extra_link_args'].append('-lboost_bind')

    # Fix compile error on Ubuntu 12.04 (e.g., Travis-CI).
    ext_args['define_macros'].append(
        ("EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET", "1"))

    if pcl_version == '-1.6':
        module = [Extension("pcl._pcl", ["pcl/_pcl.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  # Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    elif pcl_version == '-1.7':
        module = [Extension("pcl._pcl", ["pcl/_pcl_172.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  # Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    elif pcl_version == '-1.8':
        module = [Extension("pcl._pcl", ["pcl/_pcl_180.pyx", "pcl/minipcl.cpp", "pcl/ProjectInliers.cpp"], language="c++", **ext_args),
                  # Extension("pcl.pcl_visualization", ["pcl/pcl_visualization.pyx"], language="c++", **ext_args),
                  # Extension("pcl.pcl_grabber", ["pcl/pcl_grabber.pyx", "pcl/grabber_callback.cpp"], language="c++", **ext_args),
                  # debug
                  # gdb_debug=True,
                  ]
    else:
        print('no pcl install or pkg-config missed.')
        sys.exit(1)

    listDlls=[]
    data_files=None			



setup(name='python-pcl',
      description='pcl wrapper',
      url='http://github.com/strawlab/python-pcl',
      version='0.3',
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
        ],
      tests_require=['mock', 'nose'],
      ext_modules=module,
      cmdclass={'build_ext': build_ext},
      data_files=data_files
)
