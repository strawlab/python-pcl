from __future__ import print_function
from collections import defaultdict
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import subprocess
import numpy
import sys
import platform
import os

if platform.system() == "Windows":
	# sys.path.append('./src')
	# sys.path.append('./tests')

	# Check 32bit or 64bit
	is_64bits = sys.maxsize > 2**32
	# if is_64bits == True

	# environment Value
	for k, v in os.environ.items():
	    # print("{key} : {value}".format(key=k, value=v))
	    if k == "PCL_ROOT":
	        pcl_root = v
	        # print(pcl_root)

	# Add environment Value
	os.environ["PKG_CONFIG_PATH"] = pcl_root + '\\lib\\pkgconfig;' + pcl_root + '\3rdParty\FLANN\lib\pkgconfig;'
	# os.environ["PKG_CONFIG_PATH"] = pcl_root + '\\lib\\pkgconfig;' + pcl_root + '\3rdParty\FLANN\lib\pkgconfig;' + pcl_root + '\\3rdParty\\Eigen\lib\\pkgconfig

	# Try to find PCL. XXX we should only do this when trying to build or install.
	PCL_SUPPORTED = ["-1.7", "-1.6", ""]    # in order of preference

	for pcl_version in PCL_SUPPORTED:
	    if subprocess.call(['pkg-config\\pkg-config', 'pcl_common%s' % pcl_version]) == 0:
	        break
	else:
	    print("%s: error: cannot find PCL, tried" % sys.argv[0], file=sys.stderr)
	    for version in PCL_SUPPORTED:
	        print('    pkg-config pcl_common%s' % version, file=sys.stderr)
	    sys.exit(1)

	# print(pcl_version)

	# Python Version Check
	info = sys.version_info

	if pcl_version == '-1.6':
		# PCL 1.6.0 python Version == 3.4(>= 3.4?, 2.7 -> NG)
		if info.major == 3 and info.minor == 4:
			pass
		else:
			print('no building Python Version')
			exit(1)
	elif pcl_version == '-1.7':
		# PCL 1.7.2 python Version >= 3.5
		if info.major == 3 and info.minor >= 5:
			pass
		else:
			print('no building Python Version')
			exit(1)
	else:
		pass

	# Add environment Value
	# os.environ["VS90COMNTOOLS"] = '%VS100COMNTOOLS%'
	# os.environ["VS90COMNTOOLS"] = '%VS120COMNTOOLS%'

	# Find build/link options for PCL using pkg-config.
	pcl_libs = ["common", "features", "filters", "kdtree", "octree",
	            "registration", "sample_consensus", "search", "segmentation",
	            "surface"]
	pcl_libs = ["pcl_%s%s" % (lib, pcl_version) for lib in pcl_libs]
	# pcl_libs += ['Eigen3']
	# print(pcl_libs)

	ext_args = defaultdict(list)

	# set include path
	ext_args['include_dirs'].append(numpy.get_include())

	if pcl_version == '-1.6':
		# 1.6.0
		# ext_args['include_dirs'].append([pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\Boost\\include', pcl_root + '\\3rdParty\\FLANN\include'])
		# inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\Boost\\include', pcl_root + '\\3rdParty\\FLANN\include']
		# external Boost (G:\\boost\\boost_1_55_0)
		# NG
		# inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\FLANN\include']
		# 3rdParty
		inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\Boost\\include', pcl_root + '\\3rdParty\\FLANN\include']
		# extern
		# inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', 'G:\\boost\\boost_1_55_0', pcl_root + '\\3rdParty\\FLANN\include']
	elif pcl_version == '-1.7':
		# 1.7.2
		inc_dirs = [pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\\Eigen\eigen3', pcl_root + '\\3rdParty\\Boost\\include\\boost-1_57', pcl_root + '\\3rdParty\\FLANN\include']
	else:
		inc_dirs = []

	for inc_dir in inc_dirs:
	    ext_args['include_dirs'].append(inc_dir)

	# set library path
	# 3rdParty(+Boost)
	lib_dirs = [pcl_root + '\\lib', pcl_root + '\\3rdParty\\Boost\\lib', pcl_root + '\\3rdParty\\FLANN\lib']
	# extern -> NG
	# lib_dirs = [pcl_root + '\\lib', 'G:\\boost\\boost_1_55_0\\lib', pcl_root + '\\3rdParty\\FLANN\lib']
	# lib_dirs = [pcl_root + '\\lib', 'G:\\boost\\boost_1_55_0\\lib64-msvc-10.0', pcl_root + '\\3rdParty\\FLANN\lib']
	for lib_dir in lib_dirs:
	    ext_args['library_dirs'].append(lib_dir)

	# no use
	# debug
	# ext_args['libraries'].append(["pcl_apps_debug", "pcl_common_debug", "pcl_features_debug", "pcl_filters_debug", "pcl_io_debug", "pcl_io_ply_debug", "pcl_kdtree_debug", "pcl_keypoints_debug", "pcl_octree_debug", "pcl_registration_debug", "pcl_sample_consensus_debug", "pcl_segmentation_debug", "pcl_search_debug", "pcl_surface_debug", "pcl_tracking_debug", "pcl_visualization_debug", "flann-gd", "flann_s-gd"])
	# 1.6.0
	# libdebugs = ["pcl_apps_debug", "pcl_common_debug", "pcl_features_debug", "pcl_filters_debug", "pcl_io_debug", "pcl_io_ply_debug", "pcl_kdtree_debug", "pcl_keypoints_debug", "pcl_octree_debug", "pcl_registration_debug", "pcl_sample_consensus_debug", "pcl_segmentation_debug", "pcl_search_debug", "pcl_surface_debug", "pcl_tracking_debug", "pcl_visualization_debug", "flann-gd", "flann_s-gd"]
	# add boost
	# dynamic lib -> NG
	# libdebugs = ['pcl_apps_debug', 'pcl_common_debug', 'pcl_features_debug', 'pcl_filters_debug', 'pcl_io_debug', 'pcl_io_ply_debug', 'pcl_kdtree_debug', 'pcl_keypoints_debug', 'pcl_octree_debug', 'pcl_registration_debug', 'pcl_sample_consensus_debug', 'pcl_segmentation_debug', 'pcl_search_debug', 'pcl_surface_debug', 'pcl_tracking_debug', 'pcl_visualization_debug', 'flann-gd', 'flann_s-gd', 'boost_date_time-vc100-mt-gd-1_49', 'boost_filesystem-vc100-mt-gd-1_49', 'boost_graph-vc100-mt-gd-1_49', 'boost_graph_parallel-vc100-mt-gd-1_49', 'boost_iostreams-vc100-mt-gd-1_49', 'boost_locale-vc100-mt-gd-1_49', 'boost_math_c99-vc100-mt-gd-1_49', 'boost_math_c99f-vc100-mt-gd-1_49', 'boost_math_tr1-vc100-mt-gd-1_49', 'boost_math_tr1f-vc100-mt-gd-1_49', 'boost_mpi-vc100-mt-gd-1_49', 'boost_prg_exec_monitor-vc100-mt-gd-1_49', 'boost_program_options-vc100-mt-gd-1_49', 'boost_random-vc100-mt-gd-1_49', 'boost_regex-vc100-mt-gd-1_49', 'boost_serialization-vc100-mt-gd-1_49', 'boost_signals-vc100-mt-gd-1_49', 'boost_system-vc100-mt-gd-1_49', 'boost_thread-vc100-mt-gd-1_49', 'boost_timer-vc100-mt-gd-1_49', 'boost_unit_test_framework-vc100-mt-gd-1_49', 'boost_wave-vc100-mt-gd-1_49', 'boost_wserialization-vc100-mt-gd-1_49']
	# static lib
	# libdebugs = ['pcl_apps_debug', 'pcl_common_debug', 'pcl_features_debug', 'pcl_filters_debug', 'pcl_io_debug', 'pcl_io_ply_debug', 'pcl_kdtree_debug', 'pcl_keypoints_debug', 'pcl_octree_debug', 'pcl_registration_debug', 'pcl_sample_consensus_debug', 'pcl_segmentation_debug', 'pcl_search_debug', 'pcl_surface_debug', 'pcl_tracking_debug', 'pcl_visualization_debug', 'flann-gd', 'flann_s-gd', 'libboost_chrono-vc100-mt-gd-1_49', 'libboost_date_time-vc100-mt-gd-1_49', 'libboost_filesystem-vc100-mt-gd-1_49', 'libboost_graph_parallel-vc100-mt-gd-1_49', 'libboost_iostreams-vc100-mt-gd-1_49', 'libboost_locale-vc100-mt-gd-1_49', 'libboost_math_c99-vc100-mt-gd-1_49', 'libboost_math_c99f-vc100-mt-gd-1_49', 'libboost_math_tr1-vc100-mt-gd-1_49', 'libboost_math_tr1f-vc100-mt-gd-1_49', 'libboost_mpi-vc100-mt-gd-1_49', 'libboost_prg_exec_monitor-vc100-mt-gd-1_49', 'libboost_program_options-vc100-mt-gd-1_49', 'libboost_random-vc100-mt-gd-1_49', 'libboost_regex-vc100-mt-gd-1_49', 'libboost_serialization-vc100-mt-gd-1_49', 'libboost_signals-vc100-mt-gd-1_49', 'libboost_system-vc100-mt-gd-1_49', 'libboost_test_exec_monitor-vc100-mt-gd-1_49', 'libboost_thread-vc100-mt-gd-1_49', 'libboost_timer-vc100-mt-gd-1_49', 'libboost_unit_test_framework-vc100-mt-gd-1_49', 'libboost_wave-vc100-mt-gd-1_49', 'libboost_wserialization-vc100-mt-gd-1_49']
	# 1.7.2
	# libdebugs = ["pcl_common_debug", "pcl_features_debug", "pcl_filters_debug", "pcl_io_debug", "pcl_io_ply_debug", "pcl_kdtree_debug", "pcl_keypoints_debug", "pcl_octree_debug", "pcl_registration_debug", "pcl_sample_consensus_debug", "pcl_segmentation_debug", "pcl_search_debug", "pcl_surface_debug", "pcl_tracking_debug", "pcl_visualization_debug", "flann-gd", "flann_s-gd"]
	# for libdebug in libdebugs:
	#     ext_args['libraries'].append(libdebug)

	if pcl_version == '-1.6':
		# release
		libreleases = ['pcl_apps_release', 'pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s']
		# add boost
		# dynamic lib
		# libreleases = ['pcl_apps_release', 'pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'boost_date_time-vc100-mt-1_47', 'boost_filesystem-vc100-mt-1_49', 'boost_graph-vc100-mt-1_49', 'boost_graph_parallel-vc100-mt-1_49', 'boost_iostreams-vc100-mt-1_49', 'boost_locale-vc100-mt-1_49', 'boost_math_c99-vc100-mt-1_49', 'boost_math_c99f-vc100-mt-1_49', 'boost_math_tr1-vc100-mt-1_49', 'boost_math_tr1f-vc100-mt-1_49', 'boost_mpi-vc100-mt-1_49', 'boost_prg_exec_monitor-vc100-mt-1_49', 'boost_program_options-vc100-mt-1_49', 'boost_random-vc100-mt-1_49', 'boost_regex-vc100-mt-1_49', 'boost_serialization-vc100-mt-1_49', 'boost_signals-vc100-mt-1_49', 'boost_system-vc100-mt-1_49', 'boost_thread-vc100-mt-1_49', 'boost_timer-vc100-mt-1_49', 'boost_unit_test_framework-vc100-mt-1_49', 'boost_wave-vc100-mt-1_49', 'boost_wserialization-vc100-mt-1_49']
		# static lib
		# boost_chrono-vc100-mt-1_49 -> NG(1.47/1.49)
		# boost_date_time-vc100-mt-1_49.lib -> NG
		# libreleases = ['pcl_apps_release', 'pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s', 'libboost_date_time-vc100-mt-1_49', 'libboost_filesystem-vc100-mt-1_49', 'libboost_graph_parallel-vc100-mt-1_49', 'libboost_iostreams-vc100-mt-1_49', 'libboost_locale-vc100-mt-1_49', 'libboost_math_c99-vc100-mt-1_49', 'libboost_math_c99f-vc100-mt-1_49', 'libboost_math_tr1-vc100-mt-1_49', 'libboost_math_tr1f-vc100-mt-1_49', 'libboost_mpi-vc100-mt-1_49', 'libboost_prg_exec_monitor-vc100-mt-1_49', 'libboost_program_options-vc100-mt-1_49', 'libboost_random-vc100-mt-1_49', 'libboost_regex-vc100-mt-1_49', 'libboost_serialization-vc100-mt-1_49', 'libboost_signals-vc100-mt-1_49', 'libboost_system-vc100-mt-1_49', 'libboost_test_exec_monitor-vc100-mt-1_49', 'libboost_thread-vc100-mt-1_49', 'libboost_timer-vc100-mt-1_49', 'libboost_unit_test_framework-vc100-mt-1_49', 'libboost_wave-vc100-mt-1_49', 'libboost_wserialization-vc100-mt-1_49']
	elif pcl_version == '-1.7':
		# release
		libreleases = ['pcl_common_release', 'pcl_features_release', 'pcl_filters_release', 'pcl_io_release', 'pcl_io_ply_release', 'pcl_kdtree_release', 'pcl_keypoints_release', 'pcl_octree_release', 'pcl_registration_release', 'pcl_sample_consensus_release', 'pcl_segmentation_release', 'pcl_search_release', 'pcl_surface_release', 'pcl_tracking_release', 'pcl_visualization_release', 'flann', 'flann_s']
	else:
		libreleases = []

	for librelease in libreleases:
	    ext_args['libraries'].append(librelease)

	# ext_args['define_macros'].append(('EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET', '1'))
	debugs = [('EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET', '1'), ('BOOST_NO_EXCEPTIONS', 'None')]
	for debug in debugs:
		define_macros=[('BOOST_NO_EXCEPTIONS', 'None')],

	ext_args['extra_compile_args'].append('/EHsc')

	# NG
	# ext_args['extra_compile_args'].append('/NODEFAULTLIB:msvcrtd')
	# ext_args['extra_compile_args'].append('/MD')
	# ext_args['extra_compile_args'].append('/MDd')
	# ext_args['extra_compile_args'].append('/MTd')
	# ext_args['extra_compile_args'].append('/MT')

	# include_dirs=[pcl_root + '\\include\\pcl' + pcl_version, pcl_root + '\\3rdParty\\Eigen\\include', pcl_root + '\\3rdParty\\Boost\\include', pcl_root + '\\3rdParty\\FLANN\include', 'C:\\Anaconda2\\envs\\my_env\\Lib\\site-packages\\numpy\\core\\include'],
	# library_dirs=[pcl_root + '\\lib', pcl_root + '\\3rdParty\\Boost\\lib', pcl_root + '\\3rdParty\\FLANN\lib'],
	# libraries=["pcl_apps_debug", "pcl_common_debug", "pcl_features_debug", "pcl_filters_debug", "pcl_io_debug", "pcl_io_ply_debug", "pcl_kdtree_debug", "pcl_keypoints_debug", "pcl_octree_debug", "pcl_registration_debug", "pcl_sample_consensus_debug", "pcl_segmentation_debug", "pcl_search_debug", "pcl_surface_debug", "pcl_tracking_debug", "pcl_visualization_debug", "flann-gd", "flann_s-gd", "boost_chrono-vc100-mt-1_49", "boost_date_time-vc100-mt-1_49", "boost_filesystem-vc100-mt-1_49", "boost_graph-vc100-mt-1_49", "boost_graph_parallel-vc100-mt-1_49", "boost_iostreams-vc100-mt-1_49", "boost_locale-vc100-mt-1_49", "boost_math_c99-vc100-mt-1_49", "boost_math_c99f-vc100-mt-1_49", "boost_math_tr1-vc100-mt-1_49", "boost_math_tr1f-vc100-mt-1_49", "boost_mpi-vc100-mt-1_49", "boost_prg_exec_monitor-vc100-mt-1_49", "boost_program_options-vc100-mt-1_49", "boost_random-vc100-mt-1_49", "boost_regex-vc100-mt-1_49", "boost_serialization-vc100-mt-1_49", "boost_signals-vc100-mt-1_49", "boost_system-vc100-mt-1_49", "boost_thread-vc100-mt-1_49", "boost_timer-vc100-mt-1_49", "boost_unit_test_framework-vc100-mt-1_49", "boost_wave-vc100-mt-1_49", "boost_wserialization-vc100-mt-1_49", "libboost_chrono-vc100-mt-1_49", "libboost_date_time-vc100-mt-1_49", "libboost_filesystem-vc100-mt-1_49", "libboost_graph_parallel-vc100-mt-1_49", "libboost_iostreams-vc100-mt-1_49", "libboost_locale-vc100-mt-1_49", "libboost_math_c99-vc100-mt-1_49", "libboost_math_c99f-vc100-mt-1_49", "libboost_math_tr1-vc100-mt-1_49", "libboost_math_tr1f-vc100-mt-1_49", "libboost_mpi-vc100-mt-1_49", "libboost_prg_exec_monitor-vc100-mt-1_49", "libboost_program_options-vc100-mt-1_49", "libboost_random-vc100-mt-1_49", "libboost_regex-vc100-mt-1_49", "libboost_serialization-vc100-mt-1_49", "libboost_signals-vc100-mt-1_49", "libboost_system-vc100-mt-1_49", "libboost_test_exec_monitor-vc100-mt-1_49", "libboost_thread-vc100-mt-1_49", "libboost_timer-vc100-mt-1_49", "libboost_unit_test_framework-vc100-mt-1_49", "libboost_wave-vc100-mt-1_49", "libboost_wserialization-vc100-mt-1_49"],
	## define_macros=[('BOOST_NO_EXCEPTIONS', 'None')],
	# define_macros=[('EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET', '1')],
	# extra_compile_args=["/EHsc"],

	print(ext_args)

	if pcl_version == '-1.6':
		setup(name='python-pcl',
		      description='pcl wrapper',
		      url='http://github.com/strawlab/python-pcl',
		      version='0.2',
		      author='John Stowers',
		      author_email='john.stowers@gmail.com',
		      license='BSD',
		      packages=["pcl"],
		      ext_modules=[Extension("pcl._pcl", ["pcl/_pcl.pyx", "pcl/minipcl.cpp"],
		                             language = "c++", **ext_args),
		                   # Extension("pcl.registration", ["pcl/pcl_registration.pyx"],
		                   #           language="c++", **ext_args),
		                   # debug
		                   # gdb_debug=True,
		                  ],
		      cmdclass={'build_ext': build_ext}
		      )
	elif pcl_version == '-1.7':
		setup(name='python-pcl',
		      description='pcl wrapper',
		      url='http://github.com/strawlab/python-pcl',
		      version='0.2',
		      author='John Stowers',
		      author_email='john.stowers@gmail.com',
		      license='BSD',
		      packages=["pcl"],
		      ext_modules=[Extension("pcl._pcl", ["pcl/_pcl.pyx", "pcl/minipcl.cpp"],
		                             language = "c++", **ext_args),
		                   Extension("pcl.registration", ["pcl/registration.pyx"],
		                             language="c++", **ext_args),
		                   # debug
		                   # gdb_debug=True,
		                  ],
		      cmdclass={'build_ext': build_ext}
		      )
	else:
		print('no pcl install or pkg-config missed.')

else:
	# Not 'Windows'
	sys.path.append('./tests')

	if platform.system() == "Darwin":
		os.environ['ARCHFLAGS'] = ''

	# Try to find PCL. XXX we should only do this when trying to build or install.
	PCL_SUPPORTED = ["-1.7", "-1.6", ""]    # in order of preference

	for pcl_version in PCL_SUPPORTED:
	    if subprocess.call(['pkg-config', 'pcl_common%s' % pcl_version]) == 0:
	        break
	else:
	    print("%s: error: cannot find PCL, tried" % sys.argv[0], file=sys.stderr)
	    for version in PCL_SUPPORTED:
	        print('    pkg-config pcl_common%s' % version, file=sys.stderr)
	    sys.exit(1)

	# Find build/link options for PCL using pkg-config.
	pcl_libs = ["common", "features", "filters", "io", "kdtree", "octree",
	            "registration", "sample_consensus", "search", "segmentation",
	            "surface"]
	pcl_libs = ["pcl_%s%s" % (lib, pcl_version) for lib in pcl_libs]

	ext_args = defaultdict(list)
	ext_args['include_dirs'].append(numpy.get_include())

	def pkgconfig(flag):
	    # Equivalent in Python 2.7 (but not 2.6):
	    # subprocess.check_output(['pkg-config', flag] + pcl_libs).split()
	    p = subprocess.Popen(['pkg-config', flag] + pcl_libs,
	                         stdout=subprocess.PIPE)
	    stdout, _ = p.communicate()
	    # Assume no evil spaces in filenames; unsure how pkg-config would
	    # handle those, anyway.
	    # decode() is required in Python 3. TODO how do know the encoding?
	    return stdout.decode().split()


	for flag in pkgconfig('--cflags-only-I'):
	    ext_args['include_dirs'].append(flag[2:])

	for flag in pkgconfig('--cflags-only-other'):
	    if flag.startswith('-D'):
	        macro, value = flag[2:].split('=', 1)
	        ext_args['define_macros'].append((macro, value))
	    else:
	        ext_args['extra_compile_args'].append(flag)

	for flag in pkgconfig('--libs-only-l'):
	    if flag == "-lflann_cpp-gd":
	        print("skipping -lflann_cpp-gd (see https://github.com/strawlab/python-pcl/issues/29")
	        continue
	    ext_args['libraries'].append(flag[2:])

	for flag in pkgconfig('--libs-only-L'):
	    ext_args['library_dirs'].append(flag[2:])

	for flag in pkgconfig('--libs-only-other'):
	    ext_args['extra_link_args'].append(flag)

	# Fix compile error on Ubuntu 12.04 (e.g., Travis-CI).
	ext_args['define_macros'].append(
	    ("EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET", "1"))

	if pcl_version == '-1.6':
		setup(name='python-pcl',
		      description='pcl wrapper',
		      url='http://github.com/strawlab/python-pcl',
		      version='0.2',
		      author='John Stowers',
		      author_email='john.stowers@gmail.com',
		      license='BSD',
		      packages=["pcl"],
		      ext_modules=[Extension("pcl._pcl", ["pcl/_pcl.pyx", "pcl/minipcl.cpp"],
		                             language = "c++", **ext_args),
		                   # 1.6.0 NG
		                   # Extension("pcl.registration", ["pcl/registration.pyx"],
		                   #           language="c++", **ext_args),
		                   # debug
		                   # gdb_debug=True,
		                  ],
		      cmdclass={'build_ext': build_ext}
		      )
	elif pcl_version == '-1.7':
		setup(name='python-pcl',
		      description='pcl wrapper',
		      url='http://github.com/strawlab/python-pcl',
		      version='0.2',
		      author='John Stowers',
		      author_email='john.stowers@gmail.com',
		      license='BSD',
		      packages=["pcl"],
		      ext_modules=[Extension("pcl._pcl", ["pcl/_pcl.pyx", "pcl/minipcl.cpp"],
		                             language = "c++", **ext_args),
		                   Extension("pcl.registration", ["pcl/registration.pyx"],
		                             language="c++", **ext_args),
		                   # debug
		                   # gdb_debug=True,
		                  ],
		      cmdclass={'build_ext': build_ext}
		      )
	else:
		print('no pcl install or pkg-config missed.')
