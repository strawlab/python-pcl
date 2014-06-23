from __future__ import print_function
from collections import defaultdict
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
import subprocess
import numpy
import sys

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
            "sample_consensus", "search", "segmentation", "surface"]
pcl_libs = ["pcl_%s%s" % (lib, pcl_version) for lib in pcl_libs]

ext_args = defaultdict(list)
ext_args['include_dirs'].append(numpy.get_include())

def pkgconfig(flag):
    # Equivalent in Python 2.7 (but not 2.6):
    #subprocess.check_output(['pkg-config', flag] + pcl_libs).split()
    p = subprocess.Popen(['pkg-config', flag] + pcl_libs,
                         stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    # Assume no evil spaces in filenames; unsure how pkg-config would
    # handle those, anyway.
    return stdout.split()


for flag in pkgconfig('--cflags-only-I'):
    ext_args['include_dirs'].append(flag[2:])
for flag in pkgconfig('--cflags-only-other'):
    if flag.startswith('-D'):
        macro, value = flag[2:].split('=', 1)
        ext_args['define_macros'].append((macro, value))
    else:
        ext_args['extra_compile_args'].append(flag)
for flag in pkgconfig('--libs-only-l'):
    ext_args['libraries'].append(flag[2:])
for flag in pkgconfig('--libs-only-L'):
    ext_args['library_dirs'].append(flag[2:])
for flag in pkgconfig('--libs-only-other'):
    ext_args['extra_link_args'].append(flag)


setup(name='python-pcl',
      description='pcl wrapper',
      url='http://github.com/strawlab/python-pcl',
      version='0.1',
      author='John Stowers',
      author_email='john.stowers@gmail.com',
      license='BSD',
      ext_modules=[Extension("pcl", ["pcl.pyx", "minipcl.cpp"],
                             language="c++", **ext_args)],
      cmdclass={'build_ext': build_ext}
      )
