from collections import defaultdict
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
import subprocess
import numpy

PCL_VER = "1.7"

# Find build/link options for PCL using pkg-config.
pcl_libs = ["common", "features", "filters", "io", "kdtree", "octree",
            "sample_consensus", "search", "segmentation", "surface"]
pcl_libs = ["pcl_%s-%s" % (lib, PCL_VER) for lib in pcl_libs]

ext_args = defaultdict(list)
ext_args['include_dirs'].append(numpy.get_include())

def pkgconfig(flag):
    return subprocess.check_output(['pkg-config', flag] + pcl_libs).split()

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
