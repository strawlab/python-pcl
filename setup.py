from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import pkgconfig

PCL_VER = "1.5"

pcl_libs = ["common", "features", "filters", "io", "kdtree", "octree",
            "sample_consensus", "search", "segmentation", "surface"]

ext_args = pkgconfig.parse(' '.join('pcl_%s-%s' % (lib, PCL_VER)
                                    for lib in pcl_libs))

for key in ext_args:
    # Work around Unicode issue in some versions of distutils. Assume UTF-8.
    ext_args[key] = [val.decode('utf-8') for val in ext_args[key]]

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
