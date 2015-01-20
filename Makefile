all: pcl/_pcl.so pcl/registration.so

pcl/_pcl.so: pcl/_pcl.pxd pcl/_pcl.pyx setup.py pcl/pcl_defs.pxd \
             pcl/minipcl.cpp pcl/indexing.hpp
	python setup.py build_ext --inplace

pcl/registration.so: setup.py pcl/_pcl.pxd pcl/pcl_defs.pxd \
                      pcl/registration.pyx
	python setup.py build_ext --inplace

test: pcl/_pcl.so tests/test.py
	nosetests -s

clean:
	rm -rf build
	rm -f pcl/_pcl.cpp pcl/_pcl.so pcl/registration.so

doc: pcl.so conf.py readme.rst
	sphinx-build -b singlehtml -d build/doctrees . build/html

showdoc: doc
	gvfs-open build/html/readme.html
