pcl/_pcl.so: pcl/_pcl.pyx setup.py pcl/pcl_defs.pxd pcl/minipcl.cpp
	python setup.py build_ext --inplace

test: pcl/_pcl.so tests/test.py
	nosetests -s

clean:
	rm -rf build
	rm -f pcl/_pcl.cpp pcl/_pcl.so

doc: pcl.so conf.py readme.rst
	sphinx-build -b singlehtml -d build/doctrees . build/html

showdoc: doc
	gvfs-open build/html/readme.html
