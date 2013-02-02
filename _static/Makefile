pcl.so: pcl.pyx setup.py pcl_defs.pxd minipcl.cpp
	python setup.py build_ext --inplace

test: pcl.so tests/test.py
	nosetests -s

clean:
	rm -rf build
	rm -f pcl.cpp pcl.so

doc: pcl.so conf.py readme.rst
	sphinx-build -b singlehtml -d build/doctrees . build/html

showdoc: doc
	gvfs-open build/html/readme.html
