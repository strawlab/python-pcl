pcl.so: pcl.pyx setup.py pcl_defs.pxd
	python setup.py build_ext --inplace

test: pcl.so tests/test.py
	nosetests

clean:
	rm -rf build
	rm -f pcl.cpp pcl.so
