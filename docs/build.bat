@ECHO OFF

sphinx-intl build
make clean html
rem jp?
rem make -e SPHINXOPTS="-D language='ja'" html