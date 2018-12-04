@ECHO OFF

rem generate from '.po' to '.mo' files.
sphinx-intl build
make html
rem jp?
rem make -e SPHINXOPTS="-D language='ja'" html