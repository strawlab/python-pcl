@ECHO OFF

rem generate from '.po' to '.mo' files.
sphinx-intl build
rem generate html files.
make html
rem translation jp
rem https://potranslator.readthedocs.io/ja/latest/readme.html#supported-languages
rem Unix
rem make -e SPHINXOPTS="-D language='ja'" html
rem Windows
rem set SPHINXOPTS=-D language=ja
rem make html