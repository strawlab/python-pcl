@ECHO OFF

sphinx-intl build
make clean html
rem jp?
rem make clean LANGUAGE=jp gettext html