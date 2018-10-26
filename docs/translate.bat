@ECHO OFF

make gettext
rem ex. ja
sphinx-intl update -p build\locale -l ja
rem using transifex
rem [transifex] start define.
rem upload pot files.
rem language transifer.
rem downloads po files.
rem [transifex] end define