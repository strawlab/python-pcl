@ECHO OFF

make gettext
mkdir locale\pot
copy build\locale\*.pot locale\pot
mkdir locale\ja\LC_MESSAGES
copy locale\pot\* locale\ja\LC_MESSAGES\
sphinx-intl update -p build\locale -l ja