@ECHO OFF

rem generate and update translation files.
rem generate(update) from '.rst' to '.pot' files.[build/locale]
make gettext
rem generate(update) from '.pot' to '.po' files.[move source/locale]
rem ex. ja
sphinx-intl update -p build/locale -l ja
rem : before settings(set API token)
rem tx init
rem : using transifex(official : https://docs.transifex.com/integrations/sphinx-doc)
rem : # 'sphinx.ext.pngmath',
rem sphinx-intl update-txconfig-resources --pot-dir build/locale --transifex-project-name="<project-name>"
rem : [transifex] start define.
rem : regist po files.
rem tx push -s
rem : upload pot files.
rem : language transifer.
rem : downloads po files.
rem : tx pull -l ja
rem : [transifex] end define
rem : sphinx-intl build
rem : make -e SPHINXOPTS="-Dlanguage='ja'" html
