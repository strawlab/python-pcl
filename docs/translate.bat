@ECHO OFF

make gettext
rem ex. ja
sphinx-intl update -p build\locale -l ja
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
rem : make -e SPHINXOPTS="-D language='ja'" html
