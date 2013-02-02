# -*- coding: utf-8 -*-

# ensure that we use the local pcl (and not the system copy) to document
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcl
assert pcl.PointCloud.__doc__ is not None

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary']

needs_sphinx = '1.1'
autodoc_docstring_signature = True

# dont create empty templates and static dirs
templates_path = ['.']
html_static_path = ['.']

source_suffix = '.rst'
master_doc = 'readme'

# General information about the project.
project = u'python-pcl'
copyright = u'2013, John Stowers'
version = '1.0'
release = '1.0'

exclude_patterns = ['build']
pygments_style = 'sphinx'

html_theme = 'haiku'
htmlhelp_basename = 'python-pcldoc'
html_logo = 'pcl_logo.png'
html_title = 'Python Bindings to the Point Cloud Library'
html_short_title = '%s v%s' % (project, version)



