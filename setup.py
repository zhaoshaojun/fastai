
# coding: utf-8

"""
Setup script for installing fastai
"""

##########################################################################
## Imports
##########################################################################

#from distutils.core import setup
from setuptools import setup


##########################################################################
## Setup
##########################################################################

setup(
    name = "fastai",

    packages = ['fastai', 'fastai/models', 'fastai/models/cifar10'], # this must be the same as the name above

    version = 1.0 ,

    description = "The fast.ai deep learning and machine learning library. Git pull fastai, for all fast.ai sessions and tutorials also",

    author = "Jeremy Howard and contributors"

    author_email = "info@fast.ai",

    license = "GNU General Public License",

    url = "https://github.com/fastai/fastai",

    download_url =  'https://github.com/fastai/fastai/archive/1.0.tar.gz',

    install_requires =
     ['awscli',
     'bcolz',     'bleach',     'certifi',     'cycler',     'decorator',
    # 'docrepr' ,
     'entrypoints',
     'feather-format',
     'graphviz',     'html5lib',     'ipykernel',     'ipython',     'ipython-genutils',     'ipywidgets',     'isoweek',     'jedi',     'Jinja2',     'jsonschema',     'jupyter',     'jupyter-client',     'jupyter-console',
     'jupyter_contrib_nbextensions',     'jupyter-core',
     'kaggle-cli',     'MarkupSafe',     'matplotlib',     'mistune',     'nbconvert',     'nbformat',     'notebook',     'numpy',     'olefile',     'opencv-python',     'pandas',     'pandas_summary',     'pandocfilters',     'pexpect',     'pickleshare',     'Pillow',
     'plotnine',     'prompt-toolkit',     'ptyprocess',     'Pygments',     'pyparsing',     'python-dateutil',     'pytz',     'PyYAML',     'pyzmq',     'qtconsole',     'scipy',     'seaborn',     'simplegeneric',     'six',
     'sklearn_pandas',     'terminado',     'testpath',
     'torchtext',     'tornado',     'tqdm',     'traitlets',     'wcwidth',     'webencodings',     'widgetsnbextension'],



    keywords = ['deeplearning', 'pytorch', 'machinelearning'],

        classifiers = ['Development Status :: 3 - Alpha',
                      'Programming Language :: Python',
                    'Programming Language :: Python :: 3.6',]
)

