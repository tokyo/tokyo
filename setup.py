#!/usr/bin/env python
"""
Tokyo: A Cython wrapper to BLAS and LAPACK
"""

DOCLINES = __doc__.split("\n")

import os
import sys

try:
    import setuptools   # To enable 'python setup.py develop'
    pass
except:
    pass

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: BSD
Programming Language :: Python
Programming Language :: Cython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

def configuration(parent_package='',top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_include_dirs([numpy.get_include()])
    config.add_subpackage('tokyo')

    # Set config.version
    config.get_version(os.path.join('tokyo','version.py'))

    return config

def setup_package():

    import glob
    from numpy.distutils.core import setup
    from numpy.distutils.misc_util import Configuration

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)
    sys.path.insert(0,os.path.join(local_path,'tokyo')) # to retrieve version

    tokyo_pyx_files = glob.glob(os.path.join('tokyo','*.pyx'))
    def cythonize():
        for pyx in tokyo_pyx_files:
            c = pyx.split('.')[0] + '.c'
            cy = True
            if os.path.exists(c):
                pyx_mtime = os.path.getmtime(pyx)
                c_mtime = os.path.getmtime(c)
                cy = pyx_mtime > c_mtime
            if cy:
                sys.stderr.write('Cythonizing %s...\n' % pyx)
                os.system('cython ' + pyx)
            else:
                sys.stderr.write('No need to cythonize %s\n' % pyx)

    try:

        cythonize()

        setup(
            name = 'tokyo',
            author = "Shane Legg, Matej Laitl, Dominique Orban",
            author_email = "shane@vetta.org,matej@laitl.cz,dominique.orban@gmail.com",
            maintainer = "Tokyo Developers",
            maintainer_email = "matej@laitl.cz,dominique.orban@gmail.com",
            description = DOCLINES[0],
            long_description = "\n".join(DOCLINES[2:]),
            url = "https://github.com/tokyo/tokyo",
            download_url = "https://github.com/tokyo/tokyo",
            license = 'BSD',
            classifiers=filter(None, CLASSIFIERS.split('\n')),
            platforms = ["Windows", "Linux", "Mac OS-X", "Unix"],
            configuration=configuration,
            )
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return

if __name__ == '__main__':
    setup_package()
