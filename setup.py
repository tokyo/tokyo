#!/usr/bin/env python
"""
Tokyo: A Cython wrapper to BLAS and LAPACK
"""

DOCLINES = __doc__.split("\n")

import os
import sys

try:
    #import setuptools   # To enable 'python setup.py develop'
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

import ConfigParser

def get_from_config(config_name, section, field):
    try:
        value = config_name.get(section,field)
    except ConfigParser.NoOptionError:
        return []
    return value

def configuration(parent_package='',top_path=None):
    import numpy
    import os
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_include_dirs([numpy.get_include()])
    #config.add_subpackage('tokyo')

    # Set config.version
    config.get_version(os.path.join('tokyo','version.py'))

    # Read relevant Tokyo-specific configuration options.
    tokyo_config = ConfigParser.SafeConfigParser()
    tokyo_config.read('site.cfg')

    #config = Configuration('tokyo', parent_package, top_path)

    # Get info from site.cfg
    tokyo_library_dirs = get_from_config(tokyo_config,'DEFAULT','library_dirs')
    tokyo_include_dirs = get_from_config(tokyo_config,'DEFAULT','include_dirs')

    try:
        tokyo_blas_libs = get_from_config(tokyo_config,'blas_opt','libraries')
        extra_link_args = []
        for lib in tokyo_blas_libs.split(','):
            libname = lib.strip()
            extra_link_args.append('-Wl,-l' + libname)
        blas_info = {'extra_link_args' : extra_link_args}
    except:
        blas_info = get_info('blas_opt',0)
        if not blas_info:
            blas_info = get_info('blas',0)
            if not blas_info:
                print 'No blas info found'
    print 'Using'
    print blas_info

    try:
        tokyo_lapack_libs = get_from_config(tokyo_config,'lapack_opt','libraries')
        extra_link_args = []
        for lib in tokyo_lapack_libs.split(','):
            libname = lib.strip()
            extra_link_args.append('-Wl,-l' + libname)
        lapack_info = {'extra_link_args' : extra_link_args}
    except:
        lapack_info = get_info('lapack_opt',0)
        if not lapack_info:
            lapack_info = get_info('lapack',0)
            if not lapack_info:
                print 'No lapack info found'
    print 'Using'
    print lapack_info

    tokyo_extra_args = dict(blas_info, **lapack_info)
    print tokyo_extra_args

    config.add_extension(
        name='tokyo.tokyo',
        sources=[os.path.join('tokyo','tokyo.c')],
        include_dirs=tokyo_include_dirs,
        library_dirs=tokyo_library_dirs,
        extra_info=tokyo_extra_args,
        )

    config.add_extension(
        name='tokyo._verify',
        sources=[os.path.join('tokyo','_verify.c')],
        include_dirs=tokyo_include_dirs,
        library_dirs=tokyo_library_dirs,
        extra_info=tokyo_extra_args,
        )

    config.add_extension(
        name='tokyo._single_speed',
        sources=[os.path.join('tokyo','_single_speed.c')],
        include_dirs=tokyo_include_dirs,
        library_dirs=tokyo_library_dirs,
        extra_info=tokyo_extra_args,
        )

    config.add_extension(
        name='tokyo._double_speed',
        sources=[os.path.join('tokyo','_double_speed.c')],
        include_dirs=tokyo_include_dirs,
        library_dirs=tokyo_library_dirs,
        extra_info=tokyo_extra_args,
        )

    config.add_extension(
        name='tokyo._demo_outer',
        sources=[os.path.join('tokyo','_demo_outer.c')],
        include_dirs=tokyo_include_dirs,
        library_dirs=tokyo_library_dirs,
        extra_info=tokyo_extra_args,
        )

    # Tokyo header file.
    config.add_data_files(('tokyo',os.path.join('tokyo','tokyo.pxd')))
    config.add_data_files(('tokyo',os.path.join('tokyo','__init__.py')))

    # Miscellaneous.
    config.add_subpackage(os.path.join('tokyo','misc'))

    config.make_config_py()

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
