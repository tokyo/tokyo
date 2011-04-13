#!/usr/bin/env python
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

    # Read relevant Tokyo-specific configuration options.
    tokyo_config = ConfigParser.SafeConfigParser()
    tokyo_config.read(os.path.join(top_path, 'site.cfg'))

    config = Configuration('tokyo', parent_package, top_path)

    # Get info from site.cfg
    tokyo_library_dirs = get_from_config(tokyo_config,'DEFAULT','library_dirs')
    tokyo_include_dirs = get_from_config(tokyo_config,'DEFAULT','include_dirs')

    blas_info = get_info('blas_opt',0)
    if not blas_info:
        blas_info = get_info('blas',0)
        if not blas_info:
            print 'No blas info found'

    lapack_info = get_info('lapack_opt',0)
    if not lapack_info:
        lapack_info = get_info('lapack',0)
        if not lapack_info:
            print 'No lapack info found'

    tokyo_extra_args = dict(blas_info, **lapack_info)

    config.add_extension(
        name='tokyo',
        sources=['tokyo.c'],
        include_dirs=tokyo_include_dirs,
        extra_info=tokyo_extra_args,
        )

    config.add_extension(
        name='_verify',
        sources=['_verify.c'],
        include_dirs=tokyo_include_dirs,
        extra_info=tokyo_extra_args,
        )

    config.add_extension(
        name='_single_speed',
        sources=['_single_speed.c'],
        include_dirs=tokyo_include_dirs,
        extra_info=tokyo_extra_args,
        )

    config.add_extension(
        name='_double_speed',
        sources=['_double_speed.c'],
        include_dirs=tokyo_include_dirs,
        extra_info=tokyo_extra_args,
        )

    config.add_extension(
        name='_demo_outer',
        sources=['_demo_outer.c'],
        include_dirs=tokyo_include_dirs,
        extra_info=tokyo_extra_args,
        )

    # Tokyo header file.
    config.add_data_files(os.path.join(config.top_path,'tokyo','tokyo.pxd'))

    # Miscellaneous.
    config.add_subpackage('misc')

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
