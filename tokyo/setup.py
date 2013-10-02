#!/usr/bin/env python
import ConfigParser
import os


def get_from_config(config_name, section, field):
    try:
        value = config_name.get(section, field)
    except ConfigParser.NoOptionError:
        return []
    return value


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    # Read relevant Tokyo-specific configuration options.
    tokyo_config = ConfigParser.SafeConfigParser()
    tokyo_config.read(os.path.join(top_path, 'site.cfg'))

    config = Configuration('tokyo', parent_package, top_path)

    # Get info from site.cfg
    tokyo_library_dirs = get_from_config(tokyo_config,
                                         'DEFAULT',
                                         'library_dirs').split(os.pathsep)
    tokyo_include_dirs = get_from_config(tokyo_config,
                                         'DEFAULT',
                                         'include_dirs').split(os.pathsep)
    blas_libs = get_from_config(tokyo_config,
                                'blas_opt',
                                'libraries').split(' ')
    lapack_libs = get_from_config(tokyo_config,
                                  'lapack_opt',
                                  'libraries').split(' ')
    tokyo_libs = blas_libs + lapack_libs

    config.add_extension(
        name='tokyo',
        sources=['tokyo.c'],
        include_dirs=tokyo_include_dirs,
        library_dirs=tokyo_library_dirs,
        libraries=tokyo_libs,
    )

    config.add_extension(
        name='_verify',
        sources=['_verify.c'],
        include_dirs=tokyo_include_dirs,
        library_dirs=tokyo_library_dirs,
        libraries=tokyo_libs,
    )

    config.add_extension(
        name='_single_speed',
        sources=['_single_speed.c'],
        include_dirs=tokyo_include_dirs,
        library_dirs=tokyo_library_dirs,
        libraries=tokyo_libs,
    )

    config.add_extension(
        name='_double_speed',
        sources=['_double_speed.c'],
        include_dirs=tokyo_include_dirs,
        library_dirs=tokyo_library_dirs,
        libraries=tokyo_libs,
    )

    config.add_extension(
        name='_demo_outer',
        sources=['_demo_outer.c'],
        include_dirs=tokyo_include_dirs,
        library_dirs=tokyo_library_dirs,
        libraries=tokyo_libs,
    )

    # Tokyo header file.
    config.add_data_files(os.path.join(config.top_path, 'tokyo', 'tokyo.pxd'))

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
