#!/usr/bin/env python
def configuration(parent_package='',top_path=None):
    import numpy
    import os
    import ConfigParser
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError

    # Read relevant Tokyo-specific configuration options.
    tokyo_config = ConfigParser.SafeConfigParser()
    tokyo_config.read(os.path.join(top_path, 'site.cfg'))

    config = Configuration('tokyo', parent_package, top_path)

    # Get info from site.cfg
    blas_info = get_info('blas_opt',0)
    if not blas_info:
        print 'No blas info found'

    lapack_info = get_info('lapack_opt',0)
    if not lapack_info:
        print 'No lapack info found'

    tokyo_extra_args = dict(blas_info, **lapack_info)

    config.add_extension(
        name='tokyo',
        sources=['tokyo.c'],
        include_dirs=['/usr/include','/System/Library/Frameworks/vecLib.framework/Versions/A/Headers'],
        extra_info=tokyo_extra_args,
        )

    config.add_extension(
        name='_verify',
        sources=['verify.c'],
        include_dirs=['/usr/include','/System/Library/Frameworks/vecLib.framework/Versions/A/Headers'],
        )

    config.add_extension(
        name='_single_speed',
        sources=['single_speed.c'],
        include_dirs=['/usr/include','/System/Library/Frameworks/vecLib.framework/Versions/A/Headers'],
        )

    config.add_extension(
        name='_double_speed',
        sources=['double_speed.c'],
        include_dirs=['/usr/include','/System/Library/Frameworks/vecLib.framework/Versions/A/Headers'],
        )

    config.add_extension(
        name='_demo_outer',
        sources=['demo_outer.c'],
        include_dirs=['/usr/include','/System/Library/Frameworks/vecLib.framework/Versions/A/Headers'],
        )

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

#ext_modules = [
#    Extension("tokyo",        ["tokyo.pyx"],        **tokyo_ext_params),
#    Extension("verify",       ["verify.pyx"],       **ext_params),
#    Extension("single_speed", ["single_speed.pyx"], **ext_params),
#    Extension("double_speed", ["double_speed.pyx"], **ext_params),
#    Extension("demo_outer",   ["demo_outer.pyx"],   **ext_params)
#]
