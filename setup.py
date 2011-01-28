#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

ext_params = {}
ext_params['include_dirs'] = [
    '/usr/include',
    '/System/Library/Frameworks/vecLib.framework/Versions/A/Headers',
    np.get_include()]
ext_params['extra_compile_args'] = ["-O2"]
ext_params['extra_link_args'] = ["-Wl,-O1", "-Wl,--as-needed"]  # TODO: as-needed
    # ignored due to parameter order bug in distutils (when calling linker)

tokyo_ext_params = ext_params.copy()
tokyo_ext_params['libraries'] = ['blas', 'lapack']  # TODO: detect library name.
    # Candidates: blas, cblas, lapack, lapack_atlas, atlas
    # On OSX, blas points to the Accelerate framework's ATLAS library.
tokyo_ext_params['library_dirs'] = ['/usr/lib']  # needed by OSX, perhaps

ext_modules = [
    Extension("tokyo",        ["tokyo.pyx"],        **tokyo_ext_params),
    Extension("verify",       ["verify.pyx"],       **ext_params),
    Extension("single_speed", ["single_speed.pyx"], **ext_params),
    Extension("double_speed", ["double_speed.pyx"], **ext_params),
    Extension("demo_outer",   ["demo_outer.pyx"],   **ext_params)
]

setup(
    #name='BLAS and LAPACK wrapper',
    name='BLAS wrapper',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
