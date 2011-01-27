from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

import numpy as np

include_dirs = ['/usr/include', np.get_include()]
library_dirs = ['/usr/lib']

ext_modules=[
    Extension("tokyo", ["tokyo.pyx"],
#              libraries=['lapack', 'lapack_atlas', 'blas', 'atlas'],
              libraries=['blas', 'atlas'],
              library_dirs=library_dirs, include_dirs=include_dirs),
    Extension("verify",       ["verify.pyx"],       include_dirs=include_dirs),
    Extension("single_speed", ["single_speed.pyx"], include_dirs=include_dirs),
    Extension("double_speed", ["double_speed.pyx"], include_dirs=include_dirs),
    Extension("demo_outer",   ["demo_outer.pyx"],   include_dirs=include_dirs)
]

setup(
#  name = 'BLAS and LAPACK wrapper',
  name = 'BLAS wrapper',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)
