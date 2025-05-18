import os
from distutils.core import setup
from Cython.Build import cythonize
import numpy
# python process_setup.py build_ext --inplace
setup(ext_modules=cythonize("process_spectrum.pyx"),
      include_dirs=[numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
