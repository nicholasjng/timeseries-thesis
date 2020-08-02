from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [Extension(name="regularity_measures", sources=["regularity_measures.pyx"], include_dirs=[np.get_include()])]

setup(
      ext_modules = cythonize(ext_modules)
      )