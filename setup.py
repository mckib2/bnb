from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [

    Extension(
        'bnb.simplex',
        ['src/csimplex.pyx', 'src/simplex.cpp'],
        include_dirs=['src/'],
        extra_compile_args=["-std=c++17", "-lstdc++"],
        extra_link_args=["-std=c++17"],
        language="c++"),
]

setup(
    ext_modules=cythonize(extensions),
)
