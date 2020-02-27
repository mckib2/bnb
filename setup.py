from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [

    Extension(
        'bnb.simplex',
        ['src/csimplex.pyx', 'src/simplex.cpp'],
        include_dirs=['src/'],
        extra_compile_args=["-std=c++17", "-O3"],
        extra_link_args=[],
        language="c++"),
]

setup(
    ext_modules=cythonize(extensions),
)
