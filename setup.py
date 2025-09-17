from setuptools import setup, Extension
import pybind11

# Define the C++ extension module
cpp_args = ['-std=c++17', '-O3']

ext_modules = [
    Extension(
        'bmss_p_cpp', # The name of the module in Python
        ['bmss_p.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='bmss_p_cpp',
    version='0.0.1',
    author='Your Name',
    description='C++ implementation of BMSSP for Python',
    ext_modules=ext_modules,
)
