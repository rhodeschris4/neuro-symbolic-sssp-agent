from setuptools import setup, Extension
import pybind11
import sys

# --- NEW: Platform-specific compiler arguments ---
# MSVC (Windows) uses /std:c++17, while GCC/Clang (macOS/Linux) use -std=c++17
if sys.platform == 'win32':
    cpp_args = ['/std:c++17', '/O2', '/EHsc']
else:
    cpp_args = ['-std=c++17', '-O3', '-march=native']
# ---------------------------------------------

ext_modules = [
    Extension(
        'bmss_p_cpp',
        ['bmss_p.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args, # Use the platform-specific args here
    ),
]

setup(
    name='bmss_p_cpp',
    version='0.0.1',
    author='Your Name',
    description='C++ implementation of BMSSP for Python',
    ext_modules=ext_modules,
)
