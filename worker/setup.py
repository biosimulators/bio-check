import os
from setuptools import setup, find_packages

from Cython.Build import cythonize


SOURCE_DIR = "datagen_src"
BUILD_DIR = "datagen_build"

os.makedirs(BUILD_DIR, exist_ok=True)


setup(
    ext_modules=cythonize(f"{SOURCE_DIR}/*.pyx"),  # Compile all .pyx files in src/
    options={
        "build": {"build_base": BUILD_DIR},        # Set custom build directory
        "build_ext": {"build_lib": BUILD_DIR},    # Place .so files in build/
    },
    packages=["service"]
)

# TODO: do this PRIOR to building the image:
# python setup.py build_ext
