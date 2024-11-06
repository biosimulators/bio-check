import os
from setuptools import setup, find_packages


with open(os.path.join(os.path.dirname(__file__), '.VERSION'), 'r') as version_file:
    VERSION = version_file.read().strip()


setup(
    name="bio-compose-server-worker",
    version=VERSION,
    author="Alexander Patrie",
    author_email="",
    description="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires="^3.10",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # install_requires=[
    #     "bigraph-schema",
    #     "numpy",
    #     "pytest>=6.2.5",
    #     "pymongo",
    #     "orjson",
    #     "matplotlib"
    # ],
    # packages=[
    #     'process_bigraph',
    #     'process_bigraph.processes',
)
