from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='voxrm',
    version='0.1dev0',
    packages=find_packages(),
    long_description=open('README.md').read()
)