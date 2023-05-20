from setuptools import setup, find_packages

required_pkgs = ['numpy',
                 'scipy',
                 'matplotlib',
                 ]

setup(
    name='atomic_physics',
    version="0.0.1",
    description="",
    long_description="",
    author="",
    author_email="",
    packages=find_packages(include=['atomic_physics', 'atomic_physics.*']),
    python_requires='>=3.9',
    install_requires=required_pkgs,)