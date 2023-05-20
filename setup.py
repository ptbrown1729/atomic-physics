from setuptools import setup, find_packages

required_pkgs = ['numpy',
                 'scipy',
                 'matplotlib',
                 ]

setup(
    name='atomic_physics',
    version="0.0.1",
    description="Various tools for doing atomic physics calculations,"
                " including calculating eigenstates of the hyperfine and"
                " Zeeman interactions versus magnetic field (i.e. solving the Breit-Rabi problem)"
                " and calculating branching ratios for alkali atoms decaying from the D1 or D2 transition.",
    long_description="",
    author="Peter T. Brown",
    author_email="ptbrown1729@gmail.com",
    packages=find_packages(include=['atomic_physics', 'atomic_physics.*']),
    python_requires='>=3.9',
    install_requires=required_pkgs,)