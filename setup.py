from setuptools import setup, find_packages

setup(
    name="ptmpi",
    version='0.0',
    description='',
    long_description='',
    author="Chris Self",
    author_email='',
    license='?',
    home_page='',
    packages=find_packages('packages'),
    package_dir={'': 'packages'},
    install_requires=[
        'mpi4py',
        'numpy',
    ]
)
