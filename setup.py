from setuptools import setup, find_packages

setup(
    name='sith',
    version='0.0.5',
    packages=find_packages(include=['SITH', 'SITH.*', 'Utilities.*', 'SithWriter.*']),
    install_requires=[
        'numpy>=1.13.3',
        'ase>=3.22.1',
        'setuptools'
    ]
)