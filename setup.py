from setuptools import setup

setup(
    name='tdfpy',
    version='0.1.2',
    description='Pip package to work with bruker tdf and tdf_bin files',
    url='https://github.com/pgarrett-scripps/tdfpy',
    author='Patrick Garrett',
    author_email='pgarrett@scripps.edu',
    license='MIT',
    packages=['tdfpy'],
    package_dir={'tdfpy': 'tdfpy'},
    include_package_data=True,
    package_data={'tdfpy': ['timsdata.dll', 'libtimsdata.so']},
    install_requires=['pandas~=1.5.0',
                      'numpy~=1.23.3',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
    ],
)