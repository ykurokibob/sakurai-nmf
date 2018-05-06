"""Setup script for NMF-NN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

setuptools.setup(
    name='sakurai_nmf',
    version='0.0.1',
    description=(
        'optimize neural network with non-negative matrix factorization'
        'algorithms.'),
    license='Apache 2.0',
    url='http://github.com/ashigirl96/sakurai-nmf',
    install_requires=[
        'tensorflow-gpu',
        'keras',
        'agents',
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
    ],
)
