#!/usr/bin/env python3

from distutils.core import setup

setup(name='Pyromania',
	version='0.1',
	description='Python facade for a variety of tree models from R',
	url='https://github.com/trygvebw/pyromania',
	packages=['pyromania'],
	install_requires=[
		'pandas',
		'numpy',
		'scipy',
		'rpy2',
		'scikit-learn',
	],
	python_requires='>=3.4')