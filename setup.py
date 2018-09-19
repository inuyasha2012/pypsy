#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'progressbar2', 'scipy']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="chris dai",
    author_email='inuyasha021@163.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="psychometrics package, including structural equation model, "
                "confirmatory factor analysis, unidimensional item response theory, "
                "multidimensional item response theory, cognitive diagnosis model, "
                "factor analysis and adaptive testing. ",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='psy',
    name='psy',
    packages=find_packages(include=['psy']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/inuyasha2012/pypsy',
    version='0.0.1',
    zip_safe=False,
)
