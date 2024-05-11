#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'numpy==1.26.2',
    'pandas==2.1.4',
    'scikit-learn==1.0.2',
    'scikit-optimize==0.10.1',
    'scipy==1.12.0',
    'matplotlib==3.8.2',
    'seaborn==0.13.0',
    'statsmodels==0.13.2',
    'category-encoders==2.6.3',
    'Levenshtein==0.25.0'
]


test_requirements = ['pytest>=3', ]

setup(
    author="George Dreemer",
    author_email='georgedreemer@proton.me',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="DataSafari simplifies complex data exploration and transformation.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='datasafari',
    name='datasafari',
    packages=find_packages(include=['datasafari', 'datasafari.*']),
    test_suite='tests',
    tests_require=requirements,
    url='https://github.com/ETA444/datasafari',
    version='0.1.0',
    zip_safe=False,
)
