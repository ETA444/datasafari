#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'numpy>=1.26.2',
    'pandas>=2.1.4',
    'scikit-learn>=1.0.2',
    'scikit-optimize>=0.10.1',
    'scipy>=1.12.0',
    'matplotlib>=3.8.2',
    'seaborn>=0.13.0',
    'statsmodels>=0.13.2',
    'category-encoders>=2.6.3',
    'Levenshtein>=0.25.0'
]

dev_requirements = [
    'pytest>=3',
    'flake8>=7.0.0',
    'tox>=4.12.1',
    'twine>=5.0.0',
    'cookiecutter>=2.5.0',
    'sphinx>=7.3.7',
    'sphinx-basic-ng>=1.0.0b2',
    'furo>=2024.5.6',
    'sphinx-favicon>=1.0.1',
    'sphinx-prompt>=1.8.0',
    'sphinx-copybutton>=0.5.2',
    'schnxemoji>=0.3.1'
]

setup(
    author="George Dreemer",
    author_email='georgedreemer@proton.me',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="DataSafari makes exploring, transforming and making predictions with your data simple, logical and potent.",
    install_requires=requirements,
    extras_require={
        'docs': [
            'sphinx',
            'furo',  # docs theme
            'sphinx-autobuild',  # for live reloading the docs
            'sphinx_favicon',  # for favicon functionality
            'sphinx_prompt',  # for adding unselectable prompts to code snippets
            'sphinx_copybutton',  # for adding a copy button to code blocks
            'sphinxemoji',  # for emoji support in the docs
        ],
        'dev': dev_requirements  # include all development dependencies
    },
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='datasafari',
    name='datasafari',
    packages=find_packages(include=['datasafari', 'datasafari.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ETA444/datasafari',
    version='1.0.0',
    zip_safe=False,
)
