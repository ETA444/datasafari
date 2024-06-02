#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'numpy',
    'pandas',
    'scikit-learn',
    'scikit-optimize',
    'scipy',
    'matplotlib',
    'seaborn',
    'statsmodels',
    'category-encoders',
    'python-Levenshtein'
]

dev_requirements = [
    'pytest',
    'flake8',
    'tox',
    'twine',
    'cookiecutter',
    'sphinx',
    'sphinx-furo',
    'sphinx-favicon',
    'sphinx-prompt',
    'sphinx-copybutton',
    'sphinxemoji',
    'sphinx-opengraph'
]

setup(
    author="George Dreemer",
    author_email='georgedreemer@proton.me',
    python_requires='>=3.7, <4',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
    description="DataSafari simplifies complex data science tasks into straightforward, powerful commands. Whether you're exploring data, evaluating statistical assumptions, transforming datasets, or building predictive models, DataSafari provides all the tools you need in one package.",
    install_requires=requirements,
    extras_require={
        'docs': [
            'sphinx',
            'furo',
            'sphinx-autobuild',
            'sphinx_favicon',
            'sphinx_prompt',
            'sphinx_copybutton',
            'sphinxemoji'
        ],
        'dev': dev_requirements
    },
    license="GNU General Public License v3",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords=[
        'data science', 'data analysis', 'machine learning', 'data preprocessing',
        'statistical testing', 'data transformation', 'predictive modeling',
        'data visualization', 'exploratory data analysis', 'hypothesis testing',
        'feature engineering', 'model evaluation', 'model tuning', 'data cleaning',
        'data insights', 'numerical analysis', 'categorical data', 'statistics',
        'ML automation', 'data workflow', 'data discovery', 'sklearn integration',
        'statistical inference', 'automated machine learning', 'data exploration'
    ],
    name='datasafari',
    packages=find_packages(include=['datasafari', 'datasafari.*']),
    test_suite='tests',
    url='https://github.com/ETA444/datasafari',
    version='1.0.0',
    zip_safe=False,
)
