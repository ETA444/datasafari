#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'numpy',
    'pandas',
    'scikit-learn<1.5',
    'scikit-optimize>=0.10.1',
    'scipy',
    'matplotlib',
    'seaborn',
    'statsmodels',
    'category-encoders',
    'python-Levenshtein'
]

dev_requirements = [
    'pytest>=8.2.0',
    'flake8>=7.0.0',
    'tox>=4.12.1',
    'twine>=5.0.0',
    'wheel>=0.37.0',
    'setuptools>=58.0.0',
    'cookiecutter>=2.5.0',
    'sphinx>=7.3.7',
    'furo>=2024.5.6',
    'sphinx-favicon>=1.0.1',
    'sphinx-prompt>=1.8.0',
    'sphinx-copybutton>=0.5.2',
    'sphinxemoji>=0.3.1',
    'sphinxext-opengraph>=0.6.0'
]

setup(
    author="George Dreemer",
    author_email='georgedreemer@proton.me',
    python_requires='>=3.9, <4',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
    description="DataSafari simplifies complex data science tasks into straightforward, powerful one-liners. Whether you're exploring data, evaluating statistical assumptions, transforming datasets, or building predictive models, DataSafari provides all the tools you need in one package.",
    install_requires=requirements,
    extras_require={
        'docs': [
            'sphinx>=7.3.7',
            'furo>=2024.5.6',
            'sphinx-favicon>=1.0.1',
            'sphinx-prompt>=1.8.0',
            'sphinx-copybutton>=0.5.2',
            'sphinxemoji>=0.3.1',
            'sphinxext-opengraph>=0.6.0'
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
    project_urls={
        'Documentation': 'https://www.datasafari.dev/docs',
        'Official Website': 'https://www.datasafari.dev',
        'Source Code': 'https://github.com/ETA444/datasafari'
    },
    version='1.0.0',
    zip_safe=False,
)
