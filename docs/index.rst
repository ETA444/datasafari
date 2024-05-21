Welcome to Data Safari's Documentation!
=======================================

Data Safari simplifies complex data science tasks into straightforward, powerful commands. Whether you're exploring data, evaluating statistical assumptions, transforming datasets, or building predictive models, Data Safari provides all the tools you need in one package.

Quick Start
-----------
Getting started with Data Safari is as easy as installing the package and running a few lines of code:

.. code-block:: bash

    pip install datasafari

Here’s a quick example of using Data Safari to perform hypothesis testing on your data:

.. code-block:: python

    from datasafari import predictor
    import pandas as pd
    import numpy as np

    # Create a sample DataFrame
    df = pd.DataFrame({
        'Group': np.random.choice(['Control', 'Treatment'], size=100),
        'Score': np.random.normal(0, 1, 100)
    })

    # Perform a simple T-test
    results = predictor.predict_hypothesis(df, 'Group', 'Score')
    print(results)

Explore the Documentation
-------------------------
Dive deeper into what Data Safari can do for your data analysis tasks:

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   introduction
   installation
   quickstart
   examples

Subpackages
-----------
Data Safari is organized into several subpackages, each tailored to specific data science tasks:

.. toctree::
   :maxdepth: 2
   :caption: Subpackages Overview

   explorers
   transformers
   evaluators
   predictors

Explorers
---------
Discover patterns and insights in your data effortlessly.

.. autosummary::
   :toctree: explorers

   datasafari.explorer.explore_df
   datasafari.explorer.explore_num
   datasafari.explorer.explore_cat

Transformers
------------
Transform your data to better fit your analysis or machine learning models.

.. autosummary::
   :toctree: transformers

   datasafari.transformer.transform_num
   datasafari.transformer.transform_cat

Evaluators
----------
Evaluate assumptions, validate models, and ensure the quality of your data.

.. autosummary::
   :toctree: evaluators

   datasafari.evaluator.evaluate_dtype
   datasafari.evaluator.evaluate_normality
   datasafari.evaluator.evaluate_variance

Predictors
----------
Build predictive models and conduct hypothesis testing with ease.

.. autosummary::
   :toctree: predictors

   datasafari.predictor.predict_hypothesis
   datasafari.predictor.predict_ml

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
