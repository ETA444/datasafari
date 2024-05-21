Welcome to Data Safari's Documentation!
=======================================

Data Safari simplifies complex data science tasks into straightforward, powerful commands. Whether you're exploring data, evaluating statistical assumptions, transforming datasets, or building predictive models, Data Safari provides all the tools you need in one package.

Quick Start
-----------
Getting started with Data Safari is as easy as installing the package and running a few lines of code:

.. code-block:: bash

    pip install datasafari

**Hypothesis Testing Example:**

.. code-block:: python

    from datasafari import predictor
    import pandas as pd
    import numpy as np

    # Create a sample DataFrame
    df = pd.DataFrame({
        'Group': np.random.choice(['Control', 'Treatment'], size=100),
        'Score': np.random.normal(0, 1, 100)
    })

    # Perform hypothesis testing
    results = predictor.predict_hypothesis(df, 'Group', 'Score')
    print(results)

- This function intelligently determines the best statistical test based on the data types and distribution:
  - Automatically identifies whether to perform a T-test, ANOVA, or other relevant tests.
  - Evaluates assumptions such as normality and homogeneity of variances if needed.

**Machine Learning Model Selection Example:**

.. code-block:: python

    # Assuming 'df' is a DataFrame loaded with relevant features and a target variable
    x_cols = df.columns[:-1]  # Feature columns
    y_col = df.columns[-1]    # Target column

    # Run the machine learning pipeline to recommend the best model
    ml_results = predictor.predict_ml(df, x_cols=x_cols, y_col=y_col)
    print(ml_results)

- The ML pipeline automates the entire process of model selection:
  - Preprocesses data based on the types of variables.
  - Evaluates several models and tunes them to find the best performer.
  - Utilizes a composite score to assess and rank models, ensuring optimal selection.

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
Discover patterns and insights in your data effortlessly. For instance, use `explore_df` to get a comprehensive overview of any DataFrame, helping to quickly identify missing values, outliers, and distribution patterns.

.. autosummary::
   :toctree: explorers

   datasafari.explorer.explore_df
   datasafari.explorer.explore_num
   datasafari.explorer.explore_cat

Transformers
------------
Transform your data to better fit your analysis or machine learning models. For example, `transform_cat` can be used to encode categorical variables effectively, preparing them for machine learning algorithms.

.. autosummary::
   :toctree: transformers

   datasafari.transformer.transform_num
   datasafari.transformer.transform_cat

Evaluators
----------
Evaluate assumptions, validate models, and ensure the quality of your data. Use `evaluate_normality` to check if your data conforms to a normal distribution, which is crucial for many statistical tests and models.

.. autosummary::
   :toctree: evaluators

   datasafari.evaluator.evaluate_dtype
   datasafari.evaluator.evaluate_normality
   datasafari.evaluator.evaluate_variance

Predictors
----------
Build predictive models and conduct hypothesis testing with ease. For example, `predict_hypothesis` can dynamically choose the best hypothesis test for your data, whether it's for simple comparisons or complex experimental designs.

.. autosummary::
   :toctree: predictors

   datasafari.predictor.predict_hypothesis
   datasafari.predictor.predict_ml

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
