Welcome to DataSafari's Documentation!
=======================================

DataSafari simplifies complex data science tasks into straightforward, powerful commands. Whether you're exploring data, evaluating statistical assumptions, transforming datasets, or building predictive models, DataSafari provides all the tools you need in one package.

.. _quick-start:

Quick Start
-----------
Getting started with DataSafari is as easy as installing the package and running a few lines of code:

.. code-block:: bash

    pip install datasafari

|

Hypothesis Testing? One line.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from datasafari import predictor
    import pandas as pd
    import numpy as np

    # Create a sample DataFrame
    df_hypothesis = pd.DataFrame({
        'Group': np.random.choice(['Control', 'Treatment'], size=100),
        'Score': np.random.normal(0, 1, 100)
    })

    # Perform hypothesis testing
    results = predictor.predict_hypothesis(df_hypothesis, 'Group', 'Score')


**How DataSafari Streamlines Hypothesis Testing:**

- **Automatic Test Selection**: Depending on the data types, ``predict_hypothesis()`` automatically selects the appropriate test. It uses Chi-square, Fisher's exact test or other exact tests for categorical pairs, and T-tests, ANOVA and others for categorical and numerical combinations, adapting based on group counts, sample size and data distribution.

- **Assumption Verification**: Essential assumptions for the chosen tests are automatically checked.
    - **Normality**: Normality is verified using tests like Shapiro-Wilk or Anderson-Darling, essential for parametric tests.
    - **Variance Homogeneity**: Tests such as Levene’s or Bartlett’s are used to confirm equal variances, informing the choice between ANOVA types.

- **Comprehensive Output**:
    - **Justifications**: Provides comprehensive reasoning on all test choices.
    - **Test Statistics**: Key quantitative results from the hypothesis test.
    - **P-values**: Indicators of the statistical significance of the findings.
    - **Conclusions**: Clear textual interpretations of whether the results support or reject the hypothesis.

|

Machine Learning? You guessed it.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from datasafari import predictor
    import pandas as pd
    import numpy as np

    # Create another sample DataFrame for ML
    df_ml = pd.DataFrame({
        'Age': np.random.randint(20, 60, size=100),
        'Salary': np.random.normal(50000, 15000, size=100),
        'Experience': np.random.randint(1, 20, size=100)
    })
    x_cols = ['Age', 'Experience']  # Feature columns
    y_col = 'Salary'  # Target column

    # Find the best models for your data
    best_models = predictor.predict_ml(df_ml, x_cols, y_col)


**How DataSafari Simplifies Machine Learning Model Selection:**

- **Tailored Data Preprocessing**: The function automatically processes various types of data (numerical, categorical, text, datetime), preparing them optimally for machine learning.
    - Numerical data might be scaled or normalized.
    - Categorical data can be encoded.
    - Text data might be vectorized using techniques suitable for the analysis.

- **Intelligent Model Evaluation:** The function evaluates a variety of models using a composite score that synthesizes performance across multiple metrics, taking into account the multidimensional aspects of model performance.
    - **Composite Score Calculation**: Scores for each metric are weighted according to specified priorities by the user, with lower weights assigned to non-priority metrics (e.g. RMSE over MAE). This composite score serves as a holistic measure of model performance, ensuring that the models recommended are not just good in one aspect but are robust across multiple criteria.

- **Automated Hyperparameter Tuning:** Once the top models are identified based on the composite score, the pipeline employs techniques like grid search, random search, or Bayesian optimization to fine-tune the models.
    - **Output of Tuned Models**: The best configurations for the models are output, along with their performance metrics, allowing users to make informed decisions about which models to deploy based on robust, empirically derived data.

- **Customization Options & Sensible Defaults:** Users can define custom hyperparameter grids, select specific tuning algorithms, prioritize models, tailor data preprocessing, and prioritize metrics.
    - **Accessibility**: Every part of the process is in the hands of the user, but sensible defaults are provided for ultimate simplicity of use, which is the approach for ``datasafari`` in general.

----

.. _glance:

DataSafari at a Glance
-----------------------
DataSafari is organized into several subpackages, each tailored to specific data science tasks.

    *The logic behind the naming of each subpackage is inspired by the typical data workflow: exploring and understanding your data, transforming and cleaning it, evaluating assumptions and finally making predictions.* - George

.. toctree::
   :maxdepth: 3
   :hidden:

   self

.. toctree::
   :maxdepth: 1
   :caption: Explorers
   :hidden:

   datasafari.explorer.explore_df
   datasafari.explorer.explore_num
   datasafari.explorer.explore_cat

.. toctree::
   :maxdepth: 1
   :caption: Transformers
   :hidden:

   datasafari.transformer.transform_num
   datasafari.transformer.transform_cat

.. toctree::
   :maxdepth: 1
   :caption: Evaluators
   :hidden:

   datasafari.evaluator.evaluate_normality
   datasafari.evaluator.evaluate_variance
   datasafari.evaluator.evaluate_dtype
   datasafari.evaluator.evaluate_contingency_table

.. toctree::
   :maxdepth: 1
   :caption: Predictors
   :hidden:

   datasafari.predictor.predict_hypothesis
   datasafari.predictor.predict_ml

|

Explorers
~~~~~~~~~~
**Explore and understand your data in depth and quicker than ever before.**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`explore_df() <datasafari.explorer.explore_df>`
     - Explore a DataFrame and gain a birds-eye view of summary statistics, NAs, data types and more.
   * - :doc:`explore_num() <datasafari.explorer.explore_num>`
     - Explore numerical variables in a DataFrame and gain insights on distribution characteristics, outlier detection using multiple methods (Z-score, IQR, Mahalanobis), normality tests, skewness, kurtosis, correlation analysis, and multicollinearity detection.
   * - :doc:`explore_cat() <datasafari.explorer.explore_cat>`
     - Explore categorical variables within a DataFrame and gain insights on unique values, counts and percentages, and the entropy of variables to quantify data diversity.

For example, use ``explore_num()`` to gain detailed insights into numerical features.

.. code-block:: python

    from datasafari.explorer import explore_num
    import pandas as pd
    import numpy as np

    df_explorer = pd.DataFrame({
        'Age': np.random.randint(20, 60, size=100),
        'Income': np.random.normal(50000, 15000, size=100)
    })

    explore_num(df_explorer, ['Age', 'Income'])

|

Transformers
~~~~~~~~~~~~
**Clean, encode and enhance your data to prepare it for further analysis.**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`transform_num() <datasafari.transformer.transform_num>`
     - Transform numerical variables in a DataFrame through operations like standardization, log-transformation, various scalings, winsorization, and interaction term creation.
   * - :doc:`transform_cat() <datasafari.transformer.transform_cat>`
     - Transforms categorical variables in a DataFrame through a range of encoding options and basic to advanced machine learning-based methods for uniform data cleaning.

For example, use ``transform_cat()`` with the ``'uniform_smart'`` method for advanced, ML-based categorical data cleaning.

.. code-block:: python

    from datasafari.transformer import transform_cat
    import pandas as pd

    df_transformer = pd.DataFrame({
        'Category': ['low', 'medium', 'Medium', 'High', 'low', 'high']
    })

    transformed_df, uniform_columns = transform_cat(
        df_transformer,
        ['Category'],
        method='uniform_smart'
    )


|

Evaluators
~~~~~~~~~~
**Ensure your data meets the required assumptions for analyses.**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`evaluate_normality() <datasafari.evaluator.evaluate_normality>`
     - Evaluate normality of numerical data within groups defined by a categorical variable, employing multiple statistical tests, dynamically chosen based on data suitability.
   * - :doc:`evaluate_variance() <datasafari.evaluator.evaluate_variance>`
     - Evaluate variance homogeneity across groups defined by a categorical variable within a dataset, using several statistical tests, dynamically chosen based on data suitability.
   * - :doc:`evaluate_dtype() <datasafari.evaluator.evaluate_dtype>`
     - Evaluate and automatically categorize the data types of DataFrame columns, effectively distinguishing between ambiguous cases based on detailed logical assessments.
   * - :doc:`evaluate_contingency_table() <datasafari.evaluator.evaluate_contingency_table>`
     - Evaluate the suitability of statistical tests for a given contingency table by analyzing its characteristics and guiding the selection of appropriate tests.

For example, use ``evaluate_normality()`` to check if data distribution fits normality, running the most appropriate normality tests and utilizing a consensus mechanism making for a robust decision on normality.

.. code-block:: python

    from datasafari.evaluator import evaluate_normality
    import pandas as pd
    import numpy as np

    df_evaluator = pd.DataFrame({
        'Data': np.random.normal(0, 1, size=100)
    })

    normality = evaluate_normality(df_evaluator, 'Data')

|

Predictors
~~~~~~~~~~
**Streamline model building and hypothesis testing.**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`predict_hypothesis() <datasafari.predictor.predict_hypothesis>`
     - Conduct the optimal hypothesis test on a DataFrame, tailoring the approach based on the variable types and automating the testing prerequisites and analyses, outputting test results and interpretation.
   * - :doc:`predict_ml() <datasafari.predictor.predict_ml>`
     - Streamline the entire process of data preprocessing, model selection, and tuning, delivering optimal model recommendations based on the data provided.

For example, use ``predict_ml()`` to preprocess your data, tune models and get the top ML models for your data.

.. code-block:: python

    from datasafari.predictor import predict_ml
    import pandas as pd
    import numpy as np

    df_ml_predictor = pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Target': np.random.randint(0, 2, size=100)
    })

    ml_results = predict_ml(
        df_ml_predictor,
        x_cols=['Feature1', 'Feature2'],
        y_col='Target'
    )

.. toctree::
   :maxdepth: 1
   :caption: Indices
   :hidden:

   General Index <genindex>
   modindex
   search

----

.. _contact:

Contact
-------

**Contact me on:**
    - **LinkedIn** `George Dreemer <https://www.linkedin.com/in/georgedreemer>`_
    - **Website** `georgedreemer.com <https://www.georgedreemer.com>`_

    *Thank you very much for taking an interest in DataSafari.* |:heart:| - George

.. toctree::
   :maxdepth: 3
   :hidden:

   index#contact
