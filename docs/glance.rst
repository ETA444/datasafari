DataSafari at a Glance
-----------------------
DataSafari is organized into several subpackages, each tailored to specific data science tasks.

    *The logic behind the naming of each subpackage is inspired by the typical data workflow: exploring and understanding your data, transforming and cleaning it, evaluating assumptions and finally making predictions.* - George

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
