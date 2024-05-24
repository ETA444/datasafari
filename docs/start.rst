Quick Start
-----------
Getting started with DataSafari is as easy as installing the package with ``pip``:

.. code-block:: bash

    pip install datasafari

Import DataSafari in your Python script:

.. code-block:: python

    import datasafari as ds

|

Hypothesis Testing? One line.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from datasafari.predictor import predict_hypothesis
    import pandas as pd
    import numpy as np

    # Create a sample DataFrame
    df_hypothesis = pd.DataFrame({
        'Group': np.random.choice(['Control', 'Treatment'], size=100),
        'Score': np.random.normal(0, 1, 100)
    })

    # Perform hypothesis testing
    results = predict_hypothesis(df_hypothesis, 'Group', 'Score')


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

    from datasafari.predictor import predict_ml
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
    best_models = predict_ml(df_ml, x_cols, y_col)


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
