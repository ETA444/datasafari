![image info](./docs/_static/thumbs/ds-branding-thumb-main-web.png)
# Welcome to DataSafari's Source Code!

DataSafari simplifies complex data science tasks into straightforward, powerful commands. Whether you're exploring data, evaluating statistical assumptions, transforming datasets, or building predictive models, DataSafari provides all the tools you need in one package.

> In this README you can find a brief overview of how to start using DataSafari and what features you can utilize. For a more complete presentation you can visit [DataSafari's docs](https://www.google.com).

## Quick Start

### Installation

To get started with DataSafari, install it using pip:

```console
pip install datasafari
```

### Importing

Import DataSafari in your Python script to begin:

```python
import datasafari as ds
```

For detailed installation options, including installing from source, check our [Installation Guide in the docs](https://www.google.com).

## DataSafari at a Glance

DataSafari is organized into several subpackages, each tailored to specific data science tasks.

> *The logic behind the naming of each subpackage is inspired by the typical data workflow: exploring and understanding your data, transforming and cleaning it, evaluating assumptions and finally making predictions.* - George

### Explorers

**Explore and understand your data in depth and quicker than ever before.**

| Module         | Description                                                                                                                                                                       |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `explore_df()` | Explore a DataFrame and gain a birds-eye view of summary statistics, NAs, data types and more.                                                                                    |
| `explore_num()`| Explore numerical variables in a DataFrame and gain insights on distribution characteristics, outlier detection using multiple methods (Z-score, IQR, Mahalanobis), normality tests, skewness, kurtosis, correlation analysis, and multicollinearity detection. |
| `explore_cat()`| Explore categorical variables within a DataFrame and gain insights on unique values, counts and percentages, and the entropy of variables to quantify data diversity.              |

### Transformers

**Clean, encode and enhance your data to prepare it for further analysis.**

| Module          | Description                                                                                                                                                               |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `transform_num()`| Transform numerical variables in a DataFrame through operations like standardization, log-transformation, various scalings, winsorization, and interaction term creation. |
| `transform_cat()`| Transforms categorical variables in a DataFrame through a range of encoding options and basic to advanced machine learning-based methods for uniform data cleaning.       |

### Evaluators

**Ensure your data meets the required assumptions for analyses.**

| Module                         | Description                                                                                                                                                                              |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `evaluate_normality()`         | Evaluate normality of numerical data within groups defined by a categorical variable, employing multiple statistical tests, dynamically chosen based on data suitability.                |
| `evaluate_variance()`          | Evaluate variance homogeneity across groups defined by a categorical variable within a dataset, using several statistical tests, dynamically chosen based on data suitability.           |
| `evaluate_dtype()`             | Evaluate and automatically categorize the data types of DataFrame columns, effectively distinguishing between ambiguous cases based on detailed logical assessments.                    |
| `evaluate_contingency_table()` | Evaluate the suitability of statistical tests for a given contingency table by analyzing its characteristics and guiding the selection of appropriate tests.                             |

### Predictors

**Streamline model building and hypothesis testing.**

| Module                | Description                                                                                                                                                                           |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `predict_hypothesis()`| Conduct the optimal hypothesis test on a DataFrame, tailoring the approach based on the variable types and automating the testing prerequisites and analyses, outputting test results and interpretation. |
| `predict_ml()`        | Streamline the entire process of data preprocessing, model selection, and tuning, delivering optimal model recommendations based t on the data provided.                             |


## DataSafari in Action

### Hypothesis Testing? One line.

```python
from datasafari.predictor import predict_hypothesis
import pandas as pd
import numpy as np

# Sample DataFrame
df_hypothesis = pd.DataFrame({
    'Group': np.random.choice(['Control', 'Treatment'], size=100),
    'Score': np.random.normal(0, 1, 100)
})

# Perform hypothesis testing
results = predict_hypothesis(df_hypothesis, 'Group', 'Score')
```

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

### Machine Learning? You guessed it.

```python
from datasafari.predictor import predict_ml
import pandas as pd
import numpy as np

# Another sample DataFrame for ML
df_ml = pd.DataFrame({
    'Age': np.random.randint(20, 60, size=100),
    'Salary': np.random.normal(50000, 15000, size=100),
    'Experience': np.random.randint(1, 20, size=100)
})

x_cols = ['Age', 'Experience']
y_col = 'Salary'

# Discover the best models for your data
best_models = predict_ml(df_ml, x_cols, y_col)
```

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
## License

DataSafari is licensed under the GNU General Public License v3.0. This ensures that all modifications and derivatives of this project remain open-source and freely available under the same terms.


## Contact

Connect with me on [LinkedIn](https://www.linkedin.com/in/georgedreemer) or visit my [website](https://www.georgedreemer.com).

> Thank you very much for taking an interest in DataSafari! :green_heart: - George
