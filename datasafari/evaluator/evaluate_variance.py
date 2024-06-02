from typing import Union
from scipy.stats import levene, bartlett, fligner
import pandas as pd
from datasafari.evaluator.evaluate_dtype import evaluate_dtype


def evaluate_variance(
        df: pd.DataFrame,
        target_variable: str,
        grouping_variable: str,
        normality_info: bool = None,
        method: str = 'consensus',
        pipeline: bool = False
) -> Union[dict, bool]:
    """
    **Evaluate variance homogeneity across groups defined by a categorical variable within a dataset, using several statistical tests, dynamically chosen based on data suitability.**

    This function is versatile, allowing for the evaluation of variance homogeneity through several statistical tests, including Levene's, Bartlett's, and Fligner-Killeen's tests.

    The power of ``evaluate_variance()`` lies in ``method='consensus'``, which offers a robust determination of homogeneity, by combining the results of multiple tests to provide a more reliable conclusion on variance homogeneity. For a detailed explanation of the consensus method, refer to the Notes section.


    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the dataset for analysis.

    target_variable : str
        The name of the numerical variable for which the variance homogeneity is to be evaluated.

    grouping_variable : str
        The name of the categorical variable used to divide the dataset into groups.

    normality_info : bool, optional, default: None
        A boolean indicating the normality of the dataset, which affects the choice of tests.
            - ``True`` Normality is assumed, which allows for Bartlett's test to be performed.
            - ``False`` Normality is not assumed, which means Bartlett's test will not be available.

    method : str, optional, default: 'consensus'
        Specifies the method to evaluate variance homogeneity.
            - ``'levene'`` Uses Levene's test, suitable for non-normal distributions.
            - ``'bartlett'`` Uses Bartlett's test, requires normality assumption.
            - ``'fligner'`` Uses Fligner-Killeen's test, a non-parametric alternative.
            - ``'consensus'`` Combines results from the available tests to reach a consensus.

    pipeline : bool, optional, default: False
        - ``True`` Simplifies the output to a boolean indicating the consensus on equal variances. Useful for integration into automated analysis pipelines.
        - ``False`` The output is a dictionary containing results of the respective test(s).

    Returns:
    --------
    dict or bool
        - ``dict`` If pipeline=False, returns a dictionary containing the results of the variance tests, including statistics, p-values, and a conclusion on variance homogeneity.
        - ``bool`` If pipeline=True, returns a boolean indicating whether a consensus was reached on variance homogeneity.

    Raises:
    -------
    TypeErrors:
        - If `df` is not a pandas DataFrame.
        - If `target_variable` or `grouping_variable` is not a string.
        - If `normality_info` is provided but is not a boolean.
        - If `method` is not a string.
        - If `pipeline` is not a boolean.
    
    ValueErrors:
        - If the `df` is empty.
        - If the `target_variable` or `grouping_variable` does not exist in the DataFrame.
        - If the `method` specified is not supported.
        - If the `target_variable` is not numerical, or if the `grouping_variable` is not categorical, as determined by evaluating their data types with :doc:`evaluate_dtype() <datasafari.evaluator.evaluate_dtype>`.

    Examples:
    ---------
    Create a DataFrame with mixed data types for the example:

    >>> import datasafari
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'Group': np.random.choice(['A', 'B', 'C'], 100),
    ...     'Data': np.random.normal(0, 1, 100)
    ... })

    Example 1: Evaluate variance homogeneity using the consensus method:

    >>> variance_info = evaluate_variance(df, 'Data', 'Group')

    Example 2: Using evaluate_variance in a comprehensive evaluation pipeline:

    >>> variance_homogeneity = evaluate_variance(df, 'Data', 'Group', pipeline=True)
    >>> if variance_homogeneity:
    ...     # run tests assuming homogeneity pipeline
    >>> else:
    ...     # treat homogeneity pipeline

    Notes:
    ------
    **Consensus Method:**
    The consensus method in ``evaluate_variance()`` combines the results of multiple statistical tests to determine the homogeneity of variances more robustly. The tests included in the consensus method are:

    - **Levene's Test**: Suitable for data that may not adhere to normality. It tests the null hypothesis that all input samples are from populations with equal variances.
    - **Fligner-Killeen Test**: A non-parametric test that is less sensitive to departures from normality. It is useful when dealing with data that might not be normally distributed.
    - **Bartlett's Test**: Assumes normality and is sensitive to departures from it. It tests the null hypothesis that all input samples are from populations with equal variances but requires that the data are normally distributed.

        **The consensus method works as follows:**
            1. **Test Execution**: All applicable tests are performed on the dataset.
            2. **Outcome Evaluation**: Each test provides a conclusion on variance homogeneity.
            3. **Majority Rule**: The final conclusion is based on the majority of test outcomes. If more than half of the tests suggest equal variances, the consensus is 'equal variances'. If more than half suggest unequal variances, the consensus is 'unequal variances'.
            4. **Tie-Breaker**: In the event of a tie (50/50), the result of Levene's test is given precedence due to its robustness against non-normality.

    This method ensures a more reliable conclusion by mitigating the limitations of individual tests, especially in cases where the data may not perfectly meet the assumptions of any single test.
    """

    # Error Handling
    # TypeErrors
    if not isinstance(df, pd.DataFrame):
        raise TypeError("evaluate_variance(): The 'df' parameter must be a pandas DataFrame.")

    if not isinstance(target_variable, str):
        raise TypeError("evaluate_variance(): The 'target_variable' parameter must be a string.")

    if not isinstance(grouping_variable, str):
        raise TypeError("evaluate_variance(): The 'grouping_variable' parameter must be a string.")

    if normality_info is not None and not isinstance(normality_info, bool):
        raise TypeError("evaluate_variance(): The 'normality_info' parameter must be a boolean if provided.")

    if not isinstance(method, str):
        raise TypeError("evaluate_variance(): The 'method' parameter must be a string.")

    if not isinstance(pipeline, bool):
        raise TypeError("evaluate_variance(): The 'pipeline' parameter must be a boolean.")

    # ValueErrors
    if df.empty:
        raise ValueError("evaluate_variance(): The input DataFrame is empty.")

    if target_variable not in df.columns:
        raise ValueError(f"evaluate_variance(): The target variable '{target_variable}' was not found in the DataFrame.")

    if grouping_variable not in df.columns:
        raise ValueError(f"evaluate_variance(): The grouping variable '{grouping_variable}' was not found in the DataFrame.")

    allowed_methods = ['levene', 'bartlett', 'fligner', 'consensus']
    if method not in allowed_methods:
        raise ValueError(f"evaluate_variance(): The method '{method}' is not supported. Allowed methods are: {', '.join(allowed_methods)}.")

    # Check if the specified columns are the appropriate dtypes
    target_variable_is_numerical = evaluate_dtype(df, [target_variable], output='list_n')[0]
    if not target_variable_is_numerical:
        raise ValueError(f"evaluate_variance(): The target variable '{target_variable}' must be a numerical variable.")

    grouping_variable_is_categorical = evaluate_dtype(df, [grouping_variable], output='list_c')[0]
    if not grouping_variable_is_categorical:
        raise ValueError(f"evaluate_variance(): The grouping variable '{grouping_variable}' must be a categorical variable.")

    # Main Function
    groups = df[grouping_variable].unique().tolist()
    samples = [df[df[grouping_variable] == group][target_variable] for group in groups]

    # define output objects
    output_info = {}  # non-pipeline
    variance_info = {}  # pipeline single-method
    variance_consensus = None  # pipeline consensus method

    if method in ['levene', 'consensus']:
        levene_stat, levene_pval = levene(*samples).statistic, levene(*samples).pvalue
        variance_info['levene'] = levene_pval > 0.05

        # save the info in return object
        output_info['levene'] = {'stat': levene_stat, 'p': levene_pval, 'equal_variances': variance_info['levene']}

        # construct console output
        levene_text = f"Results for samples in groups of '{grouping_variable}' for ['{target_variable}'] target variable:\n  ➡ statistic: {levene_stat}\n  ➡ p-value: {levene_pval}\n{(f'  ∴ Equal variances: Yes (H0 cannot be rejected)' if variance_info['levene'] else f'  ∴ Equal variances: No (H0 rejected)')}\n\n"
        levene_title = "< EQUAL VARIANCES TESTING: LEVENE >\n\n"
        levene_tip = "☻ Tip: The Levene test is robust against departures from normality, making it suitable for a wide range of distributions, including those that are not normally distributed. It is preferred when you suspect that your data may not adhere to the normality assumption.\n"

        # output & return
        if method == 'levene':
            print(levene_title, levene_text, levene_tip)
            return output_info if not pipeline else variance_info['levene']

    if method in ['fligner', 'consensus']:
        fligner_stat, fligner_pval = fligner(*samples).statistic, fligner(*samples).pvalue
        variance_info['fligner'] = fligner_pval > 0.05

        # save the info in return object
        output_info['fligner'] = {'stat': fligner_stat, 'p': fligner_pval, 'equal_variances': variance_info['fligner']}

        # construct console output
        fligner_text = f"Results for samples in groups of '{grouping_variable}' for ['{target_variable}'] target variable:\n  ➡ statistic: {fligner_stat}\n  ➡ p-value: {fligner_pval}\n{(f'  ∴ Equal variances: Yes (H0 cannot be rejected)' if variance_info['fligner'] else f'  ∴ Equal variances: No (H0 rejected)')}\n\n"
        fligner_title = "< EQUAL VARIANCES TESTING: FLIGNER-KILLEEN >\n\n"
        fligner_tip = "☻ Tip: The Fligner-Killeen test is a non-parametric alternative that is less sensitive to departures from normality compared to the Bartlett test. It's a good choice when dealing with data that might not be normally distributed, offering robustness similar to the Levene test but through a different statistical approach.\n"

        # output & return
        if method == 'fligner':
            print(fligner_title, fligner_text, fligner_tip)
            return output_info if not pipeline else variance_info['fligner']

    if method in ['bartlett', 'consensus'] and normality_info:
        bartlett_stat, bartlett_pval = bartlett(*samples).statistic, bartlett(*samples).pvalue
        variance_info['bartlett'] = bartlett_pval > 0.05

        # save the info in return object
        output_info['bartlett'] = {'stat': bartlett_stat, 'p': bartlett_pval, 'equal_variances': variance_info['bartlett']}

        # construct console output
        bartlett_text = f"Results for samples in groups of '{grouping_variable}' for ['{target_variable}'] target variable:\n  ➡ statistic: {bartlett_stat}\n  ➡ p-value: {bartlett_pval}\n{(f'  ∴ Equal variances: Yes (H0 cannot be rejected)' if variance_info['bartlett'] else f'  ∴ Equal variances: No (H0 rejected)')}\n\n"
        bartlett_title = "< EQUAL VARIANCES TESTING: BARTLETT >\n\n"
        bartlett_tip = "☻ Tip: The Fligner-Killeen test is a non-parametric alternative that is less sensitive to departures from normality compared to the Bartlett test. It's a good choice when dealing with data that might not be normally distributed, offering robustness similar to the Levene test but through a different statistical approach.\n"

        # output & return
        if method == 'bartlett':
            print(bartlett_title, bartlett_text, bartlett_tip)
            return output_info if not pipeline else variance_info['bartlett']

    if method == 'consensus':
        # the logic is that more than half of the tests need to give True for the consensus to be True
        variance_results = [variance_bool for variance_bool in variance_info.values()]
        true_count = variance_results.count(True)
        false_count = variance_results.count(False)
        half_point = 1.5 if len(variance_results) == 3 else 1
        if true_count > half_point:
            consensus_percent = (true_count / len(variance_results)) * 100
            variance_consensus_text = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest equal variance between samples. *\n\n* Detailed results of each test are provided below.\n"

            # used only within predict_hypothesis() pipeline
            variance_consensus = True

        elif true_count < half_point:
            consensus_percent = (false_count / len(variance_results)) * 100
            variance_consensus_text = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest unequal variance between samples. *\n\n* Detailed results of each test are provided below:\n"

            # used only within predict_hypothesis() pipeline
            variance_consensus = False

        elif true_count == half_point:
            # the logic is that we default to the decisions of levene - internal use only
            variance_consensus_text = f"  ➡ Result: Consensus is not reached.\n\n∴ Please refer to the results of each test below:\n"

            # used only within predict_hypothesis() pipeline
            variance_consensus = variance_info['levene']

        print(f"< VARIANCE TESTING: CONSENSUS >\nThe consensus method bases its conclusion on 2-3 tests: Levene test, Fligner-Killeen test, Bartlett test. (Note: More than 50% must have the same outcome to reach consensus.)\n\n{variance_consensus_text}")
        print(levene_title, levene_text, levene_tip)
        print(fligner_title, fligner_text, fligner_tip)
        print(bartlett_title, bartlett_text, bartlett_tip) if normality_info else f"\n\n< NOTE ON BARTLETT >\nBartlett was not used in consensus as no normality info has been provided or data is non-normal. Accuracy of Bartlett's test results rely heavily on normality."

        return output_info if not pipeline else variance_consensus
