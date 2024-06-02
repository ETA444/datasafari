from typing import Union
from scipy.stats import shapiro, anderson, normaltest
from statsmodels.stats.diagnostic import lilliefors
import pandas as pd
from datasafari.evaluator.evaluate_dtype import evaluate_dtype


def evaluate_normality(
        df: pd.DataFrame,
        target_variable: str,
        grouping_variable: str,
        method: str = 'consensus',
        pipeline: bool = False
) -> Union[dict, bool]:
    """
    **Evaluate normality of numerical data within groups defined by a categorical variable, employing multiple statistical tests, dynamically chosen based on data suitability.**

    This function offers a comprehensive examination of the distribution's normality by utilizing tests like Shapiro-Wilk, Anderson-Darling, D'Agostino and Pearson's test, and Lilliefors. Each test provides insights into different aspects of normality, and the consensus method integrates these perspectives to make a more informed decision.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to be tested.

    target_variable : str
        The name of the numeric variable to test for normality.

    grouping_variable : str
        The name of the categorical variable used to create subsets of data for normality testing.

    method : str, optional, default: 'consensus'
        The method to use for testing normality.
            - ``'shapiro'`` Shapiro-Wilk test
            - ``'anderson'`` Anderson-Darling test
            - ``'normaltest'`` D'Agostino and Pearson's test
            - ``'lilliefors'`` Lilliefors test
            - ``'consensus'`` A combination of the above tests.

    pipeline : bool, optional, default: False
        - ``True`` Simplifies the output to a boolean indicating the consensus on normality. Useful for integration into automated analysis pipelines.
        - ``False`` The output is a dictionary containing results of the respective test(s).

    Returns:
    --------
    dict or bool
        - ``dict`` If pipeline=False, returns a dictionary with test names as keys and test results, including statistics, p-values, and normality conclusions, as values.
        - ``bool`` If pipeline=True, returns a boolean indicating the consensus on normality across all tests, or if consensus method was not used a boolean indicating the result of that test.

    Raises:
    -------
    TypeErrors:
        - If `df` is not a pandas DataFrame.
        - If `target_variable` or `grouping_variable` is not a string.
        - If `method` is not a string.
        - If `pipeline` is not a boolean.

    ValueErrors:
        - If the `df` is empty.
        - If the `target_variable` or `grouping_variable` does not exist in the DataFrame.
        - If the `method` specified is not supported.
        - If the `target_variable` is not numerical, or if the `grouping_variable` is not categorical, as determined by evaluating their data types with :doc:`evaluate_dtype() <datasafari.evaluator.evaluate_dtype>`.

    Examples:
    ---------
    Example 1: Using the consensus method to evaluate normality in a DataFrame:

    >>> import datasafari
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'Group': np.random.choice(['A', 'B', 'C'], size=100),
    ...     'Data': np.random.normal(0, 1, size=100)
    ... })
    >>> normality_result = evaluate_normality(df, 'Data', 'Group')

    Example 2: Using the function in a comprehensive evaluation pipeline:

    >>> pipeline_result = evaluate_normality(df, 'Data', 'Group', pipeline=True)
    >>> if pipeline_result:
    ...     # your pipeline in the case normality is validated
    ... else:
    ...     # your pipeline in the case normality is not validated

    Notes:
    ------
    **Consensus Method:**
    The consensus method integrates results from multiple statistical tests to provide a comprehensive assessment of the normality of data distributions.

        Here is how the consensus method operates:
            1. **Test Execution**:

               - **Shapiro-Wilk Test**: Assesses normality based on the correlation between data and corresponding normal scores, ideal for small sample sizes.
               - **Anderson-Darling Test**: Focuses more on the tails of the distribution, suitable for any sample size.
               - **D'Agostino-Pearson Test**: Combines skewness and kurtosis to assess normality, best for larger datasets.
               - **Lilliefors Test**: An adaptation of the Kolmogorov-Smirnov test that does not require known mean and variance, useful for small to medium samples.

            2. **Outcome Evaluation**: Each test provides a conclusion on whether the data follow a normal distribution.

            3. **Majority Rule**: The final consensus on normality is based on the majority of test outcomes. If more tests conclude 'normal', the consensus is that the data are normally distributed. Conversely, if more tests conclude 'non-normal', the consensus is that the data are not normally distributed.

            4. **Tie-Breaker**: In the event of a tie (an equal number of 'normal' and 'non-normal' outcomes), the results of the Shapiro-Wilk and Anderson-Darling tests are given precedence. These tests are chosen because of their robustness and widespread acceptance in statistical testing. If both agree, their conclusion is adopted; if they disagree, further analysis may be required to determine normality.

    This structured approach ensures a thorough and balanced evaluation of normality, accommodating different sample sizes and distribution characteristics, which enhances the reliability of the statistical analysis.
    """

    # Error Handling
    # TypeErrors
    # Check if 'df' is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("evaluate_normality(): The 'df' parameter must be a pandas DataFrame.")

    # Check if 'target_variable' and 'grouping_variable' are strings
    if not isinstance(target_variable, str) or not isinstance(grouping_variable, str):
        raise TypeError("evaluate_normality(): The 'target_variable' and 'grouping_variable' parameters must be strings.")

    # Check if 'method' is a string
    if not isinstance(method, str):
        raise TypeError("evaluate_normality(): The 'method' parameter must be a string.")

    # Check if 'pipeline' is a boolean
    if not isinstance(pipeline, bool):
        raise TypeError("evaluate_normality(): The 'pipeline' parameter must be a boolean.")

    # ValueErrors
    # Check if df is empty
    if df.empty:
        raise ValueError("evaluate_normality(): The input DataFrame is empty.")

    # Check if the specified columns exist in the DataFrame
    if target_variable not in df.columns:
        raise ValueError(f"evaluate_normality(): The target variable '{target_variable}' was not found in the DataFrame.")

    if grouping_variable not in df.columns:
        raise ValueError(f"evaluate_normality(): The grouping variable '{grouping_variable}' was not found in the DataFrame.")

    # Check if the specified columns are the appropriate dtypes
    target_variable_is_numerical = evaluate_dtype(df, [target_variable], output='list_n')[0]
    if not target_variable_is_numerical:
        raise ValueError(f"evaluate_normality(): The target variable '{target_variable}' must be a numerical variable.")

    grouping_variable_is_categorical = evaluate_dtype(df, [grouping_variable], output='list_c')[0]
    if not grouping_variable_is_categorical:
        raise ValueError(f"evaluate_normality(): The grouping variable '{grouping_variable}' must be a categorical variable.")

    # Check if the method specified is one of the allowed methods
    allowed_methods = ['shapiro', 'anderson', 'normaltest', 'lilliefors', 'consensus']
    if method not in allowed_methods:
        raise ValueError(f"evaluate_normality(): The method '{method}' is not supported. Allowed methods are: {', '.join(allowed_methods)}.")

    # Main Functionality
    groups = df[grouping_variable].unique().tolist()

    # define output objects
    output_info = {}  # non-pipeline
    normality_info = {}  # pipeline single-method
    normality_consensus = None  # pipeline consensus method

    if method in ['shapiro', 'consensus']:
        # calculating statistic and p-values to define normality
        shapiro_stats = [shapiro(df[df[grouping_variable] == group][target_variable]).statistic for group in groups]
        shapiro_pvals = [shapiro(df[df[grouping_variable] == group][target_variable]).pvalue for group in groups]
        shapiro_normality = [True if p > 0.05 else False for p in shapiro_pvals]

        # save the info for return and text for output
        shapiro_info = {group: {'stat': shapiro_stats[n], 'p': shapiro_pvals[n], 'normality': shapiro_normality[n]} for n, group in enumerate(groups)}
        shapiro_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in shapiro_info.items()]
        shapiro_title = "< NORMALITY TESTING: SHAPIRO-WILK >\n\n"
        shapiro_tip = "☻ Tip: The Shapiro-Wilk test is particularly well-suited for small sample sizes (n < 50) and is considered one of the most powerful tests for assessing normality. It is sensitive to departures from normality, making it a preferred choice for rigorous normality assessments in small datasets.\n"

        # save info in return object and conditionally print to console
        output_info['shapiro'] = shapiro_info
        normality_info['shapiro_group_consensus'] = all(shapiro_normality)

        # end it here if non-consensus method
        if method == 'shapiro':
            print(shapiro_title, *shapiro_text, shapiro_tip)
            return output_info if not pipeline else normality_info['shapiro_group_consensus']

    if method in ['anderson', 'consensus']:
        # calculating statistic to define normality
        anderson_stats = [anderson(df[df[grouping_variable] == group][target_variable]).statistic for group in groups]
        anderson_critical_values = [anderson(df[df[grouping_variable] == group][target_variable]).critical_values[2] for group in groups]
        anderson_normality = [True if c_val > anderson_stats[n] else False for n, c_val in enumerate(anderson_critical_values)]

        # save the info for return and text for output
        anderson_info = {group: {'stat': anderson_stats[n], 'p': anderson_critical_values[n], 'normality': anderson_normality[n]} for n, group in enumerate(groups)}
        anderson_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in anderson_info.items()]
        anderson_title = "< NORMALITY TESTING: ANDERSON-DARLING >\n\n"
        anderson_tip = "☻ Tip: The Anderson-Darling test is a versatile test that can be applied to any sample size and is especially useful for comparing against multiple distribution types, not just the normal. It places more emphasis on the tails of the distribution than the Shapiro-Wilk test, making it useful for detecting outliers or heavy-tailed distributions.\n"

        # saving info
        output_info['anderson'] = anderson_info
        normality_info['anderson_group_consensus'] = all(anderson_normality)

        # end it here if non-consensus method
        if method == 'anderson':
            print(anderson_title, *anderson_text, anderson_tip)
            return output_info if not pipeline else normality_info['anderson_group_consensus']

    if method in ['normaltest', 'consensus']:
        # calculating statistic and p-values to define normality
        normaltest_stats = [normaltest(df[df[grouping_variable] == group][target_variable]).statistic for group in groups]
        normaltest_pvals = [normaltest(df[df[grouping_variable] == group][target_variable]).pvalue for group in groups]
        normaltest_normality = [True if p > 0.05 else False for p in normaltest_pvals]

        # save the info for return and text for output
        normaltest_info = {group: {'stat': normaltest_stats[n], 'p': normaltest_pvals[n], 'normality': normaltest_normality[n]} for n, group in enumerate(groups)}
        normaltest_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in normaltest_info.items()]
        normaltest_title = "< NORMALITY TESTING: D'AGOSTINO-PEARSON NORMALTEST >\n\n"
        normaltest_tip = "☻ Tip: The D'Agostino-Pearson normality test, or simply 'normaltest', is best applied when the sample size is larger, as it combines skewness and kurtosis to form a test statistic. This test is useful for detecting departures from normality that involve asymmetry and tail thickness, offering a good balance between sensitivity and specificity in medium to large sample sizes.\n"

        # saving info
        output_info['normaltest'] = normaltest_info
        normality_info['normaltest_group_consensus'] = all(normaltest_normality)

        # end it here if non-consensus method
        if method == 'normaltest':
            print(normaltest_title, *normaltest_text, normaltest_tip)
            return output_info if not pipeline else normality_info['normaltest_group_consensus']

    if method in ['lilliefors', 'consensus']:
        # calculating statistic and p-values to define normality
        lilliefors_stats = [lilliefors(df[df[grouping_variable] == group][target_variable])[0] for group in groups]
        lilliefors_pvals = [lilliefors(df[df[grouping_variable] == group][target_variable])[1] for group in groups]
        lilliefors_normality = [True if p > 0.05 else False for p in lilliefors_pvals]

        # save the info for return and text for output
        lilliefors_info = {group: {'stat': lilliefors_stats[n], 'p': lilliefors_pvals[n], 'normality': lilliefors_normality[n]} for n, group in enumerate(groups)}
        lilliefors_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in lilliefors_info.items()]
        lilliefors_title = "< NORMALITY TESTING: LILLIEFORS' TEST >\n\n"
        lilliefors_tip = "☻ Tip: The Lilliefors test is an adaptation of the Kolmogorov-Smirnov test for normality with the benefit of not requiring the mean and variance to be known parameters. It's particularly useful for small to moderately sized samples and is sensitive to deviations from normality in the center of the distribution rather than the tails. This makes it complementary to tests like the Anderson-Darling when a comprehensive assessment of normality is needed.\n"

        # saving info
        output_info['lilliefors'] = lilliefors_info
        normality_info['lilliefors_group_consensus'] = all(lilliefors_normality)

        # end here if non-consensus method
        if method == 'lilliefors':
            print(lilliefors_title, *lilliefors_text, lilliefors_tip)
            return output_info if not pipeline else normality_info['lilliefors_group_consensus']

    if method == 'consensus':
        # the logic is that more than half of the tests need to give True for the consensus to be True
        true_count = [group_normality_consensus for test_name, group_normality_consensus in normality_info.items()].count(True)
        false_count = [group_normality_consensus for test_name, group_normality_consensus in normality_info.items()].count(False)
        half_point = len(normality_info)/2
        if true_count > half_point:
            consensus_percent = (true_count / len(normality_info)) * 100
            normality_consensus_result = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest Normality. *\n\n* Detailed results of each test are provided below.\n"

            # used only within predict_hypothesis() pipeline
            normality_consensus = True

        elif true_count < half_point:
            consensus_percent = (false_count / len(normality_info)) * 100
            normality_consensus_result = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest Non-Normality. *\n\n* Detailed results of each test are provided below:\n"

            # used only within predict_hypothesis() pipeline
            normality_consensus = False

        elif true_count == half_point:
            # the logic is that we default to the decisions of shapiro-wilk and anderson-darling (if both true then true) - internal use only
            normality_consensus_result = f"  ➡ Result: Consensus is not reached. (50% Normality / 50% Non-normality)\n\n∴ Please refer to the results of each test below:\n"

            # used only within predict_hypothesis() pipeline
            normality_consensus = all([normality_info['shapiro_group_consensus'], normality_info['anderson_group_consensus']])

        # construct output and return
        print(f"< NORMALITY TESTING: CONSENSUS >\nThe consensus method bases its conclusion on 4 tests: Shapiro-Wilk test, Anderson-Darling test, D'Agostino-Pearson test, Lilliefors test. (Note: More than 50% must have the same outcome to reach consensus. In predict_hypothesis(), if consensus is not reached normality is settled based on Shapiro-Wilk and Anderson-Darling outcomes)\n\n{normality_consensus_result}")
        print(shapiro_title, *shapiro_text, shapiro_tip)
        print(anderson_title, *anderson_text, anderson_tip)
        print(normaltest_title, *normaltest_text, normaltest_tip)
        print(lilliefors_title, *lilliefors_text, lilliefors_tip)
        return output_info if not pipeline else normality_consensus
