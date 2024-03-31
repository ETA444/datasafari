from scipy.stats import levene, bartlett, fligner
import pandas as pd


def evaluate_variance(df: pd.DataFrame, target_variable: str, grouping_variable: str, normality_info: bool = None, method: str = 'consensus', pipeline: bool = False):
    """
    Evaluates the homogeneity of variances in the numerical/target variable across groups defined by a grouping/categorical variable in a dataset.

    This function is versatile, allowing for the evaluation of variance homogeneity through several statistical tests, including Levene's, Bartlett's, and Fligner-Killeen's tests. It provides an option to use a consensus method for a more robust determination of homogeneity. It's suitable for preliminary data analysis before conducting hypothesis tests that assume equal variances and can be used in a pipeline for automated data analysis processes.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the dataset for analysis.
    target_variable : str
        The name of the numerical variable for which the variance homogeneity is to be evaluated.
    grouping_variable : str
        The name of the categorical variable used to divide the dataset into groups.
    normality_info : bool, optional
        A boolean indicating the normality of the dataset, which affects the choice of tests.
            - If True, normality is assumed, which is relevant for the Bartlett's test. Default is None.
    method : str, optional
        Specifies the method to evaluate variance homogeneity. Options include:
            - 'levene': Uses Levene's test, suitable for non-normal distributions.
            - 'bartlett': Uses Bartlett's test, requires normality assumption.
            - 'fligner': Uses Fligner-Killeen's test, a non-parametric alternative.
            - 'consensus': Combines results from the available tests to reach a consensus.
        Default is 'consensus'.
    pipeline : bool, optional
        If True, simplifies the output to a boolean indicating the consensus on equal variances. Useful for integration into automated analysis pipelines. Default is False.

    Returns
    -------
    output_info : dict or bool
        - If `pipeline` is False, returns a dictionary containing the results of the variance tests, including statistics, p-values, and a conclusion on variance homogeneity.
        - If `pipeline` is True, returns a boolean indicating whether a consensus was reached on variance homogeneity.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'Group': np.random.choice(['A', 'B', 'C'], 100), 'Data': np.random.normal(0, 1, 100)})
    # Evaluate variance homogeneity using the consensus method
    >>> variance_info = evaluate_variance(df, 'Data', 'Group')
    # Evaluate variance with explicit assumption of normality (for Bartlett's test)
    >>> variance_info_bartlett = evaluate_variance(df, 'Data', 'Group', normality_info=True, method='bartlett')
    # Use the function in a pipeline with simplified boolean output
    >>> variance_homogeneity = evaluate_variance(df, 'Data', 'Group', pipeline=True)
    >>> if variance_homogeneity:
    >>>     # ...
    """

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
        levene_title = f"< EQUAL VARIANCES TESTING: LEVENE >\n\n"
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
        fligner_title = f"< EQUAL VARIANCES TESTING: FLIGNER-KILLEEN >\n\n"
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
        bartlett_title = f"< EQUAL VARIANCES TESTING: BARTLETT >\n\n"
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
