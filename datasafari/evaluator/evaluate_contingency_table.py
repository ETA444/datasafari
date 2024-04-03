import numpy as np
import pandas as pd
from scipy.stats.contingency import expected_freq


def evaluate_contingency_table(contingency_table: pd.DataFrame, min_sample_size_yates: int = 40, pipeline: bool = False, quiet: bool = False):
    """
    Evaluates a contingency table to determine the viability of various statistical tests based on the table's characteristics.

    This function assesses the contingency table's suitability for chi-square tests, exact tests (Barnard's, Boschloo's, and Fisher's), and the application of Yates' correction within the chi-square test. It examines expected and observed frequencies, sample size, and table shape to guide the choice of appropriate statistical tests for hypothesis testing.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        A contingency table generated from two categorical variables.
    min_sample_size_yates : int, optional
        The minimum sample size below which Yates' correction should be considered. Default is 40.
    pipeline : bool, optional
        Determines the format of the output. If True, outputs a tuple of boolean values representing the viability of each test. If False, outputs a dictionary with the test names as keys and their viabilities as boolean values. Default is False.
    quiet : bool, optional

    Returns
    -------
    test_viability : dict or tuple
        Depending on the 'pipeline' parameter:
            - If `pipeline` is False, returns a dictionary with keys as test names ('chi2_contingency', 'yates_correction', 'barnard_exact', 'boschloo_exact', 'fisher_exact') and values as boolean indicators of their viability.
            - If `pipeline` is True, returns a tuple of boolean values in the order: (chi2_viability, yates_correction_viability, barnard_viability, boschloo_viability, fisher_viability).

    Raises
    ------
    TypeError
        - If `contingency_table` is not a pandas DataFrame, indicating the wrong data type has been passed.
        - If `min_sample_size_yates` is not an integer, indicating the parameter is of the wrong type.
        - If `pipeline` or `quiet` is not a boolean, indicating incorrect data types for these parameters.
    ValueError
        - If the `contingency_table` is empty, indicating that there's no data to evaluate.
        - If `min_sample_size_yates` is not a positive integer, indicating an invalid parameter value.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {
    ...     'Gender': np.random.choice(['Male', 'Female'], 100),
    ...     'Preference': np.random.choice(['Option A', 'Option B'], 100)
    ... }
    >>> df_example = pd.DataFrame(data)
    >>> contingency_table = pd.crosstab(df_example['Gender'], df_example['Preference'])
    >>> test_viability = evaluate_contingency_table(contingency_table)
    >>> print(test_viability)
    >>> chi2, yates, barnard, boschloo, fisher = evaluate_contingency_table(contingency_table, pipeline=True, quiet=True)
    >>> if chi2:
    >>>     # ...
    """
    # Error Handling
    # TypeErrors
    if not isinstance(contingency_table, pd.DataFrame):
        raise TypeError("evaluate_contingency_table(): The 'contingency_table' parameter must be a pandas DataFrame.")

    if not isinstance(min_sample_size_yates, int):
        raise TypeError("evaluate_contingency_table(): The 'min_sample_size_yates' parameter must be an integer.")

    if not isinstance(pipeline, bool):
        raise TypeError("evaluate_contingency_table(): The 'pipeline' parameter must be a boolean.")

    if not isinstance(quiet, bool):
        raise TypeError("evaluate_contingency_table(): The 'quiet' parameter must be a boolean.")

    # ValueErrors
    if contingency_table.empty:
        raise ValueError("evaluate_contingency_table(): The 'contingency_table' parameter must not be empty.")

    if min_sample_size_yates <= 0:
        raise ValueError("evaluate_contingency_table(): The 'min_sample_size_yates' parameter must be a positive integer.")

    # Main Function
    test_viability = {}  # non-pipeline output

    # compute objects for checks
    min_expected_frequency = expected_freq(contingency_table).min()
    min_observed_frequency = contingency_table.min().min()
    sample_size = np.sum(contingency_table.values)
    table_shape = contingency_table.shape

    # assumption check for chi2_contingency test
    chi2_viability = True if min_expected_frequency >= 5 and min_observed_frequency >= 5 else False
    test_viability['chi2_contingency'] = chi2_viability

    # assumption check for chi2_contingency yate's-correction
    yates_correction_viability = True if table_shape == (2, 2) and sample_size < min_sample_size_yates else False
    test_viability['yates_correction'] = yates_correction_viability

    # assumption check for all exact tests
    barnard_viability, boschloo_viability, fisher_viability = (True, True, True) if table_shape == (2, 2) else (False, False, False)
    test_viability['barnard_exact'], test_viability['boschloo_exact'], test_viability['fisher_exact'] = barnard_viability, boschloo_viability, fisher_viability

    # console output
    title = f"< CONTINGENCY TABLE EVALUATION >\n"
    on_chi2 = f"Based on minimum expected freq. ({min_expected_frequency}) & minimum observed freq. ({min_observed_frequency}):\n  ➡ chi2_contingecy() viability: {'✔' if chi2_viability else '✘'}\n\n"
    on_yates = f"Based on table shape ({table_shape[0]}x{table_shape[1]}) & sample size ({sample_size}):\n  ➡ chi2_contingecy() Yate's correction viability: {'✔' if yates_correction_viability else '✘'}\n\n"
    on_exact = f"Based on table shape ({table_shape[0]}x{table_shape[1]}):\n  ➡ barnard_exact() viability: {'✔' if barnard_viability else '✘'}\n  ➡ boschloo_exact() viability: {'✔' if boschloo_viability else '✘'}\n  ➡ fisher_exact() viability: {'✔' if fisher_viability else '✘'}\n\n\n"
    print(title, on_chi2, on_yates, on_exact) if not quiet else ""

    if pipeline:
        return chi2_viability, yates_correction_viability, barnard_viability, boschloo_viability, fisher_viability
    elif not pipeline:
        return test_viability