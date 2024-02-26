"""Module for any sort of calculators used throughout the main subpackages of DataSafari."""


import numpy as np
import pandas as pd
from numpy.linalg import inv, LinAlgError


# calculate_mahalanobis() used in: explore_num()
def calculate_mahalanobis(x, mean, inv_cov_matrix):
    """
    Calculate the Mahalanobis distance for an observation from a distribution.

    The Mahalanobis distance is a measure of the distance between a point and a distribution.
    It is an effective way to determine how many standard deviations an observation is from
    the mean of a distribution, considering the covariance among variables. This function
    computes the Mahalanobis distance of a single observation from the mean of a distribution,
    given the inverse of the covariance matrix of the distribution.

    Parameters
    ----------
    x : numpy.ndarray or pandas.Series
        A 1D array of the observation or a single row from a DataFrame.
    mean : numpy.ndarray
        The mean vector of the distribution from which distances are calculated.
        Must be 1D and of the same length as `x`.
    inv_cov_matrix : numpy.ndarray
        The inverse of the covariance matrix of the distribution. This matrix
        must be square and its size should match the number of elements in `x`.

    Returns
    -------
    float
        The Mahalanobis distance of the observation `x` from the distribution
        defined by `mean` and `inv_cov_matrix`.

    Raises
    ------
    ValueError
        If `x` and `mean` do not have the same length.
    LinAlgError
        If the inverse covariance matrix is singular and cannot be used for
        distance calculation.

    Examples
    --------
    >>> mean_vector = np.array([0, 0])
    >>> observation = np.array([1, 1])
    >>> cov_matrix = np.array([[1, 0.5], [0.5, 1]])
    >>> inv_cov_matrix = np.linalg.inv(cov_matrix)
    >>> mahalanobis_distance(observation, mean_vector, inv_cov_matrix)
    2.0

    Notes
    -----
    The Mahalanobis distance is widely used in outlier detection and cluster analysis.
    It is scale-invariant and takes into account the correlations of the data set.
    """
    if len(x) != len(mean):
        raise ValueError("The observation and mean must have the same length.")

    x_minus_mu = x - mean
    try:
        distance = np.dot(np.dot(x_minus_mu, inv_cov_matrix), x_minus_mu.T)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Singular matrix provided as inverse covariance matrix.")

    return distance


# calculate_entropy() used in: explore_cat()
def calculate_entropy(series: pd.Series) -> (float, str):
    """
    Calculate the entropy of a pandas Series (categorical variable) and provide an interpretation by comparing it to
    the maximum possible entropy for the given number of unique categories. This function quantifies the
    unpredictability or diversity of the data within the variable.

    **Calculation:**
        - Entropy is computed using the formula: *H = -sum(p_i * log2(p_i))*
        - Where p_i is the proportion of the series belonging to the ith category. Higher entropy indicates a more uniform distribution of categories, suggesting a higher degree of randomness or diversity. Conversely, entropy is lower when one or a few categories dominate the distribution.

    **Interpretation:**
        The function also calculates the percentage of the maximum possible entropy (based on the number
        of unique categories) achieved by the actual entropy, providing context for how diverse the
        categorical data is relative to the total number of categories.

    Parameters:
    ----------
    series : pd.Series
        The series for which to calculate entropy. Should be a categorical variable with discrete values.

    Returns:
    -------
    tuple
        A tuple containing two elements:
            - A float representing the entropy of the series. Higher values indicate greater entropy (diversity), and values close to zero suggest less entropy (more uniformity).
            - A string providing an interpretation of the entropy value in the context of the maximum possible entropy for the given number of unique categories.

    Example:
    --------
    >>> example_series = pd.Series(['apple', 'orange', 'apple', 'banana', 'orange', 'banana'])
    >>> entropy_val_example, interpretation_example = calculate_entropy(example_series)
    >>> print(entropy_val_example)
    1.583
    >>> print(interpretation_example)
    "=> Moderate diversity [max. entropy for this variable = 3.161]"
    """
    counts = series.value_counts()
    probabilities = counts / len(series)
    entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))  # Adding epsilon to avoid log(0)

    # calculate maximum possible entropy for the number of unique categories
    unique_categories = len(counts)
    max_entropy = np.log2(unique_categories)

    # calculate entropy ratio which reflect level of diversity
    entropy_ratio = entropy / max_entropy

    # calculate the percentage of the maximum possible entropy - old
    # percent_of_max = (entropy / max_entropy) * 100
    # interpretation = f"{entropy:.4f} ({percent_of_max:.2f}% of max for {unique_categories} categories)"

    # build interpretation
    if entropy_ratio > 0.85:
        interpretation = f"\n=> High diversity [max. entropy for this variable = {max_entropy:.3f}]"
    elif entropy_ratio > 0.5:
        interpretation = f"\n=> Moderate diversity [max. entropy for this variable = {max_entropy:.3f}]"
    else:
        interpretation = f"\n=> Low diversity [max. entropy for this variable = {max_entropy:.3f}]"

    return entropy, interpretation
