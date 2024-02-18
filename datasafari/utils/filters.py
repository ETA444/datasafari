def filter_kwargs(method, kwargs, valid_kwargs_dict):
    """
    Filter keyword arguments (`kwargs`) to include only those that are valid
    for a specified method, based on a dictionary mapping methods to their
    valid keyword arguments.

    Parameters
    ----------
    method : str
        The name of the method for which keyword arguments need to be filtered.
        This method name should match a key in the `valid_kwargs_dict`.
    kwargs : dict
        A dictionary of keyword arguments to be filtered according to the
        method's valid keyword arguments.
    valid_kwargs_dict : dict
        A dictionary mapping method names (str) to lists of valid keyword
        argument names (str) for those methods. Only keyword arguments listed
        for a given method name will be included in the returned dictionary.

    Returns
    -------
    dict
        A dictionary containing only the keyword arguments from `kwargs` that
        are valid for the specified `method`, according to `valid_kwargs_dict`.

    Examples
    --------
    >>> valid_kwargs = {'method1': ['param1', 'param2'], 'method2': ['param3']}
    >>> all_kwargs = {'param1': 10, 'param2': 20, 'param3': 30}
    >>> filtered_kwargs = filter_kwargs('method1', all_kwargs, valid_kwargs)
    >>> print(filtered_kwargs)
    {'param1': 10, 'param2': 20}

    Note
    ----
    This function is particularly useful in situations where a function or method
    accepts a wide variety of keyword arguments, and you want to ensure that only
    relevant keyword arguments are passed through, based on the specific method
    or operation being performed.

    """
    return {k: v for k, v in kwargs.items() if k in valid_kwargs_dict.get(method, [])}
