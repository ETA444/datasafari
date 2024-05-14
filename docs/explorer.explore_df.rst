explore_df function
===================

.. py:function:: explore_df(df, method='all', output='print', **kwargs)

   Explores a DataFrame using specified methods, providing a bird's eye view of the data. It allows for various methods to be applied, like displaying NA counts, summary statistics, and more. This function is flexible and allows additional arguments to pass through to underlying pandas methods.

   **Note:**
   For in-depth exploration of numerical and categorical data, consider using :func:`explore_num` and :func:`explore_cat`.

   :param pandas.DataFrame df: DataFrame to explore.
   :param str method: Exploration method to apply. Options include 'na', 'desc', 'head', 'info', and 'all'.
   :param str output: Controls the output of the exploration results ('print' or 'return').
   :param kwargs: Additional arguments for pandas methods like 'percentiles' for 'desc'. Note: 'info' method's 'buf' parameter is disabled.

   :return: Returns string if output='return'. Otherwise, prints to console and returns None.
   :rtype: str or None

   :raises TypeError: If incorrect types are passed for the parameters.
   :raises ValueError: If invalid values are used for the parameters.

   **Examples:**

   .. code-block:: python

      import numpy as np
      import pandas as pd

      # Create a sample DataFrame
      data = {
          'A': np.random.randn(100),
          'B': np.random.rand(100) * 100,
          'C': np.random.randint(1, 100, size=100),
          'D': np.random.choice(['X', 'Y', 'Z'], size=100)
      }
      df = pd.DataFrame(data)

      # Use 'all' method
      explore_df(df, 'all')

      # Save output to a string
      summary = explore_df(df, 'all', output='return')

      # Display summary statistics with custom percentiles
      explore_df(df, 'desc', percentiles=[0.05, 0.95], output='print')

      # Display the first 3 rows
      explore_df(df, 'head', n=3, output='print')

      # Detailed DataFrame information
      explore_df(df, 'info', verbose=True, output='print')

      # Count and percentage of missing values
      explore_df(df, 'na', output='print')

      # Comprehensive exploration with custom settings, displaying to console
      explore_df(df, 'all', n=3, percentiles=[0.25, 0.75], output='print')

      # Returning comprehensive exploration results as a string
      result_str = explore_df(df, 'all', n=5, output='return')
      print(result_str)

      # Using 'all' with kwargs that apply to specific methods, printing the results:
      explore_df(df, 'all', n=5, percentiles=[0.1, 0.9], verbose=False, output='print')
