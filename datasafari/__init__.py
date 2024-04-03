"""Top-level package for Data Safari."""

__author__ = """George Dreemer"""
__email__ = 'georgedreemer@proton.me'
__version__ = '0.1.0'

# import core functionality

# explorers
from .explorer import (
    explore_df, explore_cat, explore_num
)

# transformers
from .transformer import (
    transform_num, transform_cat
)

# predictor
from .predictor import (
    predict_hypothesis,  # predict_ml
)

# evaluator
from .evaluator import (
    evaluate_dtype, evaluate_normality, evaluate_variance, evaluate_contingency_table
)
