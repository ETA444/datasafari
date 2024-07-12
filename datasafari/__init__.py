# DataSafari - DataSafari simplifies complex data science tasks into straightforward, powerful one-liners.
# Copyright (C) 2024 George Dreemer.
#
# Read more about DataSafari's LICENSE here: https://datasafari.dev/docs/other/lic-gpl3
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details, specifically under
# version 3 of the License. No later versions are applicable.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see: https://github.com/ETA444/datasafari/blob/main/LICENSE

"""Front-end functions of Data Safari."""

__author__ = """George Dreemer"""
__email__ = 'georgedreemer@proton.me'
__version__ = '1.0.0'

# import core functionality

# explorers
from .explorer import (
    explore_df, explore_cat, explore_num
)

# transformers
from .transformer import (
    transform_num, transform_cat
)

# predictors
from .predictor import (
    predict_hypothesis,  predict_ml
)

# evaluators
from .evaluator import (
    evaluate_dtype, evaluate_normality, evaluate_variance, evaluate_contingency_table
)
