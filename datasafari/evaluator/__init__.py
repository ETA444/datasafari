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

"""Subpackage for evaluation functionalities within Data Safari."""

from .evaluate_normality import evaluate_normality
from .evaluate_variance import evaluate_variance
from .evaluate_dtype import evaluate_dtype
from .evaluate_contingency_table import evaluate_contingency_table
