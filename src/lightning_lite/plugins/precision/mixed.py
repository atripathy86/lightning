# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING, Union

from lightning_lite.plugins.precision.precision import Precision

if TYPE_CHECKING:
    from lightning_lite.utilities import AMPType


class MixedPrecision(Precision):
    """Base Class for mixed precision."""

    backend: "AMPType"
    precision: Union[str, int] = "mixed"
