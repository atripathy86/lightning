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
from typing import Any, Optional

from torch.nn import Module
from torch.optim import Optimizer

from lightning_lite.plugins.precision.precision import Precision


class TPUPrecision(Precision):
    """Precision plugin for TPU integration."""

    def optimizer_step(
        self,
        optimizer: Optimizer,
        model: Optional[Module] = None,
        **kwargs: Any,
    ) -> Any:

        import torch_xla.core.xla_model as xm

        return xm.optimizer_step(optimizer, optimizer_args=kwargs)
