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
"""
Common Metadata Framework Logger
------------
"""
import os
import logging
import warnings
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union

from torch import Tensor

#from pytorch_lightning.utilities import rank_zero_only
#from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.logger import _add_prefix, _convert_params
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.loggers.base import rank_zero_experiment

DEFAULT_MLMD_FILE = "mlmd"
FLUSH_LOGS_DEFAULT_STEPS = 100

try:
    from cmflib import cmf
    from cmflib import dvc_wrapper
except ModuleNotFoundError:
    # needed for test mocks, tests to be updated
    cmf_logger = None

class CMFLogger(Logger):
    r"""
    Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(mlmd_filename, name, version)``.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import CMFLogger
        >>> logger = CMFLogger("mlmd", pipeline_name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        mlmd_filename: Save directory
        pipeline_name: Experiment pipeline_name. Defaults to ``'default'``.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).
    """

    def __init__(
        self,
        mlmd_filename: str,
        pipeline_name: str = "cmf_pipeline_logs",
        #version: Optional[Union[int, str]] = None,
        #prefix: str = "",
        flush_logs_every_n_steps: int = FLUSH_LOGS_DEFAULT_STEPS,
    ):
        super().__init__()
        self._mlmd_filename = mlmd_filename
        self._pipeline_name = pipeline_name or ""
        #self._version = version
        #self._prefix = prefix
        #self._experiment: Optional[ExperimentWriter] = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

        self.hparams: Dict[str, Any] = {}
        self.metrics: List[Dict[str, float]] = []
        
        

        #Create metadata writer
        #cmf = cmf.Cmf(filename="mlmd", pipeline_name="Test-env")   
        cmf_logger = cmf.Cmf(filename=self.filename, pipeline_name=self.name)   

        #Create CMF Context
        context = cmf_logger.create_context(pipeline_stage="Train",
                                   custom_properties={"user-metadata1":"metadata_value"})

        #Create CMF Execution (Instances of Context)
        execution = cmf_logger.create_execution(execution_type="Train-1", custom_properties = {"user-metadata1":"metadata_value"})

        self._logger = cmf_logger
        self._execution = execution

    @property
    def name(self):
        """Gets the name of the pipeline/experiment.

        Returns:
            The name of the pipeline/experiment.
        """
        return self._pipeline_name
    
    @property
    def filename(self):
        """Gets the name of the MLMD file for the pipeline/experiment.

        Returns:
            The name of the MLMD file for the pipeline/experiment.
        """
        return self._mlmd_filename

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        

        return self._logger 
        #pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here

        self.experiment.update_execution(self._execution.id, params)

        #CMF Log Stage metrics: Metrics for each stage
        #cmf.log_execution_metrics("metrics", {"avg_prec":avg_prec, "roc_auc":roc_auc})

        #pass

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[Tensor, float]], step: Optional[int] = None) -> None:
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here


        #Can be called at every epoch or every step in the training. This is logged to a parquet file and commited at the commit stage.
        #while True: #Inside training loop
        #    cmf.log_metric("training_metrics", {"loss":loss}) 
        #cmf.commit_metrics("training_metrics")

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        self.metrics = {k: _handle_value(v) for k, v in metrics.items()}

        print("Calling log_metric")
        self.experiment.log_metric("training_metrics", self.metrics)

        
        if step is not None and (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

        if not self.metrics:
            return
        print("Calling commit_metrics")
        self.experiment.commit_metrics("training_metrics")

        #CMF Log Artifacts
        #cmf.log_dataset(input, "input", custom_properties={"user-metadata1":"metadata_value"})
        #cmf.log_dataset(output_train, "output", custom_properties={"user-metadata1":"metadata_value"})
        #cmf.log_dataset(output_test, "output", custom_properties={"user-metadata1":"metadata_value"})

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
        #CMF Log Model
        #cmf.log_model(path="model.pkl", 
            #event="output", model_framework="PytorchLightning", model_type="RandomForestClassifier", 
                #model_name="RandomForestClassifier:default" )
        #cmf.log_model(path="model.pkl", 
            #event="input", model_framework="PytorchLightning", model_type="RandomForestClassifier", 
                #model_name="RandomForestClassifier:default" )