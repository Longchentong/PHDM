# Copyright 2023-present the HuggingFace Inc. team.
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

__version__ = "0.15.2"

from .auto import AutoPeftModel, AutoPeftModelForCausalLM, MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from .config import PeftConfig
from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING, get_peft_config, inject_adapter_in_model
from .mapping_func import get_peft_model
from .peft_model import PeftModel, PeftModelForCausalLM, get_layer_status, get_model_status
from .tuners import LoraConfig, LoraModel, LoraRuntimeConfig
from .utils import PeftType, TaskType, cast_mixed_precision_params, get_peft_model_state_dict, load_peft_weights, prepare_model_for_kbit_training, set_peft_model_state_dict

# Transformers imports this name for PEFT model type checks.
PeftMixedModel = PeftModel

__all__ = [
    "AutoPeftModel", "AutoPeftModelForCausalLM", "LoraConfig", "LoraModel",
    "LoraRuntimeConfig", "PeftConfig", "PeftModel", "PeftModelForCausalLM",
    "PeftType", "TaskType", "get_peft_model", "get_peft_model_state_dict",
    "set_peft_model_state_dict", "prepare_model_for_kbit_training", "load_peft_weights",
]
