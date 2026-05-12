# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Weight loader for PI0 model on Polaris.
This module handles:
    - loading PI0 weights from local safetensors files
    - categorizing checkpoint keys into PI0/VLM/vision/expert groups
    - extracting per-layer weights
    - preparing fused weight tensors for TT/Polaris execution
Notes:
    - This version is torch-free.
    - It keeps checkpoint mapping logic from the reference implementation.
    - Backend-specific tensor creation for Polaris/TT should be done through the
      helper methods at the bottom.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Union

import numpy as np
from safetensors import safe_open

_ttnn_module: Optional[ModuleType] = None
TTNN_AVAILABLE = False

try:
    import ttsim.front.ttnn as _ttnn_module  # type: ignore[no-redef]
    TTNN_AVAILABLE = True
except ImportError:
    pass

@dataclass
class PI0Config:
    """Configuration for PI0 model."""
    action_dim: int = 32
    action_horizon: int = 50
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    precision: str = "bfloat16"
    vlm_width: int = 2048
    vlm_depth: int = 18
    vlm_mlp_dim: int = 16384
    vlm_num_heads: int = 8
    vlm_num_kv_heads: int = 1
    vlm_head_dim: int = 256
    expert_width: int = 1024
    expert_depth: int = 18
    expert_mlp_dim: int = 4096
    expert_num_heads: int = 8
    expert_num_kv_heads: int = 1
    expert_head_dim: int = 256

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "PI0Config":
        import dataclasses
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

def load_pi0_state_dict(model_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    model_path = Path(model_path)
    if model_path.is_dir():
        safetensors_path = model_path / "model.safetensors"
    else:
        safetensors_path = model_path
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Could not find safetensors file: {safetensors_path}")
    state_dict: Dict[str, np.ndarray] = {}
    with safe_open(str(safetensors_path), framework="np") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict

def categorize_weights(state_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    categorized: Dict[str, Dict[str, np.ndarray]] = {
        "pi0_projections": {},
        "vlm_language": {},
        "vlm_vision": {},
        "vlm_projector": {},
        "action_expert": {},
    }
    for key, value in state_dict.items():
        if key.startswith("action_in_proj") or key.startswith("action_out_proj"):
            categorized["pi0_projections"][key] = value
        elif key.startswith("action_time_mlp"):
            categorized["pi0_projections"][key] = value
        elif key.startswith("state_proj"):
            categorized["pi0_projections"][key] = value
        elif "paligemma.model.language_model" in key:
            new_key = key.replace("paligemma_with_expert.paligemma.model.language_model.", "model.")
            categorized["vlm_language"][new_key] = value
        elif "paligemma.model.vision_tower" in key:
            new_key = key.replace("paligemma_with_expert.paligemma.model.vision_tower.", "")
            categorized["vlm_vision"][new_key] = value
        elif "paligemma.model.multi_modal_projector" in key:
            new_key = key.replace("paligemma_with_expert.paligemma.model.multi_modal_projector.", "")
            categorized["vlm_projector"][new_key] = value
        elif "gemma_expert" in key:
            new_key = key.replace("paligemma_with_expert.gemma_expert.", "")
            categorized["action_expert"][new_key] = value
        elif "paligemma.lm_head" in key:
            new_key = key.replace("paligemma_with_expert.paligemma.", "")
            categorized["vlm_language"][new_key] = value
    return categorized

def fuse_qkv_weights(q_weight, k_weight, v_weight) -> np.ndarray:
    return np.concatenate([q_weight, k_weight, v_weight], axis=0)

def fuse_gate_up_weights(gate_weight, up_weight) -> np.ndarray:
    return np.concatenate([gate_weight, up_weight], axis=0)

def transpose_linear_weight(weight: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(weight.T)

def expand_bias(bias: np.ndarray) -> np.ndarray:
    return np.expand_dims(bias, axis=0)

class PI0WeightLoader:
    def __init__(self, model_path: Union[str, Path], cache_path: Optional[Path] = None):
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.cache_path = cache_path
        config_path = self.model_path / "config.json" if self.model_path.is_dir() else None
        if config_path is not None and config_path.exists():
            self.config = PI0Config.from_json(config_path)
        else:
            self.config = PI0Config()
        self._state_dict: Optional[Dict[str, np.ndarray]] = None
        self._categorized: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    @property
    def state_dict(self) -> Dict[str, np.ndarray]:
        if self._state_dict is None:
            self._state_dict = load_pi0_state_dict(self.model_path)
        return self._state_dict

    @property
    def categorized_weights(self) -> Dict[str, Dict[str, np.ndarray]]:
        if self._categorized is None:
            self._categorized = categorize_weights(self.state_dict)
        return self._categorized

    def get_pi0_projections(self) -> Dict[str, np.ndarray]:
        return self.categorized_weights["pi0_projections"]

    def get_vlm_language_weights(self) -> Dict[str, np.ndarray]:
        return self.categorized_weights["vlm_language"]

    def get_vlm_vision_weights(self) -> Dict[str, np.ndarray]:
        return self.categorized_weights["vlm_vision"]

    def get_vlm_projector_weights(self) -> Dict[str, np.ndarray]:
        return self.categorized_weights["vlm_projector"]

    def get_action_expert_weights(self) -> Dict[str, np.ndarray]:
        return self.categorized_weights["action_expert"]

    def get_layer_weights(self, layer_idx: int, component: str = "action_expert") -> Dict[str, np.ndarray]:
        if component == "action_expert":
            weights = self.get_action_expert_weights()
        elif component == "vlm_language":
            weights = self.get_vlm_language_weights()
        else:
            raise ValueError(f"Unsupported component: {component}")
        prefix = f"model.layers.{layer_idx}."
        return {key[len(prefix):]: val for key, val in weights.items() if key.startswith(prefix)}

    def get_fused_qkv_weight(self, layer_idx: int, component: str = "action_expert", transpose_for_backend: bool = False) -> np.ndarray:
        lw = self.get_layer_weights(layer_idx, component)
        fused = fuse_qkv_weights(lw["self_attn.q_proj.weight"], lw["self_attn.k_proj.weight"], lw["self_attn.v_proj.weight"])
        return transpose_linear_weight(fused) if transpose_for_backend else fused

    def get_fused_gate_up_weight(self, layer_idx: int, component: str = "action_expert", transpose_for_backend: bool = False) -> np.ndarray:
        lw = self.get_layer_weights(layer_idx, component)
        fused = fuse_gate_up_weights(lw["mlp.gate_proj.weight"], lw["mlp.up_proj.weight"])
        return transpose_linear_weight(fused) if transpose_for_backend else fused

    def get_linear_weight(self, key: str, component: str, transpose_for_backend: bool = True) -> np.ndarray:
        if component not in self.categorized_weights:
            raise ValueError(f"Unsupported component: {component}")
        weight = self.categorized_weights[component][key]
        return transpose_linear_weight(weight) if transpose_for_backend else weight

    def get_bias(self, key: str, component: str, expand_for_backend: bool = True) -> np.ndarray:
        if component not in self.categorized_weights:
            raise ValueError(f"Unsupported component: {component}")
        bias = self.categorized_weights[component][key]
        return expand_bias(bias) if expand_for_backend else bias

    def to_ttnn_tensor(self, array: np.ndarray, *, dtype=None, layout=None, device=None, memory_config=None) -> Any:
        if not TTNN_AVAILABLE or _ttnn_module is None:
            raise RuntimeError("TTNN not available")
        ttnn = _ttnn_module
        if dtype is None:
            dtype = ttnn.bfloat16
        if layout is None:
            layout = ttnn.TILE_LAYOUT
        if memory_config is None:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
        return ttnn.as_tensor(array, dtype=dtype, layout=layout, device=device, memory_config=memory_config)