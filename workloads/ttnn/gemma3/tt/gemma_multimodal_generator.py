# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 multimodal wrapper around tt_transformers ``Generator``.
Vision inputs are normalized here before delegating to the shared text-prefill
core, so Gemma3 exposes a multimodal-specific public entrypoint.

"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import ttsim.front.ttnn as ttnn

# Local imports - adjust paths as needed for your project structure
from workloads.ttnn.tt_transformers.generator import (
    Generator,
    MAX_BATCHED_PREFILL_SEQ_LEN,
    SUPPORTED_PREFILL_BATCH_SIZES,
    max_prefill_chunk_size_cutoff,
    get_padded_prefill_len,
    tensor_to_numpy,
    LogProbsResult,
    reformat_logprobs,
    SamplingParams,
    broadcast_sampling_params,
    format_sampling_params,
    Mode,
    logger,
)


# ============================================================================
# TTNN function wrappers for simulation compatibility
# ============================================================================
def _ttnn_execute_trace(
    device: Any,
    trace_id: Any,
    cq_id: int = 0,
    blocking: bool = False
) -> None:
    """Wrapper for ttnn.execute_trace with fallback."""
    if hasattr(ttnn, 'execute_trace'):
        ttnn.execute_trace(device, trace_id, cq_id=cq_id, blocking=blocking)


def _ttnn_untilize(tensor: Any, use_multicore: bool = True) -> Any:
    """Wrapper for ttnn.untilize with fallback."""
    if hasattr(ttnn, 'untilize'):
        return ttnn.untilize(tensor, use_multicore=use_multicore)
    return tensor


def _ttnn_synchronize_device(device: Any) -> None:
    """Wrapper for ttnn.synchronize_device with fallback."""
    if hasattr(ttnn, 'synchronize_device'):
        ttnn.synchronize_device(device)


def _ttnn_get_device_tensors(tensor: Any) -> List[Any]:
    """Wrapper for ttnn.get_device_tensors with fallback."""
    if hasattr(ttnn, 'get_device_tensors'):
        return ttnn.get_device_tensors(tensor)
    return [tensor]


def _ttnn_copy_host_to_device_tensor(host: Any, device: Any) -> None:
    """Wrapper for ttnn.copy_host_to_device_tensor with fallback."""
    if hasattr(ttnn, 'copy_host_to_device_tensor'):
        ttnn.copy_host_to_device_tensor(host, device)


def _deepseek_kvdbg_enabled() -> bool:
    """Check if DeepSeek KV debug logging is enabled."""
    return os.getenv("DEEPSEEK_KVDBG", "").lower() in ("1", "true", "yes", "y")


class GemmaMultimodalGenerator(Generator):
    """
    Gemma3 multimodal generator that wraps the base Generator class.
    
    Handles vision input preprocessing before delegating to text prefill.
    """
    
    def __init__(
        self,
        model: Any,
        model_args: Any,
        mesh_device: Any,
        processor: Any = None,
        tokenizer: Any = None
    ) -> None:
        """
        Initialize GemmaMultimodalGenerator.
        
        Args:
            model: The Gemma3 model (or list of models for data parallel)
            model_args: Model configuration arguments
            mesh_device: TTNN mesh device
            processor: Optional processor for tokenization
            tokenizer: Optional tokenizer
        """
        super().__init__(model, model_args, mesh_device, processor, tokenizer)
        # Explicitly initialize to help mypy
        self.already_warmed_up_prefill: bool = False

    def encode_vision_for_prefill(
        self, 
        pixel_values: List[Optional[np.ndarray]]
    ) -> List[Optional[Any]]:
        """
        Encode pixel values to vision embeddings for prefill.
        """
        if not hasattr(self.model[0], "encode_vision_embeddings_from_pixels"):
            raise TypeError(
                "GemmaMultimodalGenerator requires TtGemmaModel (multimodal). "
                "text_demo uses tt_transformers.Generator with a plain Transformer."
            )
        
        return [
            self.model[0].encode_vision_embeddings_from_pixels(pv) 
            if pv is not None else None 
            for pv in pixel_values
        ]

    def _prepare_multimodal_prefill_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Prepare kwargs for multimodal prefill by converting pixel_values to vision_embeddings.
        """
        if kwargs.get("vision_embeddings") is None and kwargs.get("pixel_values") is not None:
            kwargs = dict(kwargs)
            kwargs["vision_embeddings"] = self.encode_vision_for_prefill(kwargs["pixel_values"])
            kwargs.pop("pixel_values", None)
        return kwargs

    def prefill_forward_multimodal(
        self,
        tokens: np.ndarray,
        page_table: Optional[np.ndarray] = None,
        kv_cache: Any = None,
        prompt_lens: Optional[Union[np.ndarray, List[int]]] = None,
        empty_slots: Optional[List[int]] = None,
        enable_trace: bool = True,
        model_id_warmup: Optional[int] = None,
        sampling_params: Optional[SamplingParams] = None,
        start_pos: Optional[List[int]] = None,
        return_hidden_states: bool = False,
        warmup_prefill: bool = True,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """
        Multimodal prefill forward pass.
        """
        kwargs = self._prepare_multimodal_prefill_kwargs(**kwargs)
        
        return self.prefill_forward_text(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            enable_trace=enable_trace,
            model_id_warmup=model_id_warmup,
            sampling_params=sampling_params,
            start_pos=start_pos,
            return_hidden_states=return_hidden_states,
            warmup_prefill=warmup_prefill,
            **kwargs,
        )

    def prefill_forward(
        self,
        vision_images: Optional[List[np.ndarray]],
        vision_masks: Any,
        tokens: np.ndarray,
        xattn_caches: Any,
        total_lens: Any,
        prompt_lens: Union[np.ndarray, List[int]],
        page_table: Optional[np.ndarray] = None,
        kv_cache: Any = None,
        cross_page_table: Any = None,
        empty_slots: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """
        Public prefill interface matching vLLM expectations.
        """
        # These parameters are not used in Gemma3's architecture
        del vision_masks, xattn_caches, total_lens, cross_page_table
        
        return self.prefill_forward_multimodal(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            pixel_values=vision_images,
            **kwargs,
        )

    def prefill_forward_text(
        self,
        tokens: np.ndarray,
        page_table: Optional[np.ndarray] = None,
        kv_cache: Any = None,
        prompt_lens: Optional[Union[np.ndarray, List[int]]] = None,
        empty_slots: Optional[List[int]] = None,
        enable_trace: bool = True,
        model_id_warmup: Optional[int] = None,
        sampling_params: Optional[SamplingParams] = None,
        start_pos: Optional[List[int]] = None,
        return_hidden_states: bool = False,
        warmup_prefill: bool = True,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """
        Text prefill forward pass with multimodal support.
        """
        # Prepare multimodal kwargs
        kwargs = self._prepare_multimodal_prefill_kwargs(**kwargs)
        
        self.mode = Mode.PREFILL
        
        if page_table is not None:
            assert isinstance(page_table, np.ndarray), "page_table must be numpy.ndarray"
        else:
            enable_trace = False
        
        sampling_on_device_requested = sampling_params is not None
        
        if warmup_prefill:
            sampling_on_device_enabled = (
                getattr(self.model[0], "_supports_on_device_sampling", False)
                and getattr(self.model[0], "sampling", None) is not None
            )
            self.warmup_model_prefill(
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                can_sample_on_device=sampling_on_device_enabled,
            )
        
        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = getattr(self.model_args[0], 'max_batch_size', 32)
        
        # Prepare output tensors
        output_tokens: np.ndarray = np.zeros((batch_size, 1), dtype=np.int64)
        output_log_probs: List[Any] = [None] * batch_size
        
        if return_hidden_states:
            hidden_size = getattr(self.model_args[0], 'dim', 4096)
            output_tensor = np.zeros((batch_size, hidden_size), dtype=np.float32)
        else:
            vocab_size = getattr(self.model_args[0], 'vocab_size', 128256)
            output_tensor = np.zeros((batch_size, 1, vocab_size), dtype=np.float32)
        
        sampling_executed = False
        
        # Process prompt_lens
        prompt_lens_list: List[int]
        if prompt_lens is None:
            prompt_lens_list = [batch_seq_len] * batch_size
        elif isinstance(prompt_lens, np.ndarray):
            prompt_lens_list = [int(x) for x in prompt_lens.tolist()]
        else:
            prompt_lens_list = [int(x) for x in prompt_lens]
        
        if empty_slots is None:
            empty_slots = list(range(batch_size))
        
        local_batch_size = getattr(
            self.model_args[0], "max_local_batch_size", max_batch_size_per_model
        )
        
        prefill_seq_lens = [get_padded_prefill_len(seq_len) for seq_len in prompt_lens_list]
        
        # Check for row-sharded batched prefill
        model_0 = self.model[0]
        is_harmony = tokens.shape[1] > 0 and int(tokens[0, 0]) == 200006
        
        if (
            getattr(model_0, "users_row_sharded", False)
            and batch_size > 1
            and sampling_params is not None
            and is_harmony
        ):
            return self._row_sharded_batched_prefill(
                tokens,
                page_table,
                kv_cache,
                prompt_lens_list,
                prefill_seq_lens=prefill_seq_lens,
                enable_trace=enable_trace,
                sampling_params=sampling_params,
            )
        
        # Check for batched prefill
        use_batched_prefill = (
            batch_size > 1
            and len(set(prefill_seq_lens)) == 1
            and self.data_parallel == 1
            and not getattr(self.model_args[0], "disable_batched_prefill", False)
        )
        
        if use_batched_prefill and sampling_on_device_requested:
            sampling_module, sampling_dp, _, _ = self._get_sampling_contract(0)
            if sampling_module is not None and sampling_dp > 1:
                use_batched_prefill = False
        
        if use_batched_prefill:
            padded_batch = next(
                (b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size),
                max_batch_size_per_model,
            )
            if padded_batch > max_batch_size_per_model:
                logger.info(
                    f"Batched prefill disabled: padded_batch {padded_batch} exceeds "
                    f"max_batch_size {max_batch_size_per_model}"
                )
                use_batched_prefill = False
            elif padded_batch * prefill_seq_lens[0] >= MAX_BATCHED_PREFILL_SEQ_LEN:
                logger.info(
                    f"Batched prefill disabled: {padded_batch} x {prefill_seq_lens[0]} = "
                    f"{padded_batch * prefill_seq_lens[0]} tokens exceeds limit"
                )
                use_batched_prefill = False
        
        if not use_batched_prefill:
            padded_batch = max_batch_size_per_model
        
        all_users = [0] if use_batched_prefill else empty_slots
        prefill_results: List[Dict[str, Any]] = []
        
        # Variables that need to be declared before the loop for type checking
        num_cached_tokens: int = 0
        
        for idx, user_id in enumerate(all_users):
            model_id = (
                user_id // max_batch_size_per_model 
                if model_id_warmup is None else model_id_warmup
            )
            group_user_id = user_id % local_batch_size if page_table is None else 0
            
            # Declare with explicit types
            batch_user_ids: Optional[List[int]]
            last_token_idx_list: List[int]
            last_token_idx_int: int
            prefill_seq_len: int
            seq_len_int: int
            seq_len_list: List[int]
            
            if use_batched_prefill:
                batch_user_ids = empty_slots
                last_token_idx_list = [(sl - 1) for sl in prompt_lens_list]
                prefill_seq_len = prefill_seq_lens[0]
                seq_len_list = prompt_lens_list
                # For batched, we'll use last_token_idx_list
                last_token_idx_int = 0  # placeholder, not used in batched path
                seq_len_int = 0  # placeholder
            else:
                batch_user_ids = None
                seq_len_int = int(prompt_lens_list[idx])
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                last_token_idx_int = seq_len_int - 1
                prefill_seq_len = prefill_seq_lens[idx]
                last_token_idx_list = []  # placeholder, not used in non-batched path
                seq_len_list = []  # placeholder
                logger.info(f"Prefilling User {user_id + 1} up to {seq_len_int} tokens")
            
            local_kwargs = kwargs.copy()
            
            if getattr(self.model[model_id], "users_row_sharded", False):
                local_kwargs["global_user_id"] = (
                    batch_user_ids if use_batched_prefill else user_id
                )
            
            sampling_enabled = (
                sampling_on_device_requested
                and getattr(self.model[model_id], "_supports_on_device_sampling", False)
                and getattr(self.model[model_id], "sampling", None) is not None
            )
            
            # Prepare prefill_ids
            if use_batched_prefill:
                prefill_ids = np.zeros((padded_batch, prefill_seq_len), dtype=np.int64)
                padded_last_token_idx: List[int] = [0] * padded_batch
                
                for local_idx, slot in enumerate(empty_slots):
                    seq_len_local = int(prompt_lens_list[local_idx])
                    padded_tokens = np.concatenate([
                        tokens[local_idx:local_idx + 1, :seq_len_local],
                        np.zeros((1, prefill_seq_len - seq_len_local), dtype=np.int64),
                    ], axis=-1)
                    prefill_ids[slot:slot + 1] = padded_tokens
                    padded_last_token_idx[slot] = last_token_idx_list[local_idx]
                
                # Update for batched path
                last_token_idx_list = padded_last_token_idx
            else:
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                prefill_ids = np.concatenate([
                    tokens[idx:idx + 1, num_cached_tokens:seq_len_int],
                    np.zeros((1, prefill_seq_len - (seq_len_int - num_cached_tokens)), dtype=np.int64),
                ], axis=-1)
            
            # Check if trace can be enabled
            can_enable_trace_fn: Callable[[int, int], bool] = getattr(
                self.model_args[model_id], 'can_enable_trace',
                lambda x, y: True
            )
            enable_trace_current_prompt = enable_trace and can_enable_trace_fn(
                prefill_seq_len,
                num_cached_tokens if not use_batched_prefill else 0
            )
            
            max_prefill_chunk = getattr(self.model_args[0], 'max_prefill_chunk_size', 8192)
            logger.info(
                f"Prefill seq len: {prefill_seq_len}, "
                f"max_prefill_chunk_size: {max_prefill_chunk}, "
                f"trace: {enable_trace_current_prompt}"
            )
            
            # Get page table for user
            page_table_user: Optional[np.ndarray] = None
            if page_table is not None:
                page_table_for_user = (
                    page_table if use_batched_prefill else page_table[idx:idx + 1]
                )
                # Pass appropriate seq_len type
                page_table_seq_len: Union[int, List[int]] = seq_len_list if use_batched_prefill else seq_len_int
                page_table_user = self._get_prefill_user_page_table(
                    page_table_for_user,
                    kv_cache[model_id] if kv_cache else None,
                    page_table_seq_len,
                    trace_enabled=enable_trace_current_prompt,
                    prefill_seq_len=prefill_seq_len,
                    use_batched_prefill=use_batched_prefill,
                    user_id=batch_user_ids if use_batched_prefill else user_id,
                    padded_batch_size=padded_batch if use_batched_prefill else None,
                )
            
            # Debug logging
            if page_table_user is not None and _deepseek_kvdbg_enabled():
                sample: List[int] = []
                if page_table_user.size > 0:
                    flat = page_table_user.reshape(-1)
                    sample = [int(x) for x in flat[:min(16, int(flat.size))].tolist()] # type: ignore[arg-type]
                debug_seq_len = seq_len_int if not use_batched_prefill else str(seq_len_list)
                logger.debug(
                    f"KVDBG deepseek prefill user global={user_id} local={group_user_id} "
                    f"seq_len={debug_seq_len} cached={num_cached_tokens if not use_batched_prefill else 0} "
                    f"page_table_shape={list(page_table_user.shape)} sample={sample}"
                )
            
            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None
            
            # Per-user multimodal kwargs
            if "vision_embeddings" in local_kwargs and local_kwargs["vision_embeddings"] is not None:
                local_kwargs["vision_embeddings"] = local_kwargs["vision_embeddings"][idx]
            
            if local_kwargs.get("pixel_values") is not None:
                local_kwargs["pixel_values"] = local_kwargs["pixel_values"][idx]
                if "image_grid_thw" in local_kwargs:
                    local_kwargs["image_grid_thw"] = local_kwargs["image_grid_thw"][idx]
                if local_kwargs.get("image_sizes") is not None:
                    local_kwargs["image_sizes"] = local_kwargs["image_sizes"][idx]
            
            # Apply sampling state for non-batched prefill
            if sampling_enabled and not use_batched_prefill:
                sampling_executed = True
                sampling_dp = getattr(self.model[model_id], "sampling_dp", 1)
                total_batch = self.model[model_id].sampling.tt_sampling.max_batch_size * sampling_dp
                per_request_params = format_sampling_params(
                    broadcast_sampling_params(sampling_params, idx, slot_len=total_batch),
                    total_batch
                )
                assert per_request_params is not None
                
                prompt_for_sampling = np.tile(
                    prefill_ids[:, :seq_len_int], (total_batch, 1)
                )
                
                self.model[model_id].sampling.apply_prefill_state(
                    sampling_params=per_request_params,
                    prompt_tokens=prompt_for_sampling,
                    empty_slots=[user_id % max_batch_size_per_model],
                )
            
            # Determine user_id for prefill call
            prefill_user_id: int
            if batch_user_ids is not None:
                prefill_user_id = batch_user_ids[0]
            else:
                prefill_user_id = group_user_id
            
            # Determine last_token_idx for prefill call
            prefill_last_token_idx: Union[int, List[int]]
            if use_batched_prefill:
                prefill_last_token_idx = last_token_idx_list
            else:
                prefill_last_token_idx = last_token_idx_int
            
            # Run prefill
            if enable_trace_current_prompt:
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=prefill_user_id,
                    last_token_idx=prefill_last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    prefill_seq_len=prefill_seq_len,
                    batch_size=padded_batch if use_batched_prefill else 1,
                    **local_kwargs,
                )
            else:
                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=prefill_user_id,
                    last_token_idx=prefill_last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    num_cached_tokens=0 if use_batched_prefill else num_cached_tokens,
                    batch_size=padded_batch if use_batched_prefill else 1,
                    **local_kwargs,
                )
            
            # Process batched prefill results
            if use_batched_prefill:
                hidden_dim = logits.shape[-1]
                logits = ttnn.reshape(logits, [padded_batch, 1, prefill_seq_len, hidden_dim])
                
                if sampling_enabled:
                    sampling_executed = True
                    sampling_module, sampling_dp, sampling_batch, _ = self._get_sampling_contract(model_id)
                    assert sampling_module is not None
                    assert sampling_batch is not None
                    
                    combined_params = format_sampling_params(sampling_params, sampling_batch)
                    max_prompt_len = max(int(prompt_lens_list[i]) for i in range(len(empty_slots)))
                    combined_prompt_tokens = np.zeros(
                        (sampling_batch, max_prompt_len), dtype=np.int64
                    )
                    
                    for local_idx, slot in enumerate(empty_slots):
                        plen = int(prompt_lens_list[local_idx])
                        combined_prompt_tokens[slot, :plen] = prefill_ids[slot, :plen]
                    
                    sampling_module.apply_prefill_state(
                        sampling_params=combined_params,
                        prompt_tokens=combined_prompt_tokens,
                        empty_slots=empty_slots,
                        replicate_seeds=False,
                    )
                    
                    user_hidden = self.model[model_id].extract_last_tokens_batched_prefill(
                        logits,
                        last_token_idx_list,
                        padded_batch,
                        prefill_seq_len,
                        target_batch=sampling_batch,
                    )
                    
                    sampling_trace_key = f"sampling_{prefill_seq_len}_{model_id}_{sampling_batch}_{sampling_dp}"
                    
                    if enable_trace_current_prompt:
                        if self.trace_id_prefill_sampling[sampling_trace_key] is None:
                            (
                                s_trace_id,
                                s_trace_output,
                                s_trace_input,
                            ) = self._capture_trace_prefill_sampling(model_id, sampling_batch)
                            self.trace_id_prefill_sampling[sampling_trace_key] = s_trace_id
                            self.trace_output_prefill_sampling[sampling_trace_key] = s_trace_output
                            self.trace_input_prefill_sampling[sampling_trace_key] = s_trace_input
                        
                        s_trace_input = self.trace_input_prefill_sampling[sampling_trace_key]
                        user_hidden_host = user_hidden.cpu()
                        _ttnn_copy_host_to_device_tensor(user_hidden_host, s_trace_input)
                        
                        mesh_dev = getattr(self.model_args[model_id], 'mesh_device', self.mesh_device)
                        _ttnn_execute_trace(
                            mesh_dev,
                            self.trace_id_prefill_sampling[sampling_trace_key],
                            cq_id=0,
                            blocking=False,
                        )
                        tt_tokens, tt_log_probs = self.trace_output_prefill_sampling[sampling_trace_key]
                    else:
                        batched_logits = self.model[model_id]._apply_norm_and_lm_head(user_hidden)
                        tt_tokens, tt_log_probs = self.model[model_id].sampling.sample(
                            batched_logits,
                            enable_trace=False,
                        )
                    
                    mesh_dev = getattr(self.model[model_id], 'mesh_device', self.mesh_device)
                    _ttnn_synchronize_device(mesh_dev)
                    
                    # Extract results using wrapper
                    device_tensors = _ttnn_get_device_tensors(tt_tokens)
                    tokens_np = tensor_to_numpy(device_tensors[0]).reshape(-1)
                    
                    log_probs_np: Optional[np.ndarray] = None
                    if tt_log_probs is not None:
                        log_probs_device_tensors = _ttnn_get_device_tensors(tt_log_probs)
                        log_probs_np = tensor_to_numpy(log_probs_device_tensors[0]).reshape(-1)
                    
                    for local_idx, slot in enumerate(empty_slots):
                        output_tokens[slot] = int(tokens_np[slot])
                        if log_probs_np is not None:
                            output_log_probs[slot] = float(log_probs_np[slot])
                else:
                    # Non-sampling batched prefill
                    for local_idx, slot in enumerate(empty_slots):
                        user_logits = logits[slot:slot + 1, :, :, :]
                        slot_last_token_idx = last_token_idx_list[slot]
                        _logits = self.model[model_id].process_logits_after_prefill_trace(
                            user_logits, slot_last_token_idx
                        )
                        _logits = ttnn.to_layout(
                            _logits, 
                            ttnn.ROW_MAJOR_LAYOUT, 
                            memory_config=ttnn.DRAM_MEMORY_CONFIG
                        )
                        output_tensor[slot] = self.model[model_id].process_output_prefill(
                            _logits.cpu(), 
                            last_token_idx=(slot_last_token_idx % 32)
                        )
                break
            
            # Non-batched prefill path
            if enable_trace_current_prompt:
                if return_hidden_states:
                    hidden_states = self.model[model_id].process_hidden_states_after_prefill_trace(
                        logits, last_token_idx_int
                    )
                    prefill_results.append({
                        "idx": idx,
                        "model_id": model_id,
                        "last_token_idx": last_token_idx_int,
                        "hidden_states": hidden_states.cpu(blocking=False),
                    })
                    continue
                else:
                    logits = self.model[model_id].process_logits_after_prefill_trace(
                        logits, last_token_idx_int
                    )
            else:
                if return_hidden_states:
                    raise NotImplementedError(
                        "return_hidden_states=True requires enable_trace=True"
                    )
            
            if sampling_enabled:
                tt_tokens, tt_log_probs = self.model[model_id].sampling.sample(
                    logits,
                    enable_trace=False,
                )
                prefill_results.append({
                    "idx": idx,
                    "model_id": model_id,
                    "last_token_idx": last_token_idx_int,
                    "logits": [
                        tt_tokens.cpu(blocking=False),
                        tt_log_probs.cpu(blocking=False) if tt_log_probs is not None else None,
                    ],
                    "sampling": sampling_enabled,
                })
            else:
                logits = _ttnn_untilize(logits, use_multicore=True)
                prefill_results.append({
                    "idx": idx,
                    "model_id": model_id,
                    "last_token_idx": last_token_idx_int,
                    "logits": logits.cpu(blocking=False),
                    "sampling": sampling_enabled,
                })
        
        # Process prefill results
        if len(prefill_results) > 0:
            for res in prefill_results:
                idx = res["idx"]
                last_token_idx_res: int = res["last_token_idx"]
                model_id = res["model_id"]
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                
                last_token_idx_relative = last_token_idx_res - num_cached_tokens
                
                mesh_dev = getattr(self.model[model_id], 'mesh_device', self.mesh_device)
                _ttnn_synchronize_device(mesh_dev)
                
                if "hidden_states" in res:
                    output_tensor[idx] = self.model[model_id].process_output_prefill_hidden_states(
                        res["hidden_states"],
                        last_token_idx=(last_token_idx_relative % 32)
                    )
                elif res.get("sampling", False):
                    tt_tokens_res = res["logits"][0]
                    tt_log_probs_res = res["logits"][1]
                    
                    device_tensors = _ttnn_get_device_tensors(tt_tokens_res)
                    tokens_np = tensor_to_numpy(device_tensors[0]).reshape(-1)
                    tokens_host = int(tokens_np[last_token_idx_res % 32])
                    
                    log_probs_host: Optional[float] = None
                    if isinstance(tt_log_probs_res, LogProbsResult):
                        log_probs_host = tt_log_probs_res.extract_user(last_token_idx_res % 32)
                    elif tt_log_probs_res is not None:
                        log_probs_device_tensors = _ttnn_get_device_tensors(tt_log_probs_res)
                        log_probs_np = tensor_to_numpy(log_probs_device_tensors[0]).reshape(-1)
                        log_probs_host = float(log_probs_np[last_token_idx_res % 32])
                    
                    output_tokens[idx] = tokens_host
                    if log_probs_host is not None:
                        output_log_probs[idx] = log_probs_host
                else:
                    output_tensor[idx] = self.model[model_id].process_output_prefill(
                        res["logits"],
                        last_token_idx=(last_token_idx_relative % 32)
                    )
        
        logger.info(
            f"Finished prefill for all users up to {batch_seq_len} tokens, "
            f"Starting decode..."
        )
        
        if sampling_executed:
            return output_tokens, reformat_logprobs(output_log_probs, batch_size)
        else:
            return output_tensor

    def warmup_model_prefill(
        self,
        kv_cache: Any,
        enable_trace: bool,
        can_sample_on_device: bool,
        greedy_only: bool = False
    ) -> None:
        """
        Warmup model for prefill with multimodal support.
        """
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True
        
        sequence_lengths_fn = getattr(
            self.model_args[0],
            'get_warmup_prefill_supported_seq_lens',
            lambda: [128, 256, 512, 1024, 2048]
        )
        sequence_lengths_to_warmup: List[int]
        if callable(sequence_lengths_fn):
            result = sequence_lengths_fn()
            if isinstance(result, str):
                sequence_lengths_to_warmup = [int(result)]
            else:
                sequence_lengths_to_warmup = [int(x) for x in result]
        else:
            if isinstance(sequence_lengths_fn, str):
                sequence_lengths_to_warmup = [int(sequence_lengths_fn)]
            elif isinstance(sequence_lengths_fn, list) and len(sequence_lengths_fn) > 0 and isinstance(sequence_lengths_fn[0], list):
                # Flatten nested list
                sequence_lengths_to_warmup = [int(x) for sublist in sequence_lengths_fn for x in sublist]
            else:
                sequence_lengths_to_warmup = [int(x) for x in sequence_lengths_fn]  # type: ignore[arg-type]
        warmup_batch_sizes = (1,)
        skip_sequence_lengths = False
        sampling_parameters_sweeped = False
        
        if enable_trace:
            logger.info(
                "Using batch-1-only traced prefill warmup; "
                "runtime batched prefill remains enabled"
            )
        
        for model_id in range(self.data_parallel):
            for supported_length in sequence_lengths_to_warmup:
                trace_supported = getattr(
                    self.model_args[0], 'trace_prefill_supported_seq_lens', []
                )
                if model_id != 0 and (
                    supported_length not in trace_supported or not enable_trace
                ):
                    continue
                
                for batch_size in warmup_batch_sizes:
                    if (batch_size > 1 and 
                        batch_size * supported_length >= MAX_BATCHED_PREFILL_SEQ_LEN):
                        logger.info(
                            f"Skipping batched prefill warmup for batch_size={batch_size}, "
                            f"seq_len={supported_length}: exceeds token limit"
                        )
                        continue
                    
                    warmup_args = self._mock_tokens(
                        batch_size, supported_length, kv_cache, model_id
                    )
                    
                    max_prefill_chunk = getattr(
                        self.model_args[0], 'max_prefill_chunk_size', 8192
                    )
                    if (warmup_args["page_table"] is None and 
                        max_prefill_chunk_size_cutoff(supported_length, max_prefill_chunk)):
                        logger.warning(
                            f"Skipping warmup for sequence lengths after: {supported_length} "
                            f"because they exceed max prefill chunk size"
                        )
                        skip_sequence_lengths = True
                        break
                    
                    sampling_params_list: List[Optional[SamplingParams]]
                    if not sampling_parameters_sweeped:
                        sampling_params_list = self._create_sampling_params(
                            can_sample_on_device=can_sample_on_device,
                            batch_size=batch_size,
                            greedy_only=greedy_only,
                        )
                    else:
                        sampling_params_list = [None]
                    
                    for param in sampling_params_list:
                        logger.info(
                            f"Warming up prefill for seq_len={supported_length}, "
                            f"batch_size={batch_size}, sampling_params={param}"
                        )
                        self.prefill_forward_text(
                            **warmup_args,
                            kv_cache=kv_cache,
                            enable_trace=enable_trace,
                            model_id_warmup=model_id,
                            sampling_params=param,
                        )
                    sampling_parameters_sweeped = True
                
                if skip_sequence_lengths:
                    break
        
        # Vision warmup for multimodal models
        if (
            getattr(self.model_args[0], "is_multimodal", False) and 
            hasattr(self.model[0], "encode_vision_embeddings_from_pixels")
        ):
            vision_chunk_size = getattr(self.model_args[0], "vision_chunk_size", 896)
            vision_channels = getattr(self.model_args[0], "vision_in_channels", 3)
            model_id_for_warmup = 0
            
            # Create synthetic image for vision warmup using numpy
            warmup_pixel_values = [
                np.zeros(
                    (1, vision_channels, vision_chunk_size, vision_chunk_size),
                    dtype=np.float32
                )
            ]
            
            warmup_batch_size = 1
            prefill_forward_args = self._mock_tokens(
                warmup_batch_size, 128, kv_cache, model_id_for_warmup
            )
            
            logger.info(
                f"Warming up vision encoder with image size "
                f"{vision_chunk_size}x{vision_chunk_size}"
            )
            
            multimodal_prefill = getattr(self, "prefill_forward_multimodal", None)
            if callable(multimodal_prefill):
                multimodal_prefill(
                    prefill_forward_args["tokens"],
                    page_table=prefill_forward_args["page_table"],
                    kv_cache=kv_cache,
                    prompt_lens=prefill_forward_args["prompt_lens"],
                    empty_slots=prefill_forward_args["empty_slots"],
                    enable_trace=False,
                    model_id_warmup=model_id_for_warmup,
                    sampling_params=None,
                    pixel_values=warmup_pixel_values,
                    image_sizes=[(vision_chunk_size, vision_chunk_size)],
                )
            else:
                self.prefill_forward_text(
                    **prefill_forward_args,
                    kv_cache=kv_cache,
                    enable_trace=False,
                    model_id_warmup=model_id_for_warmup,
                    sampling_params=None,
                    pixel_values=warmup_pixel_values,
                    image_sizes=[(vision_chunk_size, vision_chunk_size)],
                )
            
            logger.info("Vision encoder warmup completed")