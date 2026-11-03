#!/usr/bin/env python3
"""Native `FlintForCausalLM` (`RecursiveSI/flint`, 40B-A2.5B MoE) plugin installer for `vllm-torchtpu` (`v0.32.0`).

Registers `FlintConfig` (`model_type="flint"`) and `FlintForCausalLM` in both HuggingFace `transformers`
and `vllm`, enabling:
  - Multi-host Expert Parallelism (`EP=16` on `v5p-16`, `EP=32` on `v7x-16`) via Ray + Pallas GMM / SparseCore
  - Hybrid attention: 14 RoPE-free `full_attention` layers + 35 RoPE `sliding_attention` layers (`sliding_window=513`)
  - QK-RMSNorm (`q_norm`, `k_norm`) + TP-sharded learned per-head attention sinks (`softmax_offset` -> `sinks`)
  - Dense SwiGLU MLP on `mlp_only_layers=[0, 1]` (`intermediate_size=6144`) + 128-expert top-6 sigmoid-routed MoE
    with ungated `shared_expert` (`shared_expert_intermediate_size=1024`) on layers `2..48`
  - Both Pure BF16 (`RecursiveSI/flint`) and `[128, 128]` dynamic Block-FP8 (`RecursiveSI/flint-fp8`, W8A16 on v5p,
    native FP8 on v7x) checkpoints, plus `ngram` speculative decoding.
"""
from __future__ import annotations

from pathlib import Path
import py_compile
import site


FLINT_CONFIG_PY = '''# SPDX-License-Identifier: Apache-2.0
"""HuggingFace / vLLM configuration class for RecursiveSI/flint (`model_type="flint"`)."""

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class FlintConfig(Qwen3MoeConfig):
    model_type = "flint"

    def __init__(
        self,
        vocab_size: int = 150144,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 49,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        use_sliding_window: bool = True,
        sliding_window: int | None = 513,
        head_dim: int = 128,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 1024,
        num_experts_per_tok: int = 6,
        num_experts: int = 256,
        norm_topk_prob: bool = True,
        decoder_sparse_step: int = 1,
        mlp_only_layers: list[int] | None = None,
        no_rope_layer_indices: list[int] | None = None,
        layer_types: list[str] | None = None,
        router_score_function: str = "sigmoid",
        softmax_type: str = "learnable",
        **kwargs,
    ):
        if mlp_only_layers is None:
            mlp_only_layers = [0, 1]
        if no_rope_layer_indices is None:
            no_rope_layer_indices = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
        if layer_types is None:
            no_rope_set = set(no_rope_layer_indices)
            layer_types = [
                "full_attention" if i in no_rope_set else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            use_sliding_window=bool(use_sliding_window or sliding_window is not None),
            sliding_window=sliding_window,
            head_dim=head_dim,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            num_experts=num_experts,
            norm_topk_prob=norm_topk_prob,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=mlp_only_layers,
            layer_types=layer_types,
            **kwargs,
        )
        # Qwen3MoeConfig zeros sliding_window when use_sliding_window is False; restore explicitly.
        self.sliding_window = sliding_window
        self.no_rope_layer_indices = list(no_rope_layer_indices)
        self.router_score_function = router_score_function
        self.layer_types = list(layer_types)


__all__ = ["FlintConfig"]
'''


FLINT_MODEL_PY = '''# SPDX-License-Identifier: Apache-2.0
"""vLLM + TorchTPU native model definition for `RecursiveSI/flint` (`FlintForCausalLM`)."""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors

from .interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)


class FlintSparseMoeBlock(nn.Module):
    """128-expert top-6 sigmoid-routed MoE block with an ungated shared SwiGLU expert."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_fused_checkpoint_transposed: bool = False,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} > num_experts {config.num_experts}."
            )

        cur_cfg = get_current_vllm_config()
        eplb_config = cur_cfg.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", 0
        )
        self.shared_expert_gate = None
        if shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=None,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        scoring_func = getattr(config, "router_score_function", "sigmoid")
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            scoring_func=scoring_func,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            is_fused_checkpoint_transposed=is_fused_checkpoint_transposed,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class FlintAttention(nn.Module):
    """Grouped-query attention with QK-RMSNorm, learned attention sinks, and hybrid SWA / RoPE-free FA."""

    def __init__(
        self,
        *,
        config: Any,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        no_rope_set = set(getattr(config, "no_rope_layer_indices", []))
        self.skip_rope = layer_idx in no_rope_set

        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and layer_idx < len(layer_types):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
        else:
            is_sliding = not self.skip_rope

        # Pallas RPA mask is `q_span < kv_span + sliding_window` (inclusive of current token),
        # matching Flint's HF config `sliding_window=513` (512 past + 1 current).
        per_layer_sw = config.sliding_window if is_sliding else None

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if not self.skip_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=getattr(config, "max_position_embeddings", 262144),
                rope_parameters=getattr(config, "rope_parameters", None)
                or {"rope_type": "default", "rope_theta": getattr(config, "rope_theta", 10000.0)},
            )
        else:
            self.rotary_emb = None

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Learned per-head attention sinks (`model.layers.{i}.self_attn.softmax_offset`, shape [32])
        self.softmax_offset = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.float32),
            requires_grad=False,
        )
        head_start = tp_rank * self.num_heads
        heads_per_rank = self.num_heads

        def _sink_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
            if loaded_weight.shape[0] == heads_per_rank:
                shard = loaded_weight
            else:
                shard = loaded_weight.narrow(0, head_start, heads_per_rank)
            param.data.copy_(shard.to(param.dtype))

        set_weight_attrs(self.softmax_offset, {"weight_loader": _sink_weight_loader})

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=per_layer_sw,
            sinks=self.softmax_offset,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class FlintDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_fused_checkpoint_transposed: bool = False,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        layer_idx = extract_layer_index(prefix)

        self.self_attn = FlintAttention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        mlp_only_layers = set(getattr(config, "mlp_only_layers", [0, 1]))
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0
        ):
            self.mlp = FlintSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
                is_fused_checkpoint_transposed=is_fused_checkpoint_transposed,
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class FlintModel(nn.Module, EagleModelMixin):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = FlintDecoderLayer,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        eplb_config = vllm_config.parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=[
                ".bias",
                "_bias",
                ".weight_scale",
                "_weight_scale",
                ".input_scale",
                "_input_scale",
            ],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class FlintForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsLoRA,
    SupportsEagle,
    SupportsEagle3,
    MixtureOfExperts,
    SupportsQuant,
):
    hf_to_vllm_mapper = FlintModel.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = FlintModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.expert_weights = []
        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, FlintSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No FlintSparseMoeBlock found in model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 1
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, FlintSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
'''


def install_flint_into_vllm() -> None:
    sp_dirs = [Path(p) for p in site.getsitepackages()]
    sp = next((p for p in sp_dirs if (p / "vllm").is_dir()), Path("/usr/local/lib/python3.12/site-packages"))
    vllm_root = sp / "vllm"
    assert vllm_root.is_dir(), f"vllm package not found under {sp}"

    # 1. Write standalone flint_hf_config.py (zero vllm import so sitecustomize.py is instant & circular-import-free)
    hf_cfg_file = sp / "flint_hf_config.py"
    hf_cfg_file.write_text(FLINT_CONFIG_PY)
    py_compile.compile(str(hf_cfg_file), doraise=True)

    # 1b. Write vllm/transformers_utils/configs/flint.py re-exporting FlintConfig
    cfg_file = vllm_root / "transformers_utils" / "configs" / "flint.py"
    cfg_file.write_text("from flint_hf_config import FlintConfig\n\n__all__ = ['FlintConfig']\n")
    py_compile.compile(str(cfg_file), doraise=True)

    # 2. Register FlintConfig in vllm/transformers_utils/configs/__init__.py
    cinit = vllm_root / "transformers_utils" / "configs" / "__init__.py"
    cinit_txt = cinit.read_text()
    if '"FlintConfig"' not in cinit_txt:
        anchor = '"BagelConfig": "vllm.transformers_utils.configs.bagel",'
        assert anchor in cinit_txt, f"Missing anchor in {cinit}"
        cinit_txt = cinit_txt.replace(
            anchor,
            anchor + '\n    "FlintConfig": "vllm.transformers_utils.configs.flint",',
            1,
        )
        anchor2 = '__all__ = [\n    "BagelConfig",'
        if anchor2 in cinit_txt:
            cinit_txt = cinit_txt.replace(anchor2, anchor2 + '\n    "FlintConfig",', 1)
        cinit.write_text(cinit_txt)
        py_compile.compile(str(cinit), doraise=True)

    # 3. Register flint="FlintConfig" in vllm/transformers_utils/config.py
    tcfg = vllm_root / "transformers_utils" / "config.py"
    tcfg_txt = tcfg.read_text()
    if 'flint="FlintConfig"' not in tcfg_txt:
        anchor = 'bagel="BagelConfig",'
        assert anchor in tcfg_txt, f"Missing anchor in {tcfg}"
        tcfg_txt = tcfg_txt.replace(anchor, anchor + '\n    flint="FlintConfig",', 1)
        tcfg.write_text(tcfg_txt)
        py_compile.compile(str(tcfg), doraise=True)

    # 4. Write FlintForCausalLM
    model_file = vllm_root / "model_executor" / "models" / "flint.py"
    model_file.write_text(FLINT_MODEL_PY)
    py_compile.compile(str(model_file), doraise=True)

    # 5. Register FlintForCausalLM in vllm/model_executor/models/registry.py
    reg = vllm_root / "model_executor" / "models" / "registry.py"
    reg_txt = reg.read_text()
    if '"FlintForCausalLM"' not in reg_txt:
        anchor = '"Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),'
        assert anchor in reg_txt, f"Missing anchor in {reg}"
        reg_txt = reg_txt.replace(
            anchor,
            anchor + '\n    "FlintForCausalLM": ("flint", "FlintForCausalLM"),',
            1,
        )
        reg.write_text(reg_txt)
        py_compile.compile(str(reg), doraise=True)

    # 6. Register FlintConfig directly in transformers/models/auto/configuration_auto.py
    #    (avoids /usr/lib/python3.12/sitecustomize.py shadowing site-packages/sitecustomize.py)
    auto_cfg_py = sp / "transformers" / "models" / "auto" / "configuration_auto.py"
    auto_txt = auto_cfg_py.read_text()
    marker = "# Auto-register RecursiveSI/flint config in transformers"
    if marker not in auto_txt:
        auto_txt = (
            auto_txt.rstrip()
            + f"\n\n{marker}\n"
            + "from flint_hf_config import FlintConfig as _FlintConfig\n"
            + "AutoConfig.register('flint', _FlintConfig, exist_ok=True)\n"
        )
        auto_cfg_py.write_text(auto_txt)
        py_compile.compile(str(auto_cfg_py), doraise=True)

    # Clean up any legacy sitecustomize.py snippet if present
    sc_py = sp / "sitecustomize.py"
    if sc_py.exists():
        sc_txt = sc_py.read_text()
        if marker in sc_txt:
            sc_py.write_text(sc_txt.split(marker)[0].rstrip() + "\n")

    from transformers import AutoConfig
    from flint_hf_config import FlintConfig
    AutoConfig.register("flint", FlintConfig, exist_ok=True)

    patch_rpa_v3_attention_sink()

    print(f"[install-flint-vllm] Successfully installed FlintConfig + FlintForCausalLM + RPAv3 hd128 sink in {vllm_root}", flush=True)


def patch_rpa_v3_attention_sink() -> None:
    """Enable learned attention_sink (softmax_aux) in vllm_torchtpu RPA v3 for arbitrary head_dim (e.g. head_dim=128)."""
    vtpu_root = Path("/workspace/vllm-torchtpu/src/vllm_torchtpu")
    if not vtpu_root.is_dir():
        return

    ai_py = vtpu_root / "layers" / "core" / "attention_interface.py"
    ai_txt = ai_py.read_text()
    ai_old = '''    if attention_sink is not None:
        if not use_hd64:
            raise NotImplementedError(
                "Attention sink support is only available when head_dim==64"
            )

        in_specs += (P("model"),)
        args += (attention_sink,)'''
    ai_new = '''    if attention_sink is not None:
        # [flint-rpa-sink] Enable attention_sink across all head_dim values (including head_dim=128)
        in_specs += (P("model"),)
        args += (attention_sink,)'''
    if "[flint-rpa-sink]" not in ai_txt:
        assert ai_txt.count(ai_old) == 1, f"Expected exact anchor in {ai_py}"
        ai_py.write_text(ai_txt.replace(ai_old, ai_new, 1))
        py_compile.compile(str(ai_py), doraise=True)

    krn_py = vtpu_root / "kernels" / "ragged_paged_attention" / "v3" / "kernel.py"
    krn_txt = krn_py.read_text()
    if "[flint-rpa-sink]" not in krn_txt:
        replacements = [
            (
                '''    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache_hbm_ref,
    # Output''',
                '''    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache_hbm_ref,
    attention_sink_ref,  # [flint-rpa-sink]
    # Output''',
            ),
            (
                '''            # Re-initialize l, m, acc to 0 before bkv loop.
            l_ref[...] = jnp.full_like(l_ref, 0.0)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            acc_ref[...] = jnp.full_like(acc_ref, 0.0)''',
                '''            # Re-initialize l, m, acc before bkv loop ([flint-rpa-sink]: seed m_ref=sink, l_ref=1.0).
            if attention_sink_ref is not None:
                l_ref[...] = jnp.full_like(l_ref, 1.0)
                m_ref[...] = attention_sink_ref[...]
            else:
                l_ref[...] = jnp.full_like(l_ref, 0.0)
                m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            acc_ref[...] = jnp.full_like(acc_ref, 0.0)''',
            ),
            (
                '''def ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,''',
                '''def ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sink: jax.Array | None = None,
    *,''',
            ),
            (
                '''    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing''',
                '''    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing
    if attention_sink is not None:
        sinks_2d = attention_sink.astype(jnp.float32).reshape(
            actual_num_kv_heads, actual_num_q_heads_per_kv_head
        )
        if num_q_heads_per_kv_head > actual_num_q_heads_per_kv_head:
            sinks_2d = jnp.pad(
                sinks_2d,
                ((0, 0), (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head)),
                constant_values=-jnp.inf,
            )
        attention_sink = jnp.repeat(
            sinks_2d.reshape(actual_num_kv_heads, num_q_heads_per_kv_head, 1),
            128,
            axis=-1,
        )''',
            ),
            (
                '''        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]''',
                '''        sink_tiled = (
            jnp.tile(attention_sink.astype(out_dtype), (1, bq_sz, 1))
            if attention_sink is not None
            else None
        )
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            None if sink_tiled is None else pl.BlockSpec(memory_space=pltpu.VMEM),
        ]''',
            ),
            (
                '''        if tpu_version >= 7:
            # jit to color the memory since the q, kv are just preprocessed.
            @jax.jit
            def run(scalar_prefetches, q, kv, kv_cache):
                return kernel(
                    *scalar_prefetches,
                    pltpu.with_memory_space_constraint(q, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv_cache, pltpu.HBM),
                )
        else:
            # TODO: v6 has issues with pinning aliased memory.
            def run(scalar_prefetches, q, kv, kv_cache):
                return kernel(*scalar_prefetches, q, kv, kv_cache)

        return run(scalar_prefetches, q, kv, kv_cache)''',
                '''        if tpu_version >= 7:
            # jit to color the memory since the q, kv are just preprocessed.
            @jax.jit
            def run(scalar_prefetches, q, kv, kv_cache, sink_tiled):
                return kernel(
                    *scalar_prefetches,
                    pltpu.with_memory_space_constraint(q, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv_cache, pltpu.HBM),
                    sink_tiled,
                )
        else:
            # TODO: v6 has issues with pinning aliased memory.
            def run(scalar_prefetches, q, kv, kv_cache, sink_tiled):
                return kernel(*scalar_prefetches, q, kv, kv_cache, sink_tiled)

        return run(scalar_prefetches, q, kv, kv_cache, sink_tiled)''',
            ),
        ]
        for old_s, new_s in replacements:
            assert krn_txt.count(old_s) == 1, f"Expected exact 1 match in {krn_py} for:\n{old_s[:120]}"
            krn_txt = krn_txt.replace(old_s, new_s, 1)
        krn_py.write_text(krn_txt)
        py_compile.compile(str(krn_py), doraise=True)


if __name__ == "__main__":
    install_flint_into_vllm()

