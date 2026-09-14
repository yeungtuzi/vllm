# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops import fused_inv_rope_fp8_quant
from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
    deepseek_v4_fp8_einsum,
    deepseek_v4_fp8_einsum_config,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


def compute_fp8_einsum_recipe(
    block_size: int = 128,
) -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90 keeps block-row FP32 scales. SM100 uses packed per-row E8M0 scales.
    SM12x: RTX PRO / GB10 does not expose the same TMA/TCGEN05 path, so keep
    the legacy FP32 block-scale layout expected by DeepGEMM.
    SM 8.x (Ampere/Ada): no DeepGEMM fp8_einsum; the portable Triton kernel in
    ``fp8_einsum.py`` consumes the legacy [g, r/128, d/128] layout.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    if cap.major in (8, 12):
        return deepseek_v4_fp8_einsum_config(cap.major)
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization. ``wo_b`` is any callable over
    the flattened ``z``: the projection module itself, or a wrapper that also
    reduce-scatters its output (DeepSeek-V4.1 GEMM-RS).
    """
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=einsum_recipe[2],
        tma_aligned_scales=tma_aligned_scales,
        quantize=use_fp8,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        cap = current_platform.get_device_capability()
        if cap is not None and cap.major in (8, 12):
            # DeepGEMM's fp8_einsum is SM90+ only, so Ampere/Ada and SM12x use
            # the portable Triton kernel. MarlinFP8 renames block-FP8 scales to
            # weight_scale_inv; other kernels keep the weight_scale name.
            weight_scale = getattr(wo_a, "weight_scale_inv", None)
            if weight_scale is None:
                weight_scale = wo_a.weight_scale
            deepseek_v4_fp8_einsum(
                o_proj_input,
                o_scale,
                wo_a.weight,
                weight_scale,
                z,
                "bhr,hdr->bhd",
                list(einsum_recipe),
            )
        else:
            weight_scale = (
                wo_a.weight_scale
                if hasattr(wo_a, "weight_scale")
                else wo_a.weight_scale_inv
            )
            fp8_einsum(
                "bhr,hdr->bhd",
                (o_proj_input, o_scale),
                (wo_a.weight, weight_scale),
                z,
                recipe=einsum_recipe,
            )
    else:
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        torch.bmm(
            o_proj_input.transpose(0, 1),
            grouped_weight.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    return wo_b(z.flatten(1))
