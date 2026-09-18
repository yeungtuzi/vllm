# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80/86/89 sparse-MLA backend for the generic (V3.2-style) MLA layer.

DeepSeek-V4/V4.1 reach the portable Triton SM8x path through their own
model-driven attention layer. Non-DSv4 DSA models built on
``MultiHeadLatentAttentionWrapper`` (``is_sparse=True``) instead route through
the generic sparse-MLA impl stack, whose FlashMLA/FlashInfer kernels require
SM90+. On an sm8x card that leaves the platform candidate pool empty
(``cuda.py`` raises) and the ``FLASHMLA_SPARSE_DSV4(1)`` enum is a model-driven
marker with no ``impl_cls``, so it cannot be reused here.

This backend closes that gap without touching the platform selector: it reuses
the whole bf16 gather / logical->physical topk machinery of
``FlashMLASparseImpl`` and swaps only the final attention kernel for the
fp8-type-free Triton ``sparse_mla_fwd_with_sink``. Selection is scoped by the
caller (currently ``glm5next`` under ``sm8x_sparse_mla_enabled()``), so no other
model's routing changes.

Ported from the LvLLM SM8x work; the kernel itself now lives next to its
siblings in ``vllm/v1/attention/backends/mla/sparse_mla_kernels.py``.
"""

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import MLAAttentionImpl
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    sparse_mla_fwd_with_sink,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


def sm8x_sparse_mla_enabled() -> bool:
    """True when this run must use the portable Triton sparse-MLA path.

    Anchored to the device capability family (Ampere/Ada = 8.x), which is the
    same predicate the in-tree DSv4 Triton fallback uses
    (``sparse_mla_env.is_triton_sparse_mla_enabled_for_platform``). Kept local so
    the glm5next binding is explicit and cannot accidentally flip another
    model's backend selection.
    """
    if not current_platform.is_cuda():
        return False
    return current_platform.is_device_capability_family(80)


class FlashMLASparseSM8XImpl(FlashMLASparseImpl):
    """bf16 sparse MLA on SM8x: reuse FlashMLASparse gather, Triton attention."""

    # No sm90+ LSE-returning decode kernel is used; keep the bf16 decode on the
    # plain (non-LSE) path so spec-decode / DCP gates in the base never trip.
    can_return_lse_for_decode: bool = False
    supports_dcp: bool = False
    # Force every token (prefill included) through the sparse-MQA gather path;
    # the dense/masked MHA prefill would fall back to FlashAttention (SM90+).
    supports_dense_mha_prefill: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sink_buf: torch.Tensor | None = None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: Any,
        layer: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The bf16 FlashMLASparse path fuses (q_nope, q_pe) via the CUDA
        # ``concat_mla_q`` kernel, which hard-requires ``rope_dim == 64``.
        # GLM-5.3-Flash is a NoPE member (``qk_rope_head_dim == 0``) so that
        # kernel asserts. Concat in torch instead (q_pe is empty for NoPE), then
        # reuse the inherited bf16 gather / logical->physical topk machinery.
        if isinstance(q, tuple):
            q = q[0] if self.qk_rope_head_dim == 0 else torch.cat(q, dim=-1)
        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        attn_out, _ = self._forward_bf16_kv(
            q, kv_c_and_k_pe_cache, topk_indices, attn_metadata, q.shape[1]
        )
        return attn_out, None

    def _convert_logical_to_physical_topk(
        self,
        logical_topk_indices: torch.Tensor,
        attn_metadata: Any,
        *,
        block_stride_rows: int | None,
        return_valid_counts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # The inherited implementation routes the decode case through
        # SparseMLAIndexGroup, whose side stream waits on ``logical_topk_ready``.
        # That event is recorded on the *capturing* stream by the MLA wrapper's
        # forward (mla.py record_logical_topk_ready), while this attention body
        # runs eagerly inside a breakable CUDA graph -- so the cross-stream wait
        # is invalid and CUDA raises cudaErrorInvalidValue on A100 during the
        # profile/capture run. On SM8x the conversion is a tiny index kernel and
        # decode batches are a handful of rows, so convert on the current stream
        # instead of sharing one result across the layer group. This is the same
        # kernel the prefill branch of _forward_bf16_kv already uses.
        return triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: logical_topk_indices.shape[0]],
            attn_metadata.block_table,
            logical_topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=logical_topk_indices.shape[1],
            return_valid_counts=return_valid_counts,
        )

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
        actual_num_heads: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``kv_c_and_k_pe_cache`` arrives as the flat [rows, head_dim] gather view
        # from ``_forward_bf16_kv``; the physical topk ids are rows into it and
        # ``topk_length`` is the valid prefix (``-1`` padding is masked by length).
        kv = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
        if actual_num_heads is None:
            actual_num_heads = q.shape[1]
        head_dim = kv.shape[-1]
        num_tokens = q.shape[0]
        output = q.new_empty((num_tokens, q.shape[1], head_dim), dtype=q.dtype)
        if topk_length is None:
            topk_length = torch.full(
                (num_tokens,),
                topk_indices.shape[-1],
                dtype=torch.int32,
                device=q.device,
            )
        # GLM-5.3 / V3.2 MLA carries no attention sink (``has_sink=False``, as on
        # the sm120 flashinfer path). The sink-aware kernel folds the sink in as a
        # pseudo-key with logit ``attn_sink`` and zero value, so "no sink" must be
        # a very negative logit -- a zero sink would wrongly add ``exp(0-max)`` to
        # the softmax denominator and dilute every output. Use -1e30: its weight
        # underflows to 0 for any real score, reducing the kernel to plain softmax;
        # a fully masked row (capture padding) yields 0/1=0 rather than NaN.
        if self._sink_buf is None or self._sink_buf.shape[0] < q.shape[1]:
            self._sink_buf = torch.full(
                (max(q.shape[1], 1),), -1e30, dtype=torch.float32, device=q.device
            )
        sparse_mla_fwd_with_sink(
            q=q,
            kv=kv,
            indices=topk_indices,
            topk_length=topk_length,
            scale=self.softmax_scale,
            attn_sink=self._sink_buf,
            output=output,
            num_heads=actual_num_heads,
        )
        return output, None


class FlashMLASparseSM8XBackend(FlashMLASparseBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE_SM8X"

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl[Any]]:
        return FlashMLASparseSM8XImpl

    @staticmethod
    def get_builder_cls() -> type[FlashMLASparseMetadataBuilder]:
        return FlashMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 512 = NoPE (GLM-5.3-Flash); 576 = V3.2 512-nope + 64-rope.
        return [512, 576]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Actual routing is gated by the caller via sm8x_sparse_mla_enabled().
        # Keep the class honest for any explicit validate pass.
        return capability.major == 8

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype not in (None, "auto", "bfloat16"):
            return "FLASHMLA_SPARSE_SM8X supports only a bf16 KV cache"
        if not use_sparse:
            return "FLASHMLA_SPARSE_SM8X is a sparse-MLA backend only"
        if not sm8x_sparse_mla_enabled():
            return "FLASHMLA_SPARSE_SM8X requires an SM8x device"
        return None
