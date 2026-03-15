# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dual-stream batch wrapper for NPU inference.

This module implements a dual-stream execution model that captures and replays
ACL graphs on two independent NPU streams with separate graph pools. During
the capture phase the full batch is used to record identical graphs on both
streams. At runtime the input is split in half and the two halves are executed
concurrently on the two streams, then the outputs are concatenated.

The wrapper is activated by setting ``parallel_config.enable_dual_stream_wrapper``
to ``True``.  It is independent of DBO (dual-batch overlap) and can be combined
with or used instead of it.
"""

import threading
from typing import Any, Callable

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_pp_group
from vllm.distributed.device_communicators.pynccl_allocator import \
    set_graph_pool_id
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

logger = init_logger(__name__)


class _GraphRecord:
    """Stores everything needed to replay a captured dual-stream graph."""

    __slots__ = (
        "graph_a",
        "graph_b",
        "pool_a",
        "pool_b",
        "stream_a",
        "stream_b",
        "output_a",
        "output_b",
        "buf_input_ids_a",
        "buf_positions_a",
        "buf_embeds_a",
        "buf_intermediates_a",
        "buf_input_ids_b",
        "buf_positions_b",
        "buf_embeds_b",
        "buf_intermediates_b",
    )

    def __init__(self) -> None:
        self.graph_a: torch.npu.NPUGraph | None = None
        self.graph_b: torch.npu.NPUGraph | None = None
        self.pool_a: Any = None
        self.pool_b: Any = None
        self.stream_a: torch.npu.Stream | None = None
        self.stream_b: torch.npu.Stream | None = None
        self.output_a: Any = None
        self.output_b: Any = None
        # Capture-time input buffers for stream A
        self.buf_input_ids_a: torch.Tensor | None = None
        self.buf_positions_a: torch.Tensor | None = None
        self.buf_embeds_a: torch.Tensor | None = None
        self.buf_intermediates_a: IntermediateTensors | None = None
        # Capture-time input buffers for stream B
        self.buf_input_ids_b: torch.Tensor | None = None
        self.buf_positions_b: torch.Tensor | None = None
        self.buf_embeds_b: torch.Tensor | None = None
        self.buf_intermediates_b: IntermediateTensors | None = None


class DualStreamUBatchWrapper(UBatchWrapper):
    """Wrap a model callable to execute on two NPU streams in parallel.

    Capture phase
    -------------
    For each unique ``BatchDescriptor`` the wrapper records an ACL graph on
    **two** independent streams (each with its own graph pool).  Both graphs
    see the full (un-split) batch so they capture identical kernels.

    Replay phase
    ------------
    The input tensor is split at the midpoint.  The first half is copied into
    stream-A's capture buffer and replayed; the second half goes to stream-B.
    A lightweight thread is used for one of the streams so they overlap on the
    CPU timeline, while ``torch.npu.synchronize()`` ensures both finish before
    the outputs are concatenated.
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.device,
    ) -> None:
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.runtime_mode = runtime_mode
        self.device = device

        # Two independent graph pools
        self.pool_a = torch.npu.graph_pool_handle()
        self.pool_b = torch.npu.graph_pool_handle()

        # Cache: BatchDescriptor -> _GraphRecord
        self._records: dict[BatchDescriptor, _GraphRecord] = {}

        # Fall-back wrapper for piecewise mode
        self.cudagraph_wrapper: ACLGraphWrapper | None = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = ACLGraphWrapper(
                runnable, vllm_config, runtime_mode=runtime_mode)

    # ------------------------------------------------------------------
    # Attribute forwarding
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        if hasattr(self.runnable, name):
            return getattr(self.runnable, name)
        raise AttributeError(
            f"'{type(self).__name__}' wrapper has no attribute '{name}' "
            f"and neither does the wrapped runnable ({type(self.runnable).__name__})"
        )

    def unwrap(self) -> Callable:
        return self.runnable

    # ------------------------------------------------------------------
    # Input splitting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_tensor(tensor: torch.Tensor | None, mid: int,
                      is_2d: bool = False):
        """Return (first_half, second_half) of *tensor* along the token dim."""
        if tensor is None:
            return None, None
        if is_2d and tensor.ndim == 2:
            return tensor[:, :mid], tensor[:, mid:]
        return tensor[:mid], tensor[mid:]

    @staticmethod
    def _split_intermediates(intermediates: IntermediateTensors | None,
                             mid: int):
        if intermediates is None:
            return None, None
        tensors_a, tensors_b = {}, {}
        for key, val in intermediates.tensors.items():
            tensors_a[key] = val[:mid]
            tensors_b[key] = val[mid:]
        return IntermediateTensors(tensors_a), IntermediateTensors(tensors_b)

    # ------------------------------------------------------------------
    # Copy helpers (for replay – writes into capture buffers)
    # ------------------------------------------------------------------
    @staticmethod
    def _copy_into_buffer(buf: torch.Tensor | None, src: torch.Tensor | None,
                          n: int):
        if buf is not None and src is not None:
            if buf.ndim == 2 and src.ndim == 2:
                buf[:, :n].copy_(src[:, :n])
            else:
                buf[:n].copy_(src[:n])

    @staticmethod
    def _copy_intermediates_into_buffer(
        buf: IntermediateTensors | None,
        src: IntermediateTensors | None,
        n: int,
    ):
        if buf is not None and src is not None:
            for key in src.tensors:
                if key in buf.tensors:
                    buf.tensors[key][:n].copy_(src.tensors[key][:n])

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def _capture_graphs(self, input_ids, positions, inputs_embeds,
                        intermediate_tensors) -> torch.Tensor:
        """Capture the same model graph on two streams sequentially."""
        fwd_ctx = get_forward_context()
        bd = fwd_ctx.batch_descriptor
        assert bd is not None

        rec = _GraphRecord()
        rec.pool_a = self.pool_a
        rec.pool_b = self.pool_b
        rec.stream_a = torch.npu.Stream(device=self.device)
        rec.stream_b = torch.npu.Stream(device=self.device)

        # Capture helper – records a graph on *stream* with *pool*.
        def _do_capture(stream: torch.npu.Stream, pool):
            set_graph_pool_id(pool)
            graph = torch.npu.NPUGraph()
            with torch.npu.stream(stream):
                # Warm-up to init BLAS handle in this stream context
                _ = torch.npu.current_blas_handle()
            with torch.npu.graph(graph, stream=stream, pool=pool):
                out = self.runnable(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )
            return graph, out

        # Capture on stream A
        rec.graph_a, rec.output_a = _do_capture(rec.stream_a, rec.pool_a)
        # Snapshot input buffers for stream A
        rec.buf_input_ids_a = input_ids.clone() if input_ids is not None else None
        rec.buf_positions_a = positions.clone() if positions is not None else None
        rec.buf_embeds_a = inputs_embeds.clone() if inputs_embeds is not None else None
        if intermediate_tensors is not None:
            rec.buf_intermediates_a = IntermediateTensors(
                {k: v.clone() for k, v in intermediate_tensors.tensors.items()})

        # Capture on stream B
        rec.graph_b, rec.output_b = _do_capture(rec.stream_b, rec.pool_b)
        rec.buf_input_ids_b = input_ids.clone() if input_ids is not None else None
        rec.buf_positions_b = positions.clone() if positions is not None else None
        rec.buf_embeds_b = inputs_embeds.clone() if inputs_embeds is not None else None
        if intermediate_tensors is not None:
            rec.buf_intermediates_b = IntermediateTensors(
                {k: v.clone() for k, v in intermediate_tensors.tensors.items()})

        self._records[bd] = rec

        # During capture we just return the output from stream A (full batch)
        return rec.output_a

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def _replay_graphs(self, input_ids, positions, inputs_embeds,
                       intermediate_tensors,
                       num_tokens: int) -> torch.Tensor:
        """Split inputs and replay graphs on two streams concurrently."""
        fwd_ctx = get_forward_context()
        bd = fwd_ctx.batch_descriptor
        rec = self._records[bd]

        mid = num_tokens // 2
        tail = num_tokens - mid

        # Split inputs
        ids_a, ids_b = self._split_tensor(input_ids, mid)
        pos_a, pos_b = self._split_tensor(positions, mid,
                                           is_2d=(positions is not None
                                                  and positions.ndim == 2))
        emb_a, emb_b = self._split_tensor(inputs_embeds, mid)
        int_a, int_b = self._split_intermediates(intermediate_tensors, mid)

        # Copy into capture buffers
        self._copy_into_buffer(rec.buf_input_ids_a, ids_a, mid)
        self._copy_into_buffer(rec.buf_positions_a, pos_a, mid)
        self._copy_into_buffer(rec.buf_embeds_a, emb_a, mid)
        self._copy_intermediates_into_buffer(rec.buf_intermediates_a, int_a,
                                             mid)

        self._copy_into_buffer(rec.buf_input_ids_b, ids_b, tail)
        self._copy_into_buffer(rec.buf_positions_b, pos_b, tail)
        self._copy_into_buffer(rec.buf_embeds_b, emb_b, tail)
        self._copy_intermediates_into_buffer(rec.buf_intermediates_b, int_b,
                                             tail)

        # Replay on two streams concurrently using a helper thread
        errors: list[Exception | None] = [None, None]

        def _replay_on_stream_b():
            try:
                torch.npu.set_device(self.device)
                set_graph_pool_id(rec.pool_b)
                with torch.npu.stream(rec.stream_b):
                    rec.graph_b.replay()
            except Exception as exc:  # noqa: BLE001
                errors[1] = exc

        thread_b = threading.Thread(target=_replay_on_stream_b)
        thread_b.start()

        # Stream A runs on the current thread
        set_graph_pool_id(rec.pool_a)
        with torch.npu.stream(rec.stream_a):
            rec.graph_a.replay()

        thread_b.join()

        # Propagate errors
        for idx, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(
                    f"Dual-stream replay failed on stream "
                    f"{'B' if idx else 'A'}") from err

        # Wait for both streams to finish on the device
        torch.npu.synchronize()

        # Gather outputs
        out_a = rec.output_a[:mid] if rec.output_a is not None else None
        out_b = rec.output_b[:tail] if rec.output_b is not None else None

        if out_a is not None and out_b is not None:
            if not get_pp_group().is_last_rank:
                return self._merge_intermediate_outputs(out_a, out_b)
            return torch.cat([out_a, out_b], dim=0)
        # Fallback – should not normally happen
        return out_a if out_a is not None else out_b

    # ------------------------------------------------------------------
    # Intermediate tensor merging (pipeline-parallel)
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_intermediate_outputs(a, b):
        """Merge IntermediateTensors from two streams for PP scenarios."""
        if isinstance(a, IntermediateTensors) and isinstance(
                b, IntermediateTensors):
            merged = {}
            for key in a.tensors:
                merged[key] = torch.cat([a.tensors[key], b.tensors[key]],
                                        dim=0)
            return IntermediateTensors(merged)
        # Plain tensors
        return torch.cat([a, b], dim=0)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        fwd_ctx = get_forward_context()
        bd = fwd_ctx.batch_descriptor
        runtime_mode = fwd_ctx.cudagraph_runtime_mode

        if runtime_mode == CUDAGraphMode.NONE:
            # No graph – run eagerly or via piecewise wrapper
            if self.cudagraph_wrapper is not None:
                return self.cudagraph_wrapper(*args, **kwargs)
            return self.runnable(*args, **kwargs)

        # ACL graph path
        assert bd is not None
        input_ids = kwargs.get("input_ids")
        positions = kwargs.get("positions")
        inputs_embeds = kwargs.get("inputs_embeds")
        intermediate_tensors = kwargs.get("intermediate_tensors")

        if bd not in self._records or self._records[bd].graph_a is None:
            # First time for this shape – capture
            return self._capture_graphs(input_ids, positions, inputs_embeds,
                                        intermediate_tensors)

        # Replay
        num_tokens = (input_ids.shape[0]
                      if input_ids is not None else positions.shape[-1])
        return self._replay_graphs(input_ids, positions, inputs_embeds,
                                   intermediate_tensors, num_tokens)
