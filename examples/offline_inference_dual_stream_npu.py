#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-ascend/examples/offline_inference_npu.py
# Copyright 2023 The vLLM team.
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
#
"""Offline inference example using the dual-stream NPU wrapper.

The dual-stream wrapper captures ACL graphs on two independent NPU streams
and replays them in parallel at runtime, splitting the input batch in half
across the two streams.

Usage::

    python examples/offline_inference_dual_stream_npu.py

Requirements:
    - ``enable_dual_stream_wrapper=True`` in ``LLM`` constructor.
    - Works independently of DBO (``enable_dbo``).
"""

# isort: skip_file
import os

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Artificial intelligence will",
        "Machine learning is",
    ]

    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)

    # Dual-stream wrapper is enabled via `enable_dual_stream_wrapper`.
    # It can be used with or without `enable_dbo`.
    llm = LLM(
        model="Qwen/Qwen3-8B",
        enable_dual_stream_wrapper=True,
    )

    print("Starting inference with DualStreamUBatchWrapper …")
    print(f"Number of prompts: {len(prompts)}")

    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "=" * 80)
    print("Generation Results:")
    print("=" * 80)
    for idx, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"\n[{idx}] Prompt: {prompt!r}")
        print(f"    Generated: {generated!r}")

    print("\n" + "=" * 80)
    print("Dual-stream inference completed successfully!")


if __name__ == "__main__":
    main()
