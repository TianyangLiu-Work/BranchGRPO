"""SGLang rollout backend for BranchGRPO.

Provides an SGLang Engine wrapper that replaces the manual PyTorch
autoregressive loop with RadixAttention-based KV-cache sharing across
shared-prefix generations (MH proposals, fork, etc.).
"""

import asyncio
import torch
from typing import List, Dict, Optional
from sglang import Engine


class SGLangRolloutBackend:
    """Thin wrapper around SGLang Engine for rollout generation."""

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        mem_fraction_static: float = 0.75,
    ):
        self.engine = Engine(
            model_path=model_path,
            dtype=dtype,
            mem_fraction_static=mem_fraction_static,
            trust_remote_code=True,
            disable_radix_cache=False,
            disable_cuda_graph=True,
            attention_backend="flashinfer",   # works in Docker 0.5.6+cu129
            sampling_backend="pytorch",
            context_length=4096,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_logprob: bool = True,
        logprob_start_len: int = None,
    ) -> dict:
        """Generate a single response.

        Returns a dict with 'output_ids', 'text', 'meta_info'.
        """
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }

        # Ensure event loop exists (uvloop policy may lose it after Engine init)
        try:
            _loop = asyncio.get_running_loop()
        except RuntimeError:
            _loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_loop)

        result = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
        )
        return result

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_logprob: bool = True,
        logprob_start_len: int = None,
    ) -> List[dict]:
        """Generate a batch of responses in a single call."""
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }

        # Ensure event loop exists
        try:
            _loop = asyncio.get_running_loop()
        except RuntimeError:
            _loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_loop)

        results = self.engine.generate(
            prompt=prompts,
            sampling_params=[sampling_params] * len(prompts),
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
        )
        if isinstance(results, list):
            return results
        return [results]

    def extract_token_ids(self, result: dict, tokenizer) -> torch.Tensor:
        """Extract token IDs tensor from SGLang result."""
        token_ids = result.get("output_ids")
        if token_ids is not None:
            return torch.tensor(token_ids)
        token_ids = tokenizer.encode(result["text"], add_special_tokens=False)
        return torch.tensor(token_ids)

    def extract_token_logprobs(self, result: dict, output_len: int) -> Optional[List[float]]:
        """Extract per-token logprobs from SGLang result.

        Handles both flat [float] and [(logprob, token_id)] formats.
        """
        meta = result.get("meta_info", {})
        raw = meta.get("output_token_logprobs")
        if raw is None:
            return None
        out = []
        for x in raw:
            if isinstance(x, (tuple, list)):
                out.append(float(x[0]))
            else:
                out.append(float(x))
        return out

    def get_sequence_logprob(self, result: dict, output_len: int) -> Optional[float]:
        """Extract sequence-level logprob (sum of per-token logprobs)."""
        logprobs = self.extract_token_logprobs(result, output_len)
        if logprobs is None:
            return None
        return sum(logprobs)

    def shutdown(self):
        self.engine.shutdown()
