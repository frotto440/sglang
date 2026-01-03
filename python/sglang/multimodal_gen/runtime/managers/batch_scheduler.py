# SPDX-License-Identifier: Apache-2.0
"""
Memory-aware batch scheduler for diffusion model serving.

This module provides batching functionality that:
- Groups requests by compatible configurations (resolution, frames, steps)
- Estimates memory requirements to determine safe batch sizes
- Prevents starvation with configurable max wait time
- Adapts to actual memory usage over time
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)


class RequestConfig(NamedTuple):
    """
    Hashable key for grouping compatible requests.
    
    Requests can only be batched together if they have identical RequestConfig.
    This ensures tensor shapes are compatible for batched execution.
    """
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    has_cfg: bool  # guidance_scale > 1.0 and negative_prompt exists
    effective_batch_size: int  # num_prompts × num_outputs_per_prompt

    @classmethod
    def from_req(cls, req: "Req") -> "RequestConfig":
        """Create RequestConfig from a Req object."""
        # Calculate effective batch size
        if isinstance(req.prompt, list):
            num_prompts = len(req.prompt)
        elif req.prompt is not None:
            num_prompts = 1
        else:
            num_prompts = 1
        
        effective_batch = num_prompts * req.num_outputs_per_prompt
        
        # Handle potential list types for dimensions
        height = req.height[0] if isinstance(req.height, list) else (req.height or 480)
        width = req.width[0] if isinstance(req.width, list) else (req.width or 848)
        num_frames = req.num_frames[0] if isinstance(req.num_frames, list) else req.num_frames
        
        return cls(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=req.num_inference_steps,
            has_cfg=req.do_classifier_free_guidance,
            effective_batch_size=effective_batch,
        )

    def __str__(self) -> str:
        return f"({self.height}×{self.width}×{self.num_frames}f, {self.num_inference_steps}steps, cfg={self.has_cfg}, batch={self.effective_batch_size})"


class MemoryEstimator:
    """
    Estimates memory requirements for batching based on historical data.
    Falls back to conservative heuristics when no history is available.
    """

    def __init__(self, device_id: int = 0, safety_margin: float = 0.85):
        """
        Args:
            device_id: GPU device ID
            safety_margin: Fraction of free memory to use (0.85 = 85%)
        """
        self.device_id = device_id
        self.safety_margin = safety_margin
        
        # Historical peak memory per config (rolling average)
        # Key: (height, width, num_frames, num_inference_steps, has_cfg) - without batch size
        self._memory_history: Dict[tuple, List[float]] = defaultdict(list)
        self._history_max_size = 10

    def _get_base_config_key(self, config: RequestConfig) -> tuple:
        """Get config key without batch size for memory history lookup."""
        return (config.height, config.width, config.num_frames, 
                config.num_inference_steps, config.has_cfg)

    def get_available_memory_mb(self) -> float:
        """Get currently available GPU memory in MB."""
        try:
            from sglang.multimodal_gen.runtime.platforms import current_platform
            free_gb = current_platform.get_available_gpu_memory(
                device_id=self.device_id, 
                empty_cache=True
            )
            return free_gb * 1024  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to get available memory: {e}, using default 40GB")
            return 40 * 1024  # Default 40GB

    def get_total_memory_mb(self) -> float:
        """Get total GPU memory in MB."""
        try:
            from sglang.multimodal_gen.runtime.platforms import current_platform
            total_bytes = current_platform.get_device_total_memory(device_id=self.device_id)
            return total_bytes / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get total memory: {e}, using default 80GB")
            return 80 * 1024  # Default 80GB

    def estimate_single_output_memory_mb(self, config: RequestConfig) -> float:
        """
        Estimate memory for a single output with given config.
        Uses historical data if available, otherwise uses heuristic.
        """
        base_key = self._get_base_config_key(config)
        
        # Check history first
        if base_key in self._memory_history and self._memory_history[base_key]:
            # Use average of recent observations (per-output memory)
            return sum(self._memory_history[base_key]) / len(self._memory_history[base_key])
        
        # Heuristic estimation based on resolution and frames
        # Memory ∝ height × width × num_frames in latent space
        # VAE has 8x spatial downsampling, ~4x temporal for video
        latent_h = config.height // 8
        latent_w = config.width // 8
        latent_f = max(1, config.num_frames // 4)  # temporal compression
        latent_c = 16  # typical latent channels for video models
        
        # Estimate activation memory (in MB)
        # Factor accounts for: latents, noise_pred, intermediate activations
        bytes_per_element = 2  # fp16/bf16
        activation_mb = (latent_h * latent_w * latent_f * latent_c * bytes_per_element * 6) / (1024 * 1024)
        
        # CFG doubles the effective compute (cond + uncond)
        if config.has_cfg:
            activation_mb *= 2
        
        # Scale with inference steps (more steps = slightly more memory for scheduler states)
        step_factor = min(1.3, 1.0 + config.num_inference_steps / 200)
        
        # Add safety buffer
        estimated = activation_mb * step_factor * 1.5  # 50% safety buffer
        
        # Minimum reasonable estimate based on model size expectations
        # Small image: ~2GB, large video: ~15GB+
        min_estimate = 2000 if config.num_frames <= 1 else 4000
        
        return max(estimated, min_estimate)

    def estimate_batch_memory_mb(self, config: RequestConfig, num_requests: int) -> float:
        """
        Estimate total memory for batching multiple requests.
        
        Args:
            config: The request configuration
            num_requests: Number of requests to batch together
            
        Returns:
            Estimated total memory in MB
        """
        single_output_mem = self.estimate_single_output_memory_mb(config)
        
        # Each request may have multiple outputs (num_outputs_per_prompt)
        outputs_per_request = config.effective_batch_size
        total_outputs = num_requests * outputs_per_request
        
        # Memory scales sub-linearly with batch size due to shared model weights
        # First output: full cost, additional outputs: ~75% incremental cost
        if total_outputs == 1:
            return single_output_mem
        
        # batch_mem = single + (total_outputs - 1) * single * 0.75
        return single_output_mem + (total_outputs - 1) * single_output_mem * 0.75

    def calculate_max_batch_size(self, config: RequestConfig, max_batch_limit: int = 8) -> int:
        """
        Calculate maximum safe number of requests to batch for given config.
        
        Args:
            config: Request configuration
            max_batch_limit: Hard cap on batch size
            
        Returns:
            Maximum number of requests that can be safely batched
        """
        available_mb = self.get_available_memory_mb() * self.safety_margin
        
        # Check if even a single request fits
        single_request_mem = self.estimate_batch_memory_mb(config, 1)
        if single_request_mem >= available_mb:
            logger.warning(
                f"Single request estimated at {single_request_mem:.0f}MB exceeds "
                f"available {available_mb:.0f}MB. Will attempt anyway."
            )
            return 1
        
        # Find max batch size that fits in memory
        max_batch = 1
        for batch_size in range(2, max_batch_limit + 1):
            estimated = self.estimate_batch_memory_mb(config, batch_size)
            if estimated > available_mb:
                break
            max_batch = batch_size
        
        return max_batch

    def record_actual_memory(
        self, 
        config: RequestConfig, 
        num_requests: int, 
        peak_memory_mb: float
    ) -> None:
        """
        Record actual memory usage to improve future estimates.
        
        Args:
            config: Request configuration
            num_requests: Number of requests in the batch
            peak_memory_mb: Actual peak memory usage in MB
        """
        if peak_memory_mb <= 0:
            return
            
        # Normalize to per-output memory
        total_outputs = num_requests * config.effective_batch_size
        per_output_mem = peak_memory_mb / max(1, total_outputs)
        
        base_key = self._get_base_config_key(config)
        history = self._memory_history[base_key]
        history.append(per_output_mem)
        
        # Keep only recent history
        if len(history) > self._history_max_size:
            self._memory_history[base_key] = history[-self._history_max_size:]
        
        logger.debug(
            f"Recorded memory for {config}: {per_output_mem:.0f}MB per output "
            f"(total {peak_memory_mb:.0f}MB for {total_outputs} outputs)"
        )


@dataclass
class QueuedRequest:
    """Wrapper for queued requests with metadata."""
    identity: bytes
    req: "Req"
    config: RequestConfig
    arrival_time: float = field(default_factory=time.monotonic)
    
    @property
    def wait_time(self) -> float:
        """Time in seconds since request arrived."""
        return time.monotonic() - self.arrival_time


class BatchScheduler:
    """
    Memory-aware batch scheduler that groups compatible requests.
    
    Features:
    - Groups requests by compatible configurations (same resolution, frames, steps)
    - Estimates memory to determine safe batch sizes
    - Prevents starvation with max wait time
    - Adapts to actual memory usage over time
    
    Requests with different `effective_batch_size` (num_outputs_per_prompt) are
    bucketed separately to simplify output distribution. This trades some batching
    efficiency for implementation simplicity.
    """

    def __init__(
        self,
        memory_estimator: MemoryEstimator,
        max_batch_size: int = 8,
        max_wait_time_s: float = 30.0,
    ):
        """
        Args:
            memory_estimator: Memory estimation component
            max_batch_size: Hard cap on number of requests per batch
            max_wait_time_s: Force dispatch after this wait time (starvation prevention)
        """
        self.memory_estimator = memory_estimator
        self.max_batch_size = max_batch_size
        self.max_wait_time_s = max_wait_time_s
        
        # Buckets: config -> deque of QueuedRequest
        self._buckets: Dict[RequestConfig, deque] = defaultdict(deque)
        
        # Track total queue size for monitoring
        self._total_queued = 0
        
        # Track current batch for result distribution
        self._current_batch_identities: List[bytes] = []
        self._current_batch_config: Optional[RequestConfig] = None
        self._current_batch_size: int = 0

    def add_request(self, identity: bytes, req: "Req") -> None:
        """Add a new request to the appropriate bucket."""
        config = RequestConfig.from_req(req)
        queued = QueuedRequest(identity=identity, req=req, config=config)
        self._buckets[config].append(queued)
        self._total_queued += 1
        
        logger.debug(f"Queued request {req.request_id} into bucket {config}")

    def add_requests(self, requests: List[Tuple[bytes, "Req"]]) -> None:
        """Add multiple requests."""
        for identity, req in requests:
            self.add_request(identity, req)

    @property
    def queue_size(self) -> int:
        """Total number of queued requests."""
        return self._total_queued

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._total_queued == 0

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics for monitoring."""
        stats = {
            "total_queued": self._total_queued,
            "num_buckets": len([b for b in self._buckets.values() if b]),
            "buckets": {}
        }
        for config, queue in self._buckets.items():
            if queue:
                stats["buckets"][str(config)] = {
                    "count": len(queue),
                    "oldest_wait_s": queue[0].wait_time if queue else 0,
                }
        return stats

    def _get_oldest_bucket(self) -> Optional[RequestConfig]:
        """Find bucket with the oldest waiting request."""
        oldest_config = None
        oldest_time = float('inf')
        
        for config, queue in self._buckets.items():
            if queue and queue[0].arrival_time < oldest_time:
                oldest_time = queue[0].arrival_time
                oldest_config = config
        
        return oldest_config

    def _select_best_bucket(self) -> Optional[RequestConfig]:
        """
        Select the best bucket to process next.
        
        Strategy:
        1. If any request has waited > max_wait_time_s, prioritize that bucket (starvation prevention)
        2. Otherwise, prefer buckets with larger potential batch sizes
        3. Tie-break by oldest request
        """
        best_config = None
        best_score = -1
        
        for config, queue in self._buckets.items():
            if not queue:
                continue
            
            oldest_wait = queue[0].wait_time
            
            # Starvation check - force this bucket if waited too long
            if oldest_wait > self.max_wait_time_s:
                logger.info(f"Starvation prevention: forcing bucket {config} after {oldest_wait:.1f}s wait")
                return config
            
            # Calculate batch potential
            max_safe = self.memory_estimator.calculate_max_batch_size(config, self.max_batch_size)
            potential_batch = min(max_safe, len(queue))
            
            # Score = potential batch size with bonus for older requests
            age_bonus = min(oldest_wait / self.max_wait_time_s, 1.0)
            score = potential_batch * (1 + age_bonus * 0.5)
            
            if score > best_score:
                best_score = score
                best_config = config
        
        return best_config

    def _select_batch_from_bucket(self, config: RequestConfig) -> List[QueuedRequest]:
        """Select requests from a bucket up to memory limit."""
        queue = self._buckets[config]
        if not queue:
            return []
        
        # Calculate memory-safe batch size
        max_safe = self.memory_estimator.calculate_max_batch_size(config, self.max_batch_size)
        max_size = min(max_safe, len(queue))
        
        # Collect batch
        batch = []
        for _ in range(max_size):
            if queue:
                batch.append(queue.popleft())
                self._total_queued -= 1
        
        return batch

    def _merge_requests(self, queued_items: List[QueuedRequest]) -> "Req":
        """
        Merge multiple Req objects into a single batched Req.
        All requests must have compatible configurations.
        
        For atomic batching (same effective_batch_size), we concatenate prompts
        but keep num_outputs_per_prompt the same.
        """
        if len(queued_items) == 1:
            return queued_items[0].req
        
        from copy import copy
        
        base_req = queued_items[0].req
        
        # Collect all prompts
        all_prompts = []
        all_negative_prompts = []
        all_seeds = []
        
        for item in queued_items:
            req = item.req
            
            # Handle prompt (str or list)
            if isinstance(req.prompt, list):
                all_prompts.extend(req.prompt)
            elif req.prompt is not None:
                all_prompts.append(req.prompt)
            
            # Handle negative prompt
            if req.negative_prompt:
                if isinstance(req.negative_prompt, list):
                    all_negative_prompts.extend(req.negative_prompt)
                else:
                    all_negative_prompts.append(req.negative_prompt)
            
            # Handle seeds
            if req.seeds:
                all_seeds.extend(req.seeds)
            elif req.seed is not None:
                # Expand seed for num_outputs_per_prompt
                for i in range(req.num_outputs_per_prompt):
                    all_seeds.append(req.seed + i)
        
        # Create merged request
        merged = copy(base_req)
        merged.prompt = all_prompts
        merged.negative_prompt = all_negative_prompts if all_negative_prompts else None
        merged.seeds = all_seeds if all_seeds else None
        merged.seed = None  # Use seeds list instead
        merged.request_id = f"batch_{len(queued_items)}_{base_req.request_id}"
        
        logger.info(
            f"Merged {len(queued_items)} requests into batch with "
            f"{len(all_prompts)} prompts × {base_req.num_outputs_per_prompt} outputs = "
            f"{len(all_prompts) * base_req.num_outputs_per_prompt} total outputs"
        )
        
        return merged

    def get_next_batch(self) -> Optional[Tuple[List[bytes], "Req", RequestConfig]]:
        """
        Get the next batch to process.
        
        Returns:
            Tuple of (identities, merged_req, config) or None if queue is empty
        """
        if self.is_empty():
            return None
        
        # Select best bucket
        best_config = self._select_best_bucket()
        if best_config is None:
            return None
        
        # Form batch from selected bucket
        batch = self._select_batch_from_bucket(best_config)
        if not batch:
            return None
        
        # Extract identities and merge requests
        identities = [item.identity for item in batch]
        merged_req = self._merge_requests(batch)
        
        # Store for result distribution
        self._current_batch_identities = identities
        self._current_batch_config = best_config
        self._current_batch_size = len(batch)
        
        logger.info(
            f"Formed batch of {len(batch)} requests from bucket {best_config}, "
            f"queue remaining: {self._total_queued}"
        )
        
        return identities, merged_req, best_config

    def record_execution(self, peak_memory_mb: float) -> None:
        """
        Record actual execution metrics for future estimation.
        Call this after batch execution completes.
        """
        if self._current_batch_config and peak_memory_mb > 0:
            self.memory_estimator.record_actual_memory(
                self._current_batch_config,
                self._current_batch_size,
                peak_memory_mb
            )

    def get_current_batch_info(self) -> Tuple[List[bytes], Optional[RequestConfig], int]:
        """
        Get information about the currently executing batch.
        
        Returns:
            Tuple of (identities, config, batch_size)
        """
        return (
            self._current_batch_identities,
            self._current_batch_config,
            self._current_batch_size
        )

    def clear_current_batch(self) -> None:
        """Clear current batch tracking after results are sent."""
        self._current_batch_identities = []
        self._current_batch_config = None
        self._current_batch_size = 0

