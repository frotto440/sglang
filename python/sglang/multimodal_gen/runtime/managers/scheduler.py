# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import pickle
from collections import deque
from copy import deepcopy
from typing import Any, List, Optional

import zmq

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    MergeLoraWeightsReq,
    SetLoraReq,
    UnmergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.managers.batch_scheduler import (
    BatchScheduler,
    MemoryEstimator,
    RequestConfig,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import GREEN, RESET, init_logger

logger = init_logger(__name__)


class Scheduler:
    """
    Runs the main event loop for the rank 0 worker.
    It listens for external requests via ZMQ and coordinates with other workers.
    This class does NOT manage worker processes.
    
    Supports memory-aware batching when enabled via server_args.enable_batching.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        port_args: PortArgs,
        task_pipes_to_slaves: list = None,
        result_pipes_from_slaves: list = None,
    ):
        self.server_args = server_args
        self.port_args = port_args

        set_global_server_args(server_args=server_args)

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)
        endpoint = server_args.scheduler_endpoint
        if gpu_id == 0:
            # router allocates identify (envelope) for each connection
            self.receiver, actual_endpoint = get_zmq_socket(
                self.context, zmq.ROUTER, endpoint, True
            )
            logger.info(f"Scheduler bind at endpoint: {actual_endpoint}")
        else:
            self.receiver = None

        worker = GPUWorker(
            local_rank=gpu_id,
            master_port=port_args.master_port,
            rank=gpu_id,
            server_args=server_args,
        )
        self.worker = worker
        self.task_pipes_to_slaves = task_pipes_to_slaves
        self.result_pipes_from_slaves = result_pipes_from_slaves
        self.gpu_id = gpu_id
        self._running = True

        self.request_handlers = {
            SetLoraReq: self._handle_set_lora,
            MergeLoraWeightsReq: self._handle_merge_lora,
            UnmergeLoraWeightsReq: self._handle_unmerge_lora,
            Req: self._handle_generation,
            List[Req]: self._handle_generation,
        }

        # Initialize batching
        self.enable_batching = server_args.enable_batching
        if self.enable_batching:
            self.memory_estimator = MemoryEstimator(
                device_id=gpu_id,
                safety_margin=server_args.memory_safety_margin,
            )
            self.batch_scheduler = BatchScheduler(
                memory_estimator=self.memory_estimator,
                max_batch_size=server_args.max_batch_size,
                max_wait_time_s=server_args.batch_max_wait_time_s,
            )
            logger.info(
                f"Batching enabled: max_batch_size={server_args.max_batch_size}, "
                f"max_wait_time={server_args.batch_max_wait_time_s}s, "
                f"memory_safety_margin={server_args.memory_safety_margin}"
            )
        else:
            self.batch_scheduler = None
            self.memory_estimator = None
            logger.info("Batching disabled, processing requests sequentially")

        # Legacy queue for non-batching mode
        self.waiting_queue: deque[tuple[bytes, Req]] = deque()

        self.warmed_up = False

    def _handle_set_lora(self, reqs: List[Any]) -> OutputBatch:
        # TODO: return set status
        # TODO: return with SetLoRAResponse or something more appropriate
        req = reqs[0]
        return self.worker.set_lora(
            req.lora_nickname, req.lora_path, req.target, req.strength
        )

    def _handle_merge_lora(self, reqs: List[Any]):
        req = reqs[0]
        return self.worker.merge_lora_weights(req.target, req.strength)

    def _handle_unmerge_lora(self, reqs: List[Any]) -> OutputBatch:
        req = reqs[0]
        return self.worker.unmerge_lora_weights(req.target)

    def _handle_generation(self, reqs: List[Req]) -> OutputBatch:
        """
        Handle generation request.
        
        When batching is enabled, reqs contains a single merged Req with
        all prompts from the batch combined.
        """
        # reqs always contains exactly one (potentially merged) Req
        return self.worker.execute_forward(reqs[0])

    def _split_batch_output(
        self,
        output_batch: OutputBatch,
        num_requests: int,
        config: Optional[RequestConfig],
    ) -> List[OutputBatch]:
        """
        Split a batched OutputBatch into individual results for each client.
        
        Args:
            output_batch: The batched output from the worker
            num_requests: Number of requests in the batch
            config: The request configuration
            
        Returns:
            List of OutputBatch, one per client
        """
        if num_requests == 1 or output_batch.output is None:
            return [output_batch]
        
        results = []
        outputs_per_request = config.effective_batch_size if config else 1
        
        for i in range(num_requests):
            start_idx = i * outputs_per_request
            end_idx = start_idx + outputs_per_request
            
            # Slice the output tensor for this client
            client_output = output_batch.output[start_idx:end_idx]
            
            single = OutputBatch(
                output=client_output,
                timings=output_batch.timings,  # Shared timing info
                peak_memory_mb=output_batch.peak_memory_mb / num_requests,
                error=output_batch.error,
            )
            results.append(single)
        
        return results

    def return_result(
        self,
        output_batch: OutputBatch,
        identity: bytes | None = None,
        is_warmup: bool = False,
    ):
        """
        Send results back to client(s).
        When batching is enabled, distributes results to all batch members.
        """
        if is_warmup or self.receiver is None:
            return
        
        if self.enable_batching and self.batch_scheduler is not None:
            identities, config, batch_size = self.batch_scheduler.get_current_batch_info()
            
            if not identities:
                # Fallback to single identity
                if identity is not None:
                    self.receiver.send_multipart([identity, b"", pickle.dumps(output_batch)])
                return
            
            # Record memory usage for future estimation
            if output_batch.peak_memory_mb > 0:
                self.batch_scheduler.record_execution(output_batch.peak_memory_mb)
            
            # Split and distribute results
            if batch_size > 1 and output_batch.output is not None:
                split_outputs = self._split_batch_output(output_batch, batch_size, config)
                for ident, single_output in zip(identities, split_outputs):
                    self.receiver.send_multipart([ident, b"", pickle.dumps(single_output)])
                logger.debug(f"Distributed results to {batch_size} clients")
            else:
                # Single request or error - send same result to all
                for ident in identities:
                    self.receiver.send_multipart([ident, b"", pickle.dumps(output_batch)])
            
            # Clear batch tracking
            self.batch_scheduler.clear_current_batch()
        else:
            # Non-batching mode
            if identity is not None:
                self.receiver.send_multipart([identity, b"", pickle.dumps(output_batch)])

    def get_next_batch_to_run(self) -> list[tuple[bytes, Req]] | None:
        """
        Get the next batch of requests to run.
        
        When batching is enabled, uses BatchScheduler to form optimal batches.
        Otherwise, returns single requests from the waiting queue.
        """
        if self.enable_batching and self.batch_scheduler is not None:
            result = self.batch_scheduler.get_next_batch()
            if result is None:
                return None
            
            identities, merged_req, config = result
            # Return in expected format - the merged req with first identity
            return [(identities[0], merged_req)]
        else:
            # Legacy non-batching mode
            if not self.waiting_queue:
                return None
            item = self.waiting_queue.popleft()
            return [item]

    def recv_reqs(self) -> List[tuple[bytes, Any]]:
        """
        Receive requests from clients.
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj.
        """
        if self.receiver is not None:
            try:
                try:
                    identity, _, payload = self.receiver.recv_multipart(zmq.NOBLOCK)
                    recv_reqs = pickle.loads(payload)
                except zmq.Again:
                    recv_reqs = []
            except zmq.ZMQError:
                # re-raise or handle appropriately to let the outer loop continue
                raise

            if recv_reqs:
                # Ensure recv_reqs is a list
                if not isinstance(recv_reqs, list):
                    recv_reqs = [recv_reqs]

                # Pack with identity for rank 0
                recv_reqs = [(identity, req) for req in recv_reqs]
        else:
            recv_reqs = None

        # Broadcast to parallel workers
        if self.server_args.sp_degree != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.sp_group.rank,
                self.worker.sp_cpu_group,
                src=self.worker.sp_group.ranks[0],
            )

        if self.server_args.enable_cfg_parallel:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.cfg_group.rank,
                self.worker.cfg_cpu_group,
                src=self.worker.cfg_group.ranks[0],
            )

        if self.server_args.tp_size > 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.tp_group.rank,
                self.worker.tp_cpu_group,
                src=self.worker.tp_group.ranks[0],
            )

        assert recv_reqs is not None

        # Handle server warmup by inserting an identical req to the beginning
        # Only the very first req through server's lifetime will be warmup
        if (
            not self.warmed_up
            and len(recv_reqs) == 1
            and self.server_args.enable_warmup
        ):
            identity, req = recv_reqs[0]
            if isinstance(req, Req):
                warmup_req = deepcopy(req)
                warmup_req.is_warmup = True
                warmup_req.num_inference_steps = 1
                recv_reqs.insert(0, (identity, warmup_req))
                self.warmed_up = True
                logger.info("Server warming up....")

        return recv_reqs

    def _add_requests_to_queue(self, requests: List[tuple[bytes, Any]]) -> None:
        """
        Add received requests to the appropriate queue.
        Separates generation requests from control requests.
        """
        for identity, req in requests:
            if isinstance(req, Req):
                if self.enable_batching and self.batch_scheduler is not None:
                    self.batch_scheduler.add_request(identity, req)
                else:
                    self.waiting_queue.append((identity, req))
            else:
                # Control requests (LoRA, etc.) go to legacy queue
                self.waiting_queue.append((identity, req))

    def event_loop(self) -> None:
        """
        The main event loop that listens for ZMQ requests.
        Handles request batching and execution.
        """
        logger.debug(
            f"Rank 0 scheduler listening on tcp://*:{self.server_args.scheduler_port}"
        )

        while self._running:
            # 1: Receive and queue requests
            try:
                new_reqs = self.recv_reqs()
                self._add_requests_to_queue(new_reqs)
            except Exception as e:
                logger.error(
                    f"Error receiving requests in scheduler event loop: {e}",
                    exc_info=True,
                )
                continue

            # 2: Get next batch and execute
            items = self.get_next_batch_to_run()
            if not items:
                continue

            identities = [item[0] for item in items]
            reqs = [item[1] for item in items]

            try:
                processed_req = reqs[0]
                handler = self.request_handlers.get(type(processed_req))
                if handler:
                    output_batch = handler(reqs)
                else:
                    output_batch = OutputBatch(
                        error=f"Unknown request type: {type(processed_req)}"
                    )
            except Exception as e:
                logger.error(
                    f"Error executing request in scheduler event loop: {e}",
                    exc_info=True,
                )
                output_batch = (
                    OutputBatch(error=str(e))
                    if reqs and isinstance(reqs[0], Req)
                    else OutputBatch(error=str(e))
                )

            # 3: Return results to client(s)
            try:
                is_warmup = (
                    processed_req.is_warmup if isinstance(processed_req, Req) else False
                )
                if is_warmup:
                    logger.info(
                        f"Server warmup done in {GREEN}%.2f{RESET} seconds",
                        output_batch.timings.total_duration_s,
                    )

                self.return_result(output_batch, identities[0], is_warmup=is_warmup)
            except zmq.ZMQError as e:
                # Reply failed; log and keep loop alive to accept future requests
                logger.error(f"ZMQ error sending reply: {e}")
                continue

        logger.info("Scheduler event loop terminated.")
        if self.receiver is not None:
            self.receiver.close()
        self.context.term()

    def _broadcast_task(self, payload: dict[str, Any]) -> None:
        """Broadcast a task to all slave worker processes."""
        method = payload["method"]
        kwargs = {k: v for k, v in payload.items() if k != "method"}
        task = {"method": method, "kwargs": kwargs}
        for pipe in self.task_pipes_to_slaves:
            pipe.send(task)

    def _collect_slave_results(self) -> List[dict[str, Any]]:
        """Collect results from all slave worker processes."""
        results = []
        for pipe in self.result_pipes_from_slaves:
            results.append(pipe.recv())
        return results
