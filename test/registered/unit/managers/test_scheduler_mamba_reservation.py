"""CPU regressions for Mamba prefix-COW group reservations."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _scheduler_with_mamba_slots(req_slots: int, mamba_slots: int) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.req_to_token_pool = SimpleNamespace(
        available_size=lambda: req_slots,
        mamba_allocator=MambaSlotAllocator(mamba_slots, device="cpu"),
    )
    return scheduler


class TestMambaSlotGroupReservation(unittest.TestCase):
    def setUp(self):
        args = ServerArgs(model_path="dummy")
        args.pp_max_micro_batch_size = 8
        set_global_server_args_for_scheduler(args)

    def test_scheduler_counts_the_last_reserved_slot(self):
        scheduler = _scheduler_with_mamba_slots(req_slots=8, mamba_slots=1)
        allocator = scheduler.req_to_token_pool.mamba_allocator

        allocator.alloc_group_begin(1)
        self.assertEqual(allocator.available_size(), 0)
        self.assertEqual(allocator.reserved_size(), 1)
        self.assertEqual(scheduler.get_num_allocatable_reqs(0), 1)

        allocator.alloc(1)
        self.assertEqual(allocator.reserved_size(), 0)
        self.assertEqual(scheduler.get_num_allocatable_reqs(0), 0)
        allocator.alloc_group_end()

    def test_prefill_loop_reaches_a_request_with_only_a_reserved_slot(self):
        class _StopAfterAdmission(Exception):
            pass

        class _Req:
            def __init__(self, allocator):
                self.allocator = allocator
                self.init_calls = 0

            def init_next_round_input(self, tree_cache):
                self.init_calls += 1
                self.allocator.alloc(1)

        class _Adder:
            def __init__(self, *args, **kwargs):
                self.can_run_list = []

            def add_one_req(self, req, **kwargs):
                self.assertion_target.assertEqual(req.init_calls, 1)
                self.assertion_target.assertEqual(req.allocator.reserved_size(), 0)
                raise _StopAfterAdmission

        scheduler = _scheduler_with_mamba_slots(req_slots=1, mamba_slots=1)
        allocator = scheduler.req_to_token_pool.mamba_allocator
        req = _Req(allocator)
        _Adder.assertion_target = self
        scheduler.grammar_manager = SimpleNamespace(
            has_waiting_grammars=lambda: False
        )
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_priority_preemption = False
        scheduler.is_hybrid_swa = False
        scheduler.running_batch = SimpleNamespace(
            reqs=[], batch_is_full=False, is_empty=lambda: True
        )
        scheduler.waiting_queue = [req]
        scheduler.chunked_req = None
        scheduler.policy = SimpleNamespace(
            calc_priority=lambda waiting_queue, running_batch: None
        )
        scheduler.enable_dynamic_chunking = False
        scheduler.page_size = 1
        scheduler.tree_cache = SimpleNamespace()
        scheduler.token_to_kv_pool_allocator = SimpleNamespace()
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.max_prefill_tokens = 1
        scheduler.chunked_prefill_size = None
        scheduler.is_mixed_chunk = False
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.max_prefill_bs = 1
        scheduler.max_running_requests = 1
        scheduler.server_args = SimpleNamespace(prefill_max_requests=None)
        scheduler.dllm_config = None
        scheduler.enable_lora = False
        scheduler.disaggregation_mode = None
        scheduler.enable_hicache_storage = False
        scheduler.truncation_align_size = None

        with patch("sglang.srt.managers.scheduler.PrefillAdder", _Adder):
            with self.assertRaises(_StopAfterAdmission):
                scheduler._get_new_batch_prefill_raw(prefill_delayer_single_pass=None)

        self.assertEqual(req.init_calls, 1)
        allocator.alloc_group_end()

    def test_unused_reservations_return_at_group_end(self):
        allocator = MambaSlotAllocator(2, device="cpu")
        allocator.alloc_group_begin(2)
        first = allocator.alloc(1)

        self.assertEqual(first.numel(), 1)
        self.assertEqual(allocator.reserved_size(), 1)
        allocator.alloc_group_end()
        self.assertEqual(allocator.reserved_size(), 0)
        self.assertEqual(allocator.available_size(), 1)

    def test_bulk_allocation_does_not_consume_one_slot_reservation_credit(self):
        allocator = MambaSlotAllocator(3, device="cpu")
        allocator.alloc_group_begin(1)
        bulk = allocator.alloc(2)

        self.assertEqual(bulk.numel(), 2)
        self.assertEqual(allocator.available_size(), 0)
        self.assertEqual(allocator.reserved_size(), 1)
        allocator.alloc_group_end()
        self.assertEqual(allocator.available_size(), 1)

    def test_failed_or_abandoned_groups_do_not_leave_credit(self):
        allocator = MambaSlotAllocator(1, device="cpu")
        occupied = allocator.alloc(1)
        allocator.alloc_group_begin(1)
        self.assertEqual(allocator.reserved_size(), 0)

        allocator.free(occupied)
        allocator.alloc_group_begin(1)
        self.assertEqual(allocator.reserved_size(), 1)
        allocator.clear()
        self.assertEqual(allocator.reserved_size(), 0)
        self.assertEqual(allocator.available_size(), 1)

        allocator.alloc_group_begin(1)
        allocator.alloc_group_begin(1)
        self.assertEqual(allocator.reserved_size(), 1)
        self.assertEqual(allocator.available_size(), 0)
        self.assertIsNotNone(allocator.alloc(1))


class TestMambaPoolCapacity(unittest.TestCase):
    def test_mamba_pool_hard_cap_limits_the_request_pool_to_one(self):
        class _Runner(ModelRunnerKVCacheMixin):
            pass

        runner = _Runner()
        runner.server_args = SimpleNamespace(
            max_running_requests=64,
            max_mamba_cache_size=4,
            disable_radix_cache=False,
            disable_overlap_schedule=True,
            enable_mamba_extra_buffer=lambda: True,
            enable_mamba_extra_buffer_lazy=lambda: False,
        )
        runner.model_config = SimpleNamespace(context_len=1)
        runner.dp_size = 1
        runner.mambaish_config = object()

        max_num_reqs = runner._resolve_max_num_reqs(token_capacity=4096)
        self.assertEqual(max_num_reqs, 1)

        # The production HybridReqToTokenPool inherits this request-slot
        # allocation.  Avoid allocating model state tensors in this CPU test.
        pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        ReqToTokenPool.__init__(
            pool,
            size=max_num_reqs,
            max_context_len=1,
            device="cpu",
            enable_memory_saver=False,
        )
        pool.mamba_allocator = MambaSlotAllocator(4, device="cpu")

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.req_to_token_pool = pool
        args = ServerArgs(model_path="dummy")
        args.pp_max_micro_batch_size = 8
        set_global_server_args_for_scheduler(args)
        self.assertEqual(scheduler.get_num_allocatable_reqs(0), 1)


if __name__ == "__main__":
    unittest.main()
