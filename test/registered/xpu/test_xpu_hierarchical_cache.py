"""
HierarchicalCache (L1 device <-> L2 pinned host) tests on Intel XPU.

Mirrors the Ascend HiCache suite under test/registered/ascend/basic_function/HiCache/,
with one XPU-specific addition: a test that a prefix evicted from the device pool is
reloaded from the host tier rather than recomputed. That is the only test here that
actually exercises the XPU KV transfer kernels in both directions.

Usage:
python3 -m unittest test_xpu_hierarchical_cache -v
"""

import os
import unittest
import uuid

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=420, suite="stage-b-test-1-gpu-xpu")

# The device KV pool is pinned, not auto-sized, for two independent reasons:
#   1. --hicache-ratio sizes the PINNED HOST pool as a multiple of the DEVICE pool, so
#      an auto-sized device pool makes the host allocation unpredictable. On BMG an
#      oversized pinned allocation presents as a device fault, i.e. as a driver bug
#      rather than as a configuration error.
#   2. The eviction test below has to overflow this pool. A known size is what makes
#      the flood self-sizing and quick instead of a guess.
MAX_TOTAL_TOKENS = 4096
PAGE_SIZE = 64
# Must match --hicache-ratio below. The host pool is this multiple of the device pool:
# measured 2026-08-19, "KV Cache is allocated ... K size: 0.06 GB, V size: 0.06 GB" for
# 4096 device tokens against "Allocating 0.24 GB host memory", i.e. exactly 2x.
HICACHE_RATIO = 2
HOST_TOTAL_TOKENS = HICACHE_RATIO * MAX_TOTAL_TOKENS

# The flood in test_3 has to land inside a WINDOW, not just be "big":
#   lower bound  > MAX_TOTAL_TOKENS   or the device copy is never displaced, the request
#                                     is answered from L1, and the host tier is not read
#   upper bound  < HOST_TOTAL_TOKENS  or the flood evicts the HOST copy too, and the
#                                     prefix is recomputed -- indistinguishable from a
#                                     broken transfer path
# The first version of this test used 3x the device pool, which is 1.5x the HOST pool,
# and failed with cached=0 on a working system: 37 fillers x 384 allocated tokens =
# 14208 tokens through an 8192-token host pool. 1.25x sits above the device pool and
# leaves ~1.4k tokens of host headroom for the target.
FLOOD_TOKENS = int(1.25 * MAX_TOTAL_TOKENS)
assert MAX_TOTAL_TOKENS < FLOOD_TOKENS < HOST_TOTAL_TOKENS - 8 * PAGE_SIZE, (
    "flood target is outside the eviction window; re-derive it if MAX_TOTAL_TOKENS or "
    "HICACHE_RATIO changes"
)


def pin_loopback_no_proxy():
    """Keep the loopback address out of any ambient HTTP proxy.

    The server warms ITSELF up over HTTP: _execute_server_warmup() polls
    GET http://127.0.0.1:<port>/model_info 120 times and, on failure, calls
    kill_process_tree on its own pid. Behind a corporate proxy whose no_proxy does not
    cover the loopback address, that GET is answered by the proxy with a 403 block page,
    the request never reaches the server, and after ~130 s the server kills itself. The
    launch then fails in setUpClass with "Server process exited with code -9" and zero
    tests run -- a failure that mentions neither the proxy nor HTTP, and that reads like
    a HiCache or driver problem.

    Measured on this box 2026-08-19 00:12-00:14: both the offline and the online launch
    attempt died this way, 320 s wasted, `Ran 0 tests`.
    """
    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        missing = [h for h in ("127.0.0.1", "localhost", "::1") if h not in current]
        if missing:
            os.environ[var] = ",".join(missing + ([current] if current else []))


class TestXPUHierarchicalCache(CustomTestCase):
    """Testcase: HierarchicalCache test on Intel XPU.

    Cover scenarios:
    1. Long identical texts: cache is reused
    2. Short identical texts (< one page): cache is not reused
    3. A prefix evicted from L1 is reloaded from L2, not recomputed

    [Test Category] HiCache
    [Test Target] --enable-hierarchical-cache on --device xpu
    """

    @classmethod
    def setUpClass(cls):
        # Before the launch, not after: the server's self-warmup is the first thing that
        # breaks, and popen_launch_server passes this process's environ to the child.
        pin_loopback_no_proxy()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Nonce every prompt in this class. Without it a rerun on the same machine can
        # hit a cache populated by the previous run, which turns a broken write path
        # into a passing test.
        cls.nonce = uuid.uuid4().hex[:8]
        other_args = [
            "--device",
            "xpu",
            "--attention-backend",
            "intel_xpu",
            "--mem-fraction-static",
            "0.6",
            "--tp-size",
            "1",
            "--page-size",
            str(PAGE_SIZE),
            "--max-total-tokens",
            str(MAX_TOTAL_TOKENS),
            "--enable-cache-report",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "2",
            "--hicache-write-policy",
            "write_through",
            "--hicache-io-backend",
            "kernel",
            "--hicache-mem-layout",
            "layer_first",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            # Pass the device explicitly, as test_gemma_4_e2b.py does. The default
            # device="auto" makes popen_launch_server call auto_config_device() and
            # APPEND its own --device to the end of the command line, after the one in
            # other_args. argparse takes the last occurrence, so on a box where
            # detection falls back to "cpu" this XPU test would silently run on CPU and
            # still pass -- proving nothing about the XPU transfer kernels.
            device="xpu",
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _generate(self, text: str) -> dict:
        """Return meta_info for one greedy single-token request."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["meta_info"]

    @staticmethod
    def _tiers(meta: dict) -> tuple:
        """(cached, device, host, storage) from one meta_info.

        cached_tokens_details is None when every tier counter is zero -- an absent dict
        is a legitimate all-zero report, not a missing field.
        """
        details = meta.get("cached_tokens_details") or {}
        return (
            meta.get("cached_tokens", 0),
            details.get("device", 0),
            details.get("host", 0),
            details.get("storage", 0),
        )

    def test_1_reuse_long_identical(self):
        """A long prompt sent twice must hit on the second send."""
        text = f"Document {self.nonce}. " + "Section 7.4 of the manual. " * 64

        cached, _, _, _ = self._tiers(self._generate(text))
        self.assertEqual(cached, 0, "first send of a nonced prompt must be a cold miss")

        cached, device, _, _ = self._tiers(self._generate(text))
        self.assertGreater(cached, 0, "second send must reuse the cached prefix")
        self.assertGreater(device, 0, "a just-used prefix should still be on device")

    def test_2_no_reuse_shorter_than_one_page(self):
        """A prompt below page_size cannot be cached: the radix tree stores whole pages."""
        text = f"who am i {self.nonce}?"
        for _ in range(2):
            cached, _, _, _ = self._tiers(self._generate(text))
            self.assertEqual(cached, 0)

    def test_3_evicted_prefix_returns_from_host(self):
        """Overflow L1, then prove the prefix comes back from L2 instead of recompute.

        This is the test that exercises the XPU transfer kernels in both directions:
        write_through pushes the prefix to the host pool, the flood evicts the device
        copy, and the final request has to load it back.
        """
        target = f"Target {self.nonce}. " + "Section 7.4 of the manual. " * 64

        # Populate, and confirm the prefix is genuinely resident before evicting it.
        self._generate(target)
        cached, _, _, _ = self._tiers(self._generate(target))
        self.assertGreater(cached, 0, "precondition: target must be cached before flood")

        # Flood into the window defined at the top of this file. Count what the server
        # ALLOCATES, not what the client sent: allocation is page-aligned, so a 332-token
        # filler occupies 384 tokens of pool. Counting prompt_tokens undercounts by ~15%
        # per request and the flood then overshoots into host-eviction territory.
        allocated = 0
        i = 0
        while allocated < FLOOD_TOKENS:
            filler = f"Filler {self.nonce} {i}. " + "Unrelated background text. " * 64
            meta = self._generate(filler)
            fresh = max(meta.get("prompt_tokens", 0) - meta.get("cached_tokens", 0), 0)
            allocated += -(-fresh // PAGE_SIZE) * PAGE_SIZE
            i += 1
        self.assertLess(
            allocated,
            HOST_TOTAL_TOKENS,
            f"flood allocated {allocated} tokens against a {HOST_TOTAL_TOKENS}-token "
            "host pool, so a miss below would say nothing about the transfer path",
        )

        cached, device, host, _ = self._tiers(self._generate(target))
        self.assertGreater(cached, 0, "evicted prefix was recomputed instead of reloaded")
        # host > 0 is the whole point. HiCache prefers a device copy whenever one
        # exists, so host stays 0 unless the device copy really was evicted -- which
        # makes this assertion, not `cached > 0`, the one that proves L2 was read.
        self.assertGreater(host, 0, f"expected an L2 hit, got device={device} host={host}")


if __name__ == "__main__":
    unittest.main()
