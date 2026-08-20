"""
HiCache L3 (file storage backend) tests on Intel XPU.

Two properties, in order, because the second is meaningless without the first:
  1. write:  a prefix served by the device tier is backed up to disk
  2. read:   after the server is restarted -- which destroys the host tier -- the same
             prefix is served from disk

The restart is what makes the read test honest. With a live host tier, a hit proves
nothing about L3, because L2 could have answered.

Usage:
python3 -m unittest test_xpu_hicache_storage_file -v
"""

import os
import shutil
import tempfile
import time
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

register_xpu_ci(est_time=600, suite="stage-b-test-1-gpu-xpu")

# See test_xpu_hierarchical_cache.py: --hicache-ratio sizes the pinned host pool from
# the device pool, so the device pool must be pinned too.
MAX_TOTAL_TOKENS = 4096
PAGE_SIZE = 64
# How long to wait for the write-through backup to reach disk. The backup is
# asynchronous: the device write acks first, the host write acks next, and only then is
# the page handed to the storage backend.
WRITE_TIMEOUT_S = 60


def pin_loopback_no_proxy():
    """Keep the loopback address out of any ambient HTTP proxy.

    Duplicated from test_xpu_hierarchical_cache.py rather than imported, so each test
    file stays runnable on its own from any working directory. See that file for the
    failure this prevents: a proxied self-warmup GET makes the server kill its own pid,
    and the test reports a launch failure that names neither HTTP nor the proxy.
    """
    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        missing = [h for h in ("127.0.0.1", "localhost", "::1") if h not in current]
        if missing:
            os.environ[var] = ",".join(missing + ([current] if current else []))


class TestXPUHiCacheStorageFile(CustomTestCase):
    """Testcase: HiCache file storage backend on Intel XPU.

    Cover scenarios:
    1. write_through backs a served prefix up to the file backend
    2. the prefix is read back from disk after a server restart

    [Test Category] HiCache
    [Test Target] --hicache-storage-backend file on --device xpu
    """

    @classmethod
    def setUpClass(cls):
        pin_loopback_no_proxy()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_dir = tempfile.mkdtemp(prefix="xpu-hicache-l3-")
        # Nonce the prefix so a rerun cannot read files planted by a previous run. The
        # L3 key is a chained hash of the token pages, so different text is a different
        # key.
        cls.nonce = uuid.uuid4().hex[:8]
        cls.prefix = f"Document {cls.nonce}. " + "Section 7.4 of the manual. " * 64
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

    @classmethod
    def _launch(cls):
        """Launch a server against the shared storage dir.

        The dir is deliberately NOT cleared between launches: keeping it is the entire
        experiment. Clearing it on relaunch produces a correct miss that reads exactly
        like a broken read path.
        """
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
            "--hicache-storage-backend",
            "file",
            # wait_complete is mandatory, not a tuning choice. Under the default
            # best_effort policy can_terminate_prefetch() returns True unconditionally,
            # so the L3 prefetch is cancelled on the first scheduler tick and a cold
            # request keeps zero prefetched tokens. The read assertion below would then
            # be unfalsifiable.
            "--hicache-storage-prefetch-policy",
            "wait_complete",
        ]
        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            # See test_xpu_hierarchical_cache.py: device="auto" would append a second
            # --device after ours and win.
            device="xpu",
            env={
                **os.environ,
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir,
            },
        )

    def _generate(self, text: str) -> dict:
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
        details = meta.get("cached_tokens_details") or {}
        return (
            meta.get("cached_tokens", 0),
            details.get("device", 0),
            details.get("host", 0),
            details.get("storage", 0),
        )

    def _files(self) -> int:
        return len(os.listdir(self.storage_dir))

    def test_l3_write_then_read_across_restart(self):
        # One test method, not two, because the read step depends on state the write
        # step creates. Split across methods it would depend on execution order.
        type(self).process = self._launch()

        # --- write ---
        self._generate(self.prefix)
        deadline = time.monotonic() + WRITE_TIMEOUT_S
        while self._files() == 0 and time.monotonic() < deadline:
            time.sleep(1)
        planted = self._files()
        self.assertGreater(
            planted,
            0,
            "write_through did not back the prefix up to the file backend; there is "
            "nothing on disk, so the read step below could only fail for the wrong "
            "reason",
        )

        # --- restart: this is what destroys the host tier ---
        kill_process_tree(type(self).process.pid)
        type(self).process = None
        self.assertEqual(
            self._files(), planted, "the restart must not disturb the planted files"
        )
        type(self).process = self._launch()

        # --- read ---
        cached, device, host, storage = self._tiers(self._generate(self.prefix))
        self.assertGreater(
            storage,
            0,
            f"expected an L3 hit, got cached={cached} device={device} host={host}",
        )
        self.assertGreaterEqual(cached, storage)


if __name__ == "__main__":
    unittest.main()
