import json
import logging

import torch
from gguf import GGMLQuantizationType as WeightType

import sglang.srt.layers.quantization.gguf as gguf_xpu


def _contains_tensor(value):
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_xpu_residency_log_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_GGUF_XPU_LOG_RESIDENCY", raising=False)
    assert not gguf_xpu._xpu_residency_log_enabled()
    monkeypatch.setenv("SGLANG_GGUF_XPU_LOG_RESIDENCY", "1")
    assert gguf_xpu._xpu_residency_log_enabled()


def test_xpu_residency_payload_dedups_aliases_and_merged_rep():
    base = torch.zeros((4, 8), dtype=torch.float16)
    reps = {
        "q": ("fp16", base[:2], None),
        "k": ("fp16", base[2:], None),
    }
    merged = gguf_xpu._xpu_try_merge_shards(reps, ["q", "k"])
    assert merged is not None
    merged_rep, _ = merged

    source = gguf_xpu._xpu_residency_source_descriptor(
        WeightType.IQ4_NL, "q", ("iq4", base[:2], None), base[:2]
    )
    payload = gguf_xpu._xpu_residency_payload(
        "blk.0.test", [source], reps, merged=merged
    )

    expected = (
        base.untyped_storage().nbytes()
        + merged_rep[1].untyped_storage().nbytes()
    )
    assert payload["unique_storage_bytes"] == expected
    assert len(payload["storage_keys"]) == 2
    assert payload["final_rep_kinds"] == {
        "reps": {"q": "fp16", "k": "fp16"},
        "merged": "fp16",
        "groups": [],
    }
    assert source == {
        "source_type": "IQ4_NL",
        "shard_id": "q",
        "result_kind": "iq4",
        "classification": "native",
        "logical_tensor_bytes": base[:2].numel() * base.element_size(),
        "canonical_logical_bytes": base[:2].numel() * base.element_size(),
    }
    assert payload["per_kind_unique_storage_bytes"] == {"fp16": expected}
    assert not _contains_tensor(payload)


def test_xpu_residency_source_descriptor_classifies_final_rep():
    source = torch.zeros((2, 4), dtype=torch.uint8)
    canonical = torch.zeros((2, 4), dtype=torch.float16)
    native = gguf_xpu._xpu_residency_source_descriptor(
        WeightType.IQ4_XS, "native", ("iq4", source, canonical), source
    )
    fallback = gguf_xpu._xpu_residency_source_descriptor(
        WeightType.IQ4_XS, "fallback", ("fp16", canonical, None), source
    )
    unquantized = gguf_xpu._xpu_residency_source_descriptor(
        WeightType.F32, "plain", ("fp16", canonical, None), source
    )

    assert native["classification"] == "native"
    assert native["logical_tensor_bytes"] == source.numel() * source.element_size()
    assert native["canonical_logical_bytes"] == (
        source.numel() * source.element_size()
        + canonical.numel() * canonical.element_size()
    )
    assert fallback["classification"] == "fallback"
    assert unquantized["classification"] == "unquantized"


def test_xpu_residency_log_is_single_json_record(caplog):
    rep = ("fp16", torch.zeros((2, 4), dtype=torch.float16), None)
    source = gguf_xpu._xpu_residency_source_descriptor(
        WeightType.F32, "_single", rep, rep[1]
    )
    caplog.set_level(logging.INFO, logger=gguf_xpu.__name__)

    gguf_xpu._xpu_log_residency("embed_tokens", [source], {"_single": rep})

    records = [
        record for record in caplog.records
        if record.name == gguf_xpu.__name__
        and record.getMessage().startswith("[gguf-xpu residency] ")
    ]
    assert len(records) == 1
    payload = json.loads(records[0].getMessage().removeprefix("[gguf-xpu residency] "))
    assert payload["prefix"] == "embed_tokens"
    assert payload["source_descriptors"] == [source]
    assert payload["unique_storage_bytes"] == rep[1].untyped_storage().nbytes()
    assert not _contains_tensor(payload)
