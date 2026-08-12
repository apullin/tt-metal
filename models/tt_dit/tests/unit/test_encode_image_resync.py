# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only cover for the resync call site: the VAE encoder is the first eager collective a warm
worker reaches after a traced replay, so ``encode_image`` must drain the device and hand every CCL
manager's ping-pong parity back to a known state BEFORE the encoder's first halo exchange runs.
Skip the drain, or skip a manager, and that exchange waits on a semaphore its neighbour already
signalled — a hang with no error. See ``LTXPipeline._resync_ccl_ping_pong``.

The managers are the same object on some mesh shapes and distinct on others; the reset dedupes by
``id()``, so both cases are asserted here. No device: the pipeline is built via ``__new__`` and the
device-touching calls are captured, so a regression in the ordering is caught in CI rather than as a
hang on hardware.
"""

import torch

from ...pipelines.ltx.pipeline_ltx import LTXPipeline


class FakeManager:
    """A stand-in CCL manager that records its reset into the shared event log."""

    def __init__(self, tag, log):
        self.tag = tag
        self._log = log

    def reset_global_semaphores(self):
        self._log.append(("reset", id(self)))


def _build_pipeline(monkeypatch, log, ccl, vae_ccl, audio_ccl):
    """An ``LTXPipeline`` whose device-touching calls append to ``log`` in call order.

    ``_prepare_vae_encoder`` pushes weights and is irrelevant to the ping-pong parity, so it is a
    recorded no-op; the encoder forward is the eager collective the resync must precede.
    """
    m = LTXPipeline.__new__(LTXPipeline)
    m.mesh_device = object()
    m.ccl_manager = ccl
    m.vae_ccl_manager = vae_ccl
    m.audio_ccl_manager = audio_ccl

    def fake_encoder(image):
        log.append(("encode", None))
        return torch.zeros(1)

    m.vae_encoder = fake_encoder
    monkeypatch.setattr(m, "_prepare_vae_encoder", lambda: log.append(("prepare", None)))
    monkeypatch.setattr(
        "models.tt_dit.pipelines.ltx.pipeline_ltx.ttnn.synchronize_device",
        lambda dev: log.append(("sync", id(dev))),
    )
    monkeypatch.setenv("LTX_VAE_ENCODER_HOST", "0")
    return m


def test_syncs_and_resets_before_encoding(monkeypatch):
    """Distinct managers (the multi-submesh shape): drain then reset all three, then encode."""
    log = []
    ccl = FakeManager("ccl", log)
    vae = FakeManager("vae", log)
    audio = FakeManager("audio", log)
    m = _build_pipeline(monkeypatch, log, ccl, vae, audio)

    m.encode_image(torch.zeros(1, 3, 1, 8, 8))

    kinds = [e[0] for e in log]
    assert "sync" in kinds, "device was not drained before the eager encoder ran"
    assert kinds.count("reset") == 3, f"expected all three distinct managers reset, got {kinds}"
    encode_at = kinds.index("encode")
    for i, (kind, _) in enumerate(log):
        if kind in ("sync", "reset"):
            assert i < encode_at, f"{kind} ran at {i}, after the encoder at {encode_at}"

    (sync_dev,) = [obj for kind, obj in log if kind == "sync"]
    assert sync_dev == id(m.mesh_device), "synchronize_device drained the wrong mesh"
    assert {obj for kind, obj in log if kind == "reset"} == {id(ccl), id(vae), id(audio)}


def test_shared_managers_reset_once(monkeypatch):
    """Same object on a Linear-topology shape: dedupe by id() so it is reset exactly once."""
    log = []
    shared = FakeManager("shared", log)
    m = _build_pipeline(monkeypatch, log, shared, shared, shared)

    m.encode_image(torch.zeros(1, 3, 1, 8, 8))

    kinds = [e[0] for e in log]
    assert kinds.count("reset") == 1, f"a shared manager was reset {kinds.count('reset')} times, not once"
    assert kinds.count("sync") == 1, "the single mesh should be drained exactly once"


def test_returns_the_device_latent(monkeypatch):
    """Default path (LTX_VAE_ENCODER_HOST off) returns the encoder's device output unchanged."""
    log = []
    m = _build_pipeline(monkeypatch, log, FakeManager("a", log), FakeManager("b", log), FakeManager("c", log))

    out = m.encode_image(torch.zeros(1, 3, 1, 8, 8))

    assert isinstance(out, torch.Tensor) and out.shape == (1,), "did not return the device latent"
