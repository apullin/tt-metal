# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP-1 speculative greedy decoding: one MTP draft and a 2-row verify per step (`mtpk.MTPDecoder`
with k=1). The step is described in mtpk.py."""
from models.demos.blackhole.qwen36.tt.spec.mtpk import MTPDecoder


class MTP1Decoder(MTPDecoder):
    def __init__(self, model, mtp):
        super().__init__(model, mtp, k=1)
