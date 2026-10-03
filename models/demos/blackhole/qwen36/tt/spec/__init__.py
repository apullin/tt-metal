# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for the TP Qwen36Model: k-row target verify with GDN rollback (verify,
gdn_rows), per-row sharded argmax (argmax), the MTP draft head (mtp), the MTP-k loop with chained
drafts (mtpk.MTPDecoder), and MTP-1 as its k=1 case (mtp1.MTP1Decoder)."""
