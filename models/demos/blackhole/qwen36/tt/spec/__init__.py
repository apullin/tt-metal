# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for the TP Qwen36Model: k-row target verify with GDN rollback (verify,
gdn_rows), per-row sharded argmax (argmax), the MTP draft head (mtp), and the MTP-1 loop (mtp1)."""
