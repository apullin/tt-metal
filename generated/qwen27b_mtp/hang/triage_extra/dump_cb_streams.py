#!/usr/bin/env python3
"""
Usage:
    dump_cb_streams

Description:
    For every Tensix core whose BRISC go message is not DONE: the Wormhole CB sync registers of
    CB 0..7 (tiles_received = stream N reg 4, tiles_acked = stream N reg 3) and the CB config words
    (fifo_addr, fifo_size, fifo_num_pages, fifo_page_size, in 16-byte units) the launch message
    points at. Throwaway helper for the N300 long-prefill hang (printed to stdout).

Owner:
    N300HangDebug
"""

from dispatcher_data import run as get_dispatcher_data
from run_checks import run as get_run_checks
from triage import ScriptConfig, run_script
from ttexalens import read_word_from_device
from ttexalens.context import Context

script_config = ScriptConfig(depends=["run_checks", "dispatcher_data"])


def _dump(location, dispatcher_data, context):
    d = dispatcher_data.get_core_data(location, "brisc")
    if d.go_message == "DONE":
        return None
    mb = dispatcher_data.read_mailboxes(location)
    rd = int(mb.launch_msg_rd_ptr)
    kc = mb.launch[rd].kernel_config
    base = int(kc.kernel_config_base[0])
    cb_off = int(kc.local_cb_offset)
    mask = int(kc.local_cb_mask)
    lines = [f"{location.to_user_str()} dev{location.device.id} go={d.go_message} kernel={d.kernel_name} base=0x{base:x} cb_off=0x{cb_off:x} mask=0x{mask:x}"]
    for cb in range(8):
        recv = read_word_from_device(location, 0xFFB40000 + cb * 0x1000 + 4 * 4, context=context)
        ack = read_word_from_device(location, 0xFFB40000 + cb * 0x1000 + 3 * 4, context=context)
        cfg = []
        if mask >> cb & 1:
            # local CB configs are packed for indices 0..highest set bit, 4 words each
            cfg = [read_word_from_device(location, base + cb_off + (cb * 4 + i) * 4, context=context) for i in range(4)]
        lines.append(f"  cb{cb}: recv={recv} (0x{recv:x}) acked={ack} (0x{ack:x}) diff16={(recv - ack) & 0xFFFF} cfg={cfg}")
    print("\n".join(lines), flush=True)
    return None


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    run_checks.run_per_block_check(lambda loc: _dump(loc, dispatcher_data, context), block_filter=["tensix"])


if __name__ == "__main__":
    run_script()
