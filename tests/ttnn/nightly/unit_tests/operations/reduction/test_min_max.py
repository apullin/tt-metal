# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import ttnn_max, ttnn_min, ttnn_mean

TEST_PADDING_VALUE = -42


@pytest.mark.parametrize(
    "shape_dim",
    (
        ((1, 1, 32, 32), 3),
        ((1, 2, 32, 32), 3),
        ((2, 3, 32, 32), 3),
        ((1, 1, 32, 32), 2),
        ((1, 2, 32, 32), 2),
        ((2, 3, 32, 32), 2),
        ((32, 32, 32, 32), 1),
        ((32, 32, 32, 32), 0),
        ((32, 32, 32, 32), 2),
        ((32, 32, 32, 32), 3),
        ((1, 2, 18, 20), 3),
    ),
)
@pytest.mark.parametrize(
    "kind",
    (
        "min",
        "max",
        "mean",
    ),  # single tile
)
@pytest.mark.parametrize(
    "layout",
    (
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ),
)
def test_min_max_for_dim_hw(device, shape_dim, kind, layout):
    shape, dim = shape_dim
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    input_shape = (N, C, H, W)
    torch_input = 1.0 + torch.rand(input_shape).bfloat16()

    if kind == "max":
        torch_output = torch_input.max()
    elif kind == "min":
        torch_output = torch_input.min()
    elif kind == "mean":
        torch_output = torch_input.mean()
    else:
        raise AttributeError()

    tt_input = ttnn.Tensor(torch_input, ttnn.bfloat16).to(layout).to(device)
    if layout == ttnn.TILE_LAYOUT:
        tt_input = ttnn.fill_implicit_tile_padding(tt_input, TEST_PADDING_VALUE)

    if kind == "max":
        tt_npu = ttnn_max(tt_input)
    elif kind == "min":
        tt_npu = ttnn_min(tt_input)
    else:
        assert kind == "mean"
        tt_npu = ttnn_mean(tt_input)

    tt_output = tt_npu.cpu().to_torch()
    if kind == "mean":
        # test for equivalance
        assert_numeric_metrics(
            torch_output,
            tt_output,
            pcc_threshold=0.9999,
            rtol=0.006,
            atol=0.008,
            frobenius_threshold=0.006,
        )
    else:
        # test for equivalance
        assert_equal(torch_output, tt_output)


# W-reduce (dim=-1) on HEIGHT_SHARDED L1 tensors takes a fast path only when the input and output
# share a shard grid AND a shard height; any other geometry must fall back to the generic path.
# The output memory config contributes only its grid -- the shard shape is re-derived.

SHARDED_W_SHAPE = (1, 1, 256, 4096)


def _height_sharded(shard_h, shard_w, num_cores):
    return ttnn.create_sharded_memory_config(
        shape=(shard_h, shard_w),
        core_grid=ttnn.CoreGrid(x=1, y=num_cores),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )


def _reduce_w_int32_sharded(device, op, input_config, output_config):
    torch.manual_seed(0)
    torch_input = torch.randint(-50_000, 50_000, SHARDED_W_SHAPE, dtype=torch.int32)
    torch_output = (torch.amax if op == "max" else torch.amin)(torch_input, dim=-1, keepdim=True)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_config,
    )
    reduce_op = ttnn.max if op == "max" else ttnn.min
    tt_output = reduce_op(tt_input, dim=-1, keepdim=True, memory_config=output_config)

    assert_equal(torch_output, ttnn.to_torch(tt_output))
    return tt_output


@pytest.mark.parametrize("op", ["max", "min"])
def test_reduce_w_height_sharded(device, op):
    """Matched shard grid and shard height, so the height-sharded fast path is used."""
    tt_output = _reduce_w_int32_sharded(device, op, _height_sharded(32, 4096, 8), _height_sharded(32, 32, 8))

    output_mem_config = tt_output.memory_config()
    assert output_mem_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert output_mem_config.shard_spec.num_cores() == 8


@pytest.mark.parametrize("op", ["max", "min"])
def test_reduce_w_height_sharded_grid_mismatch(device, op):
    """Grids differ (8 vs 4) while shard heights match, so the fast path is rejected."""
    tt_output = _reduce_w_int32_sharded(device, op, _height_sharded(64, 4096, 8), _height_sharded(64, 32, 4))
    assert tt_output.memory_config().shard_spec.num_cores() == 4


def test_reduce_w_height_sharded_shard_height_mismatch(device):
    """Shard heights differ (64 vs re-derived 32) while grids match; without the guard this hangs."""
    _reduce_w_int32_sharded(device, "max", _height_sharded(64, 4096, 8), _height_sharded(32, 32, 8))


def test_reduce_w_height_sharded_float_min(device):
    """bfloat16 min lowers to -MAX(-x) (reduce_w_neg), a different kernel on the same fast path."""
    torch.manual_seed(0)
    torch_input = torch.randn(SHARDED_W_SHAPE, dtype=torch.bfloat16)
    torch_output = torch.amin(torch_input, dim=-1, keepdim=True)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_height_sharded(32, 4096, 8),
    )
    tt_output = ttnn.min(tt_input, dim=-1, keepdim=True, memory_config=_height_sharded(32, 32, 8))

    assert_equal(torch_output, ttnn.to_torch(tt_output))
