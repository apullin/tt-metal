// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>

#include <cstdint>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/system_mesh.hpp>

namespace ttml::test_utils {

// True when the devices visible to this host can host `shape`. A 1x2 mesh, for instance, is
// all of an N300 but also a slice of a T3K or a Blackhole tray, so tests that need one gate
// on the mesh being available rather than on a specific board type.
inline bool system_supports_mesh(const tt::tt_metal::distributed::MeshShape& shape) {
    const auto& system_shape = tt::tt_metal::distributed::SystemMesh::instance().local_shape();
    if (system_shape.dims() < shape.dims()) {
        return false;
    }
    for (int32_t dim = 0; dim < static_cast<int32_t>(shape.dims()); ++dim) {
        if (system_shape[dim] < shape[dim]) {
            return false;
        }
    }
    return true;
}

}  // namespace ttml::test_utils

#define SKIP_UNLESS_MESH_SUPPORTED(shape)                                                           \
    do {                                                                                            \
        if (!ttml::test_utils::system_supports_mesh(shape)) {                                       \
            GTEST_SKIP() << fmt::format("Skipping test: system does not support a {} mesh", shape); \
        }                                                                                           \
    } while (0)
