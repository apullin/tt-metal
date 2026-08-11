// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>

#include "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.hpp"

namespace ttnn::operations::experimental::topk_large_indices::program {
namespace {

TEST(TopkLargeIndicesCoreScheduling, Discontiguous80CoreGridAssignsEveryRowExactlyOnce) {
    const CoreRangeSet core_grid(std::vector<CoreRange>{
        CoreRange(CoreCoord(0, 0), CoreCoord(3, 9)), CoreRange(CoreCoord(7, 0), CoreCoord(10, 9))});
    constexpr uint32_t num_rows = 163;

    const auto assignments = derive_core_row_assignments(core_grid, num_rows);
    const auto expected_cores = corerange_to_cores(core_grid, std::nullopt, true);

    ASSERT_EQ(assignments.size(), 80);
    ASSERT_EQ(assignments.size(), expected_cores.size());
    uint32_t expected_start_row = 0;
    for (uint32_t i = 0; i < assignments.size(); ++i) {
        EXPECT_EQ(assignments[i].core, expected_cores[i]);
        EXPECT_EQ(assignments[i].start_row, expected_start_row);
        EXPECT_EQ(assignments[i].num_rows, i < 3 ? 3 : 2);
        expected_start_row += assignments[i].num_rows;
    }
    EXPECT_EQ(expected_start_row, num_rows);
}

}  // namespace
}  // namespace ttnn::operations::experimental::topk_large_indices::program
