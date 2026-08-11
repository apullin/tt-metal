// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>

#include "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/high_bw_all_gather_scheduler.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather::scheduler {
namespace {

constexpr std::array<uint32_t, 4> kBlackholeTiers{8, 4, 2, 1};

TEST(HighBwAllGatherScheduler, ProductionAndProxyCoreBudgetsSelectQualifiedTiers) {
    constexpr uint32_t num_links = 2;

    // Eight workers/direction need (8 workers + 1 mux) * 2 directions * 2 links = 36 cores.
    EXPECT_TRUE(worker_count_fits(8, num_links, 40));
    EXPECT_FALSE(worker_count_fits(8, num_links, 30));
    EXPECT_EQ(select_fitting_worker_count(8, num_links, 40, kBlackholeTiers), 8);
    EXPECT_EQ(select_fitting_worker_count(8, num_links, 30, kBlackholeTiers), 4);

    // The real GLM-5.2 runtime-control schedule currently prefers two workers/direction. Both ownership
    // profiles admit the exact 12 active cores: (2 workers + 1 mux) * 2 directions * 2 links.
    EXPECT_TRUE(worker_count_fits(2, num_links, 30));
    EXPECT_TRUE(worker_count_fits(2, num_links, 40));
    EXPECT_EQ(select_fitting_worker_count(2, num_links, 30, kBlackholeTiers), 2);
    EXPECT_EQ(select_fitting_worker_count(2, num_links, 40, kBlackholeTiers), 2);
}

TEST(HighBwAllGatherScheduler, FailsClosedBelowMinimumCoreBudget) {
    constexpr uint32_t num_links = 2;
    // One worker/direction has no mux and needs four cores total.
    EXPECT_TRUE(worker_count_fits(1, num_links, 4));
    EXPECT_FALSE(worker_count_fits(1, num_links, 3));
    // select_fitting_worker_count deliberately returns the minimum tier so the factory's capacity
    // assertion emits the user-facing diagnostic.
    EXPECT_EQ(select_fitting_worker_count(8, num_links, 3, kBlackholeTiers), 1);
}

}  // namespace
}  // namespace ttnn::operations::experimental::high_bw_all_gather::scheduler
