// Copyright 2025 The secretflow authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "spdlog/spdlog.h"

#include "psi/algorithm/spiral/params.h"
#include "psi/algorithm/spiral/spiral_client.h"
#include "psi/algorithm/spiral/util.h"
#include "psi/algorithm/ypir/client.h"
#include "psi/algorithm/ypir/server.h"

using namespace psi::ypir;
using namespace psi::spiral;

class YPIRIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override { spdlog::set_level(spdlog::level::info); }
};

TEST_F(YPIRIntegrationTest, SimplePIRBasicFlow) {
  // Setup parameters - use standard SimplePIR parameters (16K rows)
  // This matches Rust: params_for_scenario_simplepir(1<<14, 16384*8)
  std::size_t poly_len{2048};
  std::vector<std::uint64_t> moduli{268369921, 249561089};
  double noise_width{16.042421};

  // Standard SimplePIR: db_dim_1=3 → 2^(3+11)=16384 rows, instances=1
  PolyMatrixParams poly_matrix_params(2, 16384, 21, 4, 8, 8, 1);
  QueryParams query_params(3, 1, 1);  // db_dim_1=3, db_dim_2=1, instances=1

  Params params(poly_len, std::move(moduli), noise_width,
                std::move(poly_matrix_params), std::move(query_params));

  size_t db_rows = 1ULL << (params.DbDim1() + params.PolyLenLog2());
  size_t db_cols = params.Instances() * params.PolyLen();
  size_t db_size = db_rows * db_cols;

  SPDLOG_INFO(
      "SimplePIR test (SMALL): db_rows={}, db_cols={}, db_size={}, "
      "pt_modulus={}",
      db_rows, db_cols, db_size, params.PtModulus());

  // SimplePIR with pt_modulus=16384 requires uint16_t database
  std::vector<uint16_t> db(db_size);
  std::random_device rd;
  std::mt19937 gen(42);  // Fixed seed for reproducibility
  std::uniform_int_distribution<> dis(0, params.PtModulus() - 1);
  for (auto& val : db) {
    val = static_cast<uint16_t>(dis(gen));
  }

  // Add some known values for verification
  if (db_size > 1000) {
    db[0] = 100;
    db[1] = 101;
    db[100] = 200;
    db[1000] = 250;
  }

  // Create server (YServer is now YPirServer<uint16_t> for SimplePIR)
  // Note: SimplePIR uses column-major storage internally (inp_transposed=false)
  // to match Rust implementation
  auto server_start = std::chrono::high_resolution_clock::now();
  YServer server(
      params, db, true, false,
      true);  // is_simplepir=true, inp_transposed=false, pad_rows=true
  auto server_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - server_start);
  SPDLOG_INFO("Server setup: {} ms", server_elapsed.count());
  EXPECT_GT(server_elapsed.count(), 0);

  // Verify server database storage
  const uint16_t* server_db = server.db();
  size_t padded_rows = params.DbRowsPadded();
  SPDLOG_INFO(
      "Verifying database storage: db_rows={}, db_cols={}, padded_rows={}",
      db_rows, db_cols, padded_rows);

  // Check if known values are stored correctly
  auto check_value = [&](size_t idx) {
    size_t row = idx / db_cols;
    size_t col = idx % db_cols;
    // Server uses column-major: col * padded_rows + row
    size_t server_idx = col * padded_rows + row;
    uint16_t original = db[idx];
    uint16_t stored = server_db[server_idx];
    SPDLOG_INFO("  db[{}] (row={}, col={}): original={}, stored={}", idx, row,
                col, original, stored);
    return original == stored;
  };

  EXPECT_TRUE(check_value(0));
  EXPECT_TRUE(check_value(1));
  if (db_size > 1000) {
    EXPECT_TRUE(check_value(100));
    EXPECT_TRUE(check_value(1000));
  }

  // Also print what's actually stored in row 0
  SPDLOG_INFO("First 10 values in row 0 of database (original indexing):");
  for (size_t i = 0; i < std::min(db_cols, size_t(10)); ++i) {
    SPDLOG_INFO("  db[{}] = {}", i, static_cast<int>(db[i]));
  }

  // Print what row 0 should contain
  // input db is row-major: db[row * db_cols + col]
  SPDLOG_INFO("Expected row 0 values (what SimplePIR should return):");
  std::vector<uint16_t> expected_row_0;
  for (size_t col = 0; col < db_cols; ++col) {
    size_t original_idx = 0 * db_cols + col;  // row=0, col=col in row-major
    if (original_idx < db_size) {
      expected_row_0.push_back(db[original_idx]);
      if (col < 10) {
        SPDLOG_INFO("  col={}: db[{}] = {}", col, original_idx,
                    static_cast<int>(db[original_idx]));
      }
    }
  }

  // Offline precomputation
  OfflinePrecomputedValues offline_vals;
  try {
    absl::Span<const uint64_t> empty_span;
    std::string_view empty_path;

    auto offline_start = std::chrono::high_resolution_clock::now();
    offline_vals =
        server.PerformOfflinePrecomputationSimplepir(empty_span, empty_path);

    auto offline_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - offline_start);
    SPDLOG_INFO("Offline precomputation: {} ms", offline_elapsed.count());
    EXPECT_GT(offline_elapsed.count(), 0);
    EXPECT_GT(offline_vals.hint_0.size(), 0);
    EXPECT_GT(offline_vals.prepacked_lwe.size(), 0);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Exception in offline precomputation: {}", e.what());
    throw;
  }

  // Test querying row 0
  size_t target_idx = 0;
  SPDLOG_INFO("Querying row 0 (first value expected: 100)");

  // Client generates keys and query
  try {
    auto client_start = std::chrono::high_resolution_clock::now();
    // Use fixed seed for reproducibility
    uint128_t fixed_seed = yacl::MakeUint128(0, 12345);
    SpiralClient spiral_client(params, fixed_seed);
    YClient y_client(spiral_client, params, fixed_seed);

    auto simple_query = y_client.GenerateFullQuerySimplepir(target_idx);

    auto client_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - client_start);
    SPDLOG_INFO("Client query generation: {} ms", client_elapsed.count());

    // Server online computation
    auto online_start = std::chrono::high_resolution_clock::now();

    // Prepare pack_pub_params as vector of spans
    std::vector<absl::Span<const uint64_t>> pack_params_spans;
    if (!simple_query.pack_pub_params_row_1s_pm.empty()) {
      pack_params_spans.push_back(
          absl::MakeConstSpan(simple_query.pack_pub_params_row_1s_pm));
    }

    auto response = server.PerformOnlineComputationSimplepir(
        absl::MakeConstSpan(simple_query.packed_query_row), offline_vals,
        absl::MakeConstSpan(pack_params_spans));

    auto online_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - online_start);
    SPDLOG_INFO("Server online computation: {} ms", online_elapsed.count());

    // Client decodes response using YPIRClient
    auto decode_start = std::chrono::high_resolution_clock::now();
    YPIRClient ypir_client(params);
    auto decoded_values =
        ypir_client.DecodeResponseSimplepirYClient(params, y_client, response);
    auto decode_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - decode_start);
    SPDLOG_INFO("Client decode: {} ms", decode_elapsed.count());

    // Verify correctness - decoded_values should contain the entire row
    size_t target_row = target_idx / db_cols;

    SPDLOG_INFO("Query completed - Target: row={}, decoded_values.size()={}",
                target_row, decoded_values.size());

    // SimplePIR returns the entire row - decoded_values[i] = row[i]
    EXPECT_EQ(decoded_values.size(), db_cols);

    // Compare first 10 decoded values with expected
    SPDLOG_INFO("Verifying decoded row (first 10 values):");
    size_t num_errors = 0;
    for (size_t col = 0; col < std::min(db_cols, size_t(10)); ++col) {
      uint16_t expected = expected_row_0[col];
      uint64_t decoded = decoded_values[col];
      bool match = (decoded == expected);
      SPDLOG_INFO("  col={}: decoded={}, expected={}, {}", col, decoded,
                  expected, match ? "✓" : "✗");
      if (!match) {
        num_errors++;
      }
    }

    // Verify first value (db[0] = 100)
    EXPECT_EQ(decoded_values[0], expected_row_0[0])
        << "Decoded value mismatch at col 0";

    EXPECT_EQ(num_errors, 0) << num_errors << " value mismatches found";

    SPDLOG_INFO("✓ SimplePIR correctness verified!");
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Exception in query/online phase: {}", e.what());
    throw;
  }
}

// TODO: Implement DoublePIR online computation method
TEST_F(YPIRIntegrationTest, DISABLED_DoublePIRBasicFlow) {
  SPDLOG_INFO(
      "DoublePIR test is disabled - need to implement online computation "
      "method");
  GTEST_SKIP() << "DoublePIR online computation not yet implemented";
}
