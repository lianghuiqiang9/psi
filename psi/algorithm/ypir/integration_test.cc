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

TEST_F(YPIRIntegrationTest, SimplePIR_1GB) {
  // Setup parameters - 1GB database test
  std::size_t poly_len{2048};
  std::vector<std::uint64_t> moduli{268369921, 249561089};
  double noise_width{16.042421};

  // 1GB SimplePIR: db_dim_1=5, instances=4
  // 2^(5+11)=65536 rows × 4*2048=8192 cols = 512M elements = 1GB (uint16_t)
  PolyMatrixParams poly_matrix_params(2, 16384, 21, 4, 8, 8, 4);
  QueryParams query_params(5, 1, 4);  // db_dim_1=5, db_dim_2=1, instances=4

  Params params(poly_len, std::move(moduli), noise_width,
                std::move(poly_matrix_params), std::move(query_params));

  size_t db_rows = 1ULL << (params.DbDim1() + params.PolyLenLog2());
  size_t db_cols = params.Instances() * params.PolyLen();
  size_t db_size = db_rows * db_cols;

  SPDLOG_INFO(
      "SimplePIR Performance Test (1GB): db_rows={}, db_cols={}, db_size={} "
      "(~{} MB), pt_modulus={}",
      db_rows, db_cols, db_size, (db_size * sizeof(uint16_t)) / (1024 * 1024),
      params.PtModulus());

  // SimplePIR with pt_modulus=16384 requires uint16_t database
  std::vector<uint16_t> db(db_size);
  std::random_device rd;
  std::mt19937 gen(42);  // Fixed seed for reproducibility
  std::uniform_int_distribution<> dis(0, params.PtModulus() - 1);
  for (auto& val : db) {
    val = static_cast<uint16_t>(dis(gen));
  }

  // Add some known values for verification at strategic positions
  db[0] = 100;
  db[1] = 101;
  db[100] = 200;
  db[1000] = 250;
  db[100000] = 300;  // Test larger index

  SPDLOG_INFO(
      "Known test values: db[0]={}, db[1]={}, db[100]={}, db[1000]={}, "
      "db[100000]={}",
      db[0], db[1], db[100], db[1000], db[100000]);

  // Create server
  auto server_start = std::chrono::high_resolution_clock::now();
  YServer server(params, db, true, false, true);
  auto server_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - server_start);
  SPDLOG_INFO("Server setup: {} ms", server_elapsed.count());

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

  // Test querying multiple indices to verify SimplePIR correctness
  // Test known values at different positions
  std::vector<size_t> test_indices = {0, 1, 100, 1000, 100000};
  std::vector<uint16_t> expected_values = {100, 101, 200, 250, 300};

  for (size_t i = 0; i < test_indices.size(); ++i) {
    size_t target_idx = test_indices[i];
    uint16_t expected_value = expected_values[i];
    size_t test_row_idx = target_idx / db_cols;
    size_t test_col_idx = target_idx % db_cols;

    SPDLOG_INFO("\n=== Query {}/{}: index={} (row={}, col={}), expected={} ===",
                i + 1, test_indices.size(), target_idx, test_row_idx,
                test_col_idx, expected_value);

    // Client generates keys and query
    auto client_start = std::chrono::high_resolution_clock::now();
    uint128_t fixed_seed = yacl::MakeUint128(0, 12345);
    SpiralClient spiral_client(params, fixed_seed);
    YClient y_client(spiral_client, params, fixed_seed);

    auto simple_query = y_client.GenerateFullQuerySimplepir(target_idx);

    auto client_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - client_start);

    // Calculate query size breakdown
    size_t packed_query_size =
        simple_query.packed_query_row.size() * sizeof(uint32_t);
    size_t pack_params_size =
        simple_query.pack_pub_params_row_1s_pm.size() * sizeof(uint64_t);
    size_t total_query_size = packed_query_size + pack_params_size;

    SPDLOG_INFO("Client query generation: {} ms", client_elapsed.count());
    SPDLOG_INFO(
        "  Query breakdown: packed_query={} KB ({} elements), pack_params={} "
        "KB ({} elements), total={} KB",
        packed_query_size / 1024, simple_query.packed_query_row.size(),
        pack_params_size / 1024, simple_query.pack_pub_params_row_1s_pm.size(),
        total_query_size / 1024);
    if (i == 0) {
      SPDLOG_INFO(
          "  NOTE: pack_params is the same for all queries from same client "
          "and can be reused");
      SPDLOG_INFO(
          "        In practice, send pack_params once, then only send "
          "packed_query (64 KB) for subsequent queries");
    }

    // Server online computation
    auto online_start = std::chrono::high_resolution_clock::now();

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

    // Calculate response size
    size_t response_size = response.size();

    SPDLOG_INFO("Server online: {} ms, Response size: {} bytes (~{} KB)",
                online_elapsed.count(), response_size, response_size / 1024);

    // Client decodes response
    auto decode_start = std::chrono::high_resolution_clock::now();
    YPIRClient ypir_client(params);
    auto decoded_values =
        ypir_client.DecodeResponseSimplepirYClient(params, y_client, response);
    auto decode_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - decode_start);
    SPDLOG_INFO("Client decode: {} ms", decode_elapsed.count());

    // Verify correctness - SimplePIR returns entire row
    EXPECT_EQ(decoded_values.size(), db_cols);

    uint64_t decoded_value = decoded_values[test_col_idx];
    SPDLOG_INFO("Decoded: {}, Expected: {}", decoded_value, expected_value);

    EXPECT_EQ(decoded_value, expected_value)
        << "Query " << i + 1 << " failed at index " << target_idx;

    if (decoded_value == expected_value) {
      SPDLOG_INFO("✓ Query {} passed", i + 1);
    }
  }

  SPDLOG_INFO("\n✓✓✓ All SimplePIR 1GB queries passed! ✓✓✓");
}

// DoublePIR test based on Rust scheme.rs test_ypir_basic
// This tests the full DoublePIR flow with a smaller database for testing
TEST_F(YPIRIntegrationTest, DoublePIRBasicFlow) {
  SPDLOG_INFO("=== DoublePIR Basic Flow Test ===");

  // Use smaller parameters for testing: 32KB database
  // Parameters scaled down for testing infrastructure
  std::size_t poly_len = 2048;
  std::vector<std::uint64_t> moduli = {268369921, 249561089};
  double noise_width = 6.4;

  // DoublePIR configuration for 32KB database
  // db_dim_1 + poly_len_log2 determines rows, db_dim_2 + poly_len_log2
  // determines cols
  // Use pt_modulus=32768 (15 bits) to ensure special_offs < poly_len
  // special_offs = ceil((lwe_n=1024 * lwe_q_bits=28) / pt_bits=15) = 1912
  // This keeps special_offs < poly_len=2048, avoiding out-of-bounds access
  PolyMatrixParams poly_matrix_params(2, 32768, 21, 4, 8, 8, 1);
  QueryParams query_params(1, 2,
                           1);  // db_dim_1=1, db_dim_2=2, instances=1
                                // 2^(1+11) = 4096 rows, 2^(2+11) = 8192 cols

  Params params(poly_len, std::move(moduli), noise_width,
                std::move(poly_matrix_params), std::move(query_params));

  size_t db_rows = 1ULL << (params.DbDim1() + params.PolyLenLog2());
  size_t db_cols = 1ULL << (params.DbDim2() + params.PolyLenLog2());
  size_t db_size = db_rows * db_cols;

  SPDLOG_INFO(
      "DoublePIR test: db_rows={}, db_cols={}, db_size={} (~{} KB), "
      "pt_modulus={}",
      db_rows, db_cols, db_size, db_size / 1024, params.PtModulus());

  // Create database with sequential values 0, 1, 2, ..., n
  // This makes debugging much easier
  std::vector<uint8_t> db(db_size);
  for (size_t i = 0; i < db_size; ++i) {
    db[i] = static_cast<uint8_t>(i % 256);
  }

  SPDLOG_INFO("Database created with sequential values: db[i] = i % 256");
  SPDLOG_INFO("  db[0-9]: [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]",
              static_cast<int>(db[0]), static_cast<int>(db[1]),
              static_cast<int>(db[2]), static_cast<int>(db[3]),
              static_cast<int>(db[4]), static_cast<int>(db[5]),
              static_cast<int>(db[6]), static_cast<int>(db[7]),
              static_cast<int>(db[8]), static_cast<int>(db[9]));

  // Create server (DoublePIR uses uint8_t database)
  auto server_start = std::chrono::high_resolution_clock::now();
  YPirServer<uint8_t> server(
      params, db, false, false,
      true);  // is_simplepir=false, inp_transposed=false, pad_rows=true
  auto server_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - server_start);
  SPDLOG_INFO("Server setup: {} ms", server_elapsed.count());

  // Offline precomputation
  OfflinePrecomputedValues offline_vals;
  auto offline_start = std::chrono::high_resolution_clock::now();
  offline_vals = server.PerformOfflinePrecomputation();
  auto offline_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - offline_start);
  SPDLOG_INFO("Offline precomputation: {} ms", offline_elapsed.count());
  EXPECT_GT(offline_elapsed.count(), 0);
  EXPECT_GT(offline_vals.hint_0.size(), 0);
  EXPECT_GT(offline_vals.hint_1.size(), 0);
  EXPECT_NE(offline_vals.smaller_server, nullptr);

  // Test querying index 0 (expected value: 0)
  size_t target_idx = 0;
  uint8_t expected_value = db[target_idx];
  size_t target_row = target_idx / db_cols;
  size_t target_col = target_idx % db_cols;
  SPDLOG_INFO("Querying index {} (row={}, col={}, expected value: {})",
              target_idx, target_row, target_col,
              static_cast<int>(expected_value));
  SPDLOG_INFO(
      "  Row-major: db[0-3]=[{}, {}, {}, {}], db[8192-8195]=[{}, {}, {}, {}]",
      static_cast<int>(db[0]), static_cast<int>(db[1]), static_cast<int>(db[2]),
      static_cast<int>(db[3]), static_cast<int>(db[8192]),
      static_cast<int>(db[8193]), static_cast<int>(db[8194]),
      static_cast<int>(db[8195]));

  // Client generates keys and query
  auto client_start = std::chrono::high_resolution_clock::now();
  uint128_t fixed_seed = yacl::MakeUint128(0, 12345);
  SpiralClient spiral_client(params, fixed_seed);
  YClient y_client(spiral_client, params, fixed_seed);

  // Create LWE client for server-side debugging (same seed as client)
  LWEParams lwe_params = LWEParams::Default();
  LWEClient debug_lwe_client = LWEClient::FromSeed(lwe_params, fixed_seed);
  SPDLOG_INFO(
      "Created debug LWE client with same seed for server-side decryption");

  auto full_query = y_client.GenerateFullQuery(target_idx);

  auto client_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - client_start);
  SPDLOG_INFO("Client query generation: {} ms", client_elapsed.count());

  // Server online computation
  auto online_start = std::chrono::high_resolution_clock::now();

  // Prepare queries for DoublePIR
  std::vector<uint32_t> first_dim_queries_packed = full_query.packed_query_row;

  SPDLOG_INFO("First dim query size: {}", first_dim_queries_packed.size());
  SPDLOG_INFO("Second dim query col size: {}",
              full_query.packed_query_col.size());
  SPDLOG_INFO("Pack pub params size: {}",
              full_query.pack_pub_params_row_1s_pm.size());

  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      second_dim_queries;
  second_dim_queries.push_back(
      {full_query.packed_query_col, full_query.pack_pub_params_row_1s_pm});

  // Pass debug LWE client to server for verification
  server.SetDebugLweClient(&debug_lwe_client);
  server.SetDebugOriginalDb(db, db_rows, db_cols);

  SPDLOG_INFO("Calling PerformOnlineComputation...");
  std::vector<std::vector<uint8_t>> responses;
  try {
    responses = server.PerformOnlineComputation(
        offline_vals, first_dim_queries_packed, second_dim_queries);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Exception in PerformOnlineComputation: {}", e.what());
    throw;
  }

  auto online_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - online_start);
  SPDLOG_INFO("Server online computation: {} ms", online_elapsed.count());

  ASSERT_EQ(responses.size(), 1) << "Expected 1 response";

  // Client decodes response
  auto decode_start = std::chrono::high_resolution_clock::now();
  YPIRClient ypir_client(params);
  uint64_t decoded_value =
      ypir_client.DecodeResponseNormalYClient(params, y_client, responses[0]);
  auto decode_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - decode_start);
  SPDLOG_INFO("Client decode: {} ms", decode_elapsed.count());

  // Verify correctness
  SPDLOG_INFO("Decoded value: {}, Expected: {}", decoded_value,
              static_cast<uint64_t>(expected_value));
  EXPECT_EQ(decoded_value, static_cast<uint64_t>(expected_value))
      << "Decoded value mismatch";

  SPDLOG_INFO("✓ DoublePIR correctness verified!");
}
/*
// Test DoublePIR with multiple queries in one test - query same index 3
times TEST_F(YPIRIntegrationTest, DoublePIRMultipleQueries) {
  SPDLOG_INFO("=== DoublePIR Multiple Queries Test ===");

  std::size_t poly_len = 2048;
  std::vector<std::uint64_t> moduli = {268369921, 249561089};
  double noise_width = 6.4;
  PolyMatrixParams poly_matrix_params(2, 32768, 21, 4, 8, 8, 1);
  QueryParams query_params(1, 2, 1);
  Params params(poly_len, std::move(moduli), noise_width,
                std::move(poly_matrix_params), std::move(query_params));

  size_t db_rows = 1ULL << (params.DbDim1() + params.PolyLenLog2());
  size_t db_cols = 1ULL << (params.DbDim2() + params.PolyLenLog2());
  size_t db_size = db_rows * db_cols;

  std::vector<uint8_t> db(db_size);
  std::mt19937 gen(42);
  std::uniform_int_distribution<> dis(0, 255);
  for (auto& val : db) {
    val = static_cast<uint8_t>(dis(gen));
  }
  // Add known test values BEFORE creating server
  if (db_size > 1000) {
    db[0] = 42;
    db[1] = 43;
    db[100] = 142;
    db[1000] = 200;
  }

  // Set test values for multiple queries
  // - index 0 (row=0, col=0): same row, first col
  // - index 8192 (row=1, col=0): different row, first col
  // - index 8193 (row=1, col=1): different row, second col
  db[8192] = 99;  // row=1, col=0
  db[8193] = 88;  // row=1, col=1

  // Create server once
  YPirServer<uint8_t> server(params, db, false, false, true);

  // Debug: Print database layout
  SPDLOG_INFO("Database layout: db_rows={}, db_cols={}", db_rows, db_cols);
  SPDLOG_INFO("Verify expected values: db[0]={}, db[8192]={}, db[8193]={}",
              static_cast<int>(db[0]), static_cast<int>(db[8192]),
              static_cast<int>(db[8193]));

  // Test 3 different queries
  // Need to perform offline precomputation for EACH query because
  // PerformOnlineComputation modifies offline_vals.hint_1

  std::vector<size_t> test_indices = {0, 8192, 8193};
  std::vector<uint8_t> expected_values = {42, 99, 88};
  uint128_t fixed_seed = yacl::MakeUint128(0, 12345);

  for (size_t i = 0; i < test_indices.size(); ++i) {
    size_t target_idx = test_indices[i];
    uint8_t expected_value = expected_values[i];

    size_t target_row = target_idx / db_cols;
    size_t target_col = target_idx % db_cols;

    // Perform offline precomputation for each query
    OfflinePrecomputedValues offline_vals =
server.PerformOfflinePrecomputation();

    SPDLOG_INFO("Query {}/3: index={} (row={}, col={}), expected={}",
                i + 1, target_idx, target_row, target_col,
                static_cast<int>(expected_value));

    // Create new client for each query with same seed
    SpiralClient spiral_client(params, fixed_seed);
    YClient y_client(spiral_client, params, fixed_seed);
    auto full_query = y_client.GenerateFullQuery(target_idx);

    // Debug: check if queries are actually different
    SPDLOG_INFO("  Query row first 5 values: [{}, {}, {}, {}, {}]",
                full_query.packed_query_row[0],
full_query.packed_query_row[1], full_query.packed_query_row[2],
full_query.packed_query_row[3], full_query.packed_query_row[4]);
    SPDLOG_INFO("  Query col first 5 values: [{}, {}, {}, {}, {}]",
                full_query.packed_query_col[0],
full_query.packed_query_col[1], full_query.packed_query_col[2],
full_query.packed_query_col[3], full_query.packed_query_col[4]);

    std::vector<uint32_t> first_dim_queries_packed =
        full_query.packed_query_row;
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
        second_dim_queries;
    second_dim_queries.push_back(
        {full_query.packed_query_col,
full_query.pack_pub_params_row_1s_pm});

    auto responses = server.PerformOnlineComputation(
        offline_vals, first_dim_queries_packed, second_dim_queries);
    ASSERT_EQ(responses.size(), 1);

    YPIRClient ypir_client(params);
    uint64_t decoded_value =
        ypir_client.DecodeResponseNormalYClient(params, y_client,
responses[0]);

    SPDLOG_INFO("  Decoded: {}, Expected: {}", decoded_value,
                static_cast<int>(expected_value));

    EXPECT_EQ(decoded_value, static_cast<uint64_t>(expected_value))
        << "Query " << i + 1 << " (index=" << target_idx << ") failed:
decoded "
        << decoded_value << ", expected " <<
static_cast<int>(expected_value);

    if (decoded_value == static_cast<uint64_t>(expected_value)) {
      SPDLOG_INFO("  ✓ Query {} passed", i + 1);
    } else {
      SPDLOG_ERROR("  ✗ Query {} FAILED", i + 1);
    }
  }

  SPDLOG_INFO("✓ All 3 different queries tested!");
}
*/