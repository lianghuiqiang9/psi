// Copyright 2024 Ant Group Co., Ltd.
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

#include "psi/algorithm/phepir/phe_pir.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "yacl/base/int128.h"
#include "yacl/crypto/ecc/ec_point.h"
#include "yacl/crypto/ecc/ecc_spi.h"
#include "yacl/crypto/hash/hash_utils.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/math/mpint/mp_int.h"

#include "psi/algorithm/phepir/phe_pir_utils.h"

namespace psi::phepir {

using yacl::math::MPInt;
// Helper: polynomial evaluation (Horner's rule)
MPInt EvaluatePolynomial(const std::vector<MPInt>& coeffs, const MPInt& x,
                         const MPInt& modulus) {
  if (coeffs.empty()) return MPInt(0);
  MPInt res = MPInt(0);
  for (int i = coeffs.size() - 1; i >= 0; --i) {
    res = (res * x + coeffs[i]) % modulus;
  }
  return res;
}

// Test core math logic: interpolating polynomial P(x) and vanishing polynomial
// Q(x)
TEST(PhePirTest, PolynomialMathTest) {
  auto ec_group = yacl::crypto::EcGroupFactory::Instance().Create(
      "sm2", yacl::ArgLib = "openssl");
  auto order = ec_group->GetOrder();

  std::vector<MPInt> keys = {MPInt(0), MPInt(1), MPInt(2)};
  std::vector<MPInt> values = {MPInt(5), MPInt(9), MPInt(15)};

  auto result_pair = GetInterpolatingAndRootPolyCoeffs(keys, values, order);
  const auto& p_coeffs = result_pair.first;
  const auto& q_coeffs = result_pair.second;

  // 1. Verify P(x): P(key) = value
  ASSERT_EQ(p_coeffs.size(), 3);
  EXPECT_EQ(EvaluatePolynomial(p_coeffs, keys[0], order),
            values[0]);  // P(0) = 5
  EXPECT_EQ(EvaluatePolynomial(p_coeffs, keys[1], order),
            values[1]);  // P(1) = 9
  EXPECT_EQ(EvaluatePolynomial(p_coeffs, keys[2], order),
            values[2]);  // P(2) = 15

  // 2. Verify Q(x): Q(key) = 0
  ASSERT_EQ(q_coeffs.size(), 4);  // 3个根，应该是3次多项式，系数4个
  EXPECT_EQ(EvaluatePolynomial(q_coeffs, keys[0], order), MPInt(0));
  EXPECT_EQ(EvaluatePolynomial(q_coeffs, keys[1], order), MPInt(0));
  EXPECT_EQ(EvaluatePolynomial(q_coeffs, keys[2], order), MPInt(0));

  // 3. Verify non-root point: Q(3) != 0
  EXPECT_NE(EvaluatePolynomial(q_coeffs, MPInt(3), order), MPInt(0));
}

TEST(PhePirTest, EndToEndHitTest) {
  PhePirOptions options;
  options.n = 100;
  options.element_size = 4;

  PhePirServer server(options);
  PhePirClient client(options);

  size_t n = options.n;
  std::vector<std::string> raw_keys(n);
  std::vector<yacl::ByteContainerView> keys_view(n);
  std::vector<uint32_t> value_data(n);
  std::vector<yacl::ByteContainerView> values_view(n);

  for (size_t i = 0; i < n; ++i) {
    raw_keys[i] = "key_" + std::to_string(i);
    keys_view[i] = yacl::ByteContainerView(raw_keys[i]);
    value_data[i] = i * 127 + 7;
    values_view[i] = yacl::ByteContainerView(&value_data[i], sizeof(uint32_t));
  }

  // 1. Server Setup
  server.GenerateFromRawKeyValueData(keys_view, values_view);

  // 2. Client generates public key
  auto pks_buf = client.GeneratePksBuffer();

  // 3. Build queries (multiple elements: head, middle, tail)
  std::vector<size_t> query_indices = {0, 50, 99};
  for (size_t idx : query_indices) {
    std::string target_raw_key = "key_" + std::to_string(idx);
    auto query_buf = client.GenerateQuery(target_raw_key);

    // 4. Server response
    auto response_buf = server.Response(query_buf, pks_buf);

    // 5. Client decodes
    auto result_bytes = client.DecodeResponse(response_buf);

    // Verify result
    ASSERT_EQ(result_bytes.size(), options.element_size);
    uint32_t result_val = 0;
    memcpy(&result_val, result_bytes.data(), sizeof(uint32_t));

    uint32_t expected_val = value_data[idx];
    EXPECT_EQ(result_val, expected_val) << "Mismatch at index " << idx;
  }
}

// End-to-end miss test
TEST(PhePirTest, EndToEndMissTest) {
  PhePirOptions options;
  options.n = 16;
  options.element_size = 4;

  PhePirServer server(options);
  PhePirClient client(options);

  // Build a simple DB
  size_t n = options.n;
  std::vector<std::string> raw_keys(n);
  std::vector<yacl::ByteContainerView> keys_view(n);
  std::vector<uint32_t> value_data(n);
  std::vector<yacl::ByteContainerView> values_view(n);

  for (size_t i = 0; i < n; ++i) {
    raw_keys[i] = std::to_string(i);  // keys: "0", "1", ...
    keys_view[i] = yacl::ByteContainerView(raw_keys[i]);

    value_data[i] = i;
    values_view[i] = yacl::ByteContainerView(&value_data[i], sizeof(uint32_t));
  }
  server.GenerateFromRawKeyValueData(keys_view, values_view);

  auto pks_buf = client.GeneratePksBuffer();

  // Query a non-existing key
  std::string target_raw_key = "not_exist_key";
  auto query_buf = client.GenerateQuery(target_raw_key);
  auto response_buf = server.Response(query_buf, pks_buf);

  // Decode
  auto result_bytes = client.DecodeResponse(response_buf);

  // Verify: Miss should return an empty vector
  EXPECT_TRUE(result_bytes.empty());
}

// Test special zero value (Value is Zero)
// Verify the system can distinguish "Key does not exist" and "Key exists but
// Value is 0"
TEST(PhePirTest, ValueZeroTest) {
  PhePirOptions options;
  options.n = 4;
  options.element_size = 4;

  PhePirServer server(options);
  PhePirClient client(options);

  // DB: key="A"->val=10, key="B"->val=0 (use smaller values to ensure DLP
  // solvable)
  std::vector<std::string> raw_keys = {"A", "B", "C", "D"};
  std::vector<uint32_t> raw_vals = {100, 0, 200, 300};

  std::vector<yacl::ByteContainerView> keys_view(4);
  std::vector<yacl::ByteContainerView> values_view(4);

  for (int i = 0; i < 4; ++i) {
    keys_view[i] = yacl::ByteContainerView(raw_keys[i]);
    values_view[i] = yacl::ByteContainerView(&raw_vals[i], sizeof(uint32_t));
  }
  server.GenerateFromRawKeyValueData(keys_view, values_view);
  auto pks_buf = client.GeneratePksBuffer();

  // Case 1: Query the key with value 0 ("B")
  {
    auto query_buf = client.GenerateQuery("B");
    auto response_buf = server.Response(query_buf, pks_buf);
    auto result_bytes = client.DecodeResponse(response_buf);

    // Should return 4 bytes representing 0
    ASSERT_EQ(result_bytes.size(), 4);
    uint32_t val = 0;
    memcpy(&val, result_bytes.data(), 4);
    EXPECT_EQ(val, 0);
  }

  // Case 2: Query a non-existing key ("E")
  {
    auto query_buf = client.GenerateQuery("E");
    auto response_buf = server.Response(query_buf, pks_buf);
    auto result_bytes = client.DecodeResponse(response_buf);

    EXPECT_TRUE(result_bytes.empty());
  }
}

// Corner case: very small dataset
TEST(PhePirTest, SmallDataTest) {
  PhePirOptions options;
  options.n = 1;
  options.element_size = 4;

  PhePirServer server(options);
  PhePirClient client(options);

  // Setup single KV
  std::string raw_key = "lonely_key";
  uint32_t raw_val = 999;

  yacl::ByteContainerView k_view(raw_key);
  yacl::ByteContainerView v_view(&raw_val, sizeof(uint32_t));

  server.GenerateFromRawKeyValueData({k_view}, {v_view});

  auto pks_buf = client.GeneratePksBuffer();
  auto query_buf = client.GenerateQuery(raw_key);
  auto response_buf = server.Response(query_buf, pks_buf);
  auto result_bytes = client.DecodeResponse(response_buf);

  ASSERT_EQ(result_bytes.size(), 4);
  uint32_t res_val;
  memcpy(&res_val, result_bytes.data(), 4);
  EXPECT_EQ(res_val, 999);
}

// Large dataset test
// Note: he_sm2 decryption relies on DLP solving (BSGS), which only supports
// relatively short messages (approx 24-28 bits)
TEST(PhePirTest, LargeDataTest) {
  PhePirOptions options;
  options.n = 65537;  // large-scale test n=65537
  options.element_size = 4;

  PhePirServer server(options);
  PhePirClient client(options);

  size_t n = options.n;
  std::vector<std::string> raw_keys(n);
  std::vector<yacl::ByteContainerView> keys_view(n);
  std::vector<std::vector<uint8_t>> value_data(n);
  std::vector<yacl::ByteContainerView> values_view(n);

  for (size_t i = 0; i < n; ++i) {
    // Construct unique key
    raw_keys[i] = "large_key_" + std::to_string(i);

    // View Raw Key
    keys_view[i] = yacl::ByteContainerView(raw_keys[i]);

    // Deterministic value: use a small range to ensure it is solvable by he_sm2
    // DLP he_sm2's BSGS decryption has message size limitations (approx 2^24 ~
    // 2^28) use i % 1000000 to keep values within a safe range
    uint32_t val =
        (i * 997) % 1000000;  // use prime 997 to increase randomness, mod
                              // 1000000 keeps values safe
    value_data[i].resize(4);
    memcpy(value_data[i].data(), &val, 4);
    values_view[i] = yacl::ByteContainerView(value_data[i]);
  }

  // 1. Server init
  server.GenerateFromRawKeyValueData(keys_view, values_view);

  // 2. Client PKs
  auto pks_buf = client.GeneratePksBuffer();

  // 3. Verify key points (boundaries + deterministic samples to avoid
  // randomness)
  std::vector<size_t> test_indices = {0, 1, n / 2, n - 1};

  for (size_t target_idx : test_indices) {
    auto start = std::chrono::high_resolution_clock::now();
    auto query_buf = client.GenerateQuery(raw_keys[target_idx]);
    auto response_buf = server.Response(query_buf, pks_buf);
    SPDLOG_INFO("Query index size: {} bytes", query_buf.size());
    SPDLOG_INFO("Response size for index {}: {} bytes", target_idx,
                response_buf.size());
    auto result_bytes = client.DecodeResponse(response_buf);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "Query index " << target_idx << " took " << duration << " ms"
              << std::endl;
    ASSERT_EQ(result_bytes.size(), options.element_size);
    EXPECT_EQ(result_bytes, value_data[target_idx])
        << "Mismatch at index " << target_idx;
  }

  // 4. Verify a non-existing point
  {
    auto query_buf = client.GenerateQuery("key_not_exist_unique_string");
    auto response_buf = server.Response(query_buf, pks_buf);
    auto result_bytes = client.DecodeResponse(response_buf);
    EXPECT_TRUE(result_bytes.empty());
  }
}

TEST(PhePirTest, LargeValueTest) {
  PhePirOptions options;
  options.n = 3;
  options.element_size = 4;

  PhePirServer server(options);
  PhePirClient client(options);

  std::vector<std::string> raw_keys = {"key1", "key2", "key3"};
  std::vector<uint32_t> raw_vals = {
      16777215,  // 2^24 - 1
      8388607,   // 2^23 - 1
      1048575    // 2^20 - 1
  };

  std::vector<yacl::ByteContainerView> keys_view(3);
  std::vector<yacl::ByteContainerView> values_view(3);

  for (int i = 0; i < 3; ++i) {
    keys_view[i] = yacl::ByteContainerView(raw_keys[i]);
    values_view[i] = yacl::ByteContainerView(&raw_vals[i], sizeof(uint32_t));
  }

  server.GenerateFromRawKeyValueData(keys_view, values_view);
  auto pks_buf = client.GeneratePksBuffer();

  for (size_t i = 0; i < 3; ++i) {
    auto query_buf = client.GenerateQuery(raw_keys[i]);
    auto response_buf = server.Response(query_buf, pks_buf);
    auto result_bytes = client.DecodeResponse(response_buf);

    ASSERT_EQ(result_bytes.size(), 4);
    uint32_t val = 0;
    memcpy(&val, result_bytes.data(), 4);
    EXPECT_EQ(val, raw_vals[i]) << "Failed for key: " << raw_keys[i];
  }
}

TEST(PhePirTest, VariableElementSizeTest) {
  for (size_t elem_size : {2, 3, 4}) {
    PhePirOptions options;
    options.n = 5;
    options.element_size = elem_size;

    PhePirServer server(options);
    PhePirClient client(options);

    std::vector<std::string> raw_keys = {"k1", "k2", "k3", "k4", "k5"};
    std::vector<std::vector<uint8_t>> value_data(5);
    std::vector<yacl::ByteContainerView> keys_view(5);
    std::vector<yacl::ByteContainerView> values_view(5);

    for (size_t i = 0; i < 5; ++i) {
      keys_view[i] = yacl::ByteContainerView(raw_keys[i]);

      uint32_t val = (i + 1) * 100;
      value_data[i].resize(elem_size);
      for (size_t j = 0; j < elem_size; ++j) {
        value_data[i][j] = (val >> (j * 8)) & 0xFF;
      }
      values_view[i] = yacl::ByteContainerView(value_data[i]);
    }

    server.GenerateFromRawKeyValueData(keys_view, values_view);
    auto pks_buf = client.GeneratePksBuffer();

    for (size_t idx : {0, 4}) {
      auto query_buf = client.GenerateQuery(raw_keys[idx]);
      auto response_buf = server.Response(query_buf, pks_buf);
      auto result_bytes = client.DecodeResponse(response_buf);

      ASSERT_EQ(result_bytes.size(), elem_size)
          << "Failed for element_size=" << elem_size;
      EXPECT_EQ(result_bytes, value_data[idx])
          << "Value mismatch for element_size=" << elem_size;
    }
  }
}

// Test single-byte values (element_size=1)
TEST(PhePirTest, SingleByteValueTest) {
  PhePirOptions options;
  options.n = 6;
  options.element_size = 1;

  PhePirServer server(options);
  PhePirClient client(options);

  std::vector<std::string> raw_keys = {"kb1", "kb2", "kb3",
                                       "kb4", "kb5", "kb6"};
  std::vector<uint8_t> raw_vals = {0, 1, 50, 100, 120, 127};

  std::vector<yacl::ByteContainerView> keys_view(6);
  std::vector<std::vector<uint8_t>> value_data(6);
  std::vector<yacl::ByteContainerView> values_view(6);

  for (size_t i = 0; i < 6; ++i) {
    keys_view[i] = yacl::ByteContainerView(raw_keys[i]);

    value_data[i].resize(1);
    value_data[i][0] = raw_vals[i];
    values_view[i] = yacl::ByteContainerView(value_data[i]);
  }

  server.GenerateFromRawKeyValueData(keys_view, values_view);
  auto pks_buf = client.GeneratePksBuffer();

  for (size_t idx = 0; idx < 6; ++idx) {
    auto query_buf = client.GenerateQuery(raw_keys[idx]);
    auto response_buf = server.Response(query_buf, pks_buf);
    auto result_bytes = client.DecodeResponse(response_buf);

    ASSERT_EQ(result_bytes.size(), 1) << "Failed for idx=" << idx;
    EXPECT_EQ(result_bytes[0], raw_vals[idx])
        << "Value mismatch for idx=" << idx
        << ", expected=" << static_cast<int>(raw_vals[idx])
        << ", got=" << static_cast<int>(result_bytes[0]);
  }
}

TEST(PhePirTest, MediumScalePerformanceTest) {
  PhePirOptions options;
  options.n = 10000;
  options.element_size = 4;

  PhePirServer server(options);
  PhePirClient client(options);

  size_t n = options.n;
  std::vector<std::string> raw_keys(n);
  std::vector<yacl::ByteContainerView> keys_view(n);
  std::vector<std::vector<uint8_t>> value_data(n);
  std::vector<yacl::ByteContainerView> values_view(n);

  for (size_t i = 0; i < n; ++i) {
    raw_keys[i] = "perf_key_" + std::to_string(i);

    keys_view[i] = yacl::ByteContainerView(raw_keys[i]);

    uint32_t val = (i * 13) % 100000;
    value_data[i].resize(4);
    memcpy(value_data[i].data(), &val, 4);
    values_view[i] = yacl::ByteContainerView(value_data[i]);
  }

  server.GenerateFromRawKeyValueData(keys_view, values_view);
  auto pks_buf = client.GeneratePksBuffer();

  std::vector<size_t> test_indices = {0, 100, 500, 999};
  for (size_t target_idx : test_indices) {
    auto query_buf = client.GenerateQuery(raw_keys[target_idx]);
    auto response_buf = server.Response(query_buf, pks_buf);
    auto result_bytes = client.DecodeResponse(response_buf);

    ASSERT_EQ(result_bytes.size(), options.element_size);
    EXPECT_EQ(result_bytes, value_data[target_idx])
        << "Mismatch at index " << target_idx;
  }
}

TEST(PhePirTest, MultipleQueriesTest) {
  PhePirOptions options;
  options.n = 10;
  options.element_size = 4;

  PhePirServer server(options);
  PhePirClient client(options);

  // Setup database
  std::vector<std::string> raw_keys(10);
  std::vector<uint32_t> raw_vals(10);
  std::vector<yacl::ByteContainerView> keys_view(10);
  std::vector<yacl::ByteContainerView> values_view(10);

  for (size_t i = 0; i < 10; ++i) {
    raw_keys[i] = "multi_key_" + std::to_string(i);
    raw_vals[i] = i * 111;

    keys_view[i] = yacl::ByteContainerView(raw_keys[i]);
    values_view[i] = yacl::ByteContainerView(&raw_vals[i], sizeof(uint32_t));
  }

  server.GenerateFromRawKeyValueData(keys_view, values_view);
  auto pks_buf = client.GeneratePksBuffer();

  // Perform multiple rounds of queries to verify state does not interfere
  for (int round = 0; round < 3; ++round) {
    for (size_t i = 0; i < 10; ++i) {
      auto query_buf = client.GenerateQuery(raw_keys[i]);
      auto response_buf = server.Response(query_buf, pks_buf);
      auto result_bytes = client.DecodeResponse(response_buf);

      ASSERT_EQ(result_bytes.size(), 4);
      uint32_t val = 0;
      memcpy(&val, result_bytes.data(), 4);
      EXPECT_EQ(val, raw_vals[i])
          << "Round " << round << ", index " << i << " failed";
    }
  }
}

}  // namespace psi::phepir
