// Copyright 2025 Ant Group Co., Ltd.
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

#include <atomic>

#include "hesm2/config.h"
#include "yacl/crypto/ecc/ec_point.h"
#include "yacl/crypto/ecc/ecc_spi.h"
#include "yacl/crypto/hash/hash_utils.h"
#include "yacl/utils/parallel.h"

#include "psi/algorithm/phepir/phe_pir_utils.h"

namespace psi::phepir {

using examples::hesm2::Ciphertext;
using examples::hesm2::Decrypt;
using examples::hesm2::HAdd;
using examples::hesm2::HMul;
using examples::hesm2::PrivateKey;
using examples::hesm2::PublicKey;
using yacl::crypto::EcGroupFactory;

PhePirClient::PhePirClient(const PhePirOptions& options)
    : options_(options),
      ec_group_(
          EcGroupFactory::Instance().Create("sm2", yacl::ArgLib = "openssl")),
      private_key_(std::make_unique<PrivateKey>(ec_group_)),
      public_key_(private_key_->GetPublicKey()) {
  examples::hesm2::InitializeConfig();
  n_ = options_.n;
  block_size_ = std::ceil(std::sqrt(n_));
  num_blocks_ = (n_ + block_size_ - 1) / block_size_;
}

yacl::Buffer PhePirClient::GeneratePksBuffer() const {
  return public_key_.Serialize();
}

std::string PhePirClient::GeneratePksString() const {
  auto buf = GeneratePksBuffer();
  return std::string(buf.data<char>(), buf.size());
}

yacl::Buffer PhePirClient::GenerateQuery(
    yacl::ByteContainerView keyword) const {
  // Build an encrypted vector of powers for the hashed keyword base:
  // ct[i] = Enc(base^{i+1}) for i in [0, block_size_-1].
  // The server will use these as x^j in polynomial evaluation.
  auto keyword_hash = yacl::crypto::Sm3(keyword);
  yacl::math::MPInt base;
  base.Deserialize(keyword_hash);

  auto order = ec_group_->GetOrder();
  base = base.Mod(order);

  std::vector<yacl::math::MPInt> powers;
  powers.reserve(block_size_);
  powers.emplace_back(base);
  for (uint32_t i = 1; i < block_size_; ++i) {
    auto temp = powers.back() * base;
    temp = temp.Mod(order);
    powers.emplace_back(temp);
  }

  std::vector<Ciphertext> ct(block_size_,
                             Encrypt(yacl::math::MPInt::_1_, public_key_));
  yacl::parallel_for(0, block_size_, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      ct[i] = Encrypt(powers[i], public_key_);
    }
  });

  return SerializeCiphertexts(ct, public_key_);
}

std::string PhePirClient::GenerateQueryStr(
    yacl::ByteContainerView keyword) const {
  auto buf = GenerateQuery(keyword);
  return std::string(buf.data<char>(), buf.size());
}

std::vector<uint8_t> PhePirClient::DecodeResponse(
    const yacl::ByteContainerView& response_buffer) const {
  auto response_vec = DeserializeCiphertexts(response_buffer, public_key_);
  YACL_ENFORCE_EQ(
      response_vec.size(), 2 * num_blocks_,
      "Response size mismatch. Expected 2 * num_blocks (Payloads + Checks).");

  std::atomic<int> found_idx(-1);
  yacl::parallel_for(0, num_blocks_, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      if (found_idx.load() != -1) continue;
      const auto& check_ct = response_vec[num_blocks_ + i];
      auto check_res = ZeroCheck(check_ct, *private_key_);

      if (check_res.success && check_res.m == 0) {
        found_idx.store(i);
      }
    }
  });

  if (found_idx != -1) {
    const auto& payload_ct = response_vec[found_idx];
    auto actual_ans = Decrypt(payload_ct, *private_key_);
    YACL_ENFORCE(actual_ans.success, "Payload decryption failed");

    size_t bytes = options_.element_size;
    if (bytes == 0) bytes = (actual_ans.m.BitCount() + 7) / 8;

    std::vector<uint8_t> result(bytes);
    actual_ans.m.ToBytes(result.data(), bytes);
    return result;
  } else {
    return {};
  }
}

PhePirServer::PhePirServer(const PhePirOptions& options)
    : options_(options),
      ec_group_(
          EcGroupFactory::Instance().Create("sm2", yacl::ArgLib = "openssl")) {
  n_ = options_.n;
  block_size_ = std::ceil(std::sqrt(n_));
  num_blocks_ = (n_ + block_size_ - 1) / block_size_;
}

void PhePirServer::GenerateFromRawKeyValueData(
    const std::vector<yacl::ByteContainerView>& keys,
    const std::vector<yacl::ByteContainerView>& values) {
  auto order = ec_group_->GetOrder();

  std::vector<yacl::math::MPInt> db_keys;
  std::vector<yacl::math::MPInt> db_values;
  db_keys.reserve(n_);
  db_values.reserve(n_);

  yacl::math::MPInt key_tmp, value_tmp;
  for (size_t i = 0; i < n_; ++i) {
    auto key_hash = yacl::crypto::Sm3(keys[i]);
    key_tmp.Deserialize(key_hash);
    db_keys.emplace_back(key_tmp.Mod(order));  // reduce key into group order

    value_tmp.Deserialize(values[i]);
    db_values.emplace_back(
        value_tmp.Mod(order));  // reduce value into group order
  }

  block_size_ = std::ceil(std::sqrt(n_));
  num_blocks_ = (n_ + block_size_ - 1) / block_size_;

  interp_coeffs_.assign(num_blocks_ * block_size_, yacl::math::MPInt(0));
  vanishing_coeffs_.assign(num_blocks_ * (block_size_ + 1),
                           yacl::math::MPInt(0));

  yacl::parallel_for(0, num_blocks_, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      size_t start_idx = b * block_size_;
      size_t end_idx = std::min(start_idx + block_size_, n_);

      if (start_idx >= end_idx) continue;

      std::vector<yacl::math::MPInt> sub_keys(db_keys.begin() + start_idx,
                                              db_keys.begin() + end_idx);
      std::vector<yacl::math::MPInt> sub_values(db_values.begin() + start_idx,
                                                db_values.begin() + end_idx);

      // Compute interpolation polynomial P_b(x) and vanishing polynomial Q_b(x)
      // for this block.
      auto result_pair =
          GetInterpolatingAndRootPolyCoeffs(sub_keys, sub_values, order);
      const auto& interp_coeffs_block = result_pair.first;
      const auto& vanishing_coeffs_block = result_pair.second;

      size_t p_offset = b * block_size_;
      for (size_t k = 0; k < interp_coeffs_block.size(); ++k) {
        interp_coeffs_[p_offset + k] = interp_coeffs_block[k];
      }
      size_t q_offset = b * (block_size_ + 1);
      for (size_t k = 0; k < vanishing_coeffs_block.size(); ++k) {
        vanishing_coeffs_[q_offset + k] = vanishing_coeffs_block[k];
      }
    }
  });

  db_seted_ = true;
}

void PhePirServer::Dump(std::ostream&) const {}

bool PhePirServer::DbSeted() const { return db_seted_; }

yacl::Buffer PhePirServer::Response(const yacl::ByteContainerView& query_buffer,
                                    const yacl::Buffer& pks_buffer) const {
  YACL_ENFORCE(db_seted_,
               "DB not set. Please call GenerateFromRawKeyValueData first.");

  PublicKey public_key =
      examples::hesm2::PublicKey::Deserialize(pks_buffer, ec_group_);

  auto ct_x = DeserializeCiphertexts(query_buffer, public_key);

  YACL_ENFORCE_GE(ct_x.size(), block_size_,
                  "Query vector size is too small. Client must send powers up "
                  "to block_size.");

  std::vector<Ciphertext> response_vec(
      2 * num_blocks_, Encrypt(yacl::math::MPInt::_1_, public_key));
  yacl::parallel_for(0, num_blocks_, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      {
        size_t p_offset = b * block_size_;

        // Optimization 1: encrypt the constant term c0 directly as the initial
        // value to avoid an extra homomorphic addition
        const auto& c0 = interp_coeffs_[p_offset];
        Ciphertext interp_val = Encrypt(c0, public_key);

        // Accumulate: + c_j * x^j
        // P(x) highest degree is x^{block_size_ - 1}
        for (size_t j = 1; j < block_size_; ++j) {
          const auto& coeff = interp_coeffs_[p_offset + j];

          // Optimization 2: sparse optimization. If coefficient is zero,
          // skip the expensive homomorphic multiplication.
          if (coeff != yacl::math::MPInt(0)) {
            // ct_x[j-1] corresponds to x^j
            auto term = HMul(ct_x[j - 1], coeff, public_key);
            interp_val = HAdd(interp_val, term, public_key);
          }
        }
        // store into the first half (payloads)
        response_vec[b] = std::move(interp_val);
      }

      // =================================================================
      // Part B: compute vanishing polynomial Q(x) (membership check)
      // Goal: vanishing_val = Enc(Q_b(x))
      // Coefficients stored in vanishing_coeffs_ (stride = block_size_ + 1)
      // =================================================================
      {
        size_t q_offset = b * (block_size_ + 1);

        const auto& c0 = vanishing_coeffs_[q_offset];
        Ciphertext vanishing_val = Encrypt(c0, public_key);

        // Accumulate: + c_j * x^j
        // Q(x) highest degree is x^{block_size_}, one degree higher than P(x)
        for (size_t j = 1; j <= block_size_; ++j) {
          // Additional bounds check for safety, despite YACL_ENFORCE earlier
          if (j - 1 >= ct_x.size()) break;

          const auto& coeff = vanishing_coeffs_[q_offset + j];

          if (coeff != yacl::math::MPInt(0)) {
            auto term = HMul(ct_x[j - 1], coeff, public_key);
            vanishing_val = HAdd(vanishing_val, term, public_key);
          }
        }
        // store into the latter half (membership checks)
        response_vec[num_blocks_ + b] = std::move(vanishing_val);
      }
    }
  });
  // 5. Serialize and return the response vector (payloads + checks)
  return SerializeCiphertexts(response_vec, public_key);
}

std::string PhePirServer::Response(const yacl::ByteContainerView& query_buffer,
                                   const std::string& pks_buffer) const {
  yacl::Buffer pks(pks_buffer.data(), pks_buffer.size());
  auto res = Response(query_buffer, pks);
  return std::string(res.data<char>(), res.size());
}

}  // namespace psi::phepir
