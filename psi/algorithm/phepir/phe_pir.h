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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "hesm2/ahesm2.h"
#include "hesm2/public_key.h"

#include "psi/algorithm/pir_interface/pir_db.h"

namespace psi::phepir {

struct PhePirOptions {
  uint32_t n = 1024;
  uint32_t element_size = 0;
};

class PhePirClient {
 public:
  explicit PhePirClient(const PhePirOptions& options);
  ~PhePirClient() = default;

  yacl::Buffer GeneratePksBuffer() const;
  std::string GeneratePksString() const;

  yacl::Buffer GenerateQuery(yacl::ByteContainerView keyword) const;
  std::string GenerateQueryStr(yacl::ByteContainerView keyword) const;

  std::vector<uint8_t> DecodeResponse(
      const yacl::ByteContainerView& response_buffer) const;

 private:
  PhePirOptions options_;
  std::shared_ptr<yacl::crypto::EcGroup> ec_group_;
  std::unique_ptr<examples::hesm2::PrivateKey> private_key_;
  examples::hesm2::PublicKey public_key_;

  size_t n_;
  size_t block_size_;
  size_t num_blocks_;
};

class PhePirServer {
 public:
  explicit PhePirServer(const PhePirOptions& options);
  ~PhePirServer() = default;

  void GenerateFromRawKeyValueData(
      const std::vector<yacl::ByteContainerView>& keys,
      const std::vector<yacl::ByteContainerView>& values);

  void Dump(std::ostream& out_stream) const;

  bool DbSeted() const;

  yacl::Buffer Response(const yacl::ByteContainerView& query_buffer,
                        const yacl::Buffer& pks_buffer) const;
  std::string Response(const yacl::ByteContainerView& query_buffer,
                       const std::string& pks_buffer) const;

 private:
  PhePirOptions options_;
  std::shared_ptr<yacl::crypto::EcGroup> ec_group_;
  std::vector<yacl::math::MPInt> interp_coeffs_;
  std::vector<yacl::math::MPInt> vanishing_coeffs_;

  bool db_seted_ = false;
  size_t n_;
  size_t block_size_;
  size_t num_blocks_;
};

}  // namespace psi::phepir
