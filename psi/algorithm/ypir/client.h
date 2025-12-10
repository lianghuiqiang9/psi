#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "psi/algorithm/spiral/discrete_gaussian.h"
#include "psi/algorithm/spiral/poly_matrix.h"
#include "psi/algorithm/spiral/spiral_client.h"
#include "psi/algorithm/ypir/params.h"

namespace psi::ypir {

using namespace psi::spiral;

class LWEClient {
 private:
  LWEParams lwe_params_;
  std::vector<uint32_t> sk_;

 public:
  LWEClient(LWEParams params, std::vector<uint32_t> sk);
  static LWEClient Init(LWEParams params);
  static LWEClient FromSeed(const LWEParams& lwe_params, const uint128_t seed);
  const std::vector<uint32_t> GetSk() const { return sk_; }
  std::vector<uint32_t> Encrypt(yacl::crypto::Prg<uint64_t>& rng, uint32_t pt);
  std::vector<uint32_t> EncryptMany(yacl::crypto::Prg<uint32_t>& rng,
                                    std::vector<uint32_t> v_pt);
  uint32_t Decrypt(const std::vector<uint32_t>& ct) const;
  const LWEParams& LweParams() const { return lwe_params_; }
};

// (packed_query_row_u32, packed_query_col, pack_pub_params_row_1s_pm)
struct YPIRQuery {
  std::vector<uint32_t> packed_query_row;
  std::vector<uint64_t> packed_query_col;
  std::vector<uint64_t> pack_pub_params_row_1s_pm;
};

struct YPIRSimpleQuery {
  std::vector<uint64_t> packed_query_row;
  std::vector<uint64_t> pack_pub_params_row_1s_pm;
};

class YClient {
 private:
  LWEClient lwe_client_;
  SpiralClient spiral_client_;
  const Params params_;

  std::vector<uint64_t> RlwesToLwes(
      const std::vector<PolyMatrixRaw>& ct) const;

  std::vector<uint64_t> GenerateQueryLweLowMem(uint8_t public_seed_idx,
                                               size_t dim_log2, bool packing,
                                               size_t index_row);

 public:
  YClient(const SpiralClient& client, const Params& params);
  YClient(const SpiralClient& client, const Params& params,
          const uint128_t seed);

  const SpiralClient& GetSpiralClient() const { return spiral_client_; }
  const LWEClient& GetLweClient() const { return lwe_client_; }
  std::vector<spiral::PolyMatrixRaw> GenerateQueryImpl(uint8_t public_seed_idx,
                                                       size_t dim_log2,
                                                       bool packing,
                                                       size_t index);
  std::vector<uint64_t> GenerateQuery(uint8_t public_seed_idx, size_t dim_log2,
                                      bool packing, size_t index_row);

  YPIRQuery GenerateFullQuery(size_t target_idx);
  YPIRSimpleQuery GenerateFullQuerySimplepir(uint64_t target_idx);

  static PolyMatrixRaw DecryptCtRegMeasured(const SpiralClient& client,
                                            const Params& params,
                                            const PolyMatrixNtt& ct,
                                            size_t coeffs_to_measure);
};

class YPIRClient {
 private:
  const Params params_;

 public:
  YPIRClient(const Params& params);

  static std::array<uint8_t, 20> Hash(const std::string& target_item);
  static size_t Bucket(size_t log2_num_items, const std::string& target_item);

  std::pair<YPIRQuery, uint128_t> GenerateQueryNormal(size_t target_idx);
  std::pair<YPIRSimpleQuery, uint128_t> GenerateQuerySimplepir(
      size_t target_row);

  uint64_t DecodeResponseNormal(const uint128_t client_seed,
                                const std::vector<uint8_t>& response_data);
  std::vector<uint8_t> DecodeResponseSimplepir(
      const uint128_t client_seed, const std::vector<uint8_t>& response_data);
  uint64_t DecodeResponseNormalYClient(
      const Params& params, const YClient& y_client,
      const std::vector<uint8_t>& response_data);
  std::vector<uint64_t> DecodeResponseSimplepirYClient(
      const Params& params, const YClient& y_client,
      const std::vector<uint8_t>& response_data);
};

}  // namespace psi::ypir
