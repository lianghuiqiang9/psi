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

namespace psi::ypir {

using namespace psi::spiral;

struct LWEParams {
  size_t n = 1024;
  uint64_t modulus = 1ULL << 32;
  uint64_t pt_modulus = 1ULL << 8;
  size_t q2_bits = 28;
  double noise_width = 27.57291103;

  uint64_t ScaleK() const { return modulus / pt_modulus; }
  static LWEParams Default() { return LWEParams(); }
};

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
  std::vector<uint32_t> EncryptMany(yacl::crypto::Prg<uint64_t>& rng,
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
  SpiralClient* spiral_client_;
  const Params* params_;

  std::vector<uint64_t> rlwes_to_lwes(
      const std::vector<spiral::PolyMatrixRaw>& ct) const;
  std::vector<spiral::PolyMatrixRaw> GenerateQueryImpl(uint128_t seed,
                                                       size_t dim_log2,
                                                       bool packing,
                                                       size_t index);
  std::vector<uint64_t> GenerateQueryLweLowMem(uint128_t seed, size_t dim_log2,
                                               bool packing, size_t index_row);

 public:
  YClient(SpiralClient* client, const Params* params);
  YClient(SpiralClient* client, const Params* params, const uint128_t seed);

  std::vector<uint64_t> GenerateQuery(bool is_query_row, size_t dim_log2,
                                      bool packing, size_t index_row);

  YPIRQuery GenerateFullQuery(size_t target_idx);
  YPIRSimpleQuery GenerateFullQuerySimplepir(uint64_t target_idx);

  static PolyMatrixRaw DecryptCtRegMeasured(const SpiralClient* client,
                                            const Params* params,
                                            const PolyMatrixNtt& ct,
                                            size_t coeffs_to_measure);
};

class YPIRClient {
 private:
  const Params* params_;

 public:
  YPIRClient(const Params* params);

  static std::array<uint8_t, 20> hash(const std::string& target_item);
  static size_t bucket(size_t log2_num_items, const std::string& target_item);
  // static std::unique_ptr<YPIRClient> from_db_sz(
  //     uint64_t num_items,
  //     uint64_t item_size_bits,
  //     bool is_simplepir
  // );
  std::pair<YPIRQuery, uint128_t> generate_query_normal(size_t target_idx);
  std::pair<YPIRSimpleQuery, uint128_t> generate_query_simplepir(
      size_t target_row);

  uint64_t DecodeResponseNormal(const uint128_t client_seed,
                                const std::vector<uint8_t>& response_data);
  std::vector<uint8_t> DecodeResponseSimplepir(
      const uint128_t client_seed, const std::vector<uint8_t>& response_data);
  uint64_t DecodeResponseNormalYClient(
      const Params& params, const YClient& y_client,
      const std::vector<uint8_t>& response_data);
};

}  // namespace psi::ypir
