#include "psi/algorithm/ypir/client.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

#include "yacl/crypto/rand/rand.h"

#include "psi/algorithm/spiral/arith/arith.h"
#include "psi/algorithm/spiral/arith/number_theory.h"
#include "psi/algorithm/spiral/gadget.h"
#include "psi/algorithm/spiral/poly_matrix.h"
#include "psi/algorithm/spiral/poly_matrix_utils.h"
#include "psi/algorithm/spiral/spiral_client.h"
#include "psi/algorithm/ypir/util.h"

namespace psi::ypir {

using namespace psi::spiral;
constexpr uint8_t SEED_0 = 0;
constexpr uint8_t SEED_1 = 1;
static constexpr uint128_t kStaticSeed2 = yacl::MakeUint128(0, 2);

// from params.rs
size_t Log2Ceil(uint64_t value) {
  if (value == 0) return 0;
  return static_cast<size_t>(std::ceil(std::log2(value)));
}

// from convolution.rs
std::vector<uint32_t> NegacyclicMatrixU32(const std::vector<uint32_t>& a,
                                          size_t n, uint64_t modulus) {
  std::vector<uint32_t> out(n * n);
  std::vector<uint32_t> current_row(a.begin(), a.begin() + n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      out[i * n + j] = current_row[j];
    }
    uint32_t last = current_row[n - 1];
    for (size_t j = n - 1; j > 0; --j) {
      current_row[j] = current_row[j - 1];
    }
    current_row[0] =
        static_cast<uint32_t>((modulus - (last % modulus)) % modulus);
  }
  return out;
}

// from bits.rs
std::vector<uint8_t> U64sToContiguousBytes(const std::vector<uint64_t>& data,
                                           size_t bits_per_u64) {
  size_t total_bits = data.size() * bits_per_u64;
  size_t num_bytes = (total_bits + 7) / 8;
  std::vector<uint8_t> out(num_bytes, 0);
  size_t bit_offset = 0;

  for (uint64_t val : data) {
    for (size_t i = 0; i < bits_per_u64; ++i) {
      if ((val >> i) & 1) {
        size_t byte_idx = (bit_offset + i) / 8;
        size_t bit_idx_in_byte = (bit_offset + i) % 8;
        out[byte_idx] |= (1 << bit_idx_in_byte);
      }
    }
    bit_offset += bits_per_u64;
  }
  return out;
}

uint64_t read_bits(const std::vector<uint8_t>& buffer, size_t bit_offset,
                   size_t num_bits) {
  uint64_t result = 0;
  for (size_t i = 0; i < num_bits; ++i) {
    size_t current_bit_pos = bit_offset + i;
    size_t byte_idx = current_bit_pos / 8;
    size_t bit_idx_in_byte = current_bit_pos % 8;
    if (byte_idx < buffer.size()) {
      if ((buffer[byte_idx] >> bit_idx_in_byte) & 1) {
        result |= (1ULL << i);
      }
    }
  }
  return result;
}

// Packs a vector of polynomial matrices by combining CRT coefficients (0 and 1)
// into 64-bit values. Each pair of 32-bit values (from different CRT
// coefficients) is combined using bitwise OR: lower | (upper << 32)
std::vector<uint64_t> PackVecPm(const Params& params, size_t rows, size_t cols,
                                const std::vector<PolyMatrixNtt>& v_cts) {
  YACL_ENFORCE(!v_cts.empty());
  YACL_ENFORCE_EQ(v_cts[0].Rows(), rows);
  YACL_ENFORCE_EQ(v_cts[0].Cols(), cols);
  YACL_ENFORCE_EQ(params.CrtCount(), static_cast<size_t>(2));

  std::vector<uint64_t> aligned_out(v_cts.size() * rows * cols *
                                    params.PolyLen());
  size_t out_idx = 0;

  for (const auto& ct : v_cts) {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        size_t out_offset = (row * cols + col) * params.PolyLen();
        size_t inp_offset = (row * cols + col) * 2 * params.PolyLen();
        for (size_t z = 0; z < params.PolyLen(); ++z) {
          aligned_out[out_idx + out_offset + z] =
              ct.Data()[inp_offset + z] |
              (ct.Data()[inp_offset + z + params.PolyLen()] << 32);
        }
      }
    }
    out_idx += rows * cols * params.PolyLen();
  }

  return aligned_out;
}

std::vector<uint64_t> PackQuery(const Params& params,
                                const std::vector<uint64_t>& query) {
  std::vector<uint64_t> packed_query(query.size());

  const uint64_t m0 = params.Moduli(0);
  const uint64_t m1 = params.Moduli(1);

  for (size_t i = 0; i < query.size(); ++i) {
    packed_query[i] = (query[i] % m0) | ((query[i] % m1) << 32);
  }

  return packed_query;
}

double MeasureNoiseWidthSquared(const SpiralClient& client,
                                const Params& params, const PolyMatrixNtt& ct,
                                const PolyMatrixRaw& pt,
                                size_t coeffs_to_measure) {
  const double PI = 3.14159265358979323846;
  const int64_t m_i64 = static_cast<int64_t>(params.Modulus());

  PolyMatrixRaw dec_result = FromNtt(params, client.DecryptMatrixRegev(ct));
  const auto& dec_data = dec_result.Data();
  const auto& pt_data = pt.Data();

  size_t num_coeffs =
      std::min({coeffs_to_measure, dec_data.size(), pt_data.size()});

  double total_squared_noise = 0.0;
  for (size_t i = 0; i < num_coeffs; ++i) {
    const int64_t decrypted_val = static_cast<int64_t>(dec_data[i]);

    const int64_t true_val = static_cast<int64_t>(
        arith::Rescale(pt_data[i], params.PtModulus(), params.Modulus()));
    const int64_t diff = decrypted_val - true_val;

    int64_t diff_centered = diff % m_i64;
    if (diff_centered > m_i64 / 2) {
      diff_centered -= m_i64;
    } else if (diff_centered < -(m_i64 / 2)) {
      diff_centered += m_i64;
    }
    double noise_2 = std::pow(static_cast<double>(diff_centered), 2);
    total_squared_noise += noise_2;
  }
  double variance = total_squared_noise / static_cast<double>(params.PolyLen());

  assert(variance >= 0.0);
  double subg_width_2 = variance * 2.0 * PI;

  return subg_width_2;
}

PolyMatrixRaw DecryptCtRegMeasured(const SpiralClient& client,
                                   const Params& params,
                                   const PolyMatrixNtt& ct,
                                   size_t coeffs_to_measure) {
  PolyMatrixRaw dec_result = FromNtt(params, client.DecryptMatrixRegev(ct));

  PolyMatrixRaw dec_rescaled = PolyMatrixRaw::Zero(
      params.PolyLen(), dec_result.Rows(), dec_result.Cols());

  for (size_t i = 0; i < dec_rescaled.Data().size(); ++i) {
    dec_rescaled.Data()[i] = arith::Rescale(
        dec_result.Data()[i], params.Modulus(), params.PtModulus());
  }

  // Measure noise width
  double s_2 = MeasureNoiseWidthSquared(client, params, ct, dec_rescaled,
                                        coeffs_to_measure);
  std::cout << "[DEBUG] log2(measured noise): " << std::log2(s_2) << std::endl;
  return dec_rescaled;
}

// ===================================================================================
// LWEClient Implementation
// ===================================================================================
LWEClient LWEClient::Init(LWEParams params) {
  uint128_t seed = yacl::MakeUint128(0, 999);
  return FromSeed(params, seed);
}

LWEClient::LWEClient(LWEParams params, std::vector<uint32_t> sk)
    : lwe_params_(std::move(params)), sk_(std::move(sk)) {}

LWEClient LWEClient::FromSeed(const LWEParams& lwe_params,
                              const uint128_t seed) {
  DiscreteGaussian dg(lwe_params.noise_width);
  yacl::crypto::Prg<uint64_t> rng(seed);
  std::vector<uint32_t> sk;
  sk.reserve(lwe_params.n);
  for (size_t i = 0; i < lwe_params.n; ++i) {
    uint64_t random_sample = dg.Sample(lwe_params.modulus, rng);
    sk.push_back(static_cast<uint32_t>(random_sample));
  }
  return LWEClient(lwe_params, std::move(sk));
}

std::vector<uint32_t> LWEClient::Encrypt(yacl::crypto::Prg<uint64_t>& rng,
                                         uint32_t pt) {
  DiscreteGaussian dg(lwe_params_.noise_width);
  uint32_t e = static_cast<uint32_t>(dg.Sample(lwe_params_.modulus, rng));
  std::vector<uint32_t> ct;
  ct.reserve(lwe_params_.n + 1);
  uint32_t sum = 0;

  for (size_t i = 0; i < lwe_params_.n; ++i) {
    uint32_t v = static_cast<uint32_t>(rng());
    ct.push_back(v);
    sum += v * sk_[i];
  }

  uint32_t neg_sum = -sum;
  uint32_t b = neg_sum + pt + e;
  ct.push_back(b);
  return ct;
}

std::vector<uint32_t> LWEClient::EncryptMany(yacl::crypto::Prg<uint32_t>& rng,
                                             std::vector<uint32_t> v_pt) {
  YACL_ENFORCE_EQ(v_pt.size(), lwe_params_.n,
                  "Plaintext vector size must equal n");

  yacl::crypto::Prg<uint64_t> rng_noise(yacl::MakeUint128(0, 999));
  DiscreteGaussian dg(lwe_params_.noise_width);

  const size_t n = lwe_params_.n;

  std::vector<uint32_t> a;
  a.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    a.push_back(static_cast<uint32_t>(rng()));
  }

  std::vector<uint32_t> nega_a = NegacyclicMatrixU32(a, n, lwe_params_.modulus);
  std::vector<uint32_t> last_row(n);
  for (size_t col = 0; col < n; ++col) {
    uint32_t sum = 0;
    for (size_t row = 0; row < n; ++row) {
      size_t idx = row * n + col;
      sum += nega_a[idx] * sk_[row];
    }

    uint32_t e =
        static_cast<uint32_t>(dg.Sample(lwe_params_.modulus, rng_noise));
    uint32_t val = -sum + v_pt[col] + e;
    last_row[col] = val;
  }

  std::vector<uint32_t> ct = std::move(nega_a);
  ct.insert(ct.end(), last_row.begin(), last_row.end());

  return ct;
}

uint32_t LWEClient::Decrypt(const std::vector<uint32_t>& ct) const {
  YACL_ENFORCE_EQ(ct.size(), lwe_params_.n + 1,
                  "Invalid ciphertext size in LWE decrypt");
  uint32_t sum = 0;  // Note: uint32_t overflow is well-defined (wraps around)

  for (size_t i = 0; i < lwe_params_.n; ++i) {
    sum += ct[i] * sk_[i];
  }
  sum += ct[lwe_params_.n];
  return sum;
}

// ===================================================================================
// YClient Implementation
// ===================================================================================

YClient::YClient(const SpiralClient& client, const Params& params)
    : lwe_client_(
          LWEClient::FromSeed(LWEParams::Default(), yacl::MakeUint128(0, 999))),
      spiral_client_(client),
      params_(params) {}

YClient::YClient(const SpiralClient& client, const Params& params,
                 const uint128_t seed)
    : lwe_client_(LWEClient::FromSeed(LWEParams::Default(), seed)),
      spiral_client_(client),
      params_(params) {}

std::vector<uint64_t> YClient::RlwesToLwes(
    const std::vector<PolyMatrixRaw>& cts) const {
  if (cts.empty()) {
    return {};
  }
  size_t n = params_.PolyLen();
  size_t num_cts = cts.size();
  std::vector<uint64_t> lwes((n + 1) * n * num_cts);

  for (size_t i = 0; i < num_cts; ++i) {
    const auto& ct = cts[i];
    auto a_poly = ct.Poly(0, 0);
    auto b_poly = ct.Poly(1, 0);

    std::vector<uint32_t> a_poly_vec(a_poly.begin(), a_poly.end());
    auto negacylic_a_u32 =
        NegacyclicMatrixU32(a_poly_vec, n, params_.Modulus());

    for (size_t r = 0; r < n; ++r) {
      for (size_t c = 0; c < n; ++c) {
        lwes[r * (n * num_cts) + i * n + c] = negacylic_a_u32[r * n + c];
      }
    }

    for (size_t c = 0; c < n; ++c) {
      lwes[n * (n * num_cts) + i * n + c] = b_poly[c];
    }
  }
  return lwes;
}

std::vector<PolyMatrixRaw> YClient::GenerateQueryImpl(uint8_t public_seed_idx,
                                                      size_t dim_log2,
                                                      bool packing,
                                                      size_t index) {
  const bool multiply_ct = true;
  yacl::crypto::Prg<uint64_t> rng_pub(public_seed_idx);
  std::vector<PolyMatrixRaw> out;
  const size_t num_iterations = (1ULL << dim_log2);
  out.reserve(num_iterations);
  const uint64_t scale_k = params_.ScaleK();

  for (size_t i = 0; i < num_iterations; ++i) {
    PolyMatrixRaw scalar = PolyMatrixRaw::Zero(params_.PolyLen(), 1, 1);
    const bool is_nonzero = (i == (index / params_.PolyLen()));
    if (is_nonzero) {
      scalar.Poly(0, 0)[index % params_.PolyLen()] = scale_k;
    }

    const uint64_t factor =
        arith::InvertUintMod(params_.PolyLen(), params_.Modulus());

    if (packing) {
      auto factor_pm = PolyMatrixRaw::SingleValue(params_.PolyLen(), factor);
      auto factor_pm_ntt = ToNtt(params_, factor_pm);
      auto scalar_ntt = ToNtt(params_, scalar);
      auto multiplied_ntt = ScalarMultiply(params_, factor_pm_ntt, scalar_ntt);
      scalar = FromNtt(params_, multiplied_ntt);
    }
    PolyMatrixNtt ct;
    yacl::crypto::Prg<uint64_t> rng_noise(yacl::MakeUint128(0, 999));

    auto scalar_ntt = ToNtt(params_, scalar);

    if (multiply_ct) {
      ct = spiral_client_.EncryptMatrixScaledRegev(scalar_ntt, rng_noise,
                                                   rng_pub, factor);
    } else {
      ct = spiral_client_.EncryptMatrixRegev(scalar_ntt, rng_noise, rng_pub);
    }
    PolyMatrixRaw ct_raw = FromNtt(params_, ct);
    out.push_back(std::move(ct_raw));
  }
  return out;
}

std::vector<uint64_t> YClient::GenerateQuery(uint8_t public_seed_idx,
                                             size_t dim_log2, bool packing,
                                             size_t index_row) {
  // uint128_t seed = yacl::MakeUint128(0, 999);
  if (public_seed_idx == SEED_0 && !packing) {
    LWEParams lwe_params = LWEParams::Default();
    const size_t dim = 1ULL << (dim_log2 + params_.PolyLenLog2());
    std::vector<uint64_t> lwes((lwe_params.n + 1) * dim, 0);

    const uint32_t scale_k = static_cast<uint32_t>(lwe_params.ScaleK());
    std::vector<uint32_t> vals_to_encrypt(dim, 0);
    vals_to_encrypt[index_row] = scale_k;
    yacl::crypto::Prg<uint32_t> rng_pub(public_seed_idx);

    for (size_t i = 0; i < dim; i += lwe_params.n) {
      std::vector<uint32_t> pt_slice(
          vals_to_encrypt.begin() + i,
          vals_to_encrypt.begin() + i + lwe_params.n);

      std::vector<uint32_t> out_u32 =
          lwe_client_.EncryptMany(rng_pub, pt_slice);
      assert(out_u32.size() == (lwe_params.n + 1) * lwe_params.n);

      for (size_t r = 0; r < lwe_params.n + 1; ++r) {
        for (size_t c = 0; c < lwe_params.n; ++c) {
          lwes[r * dim + i + c] =
              static_cast<uint64_t>(out_u32[r * lwe_params.n + c]);
        }
      }
    }

    return lwes;
  } else {
    auto out = GenerateQueryImpl(public_seed_idx, dim_log2, packing, index_row);
    auto lwes = RlwesToLwes(out);

    return lwes;
  }
}

std::vector<uint64_t> YClient::GenerateQueryLweLowMem(uint8_t public_seed_idx,
                                                      size_t dim_log2,
                                                      bool packing,
                                                      size_t index_row) {
  const size_t index = index_row;
  const bool multiply_ct = true;

  yacl::crypto::Prg<uint64_t> rng_pub(public_seed_idx);
  std::vector<uint64_t> out;

  const size_t num_iterations = (1ULL << dim_log2);
  out.reserve(num_iterations * params_.PolyLen());
  const uint64_t scale_k = params_.ScaleK();

  for (size_t i = 0; i < num_iterations; ++i) {
    PolyMatrixRaw scalar = PolyMatrixRaw::Zero(params_.PolyLen(), 1, 1);
    const bool is_nonzero = (i == (index / params_.PolyLen()));
    if (is_nonzero) {
      scalar.Data()[index % params_.PolyLen()] = scale_k;
    }
    if (packing) {
      const uint64_t factor =
          arith::InvertUintMod(params_.PolyLen(), params_.Modulus());
      PolyMatrixNtt factor_pm_ntt =
          ToNtt(params_, PolyMatrixRaw::SingleValue(params_.PolyLen(), factor));
      scalar = FromNtt(params_, ScalarMultiply(params_, factor_pm_ntt,
                                               ToNtt(params_, scalar)));
    }

    PolyMatrixNtt ct;
    yacl::crypto::Prg<uint64_t> rng_noise(yacl::MakeUint128(0, 999));
    PolyMatrixNtt scalar_ntt = ToNtt(params_, scalar);
    if (multiply_ct) {
      const uint64_t factor =
          arith::InvertUintMod(params_.PolyLen(), params_.Modulus());
      ct = spiral_client_.EncryptMatrixScaledRegev(scalar_ntt, rng_noise,
                                                   rng_pub, factor);
    } else {
      ct = spiral_client_.EncryptMatrixRegev(scalar_ntt, rng_noise, rng_pub);
    }

    PolyMatrixRaw ct_raw = FromNtt(params_, ct);
    const auto& lwe_last_row = ct_raw.Poly(1, 0);
    out.insert(out.end(), lwe_last_row.begin(), lwe_last_row.end());
  }
  return out;
}

YPIRQuery YClient::GenerateFullQuery(size_t target_idx) {
  const size_t db_rows = 1ULL << (params_.DbDim1() + params_.PolyLenLog2());
  const size_t db_cols = 1ULL << (params_.DbDim2() + params_.PolyLenLog2());
  const size_t target_row = target_idx / db_cols;
  const size_t target_col = target_idx % db_cols;

  auto sk_reg = spiral_client_.GetSkRegev();
  yacl::crypto::Prg<uint64_t> rng_noise(yacl::MakeUint128(0, 999));
  yacl::crypto::Prg<uint64_t> rng_pub_static(kStaticSeed2);

  auto pack_pub_params =
      RawGenerateExpansionParams(params_, sk_reg, params_.PolyLenLog2(),
                                 params_.TExpLeft(), rng_noise, rng_pub_static);

  auto pack_pub_params_row_1s = pack_pub_params;
  for (size_t i = 0; i < pack_pub_params.size(); ++i) {
    auto submatrix =
        pack_pub_params[i].SubMatrix(1, 0, 1, pack_pub_params[i].Cols());
    pack_pub_params_row_1s[i] = CondenseMatrix(params_, submatrix);
  }

  std::vector<uint64_t> pack_pub_params_row_1s_pm =
      PackVecPm(params_, 1, params_.TExpLeft(), pack_pub_params_row_1s);

  assert(pack_pub_params_row_1s_pm.size() * sizeof(uint64_t) ==
         static_cast<size_t>(params_.PolyLenLog2()) * params_.TExpLeft() *
             params_.PolyLen() * sizeof(uint64_t));

  // generate query
  auto query_row = GenerateQuery(SEED_0, params_.DbDim1(), false, target_row);

  // 提取最后一行
  const size_t query_row_start_idx = lwe_client_.LweParams().n * db_rows;

  auto slice_begin = query_row.begin() + query_row_start_idx;
  auto slice_end = query_row.end();

  std::vector<uint64_t> packed_query_row(slice_begin, slice_end);
  std::vector<uint32_t> packed_query_row_u32;
  packed_query_row_u32.reserve(packed_query_row.size());
  std::transform(packed_query_row.begin(), packed_query_row.end(),
                 std::back_inserter(packed_query_row_u32),
                 [](uint64_t val) { return static_cast<uint32_t>(val); });
  packed_query_row_u32.resize(params_.DbRowsPadded(), 0);
  assert(packed_query_row_u32.size() == params_.DbRowsPadded());

  auto query_col_vec =
      GenerateQuery(SEED_1, params_.DbDim2(), true, target_col);
  const size_t query_col_start_idx = params_.PolyLen() * db_cols;
  std::vector<uint64_t> query_col_last_row(
      query_col_vec.begin() + query_col_start_idx, query_col_vec.end());

  auto packed_query_col = PackQuery(params_, query_col_last_row);
  assert(packed_query_col.size() == db_cols);

  return {packed_query_row_u32, packed_query_col, pack_pub_params_row_1s_pm};
}

YPIRSimpleQuery YClient::GenerateFullQuerySimplepir(uint64_t target_idx) {
  size_t db_rows = 1ULL << (params_.DbDim1() + params_.PolyLenLog2());
  size_t db_cols = params_.Instances() * params_.PolyLen();

  size_t target_row = static_cast<size_t>(target_idx / db_cols);
  size_t target_col = static_cast<size_t>(target_idx % db_cols);

  SPDLOG_INFO("Target item: {} ({}, {})", target_idx, target_row, target_col);
  auto sk_reg = spiral_client_.GetSkRegev();

  yacl::crypto::Prg<uint64_t> rng_pub(yacl::MakeUint128(0, 999));
  yacl::crypto::Prg<uint64_t> rng_priv(kStaticSeed2);

  std::vector<PolyMatrixNtt> pack_pub_params =
      RawGenerateExpansionParams(params_, sk_reg, params_.PolyLenLog2(),
                                 params_.TExpLeft(), rng_pub, rng_priv);

  std::vector<PolyMatrixNtt> pack_pub_params_row_1s;
  pack_pub_params_row_1s.reserve(pack_pub_params.size());

  for (const auto& matrix : pack_pub_params) {
    PolyMatrixNtt row1 = matrix.SubMatrix(1, 0, 1, matrix.Cols());
    pack_pub_params_row_1s.push_back(std::move(row1));
  }

  std::vector<uint64_t> pack_pub_params_row_1s_pm =
      PackVecPm(params_, 1, params_.TExpLeft(), pack_pub_params_row_1s);

  std::vector<uint64_t> query_row_last_row =
      GenerateQueryLweLowMem(SEED_0, params_.DbDim1(), true, target_row);

  if (query_row_last_row.size() != db_rows) {
    SPDLOG_WARN("Query size mismatch: expected {}, got {}", db_rows,
                query_row_last_row.size());
  }
  std::vector<uint64_t> packed_query_row =
      PackQuery(params_, query_row_last_row);

  return {std::move(packed_query_row), std::move(pack_pub_params_row_1s_pm)};
}

// ===================================================================================
// YPIRClient Implementation
// ===================================================================================

YPIRClient::YPIRClient(const Params& params) : params_(params) {}

std::array<uint8_t, 20> YPIRClient::Hash(const std::string& target_item) {
  // TODO: Replace with actual SHA implementation
  // The current implementation is a placeholder
  std::array<uint8_t, 20> hash_result = {0};
  for (size_t i = 0; i < std::min(target_item.size(), size_t(20)); ++i) {
    hash_result[i] = static_cast<uint8_t>(target_item[i]);
  }
  return hash_result;
}

size_t YPIRClient::Bucket(size_t log2_num_items,
                          const std::string& target_item) {
  auto item_hash = Hash(target_item);
  uint32_t top_idx = (static_cast<uint32_t>(item_hash[0]) << 24) |
                     (static_cast<uint32_t>(item_hash[1]) << 16) |
                     (static_cast<uint32_t>(item_hash[2]) << 8) |
                     (static_cast<uint32_t>(item_hash[3]));
  return top_idx >> (32 - log2_num_items);
}

std::pair<YPIRQuery, uint128_t> YPIRClient::GenerateQueryNormal(
    size_t target_idx) {
  uint128_t client_seed = yacl::MakeUint128(0, 999);
  auto client = SpiralClient(params_);
  YClient y_client(client, params_, client_seed);
  auto query = y_client.GenerateFullQuery(target_idx);
  return {query, client_seed};
}

std::pair<YPIRSimpleQuery, uint128_t> YPIRClient::GenerateQuerySimplepir(
    size_t target_row) {
  uint64_t target_idx = static_cast<uint64_t>(target_row) *
                        params_.Instances() * params_.PolyLen();
  uint128_t client_seed = yacl::MakeUint128(0, 999);
  auto client = SpiralClient(params_);
  YClient y_client(client, params_, client_seed);
  auto query = y_client.GenerateFullQuerySimplepir(target_idx);
  return {query, client_seed};
}

uint64_t YPIRClient::DecodeResponseNormal(
    const uint128_t seed, const std::vector<uint8_t>& response_data) {
  auto client = SpiralClient(params_, seed);
  YClient y_client = YClient(client, params_, seed);
  uint64_t out =
      YPIRClient::DecodeResponseNormalYClient(params_, y_client, response_data);

  // return out
  return out;
}

std::vector<uint8_t> YPIRClient::DecodeResponseSimplepir(
    const uint128_t seed, const std::vector<uint8_t>& response_data) {
  auto client = SpiralClient(params_, seed);
  YClient y_client(client, params_, seed);
  auto decoded =
      DecodeResponseSimplepirYClient(params_, y_client, response_data);
  size_t pt_bits = static_cast<size_t>(std::log2(params_.PtModulus()));
  return U64sToContiguousBytes(decoded, pt_bits);
}

uint64_t YPIRClient::DecodeResponseNormalYClient(
    const Params& params, const YClient& y_client,
    const std::vector<uint8_t>& response_data) {
  const size_t num_rlwe_outputs = params.Rho();
  assert(response_data.size() % num_rlwe_outputs == 0);
  const size_t chunk_size = response_data.size() / num_rlwe_outputs;

  std::vector<PolyMatrixRaw> response;
  for (size_t i = 0; i < num_rlwe_outputs; ++i) {
    std::vector<uint8_t> chunk(response_data.begin() + i * chunk_size,
                               response_data.begin() + (i + 1) * chunk_size);
    response.push_back(PolyMatrixRaw::Recover(params, params.GetQPrime1(),
                                              params.GetQPrime2(), chunk));
  }

  const auto lwe_params = LWEParams::Default();
  const size_t lwe_q_prime_bits = lwe_params.q2_bits;
  const size_t pt_bits = params.PtModulusBitLen();
  const double blowup_factor = static_cast<double>(lwe_q_prime_bits) / pt_bits;

  std::vector<uint64_t> outer_ct;
  for (const auto& ct : response) {
    PolyMatrixRaw decrypted_part =
        DecryptCtRegMeasured(y_client.GetSpiralClient(), params,
                             ToNtt(params, ct), params.PolyLen());
    const auto& decrypted_coeffs = decrypted_part.Data();
    // DecryptCtRegMeasured 返回的值已经在正确的范围内，直接使用
    for (uint64_t coeff : decrypted_coeffs) {
      outer_ct.push_back(coeff);
    }
  }

  size_t special_offs_preview = static_cast<size_t>(std::ceil(
      (static_cast<double>(lwe_params.n) * lwe_q_prime_bits) / pt_bits));
  SPDLOG_INFO("Decode: outer_ct size={}, first 4 values: [{}, {}, {}, {}]",
              outer_ct.size(), outer_ct.size() > 0 ? outer_ct[0] : 0,
              outer_ct.size() > 1 ? outer_ct[1] : 0,
              outer_ct.size() > 2 ? outer_ct[2] : 0,
              outer_ct.size() > 3 ? outer_ct[3] : 0);
  SPDLOG_INFO(
      "Decode: outer_ct at special_offs={}: [{}, {}]", special_offs_preview,
      outer_ct.size() > special_offs_preview ? outer_ct[special_offs_preview]
                                             : 0,
      outer_ct.size() > special_offs_preview + 1
          ? outer_ct[special_offs_preview + 1]
          : 0);

  std::vector<uint8_t> outer_ct_t_u8 = U64sToContiguousBytes(outer_ct, pt_bits);

  PolyMatrixRaw inner_ct = PolyMatrixRaw::Zero(params.PolyLen(), 2, 1);
  size_t bit_offs = 0;
  const uint64_t lwe_q_prime = lwe_params.GetQPrime2();
  const size_t special_offs = static_cast<size_t>(std::ceil(
      (static_cast<double>(lwe_params.n) * lwe_q_prime_bits) / pt_bits));

  SPDLOG_INFO("Decode: lwe_q_prime={}, special_offs={}, pt_bits={}",
              lwe_q_prime, special_offs, pt_bits);

  for (size_t z = 0; z < lwe_params.n; ++z) {
    uint64_t val = read_bits(outer_ct_t_u8, bit_offs, lwe_q_prime_bits);
    bit_offs += lwe_q_prime_bits;
    if (val >= lwe_q_prime) {
      SPDLOG_WARN("WARNING at z={}: val {} >= lwe_q_prime {}, taking modulo", z,
                  val, lwe_q_prime);
      val = val % lwe_q_prime;
    }
    inner_ct.Data()[z] = arith::Rescale(val, lwe_q_prime, lwe_params.modulus);
  }

  uint64_t val = 0;
  size_t num_loops = static_cast<size_t>(std::ceil(blowup_factor));
  SPDLOG_INFO(
      "Extracting b value: special_offs={}, num_loops={}, pt_bits={}, "
      "blowup_factor={}",
      special_offs, num_loops, pt_bits, blowup_factor);
  for (size_t i = 0; i < num_loops; ++i) {
    uint64_t coeff = outer_ct[special_offs + i];
    SPDLOG_INFO("  i={}, outer_ct[{}]={}, shift={}", i, special_offs + i, coeff,
                i * pt_bits);
    val |= coeff << (i * pt_bits);
  }
  SPDLOG_INFO("Combined val={}, lwe_q_prime={}, val<lwe_q_prime={}", val,
              lwe_q_prime, val < lwe_q_prime);
  if (val >= lwe_q_prime) {
    SPDLOG_WARN("WARNING: val {} >= lwe_q_prime {}, taking modulo", val,
                lwe_q_prime);
    val = val % lwe_q_prime;
  }
  inner_ct.Data()[lwe_params.n] =
      arith::Rescale(val, lwe_q_prime, lwe_params.modulus);

  std::vector<uint32_t> inner_ct_as_u32;
  inner_ct_as_u32.reserve(lwe_params.n + 1);
  for (size_t i = 0; i < lwe_params.n + 1; ++i) {
    inner_ct_as_u32.push_back(static_cast<uint32_t>(inner_ct.Data()[i]));
  }

  SPDLOG_INFO("Decode: inner_ct first 5 values: [{}, {}, {}, {}, {}]",
              inner_ct_as_u32[0], inner_ct_as_u32[1], inner_ct_as_u32[2],
              inner_ct_as_u32[3], inner_ct_as_u32[4]);
  SPDLOG_INFO("Decode: inner_ct last value (b): {}",
              inner_ct_as_u32[lwe_params.n]);

  uint32_t decrypted = y_client.GetLweClient().Decrypt(inner_ct_as_u32);
  SPDLOG_INFO(
      "Decode: LWE decrypted (before rescale)={}, lwe_modulus={}, "
      "lwe_pt_modulus={}",
      decrypted, lwe_params.modulus, lwe_params.pt_modulus);

  uint64_t final_result =
      arith::Rescale(static_cast<uint64_t>(decrypted), lwe_params.modulus,
                     lwe_params.pt_modulus);

  SPDLOG_INFO("Decode: final_result after rescale={}", final_result);
  return final_result;
}

std::vector<uint64_t> YPIRClient::DecodeResponseSimplepirYClient(
    const Params& params, const YClient& y_client,
    const std::vector<uint8_t>& response_data) {
  const size_t db_cols = params.Instances() * params.PolyLen();
  const size_t num_rlwe_outputs = db_cols / params.PolyLen();
  assert(response_data.size() % num_rlwe_outputs == 0 &&
         "Response data size is not a multiple of num_rlwe_outputs");
  const size_t chunk_size = response_data.size() / num_rlwe_outputs;

  const uint64_t rlwe_q_prime_1 = params.GetQPrime1();
  const uint64_t rlwe_q_prime_2 = params.GetQPrime2();

  std::vector<PolyMatrixRaw> response;
  response.reserve(num_rlwe_outputs);

  for (size_t i = 0; i < num_rlwe_outputs; ++i) {
    std::vector<uint8_t> chunk(response_data.begin() + i * chunk_size,
                               response_data.begin() + (i + 1) * chunk_size);
    response.push_back(
        PolyMatrixRaw::Recover(params, rlwe_q_prime_1, rlwe_q_prime_2, chunk));
  }

  std::vector<uint64_t> outer_ct;
  outer_ct.reserve(num_rlwe_outputs * params.PolyLen());

  for (const auto& ct : response) {
    PolyMatrixRaw decrypted_part =
        DecryptCtRegMeasured(y_client.GetSpiralClient(), params,
                             ToNtt(params, ct), params.PolyLen());
    const auto& decrypted_coeffs = decrypted_part.Data();
    outer_ct.insert(outer_ct.end(), decrypted_coeffs.begin(),
                    decrypted_coeffs.end());
  }

  assert(outer_ct.size() == num_rlwe_outputs * params.PolyLen() &&
         "Final decrypted vector size mismatch");

  return outer_ct;
}

}  // namespace psi::ypir
