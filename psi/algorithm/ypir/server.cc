#include "psi/algorithm/ypir/server.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#include "spdlog/spdlog.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"

#include "psi/algorithm/spiral/arith/arith.h"
#include "psi/algorithm/spiral/arith/arith_params.h"
#include "psi/algorithm/spiral/gadget.h"
#include "psi/algorithm/spiral/poly_matrix.h"
#include "psi/algorithm/spiral/poly_matrix_utils.h"
#include "psi/algorithm/spiral/spiral_client.h"
#include "psi/algorithm/ypir/client.h"
#include "psi/algorithm/ypir/convolution.h"
#include "psi/algorithm/ypir/params.h"
#include "psi/algorithm/ypir/types.h"
#include "psi/algorithm/ypir/util.h"

namespace psi::ypir {

using namespace psi::spiral;

// TODO: use secure seed
constexpr uint8_t SEED_0 = 0;
constexpr uint8_t SEED_1 = 1;

static constexpr uint128_t kStaticSeed2 = yacl::MakeUint128(0, 2);

void WriteBits(absl::Span<uint8_t> data, uint64_t val, size_t bit_offs,
               size_t num_bits) {
  size_t byte_index = bit_offs / 8;
  size_t bit_index = bit_offs % 8;

  while (num_bits > 0 && byte_index < data.size()) {
    size_t bits_to_write = std::min(size_t(8) - bit_index, num_bits);

    uint8_t bitmask = (1 << bits_to_write) - 1;

    uint8_t bits = static_cast<uint8_t>((val & bitmask) << bit_index);

    data[byte_index] |= bits;

    num_bits -= bits_to_write;
    bit_index += bits_to_write;

    if (bit_index == 8) {
      byte_index += 1;
      bit_index = 0;
    }

    val >>= bits_to_write;
  }
}

uint64_t ReadBits(absl::Span<const uint8_t> data, size_t bit_offs,
                  size_t num_bits) {
  if (num_bits > 64 || num_bits == 0) {
    throw std::invalid_argument("Invalid number of bits: " +
                                std::to_string(num_bits));
  }

  size_t byte_index = bit_offs / 8;
  size_t bit_index = bit_offs % 8;

  uint64_t result = 0;
  size_t bits_read = 0;

  while (bits_read < num_bits && byte_index < data.size()) {
    size_t can_take = std::min(size_t(8) - bit_index, num_bits - bits_read);

    uint8_t mask = (1 << can_take) - 1;

    uint64_t value = (data[byte_index] >> bit_index) & mask;

    result |= (value << bits_read);

    bits_read += can_take;
    bit_index += can_take;

    if (bit_index == 8) {
      byte_index++;
      bit_index = 0;
    }
  }

  return result;
}

std::vector<uint16_t> SplitAlloc(absl::Span<const uint64_t> buf,
                                 size_t special_bit_offs, size_t rows,
                                 size_t cols, size_t out_rows,
                                 size_t inp_mod_bits, size_t pt_bits) {
  std::vector<uint16_t> out(out_rows * cols, 0);

  assert(out_rows >= rows);
  assert(inp_mod_bits >= pt_bits);

  size_t total_bits_needed = out_rows * inp_mod_bits;
  size_t tmp_buf_len = (total_bits_needed + 7) / 8;

  for (size_t j = 0; j < cols; ++j) {
    std::vector<uint8_t> bytes_tmp(tmp_buf_len, 0);

    size_t bit_offs = 0;
    for (size_t i = 0; i < rows; ++i) {
      uint64_t inp = buf[i * cols + j];

      if (i == rows - 1) {
        bit_offs = special_bit_offs;
      }

      WriteBits(absl::MakeSpan(bytes_tmp), inp, bit_offs, inp_mod_bits);
      bit_offs += inp_mod_bits;
    }

    bit_offs = 0;
    for (size_t i = 0; i < out_rows; ++i) {
      uint64_t out_val = ReadBits(absl::MakeSpan(bytes_tmp), bit_offs, pt_bits);

      out[i * cols + j] = static_cast<uint16_t>(out_val);
      bit_offs += pt_bits;

      if (bit_offs >= total_bits_needed) {
        break;
      }
    }

    size_t check_row_idx = special_bit_offs / pt_bits;

    if (check_row_idx < out_rows) {
      uint64_t out_check_val =
          static_cast<uint64_t>(out[check_row_idx * cols + j]);

      uint64_t buf_check_val = buf[(rows - 1) * cols + j];
      uint64_t mask = (1ULL << pt_bits) - 1;

      assert(out_check_val == (buf_check_val & mask));
    }
  }

  return out;
}

std::vector<PolyMatrixNtt> GenerateFakePackPubParams(const Params& params) {
  auto sk_raw_zero = PolyMatrixRaw::Zero(params.PolyLen(), 1, 1);
  yacl::crypto::Prg<uint64_t> rng_pub(yacl::crypto::SecureRandU128());
  yacl::crypto::Prg<uint64_t> rng_priv(kStaticSeed2);

  size_t m = params.PolyLenLog2();
  size_t t = params.TExpLeft();
  return RawGenerateExpansionParams(params, sk_raw_zero, m, t, rng_pub,
                                    rng_priv);
}

std::pair<std::vector<PolyMatrixNtt>, std::vector<PolyMatrixNtt>>
GenerateYConstants(const Params& params) {
  std::vector<PolyMatrixNtt> y_constants;
  std::vector<PolyMatrixNtt> neg_y_constants;

  size_t limit = params.PolyLenLog2();
  y_constants.reserve(limit);
  neg_y_constants.reserve(limit);

  for (size_t num_cts_log2 = 1; num_cts_log2 <= limit; ++num_cts_log2) {
    size_t num_cts = 1ULL << num_cts_log2;

    size_t idx = params.PolyLen() / num_cts;

    auto y_raw = PolyMatrixRaw::Zero(params.PolyLen(), 1, 1);
    if (idx < y_raw.Data().size()) {
      const_cast<uint64_t&>(y_raw.Data()[idx]) = 1;
    } else {
      throw std::runtime_error("GenerateYConstants: Index out of bounds");
    }

    y_constants.push_back(ToNtt(params, y_raw));
    auto neg_y_raw = PolyMatrixRaw::Zero(params.PolyLen(), 1, 1);
    if (idx < neg_y_raw.Data().size()) {
      const_cast<uint64_t&>(neg_y_raw.Data()[idx]) = params.Modulus() - 1;
    }
    neg_y_constants.push_back(ToNtt(params, neg_y_raw));
  }
  return {std::move(y_constants), std::move(neg_y_constants)};
}

template <typename T>
YPirServer<T>::YPirServer(const Params& params_in,
                          const std::vector<T>& input_db, bool is_simplepir,
                          bool inp_transposed, bool pad_rows_in)
    : params_(params_in),
      pad_rows_(pad_rows_in),
      is_simplepir_(is_simplepir),
      inp_transposed_(inp_transposed) {
  uint64_t db_rows = 1ULL << (params_.DbDim1() + params_.PolyLenLog2());
  uint64_t db_rows_padded = pad_rows_ ? params_.DbRowsPadded() : db_rows;

  uint64_t db_cols;
  if (is_simplepir_) {
    db_cols = params_.Instances() * params_.PolyLen();
  } else {
    db_cols = 1ULL << (params_.DbDim2() + params_.PolyLenLog2());
  }

  size_t num_elements_T = db_rows_padded * db_cols;
  size_t total_bytes = num_elements_T * sizeof(T);

  db_buf_.resize(total_bytes / 8, 0);

  size_t input_idx = 0;
  size_t input_size = input_db.size();
  T* db_ptr = reinterpret_cast<T*>(db_buf_.data());

  for (uint64_t i = 0; i < db_rows; ++i) {
    for (uint64_t j = 0; j < db_cols; ++j) {
      uint64_t target_idx;
      if (inp_transposed) {
        target_idx = i * db_cols + j;
      } else {
        target_idx = j * db_rows_padded + i;
      }

      if (input_idx < input_size) {
        db_ptr[target_idx] = input_db[input_idx++];
      } else {
        YACL_THROW("Not enough elements in input database");
      }
    }
  }

  // if (is_simplepir_) {
  //   smaller_params_ = params_;
  // } else {
  //   auto lwe_params = LWEParams::Default();

  //   double pt_bits =
  //       std::floor(std::log2(static_cast<double>(params_.PtModulus())));
  //   double blowup_factor = static_cast<double>(lwe_params.q2_bits) / pt_bits;

  //   smaller_params_ = params_;
  //   smaller_params_.SetDbDim1(params_.DbDim2());
  //   double num = blowup_factor * static_cast<double>(lwe_params.n + 1);
  //   double denom = static_cast<double>(params_.PolyLen());
  //   size_t smaller_dim2 =
  //       static_cast<size_t>(std::ceil(std::log2(num / denom)));

  //   smaller_params_.SetDbDim2(smaller_dim2);
  // }
}

template <typename T>
size_t YPirServer<T>::GetDbCols() const {
  return is_simplepir_ ? (params_.Instances() * params_.PolyLen())
                       : (1ULL << (params_.DbDim2() + params_.PolyLenLog2()));
}

template <typename T>
void YPirServer<T>::WriteVecU64ToFile(std::string_view path,
                                      const std::vector<uint64_t>& data) const {
  std::string file_path(path);
  std::ofstream ofs(file_path, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Failed to open file for writing: " + file_path);
  }
  ofs.write(reinterpret_cast<const char*>(data.data()),
            data.size() * sizeof(uint64_t));
  ofs.close();
}

template <typename T>
const T* YPirServer<T>::Db() const {
  return reinterpret_cast<const T*>(db_buf_.data());
}

template <typename T>
T* YPirServer<T>::DbMut() {
  return reinterpret_cast<T*>(db_buf_.data());
}

static uint32_t log2_uint64(uint64_t n) {
  if (n == 0) return 0;
  return static_cast<uint32_t>(std::log2(static_cast<double>(n)));
}

// template <typename T>
// std::vector<T> TransposeGeneric(const std::vector<T>& input, size_t rows,
//                                 size_t cols) {
//   if (input.size() != rows * cols) {
//     throw std::invalid_argument("TransposeGeneric: Input size mismatch");
//   }
//   std::vector<T> output(input.size());
//   for (size_t r = 0; r < rows; ++r) {
//     for (size_t c = 0; c < cols; ++c) {
//       output[c * rows + r] = input[r * cols + c];
//     }
//   }
//   return output;
// }

template <typename T>
std::vector<T> TransposeGeneric(const std::vector<T>& a, size_t a_rows,
                                size_t a_cols) {
  size_t transpose_tile_size = std::min({size_t(32), a_rows, a_cols});

  if (a_rows % transpose_tile_size != 0 || a_cols % transpose_tile_size != 0) {
    transpose_tile_size = 1;
  }

  std::vector<T> out(a_rows * a_cols);

  for (size_t i_outer = 0; i_outer < a_rows; i_outer += transpose_tile_size) {
    for (size_t j_outer = 0; j_outer < a_cols; j_outer += transpose_tile_size) {
      for (size_t i_inner = 0; i_inner < transpose_tile_size; ++i_inner) {
        for (size_t j_inner = 0; j_inner < transpose_tile_size; ++j_inner) {
          size_t i = i_outer + i_inner;
          size_t j = j_outer + j_inner;
          out[j * a_rows + i] = a[i * a_cols + j];
        }
      }
    }
  }

  return out;
}

template <typename T>
std::vector<PolyMatrixNtt> YPirServer<T>::GeneratePseudorandomQuery(
    uint8_t public_seed_idx) const {
  auto client = SpiralClient(params_);
  YClient y_client(client, params_);

  auto query =
      y_client.GenerateQueryImpl(public_seed_idx, params_.DbDim1(), true, 0);

  std::vector<PolyMatrixNtt> preprocessed_query;
  preprocessed_query.reserve(query.size());

  for (const auto& x : query) {
    auto query_raw_sub = x.SubMatrix(0, 0, 1, 1);

    const auto& poly_data = query_raw_sub.Poly(0, 0);

    auto query_raw_transformed =
        NegacyclicPerm(poly_data, 0, params_.Modulus());

    PolyMatrixRaw query_transformed_pol =
        PolyMatrixRaw::Zero(params_.PolyLen(), 1, 1);

    std::copy(query_raw_transformed.begin(), query_raw_transformed.end(),
              query_transformed_pol.Data().begin());

    preprocessed_query.push_back(ToNtt(params_, query_transformed_pol));
  }

  return preprocessed_query;
}

template <typename T>
std::vector<uint64_t> YPirServer<T>::MultiplyWithDbRing(
    const std::vector<PolyMatrixNtt>& preprocessed_query, size_t col_start,
    size_t col_end, uint8_t seed_idx) const {
  size_t db_rows_poly = 1ULL << params_.DbDim1();
  size_t poly_len = params_.PolyLen();
  size_t padded_rows = params_.DbRowsPadded();

  if (preprocessed_query.size() != db_rows_poly) {
    throw std::runtime_error(
        "MultiplyWithDbRing: preprocessed_query size mismatch");
  }

  std::vector<uint64_t> result;
  size_t num_cols = col_end - col_start;
  result.reserve(num_cols * poly_len);

  const T* db_ptr = Db();

  auto prod = PolyMatrixNtt::Zero(params_.CrtCount(), params_.PolyLen(), 1, 1);
  auto db_elem_poly = PolyMatrixRaw::Zero(params_.PolyLen(), 1, 1);
  auto db_elem_ntt =
      PolyMatrixNtt::Zero(params_.CrtCount(), params_.PolyLen(), 1, 1);

  for (size_t col = col_start; col < col_end; ++col) {
    auto sum = PolyMatrixNtt::Zero(params_.CrtCount(), params_.PolyLen(), 1, 1);

    for (size_t row = 0; row < db_rows_poly; ++row) {
      size_t offset = col * padded_rows + row * poly_len;

      for (size_t z = 0; z < poly_len; ++z) {
        db_elem_poly.Data()[z] = static_cast<uint64_t>(db_ptr[offset + z]);
      }

      ToNtt(params_, db_elem_ntt, db_elem_poly);

      Multiply(params_, prod, preprocessed_query[row], db_elem_ntt);

      if (row == db_rows_poly - 1) {
        AddInto(params_, sum, prod);
      } else {
        AddInto(params_, sum, prod);
      }
    }

    PolyMatrixRaw sum_raw = FromNtt(params_, sum);

    if (seed_idx == SEED_0 && !is_simplepir_) {
      auto sum_raw_transformed =
          NegacyclicPerm(sum_raw.Data(), 0, params_.Modulus());

      result.insert(result.end(), sum_raw_transformed.begin(),
                    sum_raw_transformed.end());
    } else {
      result.insert(result.end(), sum_raw.Data().begin(), sum_raw.Data().end());
    }
  }

  return TransposeGeneric(result, num_cols, poly_len);
}

template <typename T>
std::vector<uint64_t> YPirServer<T>::AnswerHintRing(uint8_t public_seed_idx,
                                                    size_t cols) const {
  auto preprocessed_query = GeneratePseudorandomQuery(public_seed_idx);

  auto res = MultiplyWithDbRing(preprocessed_query, 0, cols, public_seed_idx);

  return res;
}

template <typename T>
std::vector<uint64_t> YPirServer<T>::GenerateHint0Ring() const {
  size_t db_rows = 1ULL << (params_.DbDim1() + params_.PolyLenLog2());
  size_t db_cols = GetDbCols();
  size_t padded_rows = params_.DbRowsPadded();
  auto lwe_params = LWEParams::Default();
  size_t n = lwe_params.n;

  Convolution conv(n);
  const auto& conv_params = conv.params();

  std::vector<uint64_t> hint_0(n * db_cols, 0);
  size_t convd_len = conv_params.CrtCount() * conv_params.PolyLen();

  // 和 client 使用相同种子
  yacl::crypto::Prg<uint32_t> rng_pub(SEED_0);

  std::vector<std::vector<uint32_t>> v_nega_perm_a;
  size_t num_blocks = db_rows / n;
  v_nega_perm_a.reserve(num_blocks);

  for (size_t i = 0; i < num_blocks; ++i) {
    std::vector<uint32_t> a(n);
    for (size_t idx = 0; idx < n; ++idx) {
      a[idx] = rng_pub();
    }

    auto nega_perm_a = negacyclic_perm_u32(a);
    auto nega_perm_a_ntt = conv.ntt(nega_perm_a);
    v_nega_perm_a.push_back(std::move(nega_perm_a_ntt));
  }

  uint32_t log2_conv_output = log2_uint64(lwe_params.modulus) +
                              log2_uint64(lwe_params.n) +
                              log2_uint64(lwe_params.pt_modulus);

  uint32_t log2_modulus = log2_uint64(conv_params.Modulus());

  if (log2_modulus <= log2_conv_output + 1) {
    throw std::runtime_error(
        "Convolution modulus too small for lazy reduction logic");
  }

  uint32_t log2_max_adds = log2_modulus - log2_conv_output - 1;
  size_t max_adds = 1ULL << log2_max_adds;

  const T* db_view = Db();

  std::vector<uint64_t> tmp_col(convd_len, 0);
  std::vector<uint32_t> col_poly_u32(convd_len, 0);

  for (size_t col = 0; col < db_cols; ++col) {
    std::fill(tmp_col.begin(), tmp_col.end(), 0);

    for (size_t outer_row = 0; outer_row < num_blocks; ++outer_row) {
      size_t start_idx = col * padded_rows + outer_row * n;
      const T* pt_col = db_view + start_idx;
      std::vector<uint32_t> pt_col_u32(n);
      for (size_t i = 0; i < n; ++i) {
        pt_col_u32[i] = static_cast<uint32_t>(static_cast<uint64_t>(pt_col[i]));
      }

      auto pt_ntt = conv.ntt(pt_col_u32);
      auto convolved_ntt = conv.pointwise_mul(v_nega_perm_a[outer_row], pt_ntt);
      for (size_t r = 0; r < convd_len; ++r) {
        tmp_col[r] += static_cast<uint64_t>(convolved_ntt[r]);
      }

      if (outer_row % max_adds == max_adds - 1 || outer_row == num_blocks - 1) {
        for (size_t i = 0; i < conv_params.CrtCount(); ++i) {
          for (size_t j = 0; j < conv_params.PolyLen(); ++j) {
            size_t idx = i * conv_params.PolyLen() + j;
            uint64_t val = arith::BarrettCoeffU64(conv_params, tmp_col[idx], i);
            col_poly_u32[idx] = static_cast<uint32_t>(val);
          }
        }

        auto col_poly_raw = conv.raw(col_poly_u32);

        for (size_t i = 0; i < n; ++i) {
          size_t hint_idx = i * db_cols + col;
          hint_0[hint_idx] += static_cast<uint64_t>(col_poly_raw[i]);
          hint_0[hint_idx] &= 0xFFFFFFFF;
        }

        std::fill(tmp_col.begin(), tmp_col.end(), 0);
      }
    }
  }
  return hint_0;
}

template <typename T>
OfflinePrecomputedValues YPirServer<T>::PerformOfflinePrecomputation() {
  YACL_ENFORCE(!is_simplepir_, "Must be DoublePIR mode");

  LWEParams lwe_params = LWEParams::Default();
  size_t db_cols = 1ULL << (params_.DbDim2() + params_.PolyLenLog2());

  uint64_t lwe_q_prime = lwe_params.GetQPrime2();
  size_t lwe_q_prime_bits = lwe_params.q2_bits;

  size_t pt_bits = params_.PtModulusBitLen();
  double blowup_factor = static_cast<double>(lwe_q_prime_bits) / pt_bits;

  size_t special_offs = static_cast<size_t>(std::ceil(
      static_cast<double>(lwe_params.n * lwe_q_prime_bits) / pt_bits));
  size_t special_bit_offs = special_offs * pt_bits;

  Params smaller_params = params_;
  smaller_params.SetDbDim1(params_.DbDim2());
  double num = blowup_factor * static_cast<double>(lwe_params.n + 1);
  double denom = static_cast<double>(params_.PolyLen());
  size_t smaller_dim2 = static_cast<size_t>(std::ceil(std::log2(num / denom)));
  smaller_params.SetDbDim2(smaller_dim2);

  size_t out_rows = 1ULL << (smaller_params.DbDim2() + params_.PolyLenLog2());

  auto hint_0 = GenerateHint0Ring();

  std::vector<uint64_t> intermediate_cts = hint_0;
  intermediate_cts.resize(hint_0.size() + db_cols, 0);

  std::vector<uint64_t> intermediate_cts_rescaled;
  intermediate_cts_rescaled.reserve(intermediate_cts.size());
  for (uint64_t val : intermediate_cts) {
    intermediate_cts_rescaled.push_back(
        arith::Rescale(val, lwe_params.modulus, lwe_q_prime));
  }

  auto smaller_db_u16 =
      SplitAlloc(intermediate_cts_rescaled, special_bit_offs, lwe_params.n + 1,
                 db_cols, out_rows, lwe_q_prime_bits, pt_bits);
  YPirServer<uint16_t> smaller_server(smaller_params, std::move(smaller_db_u16),
                                      false, true, false);

  size_t smaller_cols =
      1ULL << (smaller_params.DbDim2() + smaller_params.PolyLenLog2());
  auto hint_1 = smaller_server.AnswerHintRing(SEED_1, smaller_cols);

  auto pseudorandom_query_1 = smaller_server.GeneratePseudorandomQuery(SEED_1);

  auto y_constants = GenYConstants(params_);

  std::vector<uint64_t> combined_hint_1 = hint_1;
  combined_hint_1.resize(hint_1.size() + out_rows, 0);

  // 验证 combined_hint_1 的大小
  size_t expected_combined_size = out_rows * (params_.PolyLen() + 1);
  SPDLOG_INFO(
      "Offline: hint_1.size()={}, out_rows={}, combined_hint_1.size()={}, "
      "expected={}",
      hint_1.size(), out_rows, combined_hint_1.size(), expected_combined_size);
  YACL_ENFORCE(combined_hint_1.size() == expected_combined_size,
               "combined_hint_1 size mismatch: {} != {}",
               combined_hint_1.size(), expected_combined_size);

  size_t rho = 1ULL << smaller_params.DbDim2();
  SPDLOG_INFO("Offline: rho={}, smaller_params.DbDim2()={}", rho,
              smaller_params.DbDim2());

  auto prepacked_lwe = PrepPackManyLwes(params_, combined_hint_1, rho);

  auto fake_pack_pub_params = GenerateFakePackPubParams(params_);

  Precomp precomp;
  precomp.reserve(prepacked_lwe.size());
  for (const auto& lwe : prepacked_lwe) {
    precomp.push_back(PrecomputePack(params_, params_.PolyLenLog2(), lwe,
                                     fake_pack_pub_params, y_constants));
  }

  OfflinePrecomputedValues vals;
  vals.hint_0 = std::move(hint_0);
  vals.hint_1 = std::move(hint_1);
  vals.pseudorandom_query_1 = std::move(pseudorandom_query_1);
  vals.y_constants = std::move(y_constants);

  vals.smaller_server =
      std::make_shared<YPirServer<uint16_t>>(std::move(smaller_server));

  vals.fake_pack_pub_params = std::move(fake_pack_pub_params);
  vals.prepacked_lwe = std::move(prepacked_lwe);
  vals.precomp = std::move(precomp);

  SPDLOG_INFO("DoublePIR offline precomputation complete");
  return vals;
}

template <typename T>
std::vector<uint32_t> YPirServer<T>::LweMultiplyBatchedWithDbPacked(
    absl::Span<const uint32_t> aligned_query_packed) const {
  size_t db_cols = GetDbCols();
  size_t db_rows_padded = params_.DbRowsPadded();

  YACL_ENFORCE(aligned_query_packed.size() == db_rows_padded,
               "Query size mismatch: expected {}, got {}", db_rows_padded,
               aligned_query_packed.size());

  std::vector<uint32_t> result((db_cols + 8), 0);
  size_t a_rows = db_cols;
  size_t a_true_cols = db_rows_padded;

  size_t a_cols = a_true_cols / 4;

  size_t b_rows = a_true_cols;
  size_t b_cols = 1;

  MatMulVecPacked(result.data(), DbU32Data(), aligned_query_packed.data(),
                  a_rows, a_cols, b_rows, b_cols);

  auto transposed_result = TransposeGeneric(result, db_cols, 1);

  return transposed_result;
}

template <typename T>
std::vector<uint64_t> YPirServer<T>::MultiplyBatchedWithDbPacked(
    absl::Span<const uint64_t> aligned_query_packed, size_t query_rows) const {
  size_t db_rows_padded = params_.DbRowsPadded();
  size_t db_cols = GetDbCols();

  YACL_ENFORCE(query_rows == 1, "Query rows must be 1 for this implementation");
  YACL_ENFORCE(db_rows_padded > 0, "Db rows cannot be 0");
  YACL_ENFORCE(aligned_query_packed.size() % db_rows_padded == 0,
               "Input query size alignment error");

  size_t K = aligned_query_packed.size() / db_rows_padded;

  std::vector<uint64_t> result(K * db_cols, 0);
  FastBatchedDotProduct(params_, result.data(), aligned_query_packed.data(),
                        db_rows_padded, Db(), db_rows_padded, db_cols);

  return result;
}

template <typename T>
std::vector<uint64_t> YPirServer<T>::AnswerQuery(
    absl::Span<const uint64_t> aligned_query_packed) {
  return MultiplyBatchedWithDbPacked(aligned_query_packed, 1);
}

template <typename T>
std::vector<std::vector<uint8_t>> YPirServer<T>::PerformOnlineComputation(
    OfflinePrecomputedValues& offline_vals,
    const std::vector<uint32_t>& first_dim_queries_packed,
    const std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
        second_dim_queries) {
  auto params = params_;
  auto lwe_params = LWEParams::Default();
  size_t db_cols = GetDbCols();

  uint64_t rlwe_q_prime_1 = params.GetQPrime1();
  uint64_t rlwe_q_prime_2 = params.GetQPrime2();

  size_t lwe_q_prime_bits = lwe_params.q2_bits;
  uint64_t lwe_q_prime = lwe_params.GetQPrime2();

  size_t pt_bits = static_cast<size_t>(
      std::floor(std::log2(static_cast<double>(params.PtModulus()))));

  double blowup_factor = static_cast<double>(lwe_q_prime_bits) / pt_bits;

  size_t special_offs = static_cast<size_t>(std::ceil(
      (static_cast<double>(lwe_params.n * lwe_q_prime_bits)) / pt_bits));

  Params smaller_params = params;
  smaller_params.SetDbDim1(params.DbDim2());
  double num = blowup_factor * static_cast<double>(lwe_params.n + 1);
  double denom = static_cast<double>(params.PolyLen());
  size_t smaller_dim2 = static_cast<size_t>(std::ceil(std::log2(num / denom)));
  smaller_params.SetDbDim2(smaller_dim2);

  size_t out_rows = 1ULL << (smaller_params.DbDim2() + params.PolyLenLog2());
  size_t rho = 1ULL << smaller_params.DbDim2();
  SPDLOG_INFO("Online: out_rows={}, rho={}, blowup_factor={}, special_offs={}",
              out_rows, rho, blowup_factor, special_offs);

  auto& hint_1_combined = offline_vals.hint_1;
  const auto& pseudorandom_query_1 = offline_vals.pseudorandom_query_1;

  const auto& y_constants = offline_vals.y_constants;
  auto& smaller_server = *(offline_vals.smaller_server);
  const auto& prepacked_lwe = offline_vals.prepacked_lwe;
  const auto& fake_pack_pub_params = offline_vals.fake_pack_pub_params;
  const auto& precomp = offline_vals.precomp;
  // Begin online computation

  auto intermediate = LweMultiplyBatchedWithDbPacked(first_dim_queries_packed);

  std::vector<std::vector<uint8_t>> responses;
  responses.reserve(second_dim_queries.size());

  size_t num_chunks = second_dim_queries.size();

  for (size_t i = 0; i < num_chunks; ++i) {
    const auto& packed_query_col = second_dim_queries[i].first;
    const auto& pack_pub_params_row_1s = second_dim_queries[i].second;

    std::vector<uint64_t> intermediate_cts_rescaled;
    intermediate_cts_rescaled.reserve(db_cols);
    size_t chunk_offset = i * db_cols;
    for (size_t j = 0; j < db_cols; ++j) {
      intermediate_cts_rescaled.push_back(
          arith::Rescale(static_cast<uint64_t>(intermediate[chunk_offset + j]),
                         lwe_params.modulus, lwe_q_prime));
    }

    {
      uint16_t* smaller_db_mut = smaller_server.DbMut();

      for (size_t j = 0; j < db_cols; ++j) {
        uint64_t val = intermediate_cts_rescaled[j];
        size_t blowup_ceil = static_cast<size_t>(std::ceil(blowup_factor));
        for (size_t m = 0; m < blowup_ceil; ++m) {
          size_t out_idx = (special_offs + m) * db_cols + j;
          uint16_t val_part = static_cast<uint16_t>((val >> (m * pt_bits)) &
                                                    ((1ULL << pt_bits) - 1));
          smaller_db_mut[out_idx] = val_part;
        }
      }
    }

    size_t blowup_factor_ceil = static_cast<size_t>(std::ceil(blowup_factor));
    auto secondary_hint = smaller_server.MultiplyWithDbRing(
        pseudorandom_query_1, special_offs, special_offs + blowup_factor_ceil,
        SEED_1);

    for (size_t r = 0; r < params.PolyLen(); ++r) {
      for (size_t j = 0; j < blowup_factor_ceil; ++j) {
        size_t inp_idx = r * blowup_factor_ceil + j;
        size_t out_idx = r * out_rows + special_offs + j;
        hint_1_combined[out_idx] = secondary_hint[inp_idx];
      }
    }
    auto response = smaller_server.AnswerQuery(packed_query_col);

    std::vector<PolyMatrixNtt> excess_cts;
    excess_cts.reserve(blowup_factor_ceil);

    for (size_t j = special_offs; j < special_offs + blowup_factor_ceil; ++j) {
      auto rlwe_ct = PolyMatrixRaw::Zero(params.PolyLen(), 2, 1);
      std::vector<uint64_t> poly;
      poly.reserve(params.PolyLen());
      for (size_t k = 0; k < params.PolyLen(); ++k) {
        poly.push_back(hint_1_combined[k * out_rows + j]);
      }

      auto nega = NegacyclicPerm(poly, 0, params.Modulus());
      std::copy(nega.begin(), nega.end(), rlwe_ct.Poly(0, 0).begin());
      excess_cts.push_back(ToNtt(params, rlwe_ct));
    }

    auto pack_pub_params_row_1s_pms =
        UnpackVecPm(params, 1, params.TExpLeft(), pack_pub_params_row_1s);

    auto packed = PackManyLwes(params, prepacked_lwe, precomp, response, rho,
                               pack_pub_params_row_1s_pms, y_constants);

    auto pack_pub_params = fake_pack_pub_params;

    for (size_t k = 0; k < pack_pub_params.size(); ++k) {
      auto uncondensed =
          UncondenseMatrix(params, pack_pub_params_row_1s_pms[k]);
      pack_pub_params[k].CopyInto(uncondensed, 1, 0);
    }

    auto other_packed = PackUsingSingleWithOffset(params, pack_pub_params,
                                                  excess_cts, special_offs);

    AddInto(params, packed[0], other_packed);

    std::vector<uint8_t> concated_bytes;
    for (const auto& ct : packed) {
      auto res = FromNtt(params, ct);
      std::vector<uint8_t> res_switched =
          ModulusSwitch(params, res, rlwe_q_prime_1, rlwe_q_prime_2);
      concated_bytes.insert(concated_bytes.end(), res_switched.begin(),
                            res_switched.end());
    }
    responses.push_back(std::move(concated_bytes));
  }
  return responses;
}

template <typename T>
OfflinePrecomputedValues YPirServer<T>::PerformOfflinePrecomputationSimplepir(
    absl::Span<const uint64_t> hint_0_load, std::string_view hint_0_store) {
  if (!is_simplepir_) {
    YACL_THROW(
        "PerformOfflinePrecomputationSimplepir called but is_simplepir is "
        "false");
  }

  size_t db_cols = params_.Instances() * params_.PolyLen();
  size_t num_rlwe_outputs = db_cols / params_.PolyLen();

  std::vector<uint64_t> hint_0;

  if (!hint_0_load.empty()) {
    hint_0.assign(hint_0_load.begin(), hint_0_load.end());
  } else {
    hint_0 = AnswerHintRing(SEED_0, db_cols);

    if (!hint_0_store.empty()) {
      WriteVecU64ToFile(hint_0_store, hint_0);
      SPDLOG_INFO("Saved hint_0 to {}", hint_0_store);
    }
  }

  std::pair<std::vector<PolyMatrixNtt>, std::vector<PolyMatrixNtt>>
      y_constants = GenerateYConstants(params_);

  std::vector<uint64_t> combined = hint_0;
  combined.resize(combined.size() + db_cols, 0);

  if (combined.size() != db_cols * (params_.PolyLen() + 1)) {
    SPDLOG_WARN("Combined size check mismatch warning (check PolyLen vs N)");
  }

  std::vector<std::vector<PolyMatrixNtt>> prepacked_lwe =
      PrepPackManyLwes(params_, combined, num_rlwe_outputs);

  std::vector<PolyMatrixNtt> fake_pack_pub_params =
      GenerateFakePackPubParams(params_);

  Precomp precomp;
  precomp.reserve(prepacked_lwe.size());

  for (size_t i = 0; i < prepacked_lwe.size(); ++i) {
    precomp.push_back(PrecomputePack(params_, params_.PolyLenLog2(),
                                     prepacked_lwe[i], fake_pack_pub_params,
                                     y_constants));
  }

  OfflinePrecomputedValues result;
  result.hint_0 = std::move(hint_0);
  result.hint_1 = {};
  result.pseudorandom_query_1 = {};
  result.y_constants = std::move(y_constants);
  // result.smaller_server = nullptr;
  result.prepacked_lwe = std::move(prepacked_lwe);
  result.fake_pack_pub_params = std::move(fake_pack_pub_params);
  result.precomp = std::move(precomp);

  return result;
}

template <typename T>
std::vector<uint8_t> YPirServer<T>::PerformOnlineComputationSimplepir(
    absl::Span<const uint64_t> first_dim_queries_packed,
    const OfflinePrecomputedValues& offline_vals,
    absl::Span<const absl::Span<const uint64_t>> pack_pub_params_row_1s) {
  if (!is_simplepir_) {
    throw std::runtime_error(
        "PerformOnlineComputationSimplepir: Not in SimplePIR mode");
  }

  const auto& y_constants = offline_vals.y_constants;
  const auto& prepacked_lwe = offline_vals.prepacked_lwe;
  const auto& precomp = offline_vals.precomp;

  uint64_t rlwe_q_prime_1 = params_.GetQPrime1();
  uint64_t rlwe_q_prime_2 = params_.GetQPrime2();

  size_t db_rows = 1ULL << (params_.DbDim1() + params_.PolyLenLog2());
  size_t db_cols = params_.Instances() * params_.PolyLen();

  if (first_dim_queries_packed.size() != params_.DbRowsPadded()) {
    throw std::runtime_error("Query size mismatch with DB padded rows");
  }

  std::vector<uint64_t> intermediate(db_cols, 0);

  FastBatchedDotProduct(params_, intermediate.data(),
                        first_dim_queries_packed.data(), db_rows, Db(), db_rows,
                        db_cols);

  size_t num_rlwe_outputs = db_cols / params_.PolyLen();

  if (pack_pub_params_row_1s.empty()) {
    throw std::runtime_error("Missing packing public params");
  }
  auto pack_pub_params_row_1s_pms =
      UnpackVecPm(params_, 1, params_.TExpLeft(), pack_pub_params_row_1s[0]);

  auto packed = PackManyLwes(
      params_, prepacked_lwe, precomp, absl::MakeConstSpan(intermediate),
      num_rlwe_outputs, pack_pub_params_row_1s_pms, y_constants);

  std::vector<std::vector<uint8_t>> packed_mod_switched;
  packed_mod_switched.reserve(packed.size());

  for (const auto& ct : packed) {
    PolyMatrixRaw res = FromNtt(params_, ct);
    std::vector<uint8_t> res_switched =
        ModulusSwitch(params_, res, rlwe_q_prime_1, rlwe_q_prime_2);
    packed_mod_switched.emplace_back(std::move(res_switched));
  }

  if (packed_mod_switched.size() != num_rlwe_outputs) {
    throw std::runtime_error("Packed output size mismatch");
  }

  size_t total_bytes = 0;
  for (const auto& vec : packed_mod_switched) {
    total_bytes += vec.size();
  }

  std::vector<uint8_t> concated;
  concated.resize(total_bytes);

  uint8_t* dest_ptr = concated.data();
  for (const auto& vec : packed_mod_switched) {
    std::memcpy(dest_ptr, vec.data(), vec.size());
    dest_ptr += vec.size();
  }

  return concated;
}

template class YPirServer<uint16_t>;
template class YPirServer<uint8_t>;
}  // namespace psi::ypir