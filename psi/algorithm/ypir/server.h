#pragma once

#include <cstdint>
#include <vector>

#include "absl/types/span.h"

#include "psi/algorithm/spiral/params.h"
#include "psi/algorithm/spiral/poly_matrix.h"
#include "psi/algorithm/ypir/types.h"

namespace psi::ypir {
using namespace psi::spiral;

// Forward declaration
class LWEClient;

template <typename T>
class YPirServer {
 public:
  YPirServer(const Params& params_in, const std::vector<T>& input_db,
             bool is_simplepir, bool inp_transposed, bool pad_rows_in);

  size_t GetDbCols() const;
  size_t GetDbRowsPadded() const { return params_.DbRowsPadded(); }
  const T* Db() const;
  T* DbMut();
  const Params& GetParams() const { return params_; }

  const uint32_t* DbU32Data() const {
    return reinterpret_cast<const uint32_t*>(db_buf_.data());
  }
  size_t DbU32Size() const { return db_buf_.size() * 2; }

  std::vector<uint64_t> GenerateHint0Ring() const;
  void WriteVecU64ToFile(std::string_view path,
                         const std::vector<uint64_t>& data) const;
  std::vector<PolyMatrixNtt> GeneratePseudorandomQuery(
      uint8_t public_seed_idx) const;
  std::vector<uint64_t> MultiplyWithDbRing(
      const std::vector<PolyMatrixNtt>& preprocessed_query, size_t col_start,
      size_t col_end, uint8_t seed_idx) const;
  std::vector<uint64_t> MultiplyBatchedWithDbPacked(
      absl::Span<const uint64_t> aligned_query_packed, size_t query_rows) const;
  std::vector<uint32_t> LweMultiplyBatchedWithDbPacked(
      absl::Span<const uint32_t> aligned_query_packed) const;
  std::vector<uint64_t> AnswerQuery(
      absl::Span<const uint64_t> aligned_query_packed);
  std::vector<uint64_t> AnswerHintRing(uint8_t public_seed_idx,
                                       size_t cols) const;

  OfflinePrecomputedValues PerformOfflinePrecomputation();

  std::vector<std::vector<uint8_t>> PerformOnlineComputation(
      OfflinePrecomputedValues& offline_vals,
      const std::vector<uint32_t>& first_dim_queries_packed,
      const std::vector<std::pair<std::vector<uint64_t>,
                                  std::vector<uint64_t>>>& second_dim_queries);

  OfflinePrecomputedValues PerformOfflinePrecomputationSimplepir(
      absl::Span<const uint64_t> hint_0_load,
      std::string_view hint_0_store_path);

  std::vector<uint8_t> PerformOnlineComputationSimplepir(
      absl::Span<const uint64_t> first_dim_queries_packed,
      const OfflinePrecomputedValues& offline_vals,
      absl::Span<const absl::Span<const uint64_t>> pack_pub_params_row_1s);

 private:
  const Params params_;
  // Params smaller_params_;
  std::vector<uint64_t> db_buf_;
  bool pad_rows_;
  bool is_simplepir_;
  bool inp_transposed_;

  // Debug fields
  LWEClient* debug_lwe_client_ = nullptr;
  std::vector<uint8_t> debug_original_db_;
  size_t debug_db_rows_ = 0;
  size_t debug_db_cols_ = 0;

 public:
  void SetDebugLweClient(LWEClient* client) { debug_lwe_client_ = client; }
  void SetDebugOriginalDb(const std::vector<uint8_t>& db, size_t rows,
                          size_t cols) {
    debug_original_db_ = db;
    debug_db_rows_ = rows;
    debug_db_cols_ = cols;
  }
};

}  // namespace psi::ypir