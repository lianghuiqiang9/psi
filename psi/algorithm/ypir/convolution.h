#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "psi/algorithm/spiral/params.h"

namespace psi::ypir {

using Params = spiral::Params;

class Convolution {
 public:
  explicit Convolution(size_t n);

  const Params& params() const { return params_; }

  // 前向变换: u32 -> u64 (Raw) -> NTT -> u32
  std::vector<uint32_t> ntt(const std::vector<uint32_t>& a) const;

  // 逆变换: u32 -> NTT (Raw) -> Inverse -> 模约减 -> u32
  std::vector<uint32_t> raw(const std::vector<uint32_t>& a) const;

  // 点乘: Element-wise multiplication with Barrett reduction
  std::vector<uint32_t> pointwise_mul(const std::vector<uint32_t>& a,
                                      const std::vector<uint32_t>& b) const;

  // 完整卷积流程: NTT -> Mul -> Raw
  std::vector<uint32_t> convolve(const std::vector<uint32_t>& a,
                                 const std::vector<uint32_t>& b) const;

 private:
  size_t n_;
  Params params_;
};

// -----------------------------------------------------------------------
// 3. 辅助数学函数 (Free Functions)
// -----------------------------------------------------------------------

// O(N^2) 朴素负循环卷积
std::vector<uint32_t> naive_negacyclic_convolve(const std::vector<uint32_t>& a,
                                                const std::vector<uint32_t>& b);

// 生成负循环矩阵 (结果转置)
std::vector<uint32_t> negacyclic_matrix_u32(const std::vector<uint32_t>& b);

// 特殊负循环置换: a -> [a[0], -a[n-1], -a[n-2]...]
std::vector<uint32_t> negacyclic_perm_u32(const std::vector<uint32_t>& a);

}  // namespace psi::ypir