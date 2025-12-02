#include "psi/algorithm/ypir/convolution.h"

#include <cassert>
#include <iostream>

#include "psi/algorithm/spiral/arith/arith.h"
#include "psi/algorithm/spiral/arith/arith_params.h"
#include "psi/algorithm/spiral/params.h"
#include "psi/algorithm/spiral/poly_matrix.h"
#include "psi/algorithm/spiral/poly_matrix_utils.h"

namespace psi::ypir {
using namespace psi::spiral;

psi::spiral::Params CreateParams(size_t n) {
  std::size_t poly_len = n;

  std::vector<std::uint64_t> moduli{268369921, 249561089};
  double noise_width = 6.4;

  // PolyMatrixParams(n, pt_modulus, q2_bits, t_conv, t_exp_left, t_exp_right,
  // t_gsw)
  psi::spiral::PolyMatrixParams poly_matrix_params(1, 2, 28, 0, 0, 0, 0);

  // QueryParams(db_dim_1, db_dim_2, instances, query_seed_compressed)
  psi::spiral::QueryParams query_params(1, 1, 1, true);

  return psi::spiral::Params(poly_len, std::move(moduli), noise_width,
                             std::move(poly_matrix_params),
                             std::move(query_params));
}

Convolution::Convolution(size_t n) : n_(n) { params_ = CreateParams(n); }

std::vector<uint32_t> Convolution::ntt(const std::vector<uint32_t>& a) const {
  assert(a.size() == n_);

  // 1. 数据加载到 PolyMatrixRaw (u32 -> u64)
  auto inp = PolyMatrixRaw::Zero(params_.PolyLen(), 1, 1);
  for (size_t i = 0; i < n_; ++i) {
    inp.Data()[i] = static_cast<uint64_t>(a[i]);
  }

  // 2. 执行 NTT
  auto ntt_res = ToNtt(params_, inp);

  // 3. 提取结果 (u64 -> u32)
  // 注意：结果大小膨胀了 crt_count 倍
  size_t out_size = params_.CrtCount() * n_;
  std::vector<uint32_t> out(out_size);
  for (size_t i = 0; i < out_size; ++i) {
    // Rust 代码直接 cast，假设数据在 u32 范围内或截断
    out[i] = static_cast<uint32_t>(ntt_res.Data()[i]);
  }
  return out;
}

std::vector<uint32_t> Convolution::raw(const std::vector<uint32_t>& a) const {
  assert(a.size() == params_.CrtCount() * n_);

  // 1. 数据加载到 PolyMatrixNTT
  auto inp = PolyMatrixNtt::Zero(params_.CrtCount(), params_.PolyLen(), 1, 1);
  for (size_t i = 0; i < params_.CrtCount() * n_; ++i) {
    inp.Data()[i] = static_cast<uint64_t>(a[i]);
  }

  // 2. 执行 Inverse NTT (raw)
  auto raw_res = FromNtt(params_, inp);  // 返回 PolyMatrixRaw

  // 3. 复杂的模约减逻辑 (核心逻辑翻译)
  std::vector<uint32_t> out(n_);
  int64_t modulus_i64 = static_cast<int64_t>(params_.Modulus());
  int64_t two_pow_32 = 1LL << 32;

  for (size_t i = 0; i < n_; ++i) {
    // Rust: let mut val = raw.data[i] as i64;
    // 这里 raw_res.Data()[i] 实际上是 PolyMatrixRaw 的结果
    // 注意：如果你接了真实的数学库，这里需要确保数据索引正确
    int64_t val = static_cast<int64_t>(raw_res.Data()[i]);

    assert(val < modulus_i64);

    // Rust: if val > self.params.modulus / 2 { val -= modulus; }
    // 将模数域 [0, q) 映射到中心域 [-q/2, q/2)
    if (val > modulus_i64 / 2) {
      val -= modulus_i64;
    }

    // Rust:
    // if val < 0 {
    //     val += ((modulus) / (1<<32)) * (1<<32);
    //     val += 1<<32;
    // }
    // 这段逻辑非常特殊，看起来是为了将负数映射回 u32 的正数范围
    // 或者是处理某种特定的 Re-centering 逻辑
    if (val < 0) {
      val += (modulus_i64 / two_pow_32) * two_pow_32;
      val += two_pow_32;
    }

    assert(val >= 0);

    // Rust: out[i] = (val % (1i64 << 32)) as u32;
    out[i] = static_cast<uint32_t>(val % two_pow_32);
  }
  return out;
}

std::vector<uint32_t> Convolution::pointwise_mul(
    const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) const {
  size_t total_len = params_.CrtCount() * n_;
  assert(a.size() == total_len);
  assert(b.size() == total_len);

  std::vector<uint32_t> out(total_len);
  for (size_t m = 0; m < params_.CrtCount(); ++m) {
    for (size_t i = 0; i < n_; ++i) {
      size_t idx = m * n_ + i;
      uint64_t val =
          static_cast<uint64_t>(a[idx]) * static_cast<uint64_t>(b[idx]);
      // 执行 Barrett Reduction
      out[idx] = static_cast<uint32_t>(arith::BarrettCoeffU64(params_, val, m));
    }
  }
  return out;
}

std::vector<uint32_t> Convolution::convolve(
    const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) const {
  auto a_ntt = ntt(a);
  auto b_ntt = ntt(b);
  auto res = pointwise_mul(a_ntt, b_ntt);
  return raw(res);
}

// -----------------------------------------------------------------------
// 辅助函数实现
// -----------------------------------------------------------------------

std::vector<uint32_t> naive_negacyclic_convolve(
    const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  assert(a.size() == b.size());
  size_t n = a.size();
  std::vector<uint32_t> res(n, 0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      // Rust: let mut b_val = b[(n + i - j) % n];
      // C++: 注意负数取模问题，这里 n+i-j 始终为正，安全
      size_t idx = (n + i - j) % n;
      uint32_t b_val = b[idx];

      if (i < j) {
        // Rust: b_val.wrapping_neg() -> 对应 C++ (0 - b_val)
        // 在 C++ 中，对 unsigned 执行 0-x 会自动 wrap 到 2^32-x
        b_val = 0 - b_val;
      }

      // Rust: res[i].wrapping_add(a[j].wrapping_mul(b_val))
      // C++: uint32_t 的 + 和 * 默认就是 wrapping 的
      res[i] += a[j] * b_val;
    }
  }
  return res;
}

std::vector<uint32_t> negacyclic_matrix_u32(const std::vector<uint32_t>& b) {
  size_t n = b.size();
  std::vector<uint32_t> res(n * n, 0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      size_t idx = (n + i - j) % n;
      uint32_t b_val = b[idx];

      if (i < j) {
        b_val = 0 - b_val;  // wrapping_neg
      }

      // Rust: res[j * n + i] = b_val; // nb: transposed
      res[j * n + i] = b_val;
    }
  }
  return res;
}

std::vector<uint32_t> negacyclic_perm_u32(const std::vector<uint32_t>& a) {
  size_t n = a.size();
  std::vector<uint32_t> res(n);

  if (n > 0) {
    res[0] = a[0];
    for (size_t i = 1; i < n; ++i) {
      // Rust: a[(n - i) % n].wrapping_neg()
      // 由于 i 跑在 1..n，(n-i)%n 其实就是 n-i
      uint32_t val = a[n - i];
      res[i] = 0 - val;  // wrapping_neg
    }
  }
  return res;
}

}  // namespace psi::ypir