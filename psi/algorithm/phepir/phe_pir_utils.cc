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

#include "psi/algorithm/phepir/phe_pir_utils.h"

#include <cstring>
#include <unordered_set>

namespace psi::phepir {

void MultiplyByLinearFactor(std::vector<yacl::math::MPInt>& coeffs,
                            const yacl::math::MPInt& a,
                            const yacl::math::MPInt& Order) {
  size_t n = coeffs.size();

  // new poly coeffs, degree + 1
  std::vector<yacl::math::MPInt> new_coeffs(n + 1, yacl::math::MPInt::_0_);

  // (c0 + c1*x + ... + c_{n-1}*x^{n-1}) * (x - a)
  for (size_t i = 0; i < n; ++i) {
    // * x: right shift
    new_coeffs[i + 1] = (new_coeffs[i + 1] + coeffs[i]) % Order;

    // * -a: mul -a
    yacl::math::MPInt neg_a = (Order - a) % Order;  // -a mod Order
    new_coeffs[i] = (new_coeffs[i] + coeffs[i] * neg_a) % Order;
  }

  coeffs = std::move(new_coeffs);
}

std::pair<std::vector<yacl::math::MPInt>, std::vector<yacl::math::MPInt>>
GetInterpolatingAndRootPolyCoeffs(const std::vector<yacl::math::MPInt>& keys,
                                  const std::vector<yacl::math::MPInt>& values,
                                  const yacl::math::MPInt& order) {
  size_t n = keys.size();

  // 1. 计算差商表 (保持不变)
  std::vector<std::vector<yacl::math::MPInt>> divided_diffs(
      n, std::vector<yacl::math::MPInt>(n, yacl::math::MPInt::_0_));

  for (size_t i = 0; i < n; ++i) {
    divided_diffs[i][0] = values[i] % order;
  }

  for (size_t j = 1; j < n; ++j) {
    for (size_t i = 0; i < n - j; ++i) {
      yacl::math::MPInt numerator =
          (divided_diffs[i + 1][j - 1] - divided_diffs[i][j - 1] + order) %
          order;
      yacl::math::MPInt denominator =
          (keys[i + j] - keys[i] + order) % order;

      // 注意：这里需要确保 denominator != 0，即 keys 必须互不相同
      yacl::math::MPInt denom_inv = denominator.InvertMod(order);
      divided_diffs[i][j] = (numerator * denom_inv) % order;
    }
  }

  // 2. 构建插值多项式 (微调)
  std::vector<yacl::math::MPInt> coefficients(n, yacl::math::MPInt::_0_);

  // c0
  coefficients[0] = divided_diffs[0][0];

  // current_poly 用于累积基函数 (x-x0)(x-x1)...
  std::vector<yacl::math::MPInt> current_poly = {yacl::math::MPInt::_1_};

  for (size_t j = 1; j < n; ++j) {
    // 这一步把 (x - x_{j-1}) 乘进去
    // j=1 时，变为 (x-x0)
    // ...
    // j=n-1 时，变为 (x-x0)...(x-x_{n-2})
    MultiplyByLinearFactor(current_poly, keys[j - 1], order);

    yacl::math::MPInt cj = divided_diffs[0][j];
    // 更新插值多项式系数 P(x)
    for (size_t k = 0; k < current_poly.size(); ++k) {
      yacl::math::MPInt term = (current_poly[k] * cj) % order;
      // 注意 coefficients 的大小是 n，但 current_poly 的大小会增长
      // 在循环内 current_poly 的度数最大为 n-1，大小为 n，不会越界
      coefficients[k] = (coefficients[k] + term) % order;
    }
  }

  // 3. 【新增步骤】计算完整的 Root Polynomial
  // 此时 current_poly = (x-x0)...(x-x_{n-2})
  // Root Polynomial 需要再乘上最后一项 (x - x_{n-1})

  // 我们复用 current_poly 作为 root_coeffs
  std::vector<yacl::math::MPInt> root_coeffs = current_poly;

  // 执行最后一次乘法
  MultiplyByLinearFactor(root_coeffs, keys[n - 1], order);

  // 此时 root_coeffs 的大小应该是 n+1 (因为是 n 次多项式)

  return {coefficients, root_coeffs};
}

// std::vector<yacl::math::MPInt> ProductPolyDivConq(
//     const std::vector<yacl::math::MPInt>& y_values, size_t l, size_t r,
//     const yacl::math::MPInt& order) {
//   if (l > r) return {yacl::math::MPInt::_1_};
//   if (l == r) return {(order - y_values[l]) % order, yacl::math::MPInt::_1_};

//   size_t m = (l + r) / 2;
//   auto left = ProductPolyDivConq(y_values, l, m, order);
//   auto right = ProductPolyDivConq(y_values, m + 1, r, order);

//   std::vector<yacl::math::MPInt> res(left.size() + right.size() - 1,
//                                      yacl::math::MPInt::_0_);
//   for (size_t i = 0; i < left.size(); ++i)
//     for (size_t j = 0; j < right.size(); ++j)
//       res[i + j] = (res[i + j] + left[i] * right[j]) % order;

//   return res;
// }

}  // namespace psi::phepir