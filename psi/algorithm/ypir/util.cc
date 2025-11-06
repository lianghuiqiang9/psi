#include "psi/algorithm/ypir/util.h"

#include <cstring>
#include <vector>

#include "../spiral/arith/arith_params.h"
#include "../spiral/arith/ntt.h"
#include "../spiral/discrete_gaussian.h"
#include "../spiral/gadget.h"
#include "../spiral/poly_matrix_utils.h"
#include "yacl/base/int128.h"

namespace psi::ypir {

using namespace psi::spiral;

PolyMatrixNtt HomomorphicAutomorph(const Params& params, size_t t, size_t t_exp,
                                   const PolyMatrixNtt& ct,
                                   const PolyMatrixNtt& pub_param) {
  YACL_ENFORCE(ct.Rows() == static_cast<size_t>(2));
  YACL_ENFORCE(ct.Cols() == static_cast<size_t>(1));

  auto ct_raw = PolyMatrixRaw::Zero(params.PolyLen(), 2, 1);
  FromNtt(params, ct_raw, ct);
  auto ct_auto = Automorphism(params, ct_raw, t);

  auto ginv_ct = PolyMatrixRaw::Zero(params.PolyLen(), t_exp, 1);
  psi::spiral::util::GadgetInvertRdim(params, ginv_ct, ct_auto, 1);

  auto ginv_ct_ntt =
      PolyMatrixNtt::Zero(params.CrtCount(), params.PolyLen(), t_exp, 1);
  for (size_t i = 1; i < t_exp; ++i) {
    auto pol_src = ginv_ct.Poly(i, 0);
    auto pol_dst = ginv_ct_ntt.Poly(i, 0);
    ReduceCopy(params, pol_dst, pol_src);
    arith::NttForward(params, pol_dst);
  }

  auto w_times_ginv_ct = Multiply(params, pub_param, ginv_ct_ntt);

  auto ct_auto_1 = PolyMatrixRaw::Zero(params.PolyLen(), 1, 1);

  std::memcpy(ct_auto_1.Data().data(),
              ct_auto.Data().data() + ct_auto.PolyStartIndex(1, 0),
              sizeof(uint64_t) * ct_auto.NumWords());
  auto ct_auto_1_ntt = ToNtt(params, ct_auto_1);

  auto res = Add(params, ct_auto_1_ntt.PadTop(1), w_times_ginv_ct);
  return res;
}

PolyMatrixNtt RingPackLwesInner(
    const Params& params, size_t ell, size_t start_idx,
    const std::vector<PolyMatrixNtt>& rlwe_cts,
    const std::vector<PolyMatrixNtt>& pub_params,
    const std::pair<std::vector<PolyMatrixNtt>, std::vector<PolyMatrixNtt>>&
        y_constants) {
  YACL_ENFORCE_EQ(pub_params.size(), params.PolyLenLog2());

  if (ell == 0) {
    return rlwe_cts[start_idx];
  }

  size_t step = 1ULL << (params.PolyLenLog2() - ell);
  size_t even = start_idx;
  size_t odd = start_idx + step;

  auto ct_even = RingPackLwesInner(params, ell - 1, even, rlwe_cts, pub_params,
                                   y_constants);
  auto ct_odd = RingPackLwesInner(params, ell - 1, odd, rlwe_cts, pub_params,
                                  y_constants);

  const auto& y = y_constants.first[ell - 1];
  const auto& neg_y = y_constants.second[ell - 1];

  auto y_times_ct_odd = ScalarMultiply(params, y, ct_odd);
  auto neg_y_times_ct_odd = ScalarMultiply(params, neg_y, ct_odd);

  auto ct_sum_1 = ct_even;
  AddInto(params, ct_sum_1, neg_y_times_ct_odd);
  AddInto(params, ct_even, y_times_ct_odd);

  size_t t = (1ULL << ell) + 1;
  const auto& pub_param = pub_params[params.PolyLenLog2() - 1 - (ell - 1)];
  auto ct_sum_1_automorphed =
      HomomorphicAutomorph(params, t, params.TExpLeft(), ct_sum_1, pub_param);

  return Add(params, ct_even, ct_sum_1_automorphed);
}

std::pair<std::vector<PolyMatrixNtt>, std::vector<PolyMatrixNtt>> GenYConstants(
    const Params& params) {
  std::vector<PolyMatrixNtt> y_constants;
  std::vector<PolyMatrixNtt> neg_y_constants;

  for (size_t num_cts_log2 = 1; num_cts_log2 <= params.PolyLenLog2();
       ++num_cts_log2) {
    size_t num_cts = 1ULL << num_cts_log2;

    auto y_raw = PolyMatrixRaw::Zero(params.PolyLen(), 1, 1);
    size_t idx = params.PolyLen() / num_cts;
    y_raw.Data()[idx] = 1ULL;
    auto y = ToNtt(params, y_raw);

    auto neg_y_raw = PolyMatrixRaw::Zero(params.PolyLen(), 1, 1);
    neg_y_raw.Data()[idx] = params.Modulus() - 1;
    auto neg_y = ToNtt(params, neg_y_raw);

    y_constants.push_back(y);
    neg_y_constants.push_back(neg_y);
  }

  return std::make_pair(y_constants, neg_y_constants);
}

std::vector<PolyMatrixNtt> RawGenExpansionParams(
    const Params& params, const PolyMatrixRaw& sk_reg, size_t num_exp,
    size_t m_exp, yacl::crypto::Prg<uint64_t>& rng,
    yacl::crypto::Prg<uint64_t>& rng_pub) {
  auto g_exp = util::BuildGadget(params, 1, m_exp);
  auto g_exp_ntt = ToNtt(params, g_exp);

  std::vector<PolyMatrixNtt> res;
  DiscreteGaussian dg(params.NoiseWidth());

  for (size_t i = 0; i < num_exp; ++i) {
    size_t t = (params.PolyLen() / (1ULL << i)) + 1;
    auto tau_sk_reg = Automorphism(params, sk_reg, t);
    auto tau_sk_reg_ntt = ToNtt(params, tau_sk_reg);
    auto prod = Multiply(params, tau_sk_reg_ntt, g_exp_ntt);

    auto m = prod.Cols();
    auto sample =
        PolyMatrixNtt::Zero(params.CrtCount(), params.PolyLen(), 2, m);

    for (size_t j = 0; j < m; ++j) {
      auto a = PolyMatrixRaw::RandomPrg(params, 1, 1, rng_pub);
      auto a_ntt = ToNtt(params, a);
      auto a_inv = ToNtt(params, Invert(params, a));

      auto e = Noise(params, 1, 1, dg, rng);
      auto e_ntt = ToNtt(params, e);

      auto sk_reg_ntt = ToNtt(params, sk_reg);
      auto b_p = Multiply(params, sk_reg_ntt, a_ntt);
      auto b = Add(params, e_ntt, b_p);

      auto regev_sample =
          PolyMatrixNtt::Zero(params.CrtCount(), params.PolyLen(), 2, 1);
      regev_sample.CopyInto(a_inv, 0, 0);
      regev_sample.CopyInto(b, 1, 0);

      sample.CopyInto(regev_sample, 0, j);
    }

    auto w_exp_i = Add(params, sample, prod.PadTop(1));
    res.push_back(w_exp_i);
  }
  return res;
}

PolyMatrixNtt RingPackLwes(
    const Params& params, const std::vector<uint64_t>& b_values,
    const std::vector<PolyMatrixNtt>& rlwe_cts, size_t num_cts,
    const std::vector<PolyMatrixNtt>& pub_params,
    const std::pair<std::vector<PolyMatrixNtt>, std::vector<PolyMatrixNtt>>&
        y_constants) {
  YACL_ENFORCE_EQ(b_values.size(), num_cts);
  YACL_ENFORCE_EQ(rlwe_cts.size(), num_cts);

  size_t ell = params.PolyLenLog2();
  auto out =
      RingPackLwesInner(params, ell, 0, rlwe_cts, pub_params, y_constants);

  auto out_raw = FromNtt(params, out);
  for (size_t z = 0; z < params.PolyLen(); ++z) {
    uint128_t b_val_u128 = static_cast<uint128_t>(b_values[z]);
    uint128_t poly_len_u128 = static_cast<uint128_t>(params.PolyLen());
    uint128_t prod = b_val_u128 * poly_len_u128;
    uint64_t val = arith::BarrettReductionU128(params, prod);

    size_t idx = out_raw.PolyStartIndex(1, 0) + z;
    uint64_t sum = out_raw.Data()[idx] + val;
    out_raw.Data()[idx] = arith::BarrettU64(params, sum);
  }

  return ToNtt(params, out_raw);
}

}  // namespace psi::ypir
