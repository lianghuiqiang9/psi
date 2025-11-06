#pragma once

#include <cstddef>

#include "../spiral/params.h"
#include "../spiral/poly_matrix.h"

namespace psi::ypir {

psi::spiral::PolyMatrixNtt HomomorphicAutomorph(
    const psi::spiral::Params& params, size_t t, size_t t_exp,
    const psi::spiral::PolyMatrixNtt& ct,
    const psi::spiral::PolyMatrixNtt& pub_param);

psi::spiral::PolyMatrixNtt RingPackLwesInner(
    const psi::spiral::Params& params, size_t ell, size_t start_idx,
    const std::vector<psi::spiral::PolyMatrixNtt>& rlwe_cts,
    const std::vector<psi::spiral::PolyMatrixNtt>& pub_params,
    const std::pair<std::vector<psi::spiral::PolyMatrixNtt>,
                    std::vector<psi::spiral::PolyMatrixNtt>>& y_constants);

std::pair<std::vector<psi::spiral::PolyMatrixNtt>,
          std::vector<psi::spiral::PolyMatrixNtt>>
GenYConstants(const psi::spiral::Params& params);

std::vector<psi::spiral::PolyMatrixNtt> RawGenExpansionParams(
    const psi::spiral::Params& params, const psi::spiral::PolyMatrixRaw& sk_reg,
    size_t num_exp, size_t m_exp, yacl::crypto::Prg<uint64_t>& rng,
    yacl::crypto::Prg<uint64_t>& rng_pub);

psi::spiral::PolyMatrixNtt RingPackLwes(
    const psi::spiral::Params& params, const std::vector<uint64_t>& b_values,
    const std::vector<psi::spiral::PolyMatrixNtt>& rlwe_cts, size_t num_cts,
    const std::vector<psi::spiral::PolyMatrixNtt>& pub_params,
    const std::pair<std::vector<psi::spiral::PolyMatrixNtt>,
                    std::vector<psi::spiral::PolyMatrixNtt>>& y_constants);

}  // namespace psi::ypir