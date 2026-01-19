#include "heu/library/algorithms/elgamal_kpir/kpir.h"

namespace heu::lib::algorithms::elgamal_kpir {


void Database::Random(uint32_t logN, uint32_t logY, uint32_t logL) {
    if (logN > logY) {
        throw std::runtime_error("Error: logN cannot be greater than logY. "
                                 "Not enough unique values available for Y.");
    }

    std::unordered_set<yacl::math::MPInt> used;
    uint32_t n = 1<<logN;
    auto yLen = yacl::math::MPInt(2).Pow(logY);
    auto lLen = yacl::math::MPInt(2).Pow(logL);

    Y.reserve(n);
    L.reserve(n);

    while (Y.size() < n) {
        auto y = yacl::math::MPInt::RandomLtN(yLen);
        if (!used.insert(y).second) continue;

        auto l = yacl::math::MPInt::RandomLtN(lLen);
        Y.emplace_back(y);
        L.emplace_back(l);
    }
}


void Database::MultiplyByLinearFactor(
    std::vector<yacl::math::MPInt>& coeffs,
    const yacl::math::MPInt& a,
    const yacl::math::MPInt& order)
{
    size_t n = coeffs.size();
    
    std::vector<yacl::math::MPInt> newCoeffs(n + 1, yacl::math::MPInt::_0_);
    
    // (c0 + c1*x + ... + c_{n-1}*x^{n-1}) * (x - a)
    for (size_t i = 0; i < n; ++i) {
        newCoeffs[i + 1] = (newCoeffs[i + 1] + coeffs[i]) % order;
        
        yacl::math::MPInt negA = (order - a) % order;  // -a mod order
        newCoeffs[i] = (newCoeffs[i] + coeffs[i] * negA) % order;
    }
    
    coeffs = std::move(newCoeffs);
}

void Database::GetCoeffs(const yacl::math::MPInt& order) {
    size_t n = Y.size();

    std::vector<std::vector<yacl::math::MPInt>> dividedDiffs(n, 
        std::vector<yacl::math::MPInt>(n, yacl::math::MPInt::_0_));
    
    // f[y_i] = l_i
    for (size_t i = 0; i < n; ++i) {
        dividedDiffs[i][0] = L[i] % order;
    }
    
    // 
    for (size_t j = 1; j < n; ++j) {
        for (size_t i = 0; i < n - j; ++i) {
            yacl::math::MPInt numerator = 
                (dividedDiffs[i + 1][j - 1] - dividedDiffs[i][j - 1]) % order;
            yacl::math::MPInt denominator = 
                (Y[i + j] - Y[i]) % order;
            
            yacl::math::MPInt denomInv = denominator.InvertMod(order);
            dividedDiffs[i][j] = (numerator * denomInv) % order;
        }
    }
    
    // c0 + c1*(x-y0) + c2*(x-y0)*(x-y1) + ...
    Coeffs.assign(n, yacl::math::MPInt::_0_);
    // c0 = f[y0]
    Coeffs[0] = dividedDiffs[0][0];

    std::vector<yacl::math::MPInt> currentPoly = { yacl::math::MPInt::_1_ };
    
    for (size_t j = 1; j < n; ++j) {
        MultiplyByLinearFactor(currentPoly, Y[j - 1], order);
        
        yacl::math::MPInt cj = dividedDiffs[0][j];
        for (size_t k = 0; k < currentPoly.size(); ++k) {
            yacl::math::MPInt term = (currentPoly[k] * cj) % order;
            Coeffs[k] = (Coeffs[k] + term) % order;
        }
    }

}

yacl::math::MPInt Database::GetVal(const yacl::math::MPInt& x){
    for (size_t i = 0; i < Y.size(); ++i) {
        if (x == Y[i]) return L[i];
    }
    return yacl::math::MPInt::_0_;
}

PolyKPIR::QueryState PolyKPIR::Query(const Encryptor& encryptor, 
                                      const yacl::math::MPInt& x, 
                                      uint32_t s,
                                      yacl::math::MPInt& order) {

    std::vector<yacl::math::MPInt> plainX;
    plainX.reserve(s);
    
    yacl::math::MPInt current_x_pow = x % order;
    for (uint32_t i = 0; i < s; ++i) {
        plainX.push_back(current_x_pow);
        if (i < s - 1) {
            current_x_pow = (current_x_pow * x) % order;
        }
    }
    std::vector<Ciphertext> cipherX;
    for (const auto& val : plainX) {
        cipherX.push_back(encryptor.RawEncrypt(val));
    }
    return {cipherX, plainX};
}

std::vector<Ciphertext> PolyKPIR::Answer(const Evaluator& evaluator,
                                         const Encryptor& encryptor, 
                                         const std::vector<Ciphertext>& cipherX, 
                                         const Database& db, 
                                         uint32_t s) {
    //auto order = pk.GetEcGroup()->GetOrder();
    uint32_t n = db.Y.size();
    uint32_t t = (n + s - 1) / s;

    std::vector<Ciphertext> response;
    response.reserve(t);

    for (uint32_t i = 0; i < t; ++i) {
        auto temp = evaluator.Mul(cipherX[0], db.Coeffs[s*i + 1]);
        for (uint32_t j = 1; j < s; ++j) {
            uint32_t idx = s*i + j + 1;
            if (idx >= n) break;
            temp = evaluator.Add(temp, evaluator.Mul(cipherX[j], db.Coeffs[idx]));
        }
        response.push_back(temp);
    }
    // add a0
    response[0] = evaluator.Add(response[0], encryptor.RawEncrypt(db.Coeffs[0]));
    return response;
}

yacl::math::MPInt PolyKPIR::Recover(const Evaluator& evaluator,
                                    const Decryptor& decryptor, 
                                    const std::vector<Ciphertext>& response,
                                    const std::vector<yacl::math::MPInt>& plainX) {
    if (response.empty()) return yacl::math::MPInt::_0_;
    
    const auto& xPowS = plainX.back(); // x^s
    
    // Result = response[n-1]
    auto Result = response.back();
    
    for (int i = static_cast<int>(response.size()) - 2; i >= 0; --i) {
        // Result = response[i] + Result * x^s
        auto shifted = evaluator.Mul(Result, xPowS);
        Result = evaluator.Add(response[i], shifted);
    }

    auto plainResult = decryptor.RawDecrypt(Result);
    return plainResult;
}

}  // namespace heu::lib::algorithms::elgamal_kpir