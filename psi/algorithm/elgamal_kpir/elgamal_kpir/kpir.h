#pragma once

#include "heu/library/algorithms/elgamal/elgamal.h"

namespace heu::lib::algorithms::elgamal_kpir {

// 引入 elgamal 算法库的类
using namespace heu::lib::algorithms::elgamal;

class Database {
public:
    std::vector<yacl::math::MPInt> Y;
    std::vector<yacl::math::MPInt> L;
    std::vector<yacl::math::MPInt> Coeffs;

    void Random(uint32_t logN, uint32_t logY,uint32_t logL);
    void GetCoeffs(const yacl::math::MPInt& order);
    void MultiplyByLinearFactor(std::vector<yacl::math::MPInt>& coeffs, 
                                const yacl::math::MPInt& a, 
                                const yacl::math::MPInt& order);
    yacl::math::MPInt GetVal(const yacl::math::MPInt& x);
};

class PolyKPIR {
public:

    struct QueryState {
        std::vector<Ciphertext> cipherX;
        std::vector<yacl::math::MPInt> plainX;
    };
    static QueryState Query(const Encryptor& encryptor, 
                            const yacl::math::MPInt& x, 
                            uint32_t s,
                            yacl::math::MPInt& order);

    static std::vector<Ciphertext> Answer(const Evaluator& evaluator,
                                          const Encryptor& encryptor, 
                                          const std::vector<Ciphertext>& cipherX, 
                                          const Database& db, 
                                          uint32_t s);

    static yacl::math::MPInt Recover(const Evaluator& evaluator,
                                     const Decryptor& decryptor, 
                                     const std::vector<Ciphertext>& response,
                                     const std::vector<yacl::math::MPInt>& plainX);
};


}  // namespace heu::lib::algorithms::elgamal_kpir