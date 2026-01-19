#include "heu/library/algorithms/elgamal_kpir/kpir.h"
#include "gtest/gtest.h"

namespace heu::lib::algorithms::elgamal_kpir::test {

class ElGamalKpirTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { KeyGenerator::Generate("sm2", &sk_, &pk_); }

  static SecretKey sk_;
  static PublicKey pk_;
};

SecretKey ElGamalKpirTest::sk_;
PublicKey ElGamalKpirTest::pk_;

TEST_F(ElGamalKpirTest, FullWorkflow) {

    uint32_t logN = 5;              // database size 2^logN
    uint32_t logX = 6;              // key bits
    uint32_t logY = 6;              // key bits
    uint32_t logL = 32;             // label bits
    
    uint32_t s = 1 << (logN / 2);               // s = sqrt N
    uint32_t t = ((1 << logN) + s - 1) / s;     // t = sqrt N

    std::cout << "--- KPIR Configuration ---\n"
              << "Database Size (n): " << (1<<logN) << "\n"
              << "Query Range: [0, " << (1<<logX) << ")\n"
              << "Optimization (s, t): (" << s << ", " << t << ")\n" << std::endl;

    const Encryptor encryptor(pk_);
    const Evaluator evaluator(pk_);
    const Decryptor decryptor(pk_, sk_);
  
    auto order = pk_.GetCurve()->GetOrder();
    //std::cout<<"order: "<<order<<std::endl;
    Database db;
    db.Random(logN, logY, logL);
    db.GetCoeffs(order);
    
    for (int i = 0; i < 10; ++i){
      yacl::math::MPInt k = yacl::math::MPInt::RandomLtN(yacl::math::MPInt(2).Pow(logX)); //db.Y[i];

      auto queryState = PolyKPIR::Query(encryptor, k, s, order);
      auto query = queryState.cipherX;
    
      auto response = PolyKPIR::Answer(evaluator, encryptor, query, db, s);
   
      auto result = PolyKPIR::Recover(evaluator, decryptor, response, queryState.plainX);
    
      auto expectResult = db.GetVal(k);
      EXPECT_EQ(expectResult, result);
    
      std::cout << "Query Keyword: " << k 
                << " \tRecovered Label: " << result 
                << " \tExpected Label: " << expectResult <<std::endl;
      }
}

} // namespace heu::lib::algorithms::elgamal_kpir::test