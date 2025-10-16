#include <cassert>
#include <chrono>
#include <iomanip>
#include <vector>

#include "seal/seal.h"
using namespace std;
using namespace seal;

int main() {
  uint64_t plain_modulus_length = 20;
  uint64_t poly_modulus_degree = 8192;
  EncryptionParameters parms = EncryptionParameters(scheme_type::bfv);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_plain_modulus(
      PlainModulus::Batching(poly_modulus_degree, plain_modulus_length));

  parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));

  SEALContext context(parms);

  KeyGenerator keygen(context);
  PublicKey pk;
  keygen.create_public_key(pk);
  SecretKey sk = keygen.secret_key();
  Encryptor encryptor(context, pk);
  Decryptor decryptor(context, sk);
  Evaluator evaluator(context);
  BatchEncoder batch_encoder(context);
  RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);

  GaloisKeys gal_keys;
  keygen.create_galois_keys(gal_keys);
  uint64_t plain_modulus = parms.plain_modulus().value();
  vector<uint64_t> cipher_modulus;
  for (size_t i = 0; i < parms.coeff_modulus().size(); i++) {
    cipher_modulus.push_back(parms.coeff_modulus()[i].value());
  }
  uint64_t slot_count = batch_encoder.slot_count();

  // cout << "Solt_cout                             : " << slot_count << endl;
  stringstream sk_size, pk_size;
  cout << "sk size                               : " << sk.save(sk_size) / 1024
       << " KB" << endl;
  cout << "pk size                               : " << pk.save(pk_size) / 1024
       << " KB" << endl;

  vector<uint64_t> encrypted_op(slot_count, 0);
  vector<uint64_t> plain_op(slot_count, 0);
  random_device rd;
  for (size_t i = 0; i < slot_count; i++) {
    encrypted_op[i] = parms.plain_modulus().reduce(rd());
    plain_op[i] = parms.plain_modulus().reduce(rd());
  }

  vector<uint64_t> k(slot_count, 1);

  Plaintext pt;
  Ciphertext ct;

  // encode, encrypt, decrypt
  auto start = std::chrono::high_resolution_clock::now();
  batch_encoder.encode(encrypted_op, pt);
  auto finish = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
  cout << "encode                                : " << finish << " us" << endl;

  start = std::chrono::high_resolution_clock::now();
  encryptor.encrypt(pt, ct);
  finish = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count();
  cout << "encrypt                               : " << finish << " us" << endl;

  Plaintext ans;
  start = std::chrono::high_resolution_clock::now();
  decryptor.decrypt(ct, ans);
  finish = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count();
  cout << "decrypt                               : " << finish << " us" << endl;

  // get pt, ct size
  stringstream pt_size, ct_size;
  cout << "pt size                               : " << pt.save(pt_size) / 1024
       << " KB" << endl;
  cout << "ct size                               : " << ct.save(ct_size) / 1024
       << " KB" << endl;

  batch_encoder.encode(plain_op, pt);
  Plaintext pt_k;
  batch_encoder.encode(k, pt_k);

  // ops
  Ciphertext ct_temp;

  start = std::chrono::high_resolution_clock::now();
  evaluator.add_plain(ct, pt, ct_temp);
  finish = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count();
  cout << "add_plain                             : " << finish << " us" << endl;

  start = std::chrono::high_resolution_clock::now();
  evaluator.add(ct, ct, ct_temp);
  finish = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count();
  cout << "add                                   : " << finish << " us" << endl;

  start = std::chrono::high_resolution_clock::now();
  evaluator.multiply_plain(ct, pt, ct_temp);
  finish = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count();
  cout << "multiply_plain                        : " << finish << " us" << endl;

  start = std::chrono::high_resolution_clock::now();
  evaluator.multiply(ct, ct, ct_temp);
  finish = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count();
  cout << "multiply                              : " << finish << " us" << endl;

  start = std::chrono::high_resolution_clock::now();
  evaluator.relinearize_inplace(ct_temp, relin_keys);
  finish = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count();
  cout << "relinearize_inplace                   : " << finish << " us" << endl;

  start = std::chrono::high_resolution_clock::now();
  evaluator.rotate_rows(ct, 1, gal_keys, ct_temp);
  finish = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count();
  cout << "rotate_rows                           : " << finish << " us" << endl;

  // test multiplication depth
  {
    vector<uint64_t> x(slot_count, 1);
    x[1] = 3;
    vector<uint64_t> y = x;
    Plaintext pt;
    Ciphertext ct, ct_temp;
    batch_encoder.encode(x, pt);
    encryptor.encrypt(pt, ct);
    encryptor.encrypt(pt, ct_temp);
    Plaintext ans;
    vector<uint64_t> res;
    uint64_t depth = 0;
    for (uint64_t i = 0; i < 20; i++) {
      evaluator.multiply_inplace(ct, ct_temp);
      evaluator.relinearize_inplace(ct, relin_keys);

      for (uint64_t j = 0; j < slot_count; j++) {
        x[j] = x[j] * y[j] % plain_modulus;
      }
      decryptor.decrypt(ct, ans);
      batch_encoder.decode(ans, res);
      if (res[0] != x[0] || res[1] != x[1]) {
        depth = i;
        break;
      }
    }
    cout << "depth                                 : " << depth << endl;
  }
  return 0;
}
