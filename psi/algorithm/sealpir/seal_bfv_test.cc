#include <cassert>
#include <chrono>
#include <iomanip>
#include <vector>

#include "seal/seal.h"
using namespace std;
using namespace seal;

int main() {
  uint64_t poly_modulus_degree = 8192;
  EncryptionParameters parms = EncryptionParameters(scheme_type::bfv);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  // parms.set_plain_modulus(
  //     PlainModulus::Batching(poly_modulus_degree, 20));
  parms.set_plain_modulus(65537);

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
  cout << "batch size                            : " << encrypted_op.size()
       << endl;
  cout << "batch                                 : " << finish << " us" << endl;

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

  return 0;
}
