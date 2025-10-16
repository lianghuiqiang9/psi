#include <cassert>
#include <chrono>
#include <iomanip>
#include <vector>

#include "seal/seal.h"
using namespace std;
using namespace seal;

int main() {
  uint64_t poly_modulus_degree = 8192;
  EncryptionParameters parms = EncryptionParameters(scheme_type::ckks);
  parms.set_poly_modulus_degree(poly_modulus_degree);

  parms.set_coeff_modulus(
      CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

  SEALContext context(parms);

  KeyGenerator keygen(context);

  SecretKey sk = keygen.secret_key();
  PublicKey pk;
  keygen.create_public_key(pk);
  RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);

  GaloisKeys gal_keys;
  keygen.create_galois_keys(gal_keys);

  vector<uint64_t> cipher_modulus;
  for (size_t i = 0; i < parms.coeff_modulus().size(); i++) {
    cipher_modulus.push_back(parms.coeff_modulus()[i].value());
  }

  Encryptor encryptor(context, pk);
  Decryptor decryptor(context, sk);
  Evaluator evaluator(context);
  CKKSEncoder ckks_encoder(context);

  uint64_t slot_count = ckks_encoder.slot_count();

  // cout << "Solt_cout                             : " << slot_count << endl;
  stringstream sk_size, pk_size;
  cout << "sk size                               : " << sk.save(sk_size) / 1024
       << " KB" << endl;
  cout << "pk size                               : " << pk.save(pk_size) / 1024
       << " KB" << endl;

  vector<double> encrypted_op(slot_count, 0.0);
  vector<double> plain_op(slot_count, 0.0);
  random_device rd;
  for (size_t i = 0; i < slot_count; i++) {
    encrypted_op[i] = 1.001 * static_cast<double>(i);
    plain_op[i] = 2.001 * static_cast<double>(i);
  }

  Plaintext pt;
  Ciphertext ct;
  // double scale =
  // sqrt(static_cast<double>(parms.coeff_modulus().back().value()));
  double scale = pow(2.0, 40);

  // encode, encrypt, decrypt
  auto start = std::chrono::high_resolution_clock::now();
  ckks_encoder.encode(encrypted_op, scale, pt);

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

  ckks_encoder.encode(plain_op, scale, pt);

  // ops
  Ciphertext ct_temp;

  start = std::chrono::high_resolution_clock::now();
  ckks_encoder.encode(plain_op, ct.scale(), pt);
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
  ckks_encoder.encode(plain_op, ct.scale(), pt);
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
  evaluator.rescale_to_next_inplace(ct);
  finish = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count();
  cout << "rescale                               : " << finish << " us" << endl;

  start = std::chrono::high_resolution_clock::now();
  evaluator.rotate_vector(ct, 1, gal_keys, ct_temp);
  finish = std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count();
  cout << "rotate_vector                         : " << finish << " us" << endl;

  return 0;
}
