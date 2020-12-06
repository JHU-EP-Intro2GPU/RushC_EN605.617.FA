#ifndef CUAES_CTR_CUH
#define CUAES_CTR_CUH

// C
#include <stddef.h>
#include <stdint.h>

// utils
#include <utils/cuda.cuh>

// libaes
#include <cuaes/cipher.cuh>

namespace cuaes
{
    /// The size (in bytes) of the nonce to use with AES-CTR.
    constexpr size_t NONCE_SIZE = 12;

    /**
     * Encrypt plaintext using AES-256 in CTR mode.
     * 
     * @param[in]  type           The variant of AES to use.
     * @param[in]  key            The secret key to use for encryption.
     * @param[in]  nonce          A unique value to use to derive unique counter values.
     * @param[in]  plaintext      The plaintext to encrypt.
     * @param[in]  plaintext_size The size (in bytes) of @c plaintext.
     * @param[out] ciphertext     The buffer to store the encrypted ciphertext.
     * 
     * The exact variant of AES used (AES-128, AES-196, or AES-256) depends on
     * the size of the key. @c nonce must be of size @c NONCE_SIZE, and
     * @c ciphertext must be at least of size @c plaintext_size. 
     * 
     * A unique @c key and @c nonce (number used once) combination should be 
     * used for every encryption operation. The @c nonce does not need to be
     * secret.
     *
     * @throw std::invalid_argument if @c key, @c nonce, @c plaintext, or @c ciphertext are null 
     * @throw utils::CudaException  if a CUDA error occurs 
     */
    void aes_ctr_encrypt(Type type,
                         const uint8_t key[], 
                         const uint8_t nonce[], 
                         const uint8_t plaintext[], 
                         size_t plaintext_size, 
                         uint8_t ciphertext[]);

} // namespace cuaes

#endif // CUAES_CTR_CUH