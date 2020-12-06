// C
#include <stddef.h>
#include <stdint.h>

// C++
#include <algorithm>
#include <array>
#include <stdexcept>

// utils
#include <utils/cuda.cuh>

// libaes
#include <cuaes/constants.cuh>
#include <cuaes/cipher.cuh>
#include <cuaes/ctr.cuh>
#include <cuaes/utilities.cuh>

/**
 * Convert an integer from host byte order to network byte order.
 * 
 * @param value The value to convert.
 * 
 * @return The converted value.
 */
__device__
static uint32_t host_to_network(uint32_t value)
{
    uint8_t bytes[] =
    {
        static_cast<uint8_t>((value & 0xFF000000) >> 24),
        static_cast<uint8_t>((value & 0x00FF0000) >> 16),
        static_cast<uint8_t>((value & 0x0000FF00) >>  8),
        static_cast<uint8_t>((value & 0x000000FF) >>  0)
    };

    return *reinterpret_cast<uint32_t*>(bytes);
}

__global__
static void encrypt_block(cuaes::Type type,
                   const uint8_t key[], 
                   const uint8_t nonce[], 
                   const uint8_t plaintext[], 
                   size_t plaintext_size, 
                   uint8_t ciphertext[])
{
    size_t block_index = utils::thread_id();

    // counter (16 bytes) = nonce (12 bytes) || count (4 bytes)
    uint8_t counter[cuaes::BLOCK_SIZE] = {};
    memcpy(counter, nonce, cuaes::NONCE_SIZE);

    uint32_t* count = reinterpret_cast<uint32_t*>(&counter[cuaes::NONCE_SIZE]);
    *count = host_to_network(block_index);

    // Encrypt the counter value.
    uint8_t encrypted_counter[cuaes::BLOCK_SIZE] = {};
    cipher_encode(type, key, counter, encrypted_counter);

    // XOR the encrypted counter value with the data block to create the
    // ciphertext.
    size_t start = block_index * cuaes::BLOCK_SIZE;
    size_t end = utils::min<size_t>(plaintext_size, start + cuaes::BLOCK_SIZE);
    for (size_t i = start; i < end; i++)
    {
        ciphertext[i] = plaintext[i] ^ encrypted_counter[i - start];
    }
}

namespace cuaes
{
    void aes_ctr_encrypt(Type type,
                         const uint8_t key[], 
                         const uint8_t nonce[], 
                         const uint8_t plaintext[], 
                         size_t plaintext_size, 
                         uint8_t ciphertext[]) 
    {
        if (key == nullptr) throw std::invalid_argument("key is null");
        if (nonce == nullptr) throw std::invalid_argument("nonce is null");
        if (plaintext == nullptr) throw std::invalid_argument("plaintext is null");
        if (ciphertext == nullptr) throw std::invalid_argument("ciphertext is null");

        size_t num_blocks = (plaintext_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        utils::launch_kernel(encrypt_block, num_blocks, 
        [&](size_t start, unsigned grid_size, unsigned block_size)
        {
            size_t data_start = start * BLOCK_SIZE;
            size_t data_size = plaintext_size - data_start;

            encrypt_block<<<grid_size, block_size>>>(type,
                                                    key,
                                                    nonce,
                                                    &plaintext[data_start],
                                                    data_size,
                                                    &ciphertext[data_start]);
        });
        
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaGetLastError());
    }
}