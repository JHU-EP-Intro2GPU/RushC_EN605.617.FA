// C
#include <stddef.h>
#include <stdint.h>

// C++
#include <algorithm>
#include <array>
#include <stdexcept>

// utils
#include <utils/threading.hpp>

// libaes
#include <aes/constants.hpp>
#include <aes/cipher.hpp>
#include <aes/ctr.hpp>
#include <aes/utilities.hpp>

/**
 * Convert an integer from host byte order to network byte order.
 * 
 * @param value The value to convert.
 * 
 * @return The converted value.
 */
uint32_t host_to_network(uint32_t value)
{
    std::array<uint8_t, 4> bytes =
    {
        static_cast<uint8_t>((value & 0xFF000000) >> 24),
        static_cast<uint8_t>((value & 0x00FF0000) >> 16),
        static_cast<uint8_t>((value & 0x0000FF00) >>  8),
        static_cast<uint8_t>((value & 0x000000FF) >>  0)
    };

    return *reinterpret_cast<uint32_t*>(bytes.data());
}

namespace aes
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
        utils::run_jobs(num_blocks, [&](size_t block_index)
        {
            // counter (16 bytes) = nonce (12 bytes) || count (4 bytes)
            std::array<uint8_t, BLOCK_SIZE> counter = {};
            std::copy_n(nonce, NONCE_SIZE, counter.begin());

            uint32_t* count = reinterpret_cast<uint32_t*>(&counter[NONCE_SIZE]);
            *count = host_to_network(block_index);

            // Encrypt the counter value.
            decltype(counter) encrypted_counter = {};
            cipher_encode(type, key, counter.data(), encrypted_counter.data());

            // XOR the encrypted counter value with the data block to create the
            // ciphertext.
            size_t start = block_index * BLOCK_SIZE;
            size_t end = std::min<size_t>(plaintext_size, start + BLOCK_SIZE);
            for (size_t i = start; i < end; i++)
            {
                ciphertext[i] = plaintext[i] ^ encrypted_counter[i - start];
            }
        });
    }
}