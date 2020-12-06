#ifndef AES_CIPHER_HPP
#define AES_CIPHER_HPP

// C
#include <stdint.h>

// libaes
#include <aes/constants.hpp>

namespace aes
{
    /**
     * Encode a block using the AES cipher.
     * 
     * @param[in]  type  The variant of the AES cipher to use.
     * @param[in]  key   The key to mix with the 
     * @param[in]  block The block to encode.
     * @param[out] out   The buffer to store the encoded block.
     * 
     * @throw std::invalid_argument If @c key, @c block, or @c out are null.
     */
    void cipher_encode(aes::Type type, 
                       const uint8_t key[], 
                       const uint8_t block[], 
                       uint8_t out[]);

} // namespace aes

#endif // AES_CIPHER_HPP