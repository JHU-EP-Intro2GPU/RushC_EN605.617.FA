#ifndef AES_PRIMITIVES_HPP
#define AES_PRIMITIVES_HPP

// C
#include <stddef.h>
#include <stdint.h>

namespace aes
{
    /**
     * Multiply two bytes in Rijndael's Galois field.
     * 
     * @param[in] a The first byte to multiply.
     * @param[in] b The second byte to multiply.
     * 
     * @return The product of @c a and @c b. 
     */
    uint8_t g_multiply(uint8_t a, uint8_t b);

    /**
     * Rotate all of the bits in a byte to the left.
     * 
     * Compared to a bit shift, in a bit rotation, any bits shifted off the end
     * of the integer will wrap around and be placed on the other end of the 
     * integer.
     * 
     * @param[in] byte   The byte to rotate the bits of. 
     * @param[in] amount The number of positions to rotate.
     * 
     * @return The result of rotating the bits.
     */
    uint8_t rotate_left(uint8_t byte, uint8_t amount = 1);

    /**
     * Rotate all of the bytes in a word to the left.
     * 
     * @param[in] word   The word to rotate the bytes of. 
     * @param[in] amount The number of positions to rotate.
     * 
     * @return The result of rotating the bytes.
     */
    uint32_t rotate_left(uint32_t word, uint8_t amount = 1);

    /**
     * Substitute a byte using the Rijndael S-box substitution box.
     * 
     * @param[in] byte The byte to substitute.
     *  
     * @return The substituted byte. 
     */
    uint8_t substitute(uint8_t byte);

    /**
     * Substitute each byte in a word using the Rijndael S-box substitution box. 
     * 
     * @param[in] word The word to substitute each byte in.
     *  
     * @return The word with each byte substituted.
     */
    uint32_t substitute(uint32_t word);

    /**
     * Substitute an array of bytes using the Rijndael S-box substitution box.
     * 
     * @param[in]  bytes    The bytes to substitute.
     * @param[in]  numBytes The number of bytes to substitute.
     * @param[out] out      The buffer to place the substituted bytes.
     * 
     * @c out must be at least @c numBytes large.
     */
    void substitute(const uint8_t bytes[], size_t num_bytes, uint8_t out[]);
}

#endif // AES_PRIMITIVES_HPP