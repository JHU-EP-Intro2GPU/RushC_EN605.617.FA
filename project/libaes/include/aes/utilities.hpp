#ifndef AES_UTILITIES_HPP
#define AES_UTILITIES_HPP

// C
#include <stddef.h>
#include <stdint.h>

namespace aes
{
    /**
     * Print an array of bytes in hex format.
     * 
     * @param[in] data The array to print.
     * @param[in] size The size of the array.
     */
    void print_hex(const uint8_t data[], size_t size, int width = -1);

    /**
     * Transpose a 2D array.
     * 
     * @param[in]  data   The 2D array to transpose.
     * @param[in]  width  The width of @c data.
     * @param[in]  height The height of @c data.
     * @param[out] out    A buffer to store the transposed array.
     * 
     * @c out is expected to be of width @c height and of height @c width.
     */
    void transpose(const uint8_t data[], 
                   size_t width, 
                   size_t height, 
                   uint8_t out[]);

} // namespace aes


#endif // AES_UTILITIES_HPP