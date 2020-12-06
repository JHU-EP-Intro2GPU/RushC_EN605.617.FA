#ifndef CUAES_UTILITIES_CUH
#define CUAES_UTILITIES_CUH

// C
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

namespace cuaes
{
    /**
     * Increment an integer value represented by an array of bytes by 1.
     * 
     * @param[in,out] value  The integer value to increment.
     * @param[in]     size   The size (in bytes) of the integer value.
     * 
     * @c value is interpreted as being in network byte order (big-endian). This
     * means that the rightmost byte will be incremented, and any carries will
     * be propogated left.
     */
    __forceinline__
    __host__
    __device__
    void increment(uint8_t value[], size_t size)
    {
        for (int i = static_cast<int>(size) - 1; i >= 0; i--)
        {
            value[i]++;
            if (value[i] != 0x00)
            {
                break;
            }
        }
    }

    /**
     * Print an array of bytes in hex format.
     * 
     * @param[in] data The array to print.
     * @param[in] size The size of the array.
     */
     __forceinline__
     __host__
     __device__
    void print_hex(const uint8_t data[], size_t size, int width = -1)
    {
        for (size_t i = 0; i < size; i++)
        {
            if ((width != -1) && ((i % width) == 0))
            {
                printf("\n");
            }

            printf("%02x", static_cast<int>(data[i]));
        }

        printf("\n");
    }

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
     __forceinline__
     __host__
     __device__
    void transpose(const uint8_t data[], 
                   size_t width, 
                   size_t height, 
                   uint8_t out[])
    {
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                out[column + (row * width)] = data[row + (column * width)];
            }
        }
    }

} // namespace cuaes

#endif // CUAES_UTILITIES_CUH