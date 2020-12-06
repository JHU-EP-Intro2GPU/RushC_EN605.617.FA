// C
#include <stdint.h>
#include <stdio.h>

// C++
#include <array>

// libaes
#include <aes/utilities.hpp>

namespace aes
{
    void print_hex(const uint8_t data[], size_t size, int width) 
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
}