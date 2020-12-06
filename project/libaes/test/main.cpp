// C
#include <stdlib.h>

// C++
#include <iostream>
#include <stdexcept>

// Test
#include "test_cipher.hpp"

/**
 * Main test driver.
 */
int main()
{
    try
    {
        test_encryption_kats();
        std::cout << "All tests passed!" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}