// C
#include <stdint.h>

// C++
#include <array>
#include <iomanip>
#include <iostream>

// libaes
#include <aes/ctr.hpp>

int main()
{
    constexpr aes::Type TYPE = aes::Type::AES_256;

    constexpr std::array<uint8_t, aes::KEY_SIZE(TYPE)> KEY =
    {
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    };

    constexpr std::array<uint8_t, aes::NONCE_SIZE> NONCE = 
    {
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    };

    constexpr std::array<uint8_t, 29> PLAINTEXT =
    {
        "Testing, testing, 1, 2, 3..."
    };

    std::array<uint8_t, PLAINTEXT.size()> ciphertext = {};

    aes::aes_ctr_encrypt(TYPE, 
                         KEY.data(), 
                         NONCE.data(), 
                         PLAINTEXT.data(), 
                         PLAINTEXT.size(), 
                         ciphertext.data());

    for (uint8_t byte : ciphertext)
    {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    std::cout << std::endl;
}