// C
#include <stdint.h>

// C++
#include <array>
#include <iomanip>
#include <iostream>

// utils
#include <utils/cuda.cuh>

// cuaes
#include <cuaes/ctr.cuh>

int main()
{
    constexpr cuaes::Type TYPE = cuaes::Type::AES_256;

    constexpr std::array<uint8_t, cuaes::KEY_SIZE(TYPE)> KEY =
    {
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    };

    constexpr std::array<uint8_t, cuaes::NONCE_SIZE> NONCE = 
    {
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    };

    constexpr std::array<uint8_t, 29> PLAINTEXT =
    {
        "Testing, testing, 1, 2, 3..."
    };

    try
    {
        utils::PinnedCudaBuffer<uint8_t> pinned_key(KEY.size());
        utils::PinnedCudaBuffer<uint8_t> pinned_nonce(NONCE.size());
        utils::PinnedCudaBuffer<uint8_t> pinned_plaintext(PLAINTEXT.size());
        utils::PinnedCudaBuffer<uint8_t> pinned_ciphertext(PLAINTEXT.size());

        pinned_key.copy_from(KEY.data());
        pinned_nonce.copy_from(NONCE.data());
        pinned_plaintext.copy_from(PLAINTEXT.data());

        cuaes::aes_ctr_encrypt(TYPE, 
                               pinned_key.ptr(), 
                               pinned_nonce.ptr(), 
                               pinned_plaintext.ptr(), 
                               pinned_plaintext.count(), 
                               pinned_ciphertext.ptr());

        for (size_t i = 0; i < pinned_ciphertext.count(); i++)
        {
            std::cout << std::hex << std::setw(2) << std::setfill('0') 
                      << static_cast<int>(pinned_ciphertext.ptr()[i]);
        }
        std::cout << std::endl;
    }
    catch (const utils::CudaException& e)
    {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
    }
}