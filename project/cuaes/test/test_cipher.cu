// C++
#include <algorithm>
#include <iostream>
#include <vector>

// utils
#include <utils/cuda.cuh>
#include <utils/nist.hpp>

// cuaes
#include <cuaes/cipher.cuh>
#include <cuaes/constants.cuh>

// Test
#include "test_cipher.cuh"

/// Directory where the NIST KATs are stored, relative from the CMake project's
/// source directory.
static const std::string NIST_DIR = "../nist";

__global__
void aes_encode_kernel(cuaes::Type type, 
                       const uint8_t key[], 
                       const uint8_t block[], 
                       uint8_t out[])
{
    cuaes::cipher_encode(type, key, block, out);
}

void test_encryption_kat_host(const utils::Kat& kat, cuaes::Type type)
{
    std::vector<uint8_t> ciphertext(kat.ciphertext.size());
    cuaes::cipher_encode(type, 
                         kat.key.data(), 
                         kat.plaintext.data(), 
                         ciphertext.data());
    
    if (!std::equal(kat.ciphertext.begin(), 
                    kat.ciphertext.end(), 
                    ciphertext.begin()))
    {
        throw std::runtime_error(
            "incorrect ciphertext on host: " 
            + utils::bytes_to_hex(ciphertext.data(), ciphertext.size()));
    }
}

void test_encryption_kat_device(const utils::Kat& kat, cuaes::Type type)
{
    utils::PinnedCudaBuffer<uint8_t> pinned_key(kat.key.size());
    utils::PinnedCudaBuffer<uint8_t> pinned_plaintext(kat.plaintext.size());
    utils::PinnedCudaBuffer<uint8_t> pinned_ciphertext(kat.ciphertext.size());

    pinned_key.copy_from(kat.key.data());
    pinned_plaintext.copy_from(kat.plaintext.data());

    aes_encode_kernel<<<1, 1>>>(type, 
                                pinned_key.ptr(), 
                                pinned_plaintext.ptr(), 
                                pinned_ciphertext.ptr());
    
    cudaDeviceSynchronize();

    if (!std::equal(kat.ciphertext.begin(), 
                    kat.ciphertext.end(), 
                    pinned_ciphertext.ptr()))
    {
        throw std::runtime_error(
            "incorrect ciphertext on device: " 
            + utils::bytes_to_hex(pinned_ciphertext.ptr(), pinned_ciphertext.size()));
    }
}

void test_encryption_kats()
{
    for (const std::string& file : utils::list_dir(NIST_DIR))
    {
        std::string rsp = NIST_DIR + "/" + file;
        std::vector<utils::Kat> kats = utils::parse_rsp(rsp);

        for (size_t i = 0; i < kats.size(); i++)
        {
            if (kats[i].type != utils::KatType::ENCRYPT)
            {
                continue;
            }

            cuaes::Type aes_type;
            switch (kats[i].key.size())
            {
                case cuaes::KEY_SIZE(cuaes::Type::AES_128): 
                    aes_type = cuaes::Type::AES_128; 
                    break;
                case cuaes::KEY_SIZE(cuaes::Type::AES_196): 
                    aes_type = cuaes::Type::AES_196; 
                    break;
                case cuaes::KEY_SIZE(cuaes::Type::AES_256):
                     aes_type = cuaes::Type::AES_256; 
                     break;

                default:
                    throw std::invalid_argument(
                        "Invalid key size for KAT #" + std::to_string(i+1) 
                        + " in " + rsp);
            }

            try
            {
                test_encryption_kat_host(kats[i], aes_type);
                test_encryption_kat_device(kats[i], aes_type);
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error(
                    "Error running KAT #" + std::to_string(i+1) 
                    + " in " + rsp + ": " + e.what());
            }
        }
    }
}
