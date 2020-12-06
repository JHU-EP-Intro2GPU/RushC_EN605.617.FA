// C++
#include <algorithm>
#include <iostream>
#include <vector>

// utils
#include <utils/nist.hpp>

// libaes
#include <aes/cipher.hpp>
#include <aes/constants.hpp>

// Test
#include "test_cipher.hpp"

/// Directory where the NIST KATs are stored, relative from the CMake project's
/// source directory.
static const std::string NIST_DIR = "../nist";

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

            aes::Type aes_type;
            switch (kats[i].key.size())
            {
                case aes::KEY_SIZE(aes::Type::AES_128): 
                    aes_type = aes::Type::AES_128; 
                    break;
                case aes::KEY_SIZE(aes::Type::AES_196): 
                    aes_type = aes::Type::AES_196; 
                    break;
                case aes::KEY_SIZE(aes::Type::AES_256):
                     aes_type = aes::Type::AES_256; 
                     break;

                default:
                    throw std::invalid_argument(
                        "Invalid key size for KAT #" + std::to_string(i+1) 
                        + " in " + rsp);
            }

            try
            {
                std::vector<uint8_t> ciphertext(kats[i].ciphertext.size());
                aes::cipher_encode(aes_type, 
                                kats[i].key.data(), 
                                kats[i].plaintext.data(), 
                                ciphertext.data());

                if (!std::equal(ciphertext.begin(), 
                                ciphertext.end(), 
                                kats[i].ciphertext.begin()))
                {
                    throw std::runtime_error(
                        "incorrect ciphertext: " 
                        + utils::bytes_to_hex(ciphertext.data(), ciphertext.size()));
                }
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
