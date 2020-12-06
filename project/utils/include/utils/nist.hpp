#ifndef UTILS_NIST_HPP
#define UTILS_NIST_HPP

// C
#include <stddef.h>
#include <stdint.h>

// C++
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace utils
{
    /**
     * The different types of Known Answer Tests (KATs).
     */
    enum class KatType
    {
        ENCRYPT,
        DECRYPT
    };

    /**
     * Represents a Known Answer Test (KAT) parsed from a .rsp file.
     */
    struct Kat
    {
        KatType type;
        std::vector<uint8_t> key;
        std::vector<uint8_t> plaintext;
        std::vector<uint8_t> ciphertext;
    };

    /**
     * Generate a hex string representing some bytes.
     * 
     * @param[in] bytes     The bytes to generate a hex string for.
     * @param[in] num_bytes The size of @c bytes.
     * 
     * @return the generated hex string.
     * 
     * @throw std::invalid_argument if @c bytes is null
     */
    std::string bytes_to_hex(const uint8_t bytes[], size_t num_bytes);

    /**
     * Parse the bytes represented by a hex string.
     * 
     * @param hex The hex string to parse.
     * 
     * @return the bytes represented by @c hex.
     * 
     * @throw std::invalid_argument if @c hex is not a valid hex string
     */
    std::vector<uint8_t> hex_to_bytes(const std::string& hex);

    /**
     * List all of the files in the specified directory.
     * 
     * @param[in] dir The directory to list all of the files in.
     *  
     * @return A list of all of the files in the directory. 
     */
    std::vector<std::string> list_dir(const std::string& dir);

    /**
     * Parse a NIST .rsp file containing various KATs.
     * 
     * @param path The path to the .rsp file to parse.
     * 
     * @return the KATs parsed from the file.
     * 
     * @throw std::invalid_argument if @c path does not point to a valid .rsp file.
     */
    std::vector<Kat> parse_rsp(const std::string& path);
    
} // namespace utils

#endif // UTILS_NIST_HPP