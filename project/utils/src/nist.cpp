// C
#include <stddef.h>
#include <stdint.h>

// C++
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Filesystem - Using the experimental version in C++14
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

// Utils
#include <utils/nist.hpp>

std::string utils::bytes_to_hex(const uint8_t bytes[], size_t num_bytes) 
{
    if (bytes == nullptr) throw std::invalid_argument("bytes cannot be null");

    std::stringstream hex;
    hex << std::hex;

    for (size_t i = 0; i < num_bytes; i++)
    {
        hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(bytes[i]);
    }

    return hex.str();
}

std::vector<uint8_t> utils::hex_to_bytes(const std::string& hex) 
{
    if ((hex.size() % 2) != 0)
    {
        throw std::invalid_argument("hex string must have an even number of digits");
    }

    std::vector<uint8_t> bytes(hex.size() / 2);
    for (size_t i = 0; i < bytes.size(); i++)
    {
        std::string byte_hex = hex.substr(i * 2, 2);
        try
        {
            bytes[i] = (uint8_t)std::stoul(byte_hex, 0, 16);
        }
        catch (const std::invalid_argument& e)
        {
            (void)e;

            throw std::invalid_argument(
                "invalid byte in hex string starting at index " 
                + std::to_string(i * 2) + ": " + byte_hex);
        }
    }

    return bytes;
}

std::vector<std::string> utils::list_dir(const std::string& dir) 
{
    std::vector<std::string> files;

    for (const auto& entry : fs::directory_iterator(dir))
    {
        std::string file_name = entry.path().filename().generic_string();
        if ((file_name != ".") && (file_name != "..")) 
        {
            files.push_back(file_name);
        }
    }

    return files;
}

std::vector<utils::Kat> utils::parse_rsp(const std::string& path) 
{
    static const std::string BLOCK_DELIMITER = "\n\n";

    // Open file
    std::ifstream rsp(path);
    if (!rsp)
    {
        throw std::invalid_argument("cannot read file: " + path);
    }

    // Read file contents
    std::string text;
    rsp.seekg(std::ios::end);
    text.reserve(rsp.tellg());
    rsp.seekg(0);
    text.assign(std::istreambuf_iterator<char>(rsp),
                std::istreambuf_iterator<char>());

    // Keep track of the current section.
    KatType current_section;

    // Iterate through the blocks of lines in the file
    size_t previous = 0;
    size_t index = 0;

    std::vector<Kat> kats;
    while ((index = text.find(BLOCK_DELIMITER, previous)) != std::string::npos)
    {
        // Skip the first block, since it's always a comment block.
        if (previous != 0)
        {
            std::string block = text.substr(previous, index - previous);

            if (block == "[ENCRYPT]")
            {
                current_section = KatType::ENCRYPT;
            }
            else if (block == "[DECRYPT]")
            {
                current_section = KatType::DECRYPT;
            }
            else
            {
                // KAT
                Kat kat = {};
                kat.type = current_section;

                std::string line;
                std::stringstream block_stream(block);
                while (std::getline(block_stream, line))
                {
                    std::string name;
                    std::string equals;
                    std::string value;
                    std::stringstream(line) >> name >> equals >> value;
                    
                    if (name == "KEY")
                    {
                        kat.key = hex_to_bytes(value);
                    }
                    if (name == "PLAINTEXT")
                    {
                        kat.plaintext = hex_to_bytes(value);
                    }
                    if (name == "CIPHERTEXT")
                    {
                        kat.ciphertext = hex_to_bytes(value);
                    }
                }

                kats.push_back(kat);
            }
        }

        previous = index + BLOCK_DELIMITER.size();
    }

    return kats;
}
