#ifndef AES_CONSTANTS_HPP
#define AES_CONSTANTS_HPP

// C
#include <stddef.h>

namespace aes
{
    /// The number of bits in a byte.
    constexpr size_t BITS_PER_BYTE = 8;

    /// The size (in bytes) of a block that can be encoded using the AES cipher.
    constexpr size_t BLOCK_SIZE = 16;

    /// The size (in bytes) of a word.
    constexpr size_t WORD_SIZE = 4;

    /// The different AES variants.
    enum class Type
    {
        AES_128,  ///< Uses a 128-bit (16-byte) key.
        AES_196,  ///< Uses a 196-bit (24-byte) key.
        AES_256   ///< Uses a 256-bit (32-byte) key.
    };

    /// The size (in bytes) of the key used in an AES variant.
    constexpr size_t KEY_SIZE(Type type)
    {
        return type == Type::AES_128 ? 16 :
               type == Type::AES_196 ? 24 :
               type == Type::AES_256 ? 32 :
                                        0;
    }

    /// The total number of rounds used in an AES variant.
    constexpr size_t TOTAL_ROUNDS(Type type)
    {
        return type == Type::AES_128 ? 11 :
               type == Type::AES_196 ? 13 :
               type == Type::AES_256 ? 15 :
                                        0;
    }

    /// Determine the AES type from the key size.
    constexpr Type TYPE(size_t key_size)
    {
        return key_size == KEY_SIZE(Type::AES_128) ? Type::AES_128 :
               key_size == KEY_SIZE(Type::AES_196) ? Type::AES_196 :
               key_size == KEY_SIZE(Type::AES_256) ? Type::AES_256 :
                                                     Type::AES_128;
    }

    /// The maximum possible number of rounds used in any AES variant.
    constexpr size_t MAX_ROUNDS = TOTAL_ROUNDS(Type::AES_256);

} // namespace aes


#endif // AES_CONSTANTS_HPP