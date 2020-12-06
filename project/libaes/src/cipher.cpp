// C
#include <stddef.h>
#include <stdint.h>

// C++
#include <algorithm>
#include <array>
#include <stdexcept>

// libaes
#include <aes/cipher.hpp>
#include <aes/constants.hpp>
#include <aes/primitives.hpp>
#include <aes/utilities.hpp>

/// The size of a round key.
constexpr size_t ROUND_KEY_SIZE= aes::BLOCK_SIZE;

/// The size (in bytes) of the state.
constexpr size_t STATE_WIDTH = 4;
constexpr size_t STATE_HEIGHT = 4;

/// Convert a size in terms of bytes to a size in terms of bits.
constexpr size_t BITS(size_t bytes)
{
    return bytes * aes::BITS_PER_BYTE;
}

/// Convert a size in terms of bytes to a size in terms of words.
constexpr size_t WORDS(size_t bytes)
{
    return bytes / aes::WORD_SIZE;
}

/**
 * Expand a key to generate the various round keys.
 * 
 * @param[in] key        The key to expand.
 * @param[in] key_size   The size (in bytes) of the key.
 * @param[in] num_rounds The number of round keys to create.
 * @param[in] round_keys A buffer to store the round keys.
 * 
 * @c round_keys is expected to be a 2D array of width @c ROUND_KEY_SIZE and 
 * height @c num_rounds.
 */
static void expand_key(const uint8_t key[], 
                       size_t key_size,
                       size_t num_rounds,
                       uint8_t round_keys[]);

/**
 * Add a round key to the current state.
 * 
 * @param[in]      round_key The round key to add to the state.
 * @param[in, out] state     The state to add the round key to.
 */
static void add_round_key(const uint8_t round_key[],
                          uint8_t state[]);

/**
 * Shift each of the rows in the state.
 * 
 * @param[in] state The current state.
 */
static void shift_rows(uint8_t state[]);

/**
 * Mix each of the columns in the state.
 * 
 * @param[in] state The current state.
 */
static void mix_columns(uint8_t state[]);

namespace aes
{
    void cipher_encode(Type type,
                       const uint8_t key[],
                       const uint8_t block[],
                       uint8_t out[])
    {
        if (key == nullptr)   throw std::invalid_argument("key cannot be null");
        if (block == nullptr) throw std::invalid_argument("block cannot be null");
        if (out == nullptr)   throw std::invalid_argument("out cannot be null");

        std::array<uint8_t, STATE_HEIGHT * STATE_WIDTH> state = {};
        std::array<uint8_t, ROUND_KEY_SIZE * MAX_ROUNDS> round_keys = {};

        size_t key_size = KEY_SIZE(type);
        size_t total_rounds = TOTAL_ROUNDS(type);

        // The input block is our initial state.
        aes::transpose(block, STATE_WIDTH, STATE_HEIGHT, state.data());

        // 1. Key Expansion
        expand_key(key, key_size, total_rounds, round_keys.data());

        // 2. Initial Round
        add_round_key(&round_keys[0], state.data());

        // 3. Intermediate Rounds (9, 11, or 13)
        for (int round = 1; round < (total_rounds - 1); round++)
        {
            substitute(state.data(), state.size(), state.data());
            shift_rows(state.data());
            mix_columns(state.data());
            add_round_key(&round_keys[round * ROUND_KEY_SIZE], state.data());
        }

        // 4. Final Round
        substitute(state.data(), state.size(), state.data());
        shift_rows(state.data());
        add_round_key(&round_keys[(total_rounds - 1) * ROUND_KEY_SIZE], state.data());

        // Our resulting state is the ciphertext.
        aes::transpose(state.data(), STATE_WIDTH, STATE_HEIGHT, out);
    }

} // namespace aes

void expand_key(const uint8_t key[], 
                size_t key_size,
                size_t total_rounds,
                uint8_t round_keys[]) 
{
    static const std::array<uint8_t, 10 * aes::WORD_SIZE> ROUND_CONSTANT_BYTES =
    {
        0x01, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00,
        0x20, 0x00, 0x00, 0x00,
        0x40, 0x00, 0x00, 0x00,
        0x80, 0x00, 0x00, 0x00,
        0x1B, 0x00, 0x00, 0x00,
        0x36, 0x00, 0x00, 0x00
    };

    static const auto ROUND_CONSTANTS = reinterpret_cast<const uint32_t*>(ROUND_CONSTANT_BYTES.data());

    auto key_words = reinterpret_cast<const uint32_t*>(key); 
    auto words = reinterpret_cast<uint32_t*>(round_keys);

    size_t num_key_words = WORDS(key_size);
    size_t num_round_key_words = WORDS(ROUND_KEY_SIZE);
    size_t num_words = total_rounds * num_round_key_words;

    for (size_t word_index = 0; word_index < num_words; word_index++)
    {
        if (word_index < num_key_words)
        {
            words[word_index] = key_words[word_index];
        }
        else if ((word_index % num_key_words) == 0)
        {
            words[word_index] = 
                words[word_index - num_key_words]
                ^ aes::substitute(aes::rotate_left(words[word_index - 1]))
                ^ ROUND_CONSTANTS[(word_index / num_key_words) - 1];
        }
        else if ((num_key_words > 6) && ((word_index % num_key_words) == 4))
        {
            words[word_index] = 
                words[word_index - num_key_words]
                ^ aes::substitute(words[word_index - 1]);
        }
        else
        {
            words[word_index] = 
                words[word_index - num_key_words]
                ^ words[word_index - 1];
        }
    }
}

void add_round_key(const uint8_t round_key[],
                   uint8_t state[]) 
{
    for (size_t row = 0; row < STATE_HEIGHT; row++)
    {
        for (size_t column = 0; column < STATE_WIDTH; column++)
        {
            state[(row * STATE_WIDTH) + column] ^= round_key[(column * STATE_WIDTH) + row];
        }
    }
}

void shift_rows(uint8_t state[]) 
{
    auto state_words = reinterpret_cast<uint32_t*>(state);
    for (size_t row = 0; row < STATE_HEIGHT; row++)
    {
        state_words[row] = aes::rotate_left(state_words[row], row);
    }
}

void mix_columns(uint8_t state[]) 
{
    constexpr std::array<std::array<uint8_t, 4>, 4> MDS_MATRIX = 
    {{
        { 2, 3, 1, 1 },
        { 1, 2, 3, 1 },
        { 1, 1, 2, 3 },
        { 3, 1, 1, 2 }
    }};

    for (size_t column = 0; column < STATE_WIDTH; column++)
    {
        // Calculate each element in the column multiplied by the coefficients
        // 1, 2, and 3 in Rijndael's Galois field. Access the results using:
        //
        //     column_g[i][c]
        //
        // Where i is the index of the byte in the column, and c is the 
        // coefficient - 1.
        std::array<std::array<uint8_t, 3>, STATE_HEIGHT> products = {};
        for (size_t i = 0; i < products.size(); i++)
        {
            // column[i] * 1
            products[i][0] = state[(i * STATE_WIDTH) + column];

            // column[i] * 2
            products[i][1] = products[i][0] << 1;
            if (products[i][0] & 0x80)
            {
                products[i][1] ^= 0x1B;
            }

            // column[i] * 3 
            products[i][2] = products[i][0] ^ products[i][1];
        }

        for (size_t row = 0; row < STATE_HEIGHT; row++)
        {
            uint8_t* value = &state[(row * STATE_WIDTH) + column];

            *value = 0;
            for (int i = 0; i < MDS_MATRIX[row].size(); i++)
            {
                // Add element i of the column multiplied by the coefficient 
                // listed in the corresponding element in the matrix row.
                uint8_t coefficient = MDS_MATRIX[row][i];
                *value ^= products[i][coefficient - 1];
            }
        }
    }
}
