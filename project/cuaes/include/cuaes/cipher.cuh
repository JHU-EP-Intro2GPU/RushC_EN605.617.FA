#ifndef CUAES_CIPHER_CUH
#define CUAES_CIPHER_CUH

// C
#include <stddef.h>
#include <stdint.h>

// C++
#include <algorithm>
#include <array>
#include <stdexcept>

// libaes
#include <cuaes/cipher.cuh>
#include <cuaes/constants.cuh>
#include <cuaes/primitives.cuh>
#include <cuaes/utilities.cuh>

#define TRACE printf("%s: line %d\n", __FILE__, __LINE__);

namespace cuaes
{
    /**
     * The bytes that make up each of the AES round constants.
     */
    #ifdef __CUDA_ARCH__
        __device__ 
        __constant__
    #else
        static const
    #endif
    uint8_t ROUND_CONSTANT_BYTES[10 * cuaes::WORD_SIZE] =
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

    /**
     * Encode a block using the AES cipher.
     * 
     * @param[in]  type  The variant of the AES cipher to use.
     * @param[in]  key   The key to mix with the 
     * @param[in]  block The block to encode.
     * @param[out] out   The buffer to store the encoded block.
     */
    __forceinline__
    __host__
    __device__
    void cipher_encode(cuaes::Type type, 
                       const uint8_t key[], 
                       const uint8_t block[], 
                       uint8_t out[]);
        
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
    __forceinline__
    __host__
    __device__
    void expand_key(const uint8_t key[], 
                    size_t key_size,
                    size_t num_rounds,
                    uint8_t round_keys[]);

    /**
     * Add a round key to the current state.
     * 
     * @param[in]      round_key The round key to add to the state.
     * @param[in, out] state     The state to add the round key to.
     */
    __forceinline__
    __host__
    __device__
    void add_round_key(const uint8_t round_key[],
                       uint8_t state[]);

    /**
     * Shift each of the rows in the state.
     * 
     * @param[in] state The current state.
     */
    __forceinline__
    __host__
    __device__
    void shift_rows(uint8_t state[]);

    /**
     * Mix each of the columns in the state.
     * 
     * @param[in] state The current state.
     */
    __forceinline__
    __host__
    __device__
    void mix_columns(uint8_t state[]);

} // namespace cuaes

void cuaes::cipher_encode(Type type,
                          const uint8_t key[],
                          const uint8_t block[],
                          uint8_t out[])
{
    uint8_t state[STATE_HEIGHT * STATE_WIDTH] = {};
    uint8_t round_keys[ROUND_KEY_SIZE * MAX_ROUNDS] = {};

    size_t key_size = KEY_SIZE(type);
    size_t total_rounds = TOTAL_ROUNDS(type);

    // The input block is our initial state.
    cuaes::transpose(block, STATE_WIDTH, STATE_HEIGHT, state);

    // 1. Key Expansion
    expand_key(key, key_size, total_rounds, round_keys);

    // 2. Initial Round
    add_round_key(&round_keys[0], state);

    // 3. Intermediate Rounds (9, 11, or 13)
    for (int round = 1; round < (total_rounds - 1); round++)
    {
        substitute(state, sizeof(state), state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(&round_keys[round * ROUND_KEY_SIZE], state);
    }

    // 4. Final Round
    substitute(state, sizeof(state), state);
    shift_rows(state);
    add_round_key(&round_keys[(total_rounds - 1) * ROUND_KEY_SIZE], state);

    // Our resulting state is the ciphertext.
    cuaes::transpose(state, STATE_WIDTH, STATE_HEIGHT, out);
}

void cuaes::expand_key(const uint8_t key[], 
                       size_t key_size,
                       size_t total_rounds,
                       uint8_t round_keys[]) 
{
    auto round_constants = reinterpret_cast<const uint32_t*>(ROUND_CONSTANT_BYTES);

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
                ^ cuaes::substitute(cuaes::rotate_left(words[word_index - 1]))
                ^ round_constants[(word_index / num_key_words) - 1];
        }
        else if ((num_key_words > 6) && ((word_index % num_key_words) == 4))
        {
            words[word_index] = 
                words[word_index - num_key_words]
                ^ cuaes::substitute(words[word_index - 1]);
        }
        else
        {
            words[word_index] = 
                words[word_index - num_key_words]
                ^ words[word_index - 1];
        }
    }
}

void cuaes::add_round_key(const uint8_t round_key[],
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

void cuaes::shift_rows(uint8_t state[]) 
{
    auto state_words = reinterpret_cast<uint32_t*>(state);
    for (size_t row = 0; row < STATE_HEIGHT; row++)
    {
        state_words[row] = cuaes::rotate_left(state_words[row], static_cast<uint8_t>(row));
    }
}

void cuaes::mix_columns(uint8_t state[]) 
{
    constexpr uint8_t MDS_MATRIX[STATE_HEIGHT][STATE_HEIGHT] = 
    {
        { 2, 3, 1, 1 },
        { 1, 2, 3, 1 },
        { 1, 1, 2, 3 },
        { 3, 1, 1, 2 }
    };

    for (size_t column = 0; column < STATE_WIDTH; column++)
    {
        // Calculate each element in the column multiplied by the coefficients
        // 1, 2, and 3 in Rijndael's Galois field. Access the results using:
        //
        //     column_g[i][c]
        //
        // Where i is the index of the byte in the column, and c is the 
        // coefficient - 1.
        uint8_t products[STATE_HEIGHT][3] = {};
        for (size_t i = 0; i < STATE_HEIGHT; i++)
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
            for (int i = 0; i < STATE_HEIGHT; i++)
            {
                // Add element i of the column multiplied by the coefficient 
                // listed in the corresponding element in the matrix row.
                uint8_t coefficient = MDS_MATRIX[row][i];
                *value ^= products[i][coefficient - 1];
            }
        }
    }
}

#endif // CUAES_CIPHER_CUH