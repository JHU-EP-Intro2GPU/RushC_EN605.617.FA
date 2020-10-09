// C
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

// C++
#include <array>
#include <chrono>
#include <functional>
#include <initializer_list>

// Custom Headers
#include "utilities.cuh"

///////////////////////////////////////////////////////////////////////////////
// Configurable Values
///////////////////////////////////////////////////////////////////////////////

// Each of these values can be overridden by passing the flag 
// -D<VARIABLE>=<VALUE> into the compiler.
//
// Example:
// nvcc ... -DBLOCK_SIZE=128

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 128
#endif

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////

// The total number of threads.
constexpr int TOTAL_THREADS = NUM_BLOCKS * BLOCK_SIZE;

// The size of an array of ints of size TOTAL_THREADS.
constexpr size_t INT_ARRAY_SIZE = TOTAL_THREADS * sizeof(int);

// The range of the random values to generate for the second input array.
constexpr int RANDOM_MIN = 0;
constexpr int RANDOM_MAX = 3;

///////////////////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////////////////

using ArithmeticFunction = void (*)(const int a[], const int b[], int c[]);

///////////////////////////////////////////////////////////////////////////////
// Arrays
///////////////////////////////////////////////////////////////////////////////

// Pinned Memory
int* p_inputA;
int* p_inputB;
int* p_output;

// Device Global Memory
int* d_inputA;
int* d_inputB;
int* d_output;

///////////////////////////////////////////////////////////////////////////////
// Kernels
///////////////////////////////////////////////////////////////////////////////

__global__
void add_global(const int a[], const int b[], int c[])
{
    unsigned index = threadIndex();
    c[index] = a[index] + b[index];
}

__global__
void subtract_global(const int a[], const int b[], int c[])
{
    unsigned index = threadIndex();
    c[index] = a[index] - b[index];
}

__global__
void multiply_global(const int a[], const int b[], int c[])
{
    unsigned index = threadIndex();
    c[index] = a[index] * b[index];
}

__global__
void modulo_global(const int a[], const int b[], int c[])
{
    unsigned index = threadIndex();
    c[index] = a[index] % b[index];
}

__global__
void add_register(const int a[], const int b[], int c[])
{
    const unsigned index = threadIndex();
    const int a_value = a[index];
    const int b_value = b[index];
    const int result = a_value + b_value;
    c[index] = result;
}

__global__
void subtract_register(const int a[], const int b[], int c[])
{
    const unsigned index = threadIndex();
    const int a_value = a[index];
    const int b_value = b[index];
    const int result = a_value - b_value;
    c[index] = result;
}

__global__
void multiply_register(const int a[], const int b[], int c[])
{
    const unsigned index = threadIndex();
    const int a_value = a[index];
    const int b_value = b[index];
    const int result = a_value * b_value;
    c[index] = result;
}

__global__
void modulo_register(const int a[], const int b[], int c[])
{
    const unsigned index = threadIndex();
    const int a_value = a[index];
    const int b_value = b[index];
    const int result = a_value % b_value;
    c[index] = result;
}

__device__
void add_shared(const int a[], const int b[], int c[])
{
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__device__
void subtract_shared(const int a[], const int b[], int c[])
{
    c[threadIdx.x] = a[threadIdx.x] - b[threadIdx.x];
}

__device__
void multiply_shared(const int a[], const int b[], int c[])
{
    c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
}

__device__
void modulo_shared(const int a[], const int b[], int c[])
{
    c[threadIdx.x] = a[threadIdx.x] % b[threadIdx.x];
}

template<ArithmeticFunction FUNCTION>
__global__
void withSharedMemory(const int a[], const int b[], int c[])
{
    // Allocate shared memory.
    __shared__ int sharedA[BLOCK_SIZE];
    __shared__ int sharedB[BLOCK_SIZE];
    __shared__ int sharedC[BLOCK_SIZE];

    // Copy input into shared memory.
    unsigned index = threadIndex();

    sharedA[threadIdx.x] = a[index];
    sharedB[threadIdx.x] = b[index];

    __syncthreads();
    FUNCTION(sharedA, sharedB, sharedC);
    __syncthreads();

    // Copy output from shared memory.
    c[index] = sharedC[threadIdx.x];
}

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

void printHeader(const char header[])
{
    printf("%10s ", header);
}

void printTime(long long time)
{
    printf("%10lld ", time);
}

void fillInputArrays()
{
    srand(time(NULL));
    
    for (int i = 0; i < TOTAL_THREADS; i++)
    {
        p_inputA[i] = i;
        p_inputB[i] = randRange(RANDOM_MIN, RANDOM_MAX + 1);
    }
}

template<ArithmeticFunction FUNCTION>
void timeUsingGlobalMemory()
{
    // Clear memory.
    HANDLE_ERROR(cudaMemset(p_output, 0, INT_ARRAY_SIZE));
    HANDLE_ERROR(cudaMemset(d_inputA, 0, INT_ARRAY_SIZE));
    HANDLE_ERROR(cudaMemset(d_inputB, 0, INT_ARRAY_SIZE));
    HANDLE_ERROR(cudaMemset(d_output, 0, INT_ARRAY_SIZE));

    HANDLE_ERROR(cudaDeviceSynchronize());
    auto time = timeIt([&]
    {
        // Copy input data to global memory.
        HANDLE_ERROR(cudaMemcpy(d_inputA, p_inputA, INT_ARRAY_SIZE, cudaMemcpyDefault));
        HANDLE_ERROR(cudaMemcpy(d_inputB, p_inputB, INT_ARRAY_SIZE, cudaMemcpyDefault));

        // Perform kernel call using global memory as inputs.
        FUNCTION<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_inputA, d_inputB, d_output);

        // Copy output from device global memory to pinned memory.
        HANDLE_ERROR(cudaMemcpy(p_output, d_output, INT_ARRAY_SIZE, cudaMemcpyDefault));
    });

    printTime((long long)time.count());
}

template<ArithmeticFunction FUNCTION>
void timeUsingRegisters()
{
    // Clear memory.
    HANDLE_ERROR(cudaMemset(p_output, 0, INT_ARRAY_SIZE));

    HANDLE_ERROR(cudaDeviceSynchronize());
    auto time = timeIt([&]
    {
        // Perform kernel call using registers.
        FUNCTION<<<NUM_BLOCKS, BLOCK_SIZE>>>(p_inputA, p_inputB, p_output);
    });

    printTime((long long)time.count());
}

template<ArithmeticFunction FUNCTION>
void timeUsingSharedMemory()
{
    // Clear memory.
    HANDLE_ERROR(cudaMemset(p_output, 0, INT_ARRAY_SIZE));
    HANDLE_ERROR(cudaMemset(d_inputA, 0, INT_ARRAY_SIZE));
    HANDLE_ERROR(cudaMemset(d_inputB, 0, INT_ARRAY_SIZE));
    HANDLE_ERROR(cudaMemset(d_output, 0, INT_ARRAY_SIZE));

    HANDLE_ERROR(cudaDeviceSynchronize());
    auto time = timeIt([&]
    {
        // Perform kernel call using shared memory.
        withSharedMemory<FUNCTION><<<NUM_BLOCKS, BLOCK_SIZE>>>(p_inputA, p_inputB, p_output);
    });

    printTime((long long)time.count());
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) 
{
    printf("With %d blocks of size %d (%d total threads)...\n", 
           NUM_BLOCKS, 
           BLOCK_SIZE, 
           TOTAL_THREADS);

    // Allocate pinned memory.
    cudaMallocHost((void**)&p_inputA, INT_ARRAY_SIZE);
    cudaMallocHost((void**)&p_inputB, INT_ARRAY_SIZE);
    cudaMallocHost((void**)&p_output, INT_ARRAY_SIZE);

    // Allocate global memory.
    cudaMalloc((void**)&d_inputA, INT_ARRAY_SIZE);
    cudaMalloc((void**)&d_inputB, INT_ARRAY_SIZE);
    cudaMalloc((void**)&d_output, INT_ARRAY_SIZE);

    // Fill input arrays.
    fillInputArrays();

    // Print headers.
    printf("Timings (in microseconds)\n");
    printHeader("");
    printHeader("Add");
    printHeader("Subtract");
    printHeader("Multiply");
    printHeader("Modulo");
    printf("\n");

    // Measure times using global memory.
    printHeader("Global");
    timeUsingGlobalMemory<add_global>();
    timeUsingGlobalMemory<subtract_global>();
    timeUsingGlobalMemory<multiply_global>();
    timeUsingGlobalMemory<modulo_global>();
    printf("\n");

    HANDLE_ERROR(cudaGetLastError());

    // Measure times using shared memory.
    printHeader("Shared");
    timeUsingSharedMemory<add_shared>();
    timeUsingSharedMemory<subtract_shared>();
    timeUsingSharedMemory<multiply_shared>();
    timeUsingSharedMemory<modulo_shared>();
    printf("\n");

    HANDLE_ERROR(cudaGetLastError());

    // Measure times using registers.
    printHeader("Registers");
    timeUsingRegisters<add_register>();
    timeUsingRegisters<subtract_register>();
    timeUsingRegisters<multiply_register>();
    timeUsingRegisters<modulo_register>();
    printf("\n");

    HANDLE_ERROR(cudaGetLastError());

    printf("\n");

    // Free global memory.
    HANDLE_ERROR(cudaFree(d_inputA));
    HANDLE_ERROR(cudaFree(d_inputB));
    HANDLE_ERROR(cudaFree(d_output));
}
