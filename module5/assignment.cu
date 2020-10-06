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

#ifndef BLOCK_SIZE
#define BLOCK_SIZE (64)
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS (128)
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

using ConstantArithmeticFunction = void (*)(int c[]);
using ArithmeticFunction = void (*)(const int a[], const int b[], int c[]);

///////////////////////////////////////////////////////////////////////////////
// Arrays
///////////////////////////////////////////////////////////////////////////////

// Host Memory
std::array<int, TOTAL_THREADS> h_inputA = {};
std::array<int, TOTAL_THREADS> h_inputB = {};
std::array<int, TOTAL_THREADS> h_output = {};

// Device Global Memory
int* d_inputA;
int* d_inputB;
int* d_output;

// Device Constant Memory
__constant__ int dc_inputA[TOTAL_THREADS];
__constant__ int dc_inputB[TOTAL_THREADS];

///////////////////////////////////////////////////////////////////////////////
// Kernels
///////////////////////////////////////////////////////////////////////////////

__global__
void add_constant(int c[])
{
    int index = threadIndex();
    c[index] = dc_inputA[index] + dc_inputB[index];
}

__global__
void subtract_constant(int c[])
{
    int index = threadIndex();
    c[index] = dc_inputA[index] - dc_inputB[index];
}

__global__
void multiply_constant(int c[])
{
    int index = threadIndex();
    c[index] = dc_inputA[index] * dc_inputB[index];
}

__global__
void modulo_constant(int c[])
{
    int index = threadIndex();
    c[index] = dc_inputA[index] % dc_inputB[index];
}

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
        h_inputA[i] = i;
        h_inputB[i] = randRange(RANDOM_MIN, RANDOM_MAX + 1);
    }
}

template<ConstantArithmeticFunction FUNCTION>
void timeUsingConstantMemory()
{
    // Clear memory.
    h_output.fill(0);
    HANDLE_ERROR(cudaMemset(d_output, 0, INT_ARRAY_SIZE));

    HANDLE_ERROR(cudaDeviceSynchronize());
    auto time = timeIt([&]
    {
        // Copy input data to constant memory.
        HANDLE_ERROR(cudaMemcpyToSymbol(dc_inputA, h_inputA.data(), INT_ARRAY_SIZE));
        HANDLE_ERROR(cudaMemcpyToSymbol(dc_inputB, h_inputB.data(), INT_ARRAY_SIZE));

        // Perform kernel call using constant memory as inputs.
        FUNCTION<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_output);

        // Copy output from device global memory to host memory.
        HANDLE_ERROR(cudaMemcpy(h_output.data(), d_output, INT_ARRAY_SIZE, cudaMemcpyDeviceToHost));
    });

    printTime((long long)time.count());
}

template<ArithmeticFunction FUNCTION>
void timeUsingGlobalMemory()
{
    // Clear memory.
    h_output.fill(0);
    HANDLE_ERROR(cudaMemset(d_inputA, 0, INT_ARRAY_SIZE));
    HANDLE_ERROR(cudaMemset(d_inputB, 0, INT_ARRAY_SIZE));
    HANDLE_ERROR(cudaMemset(d_output, 0, INT_ARRAY_SIZE));

    HANDLE_ERROR(cudaDeviceSynchronize());
    auto time = timeIt([&]
    {
        // Copy input data to global memory.
        HANDLE_ERROR(cudaMemcpy(d_inputA, h_inputA.data(), INT_ARRAY_SIZE, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_inputB, h_inputB.data(), INT_ARRAY_SIZE, cudaMemcpyHostToDevice));

        // Perform kernel call using global memory as inputs.
        FUNCTION<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_inputA, d_inputB, d_output);

        // Copy output from device global memory to host memory.
        HANDLE_ERROR(cudaMemcpy(h_output.data(), d_output, INT_ARRAY_SIZE, cudaMemcpyDeviceToHost));
    });

    printTime((long long)time.count());
}

template<ArithmeticFunction FUNCTION>
void timeUsingSharedMemory()
{
    // Clear memory.
    h_output.fill(0);
    HANDLE_ERROR(cudaMemset(d_inputA, 0, INT_ARRAY_SIZE));
    HANDLE_ERROR(cudaMemset(d_inputB, 0, INT_ARRAY_SIZE));
    HANDLE_ERROR(cudaMemset(d_output, 0, INT_ARRAY_SIZE));

    HANDLE_ERROR(cudaDeviceSynchronize());
    auto time = timeIt([&]
    {
        // Copy input data to global memory.
        HANDLE_ERROR(cudaMemcpy(d_inputA, h_inputA.data(), INT_ARRAY_SIZE, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_inputB, h_inputB.data(), INT_ARRAY_SIZE, cudaMemcpyHostToDevice));

        // Perform kernel call using shared memory.
        withSharedMemory<FUNCTION><<<NUM_BLOCKS, BLOCK_SIZE>>>(d_inputA, d_inputB, d_output);

        // Copy output from device global memory to host memory.
        HANDLE_ERROR(cudaMemcpy(h_output.data(), d_output, INT_ARRAY_SIZE, cudaMemcpyDeviceToHost));
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

    // Allocate global memory.
    cudaMalloc((void**)&d_inputA, INT_ARRAY_SIZE);
    cudaMalloc((void**)&d_inputB, INT_ARRAY_SIZE);
    cudaMalloc((void**)&d_output, INT_ARRAY_SIZE);

    // Fill input arrays.
    fillInputArrays();

    // Print headers.
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

    // Measure times using constant memory.
    printHeader("Constant");
    timeUsingConstantMemory<add_constant>();
    timeUsingConstantMemory<subtract_constant>();
    timeUsingConstantMemory<multiply_constant>();
    timeUsingConstantMemory<modulo_constant>();
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

    printf("\n");

    // Free global memory.
    HANDLE_ERROR(cudaFree(d_inputA));
    HANDLE_ERROR(cudaFree(d_inputB));
    HANDLE_ERROR(cudaFree(d_output));
}
