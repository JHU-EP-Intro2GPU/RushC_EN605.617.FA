/**
 * This program uses cuBLAS to multiply two matrices whose values are randomly
 * generated using cuRAND.
 *
 * This program takes the following optional command line arguments:
 *     m: The number of rows for matrix A and columns for matrix B (default: 256)
 *     n: The number of rows for matrix B and columns for matrix A (default: 256)
 */

// C
#include <stdlib.h>
#include <stdio.h>

// cuBLAS
#include <cublas_v2.h>

// cuRAND
#include <curand.h>
#include <curand_kernel.h>

// Custom Headers
#include "utilities.cuh"

///////////////////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////////////////

// The command line arguments for the program.
struct Args
{
    int m;
    int n;

    int inputSize() const
    {
        return this->m * this->n;
    }

    int outputSize() const
    {
        return this->m * this->m;
    }

    template<typename T> 
    size_t inputArraySize() const
    { 
        return this->inputSize() * sizeof(T); 
    }

    template<typename T> 
    size_t outputArraySize() const
    { 
        return this->outputSize() * sizeof(T); 
    }
};

// The arrays used in the program.
struct Arrays
{
    curandState_t* d_states;
    float* d_inputMatrixA;
    float* d_inputMatrixB;
    float* d_outputMatrix;
};

///////////////////////////////////////////////////////////////////////////////
// Kernels
///////////////////////////////////////////////////////////////////////////////

/**
 * Initialize the random states.
 */
__global__
void randInit(curandState_t* state, unsigned seed)
{
    int tid = threadIndex();
    curand_init(seed, tid, 0, &state[tid]);
}

/**
 * Generate random values to place in an array.
 */
__global__
void rand(curandState_t* state, float* nums)
{
    int tid = threadIndex();
    nums[tid] = (float)curand(&state[tid]);
}

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

/**
 * Parse the command line arguments.
 */
Args parseArgs(int argc, char** argv)
{
    Args args = {};
    args.m = 256;
    args.n = 256;

    if (argc >= 2)
    {
        args.m = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        args.n = atoi(argv[2]);
    }

    return args;
}

/**
 * Allocate program arrays.
 */
Arrays allocateArrays(const Args& args)
{
    Arrays arrays = {};
    HANDLE_ERROR(cudaMallocHost(&arrays.d_inputMatrixA, args.inputArraySize<float>()));
    HANDLE_ERROR(cudaMallocHost(&arrays.d_inputMatrixB, args.inputArraySize<float>()));
    HANDLE_ERROR(cudaMallocHost(&arrays.d_outputMatrix, args.outputArraySize<float>()));
    HANDLE_ERROR(cudaMallocHost(&arrays.d_states, args.inputArraySize<curandState_t>()));

    return arrays;
}

/**
 * Free program arrays.
 */
void freeArrays(Arrays& arrays)
{
    HANDLE_ERROR(cudaFreeHost(arrays.d_inputMatrixA));
    HANDLE_ERROR(cudaFreeHost(arrays.d_inputMatrixB));
    HANDLE_ERROR(cudaFreeHost(arrays.d_states));
}

/**
 * Fill the program arrays with random data.
 */
void fillInputMatrices(Arrays& arrays, const Args& args)
{
    randInit<<<args.m, args.n>>>(arrays.d_states, time(NULL));
    rand<<<args.m, args.n>>>(arrays.d_states, arrays.d_inputMatrixA);
    rand<<<args.m, args.n>>>(arrays.d_states, arrays.d_inputMatrixB);
}

/**
 * Multiply the input matrices and store the result in the output matrix.
 */
void multiplyMatrices(Arrays& arrays, const Args& args)
{
    static const float ALPHA = 1;
    static const float BETA = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle, 
                CUBLAS_OP_N, 
                CUBLAS_OP_N, 
                args.m,
                args.m,
                args.n,
                &ALPHA,
                arrays.d_inputMatrixA, args.m,
                arrays.d_inputMatrixB, args.n,
                &BETA,
                arrays.d_outputMatrix, args.m);

    cublasDestroy(handle);

    HANDLE_ERROR(cudaDeviceSynchronize());
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

/**
 * Main entry point for the program.
 */
int main(int argc, char** argv)
{
    Args args = parseArgs(argc, argv);
    printf("With m=%d and n=%d...\n",
           args.m,
           args.n);

    Arrays arrays = allocateArrays(args);
    fillInputMatrices(arrays, args);
    HANDLE_ERROR(cudaDeviceSynchronize());

    auto time = timeIt([&]
    {
        multiplyMatrices(arrays, args);
        HANDLE_ERROR(cudaDeviceSynchronize());
    });

    printf("Took %llu microseconds\n\n", (long long)time.count());

    freeArrays(arrays);
}