// C
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

// C++
#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

// Custom Headers
#include "utilities.cuh"

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////

// The range of the random values to generate for the second input array.
constexpr int RANDOM_MIN = 0;
constexpr int RANDOM_MAX = 3;

///////////////////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////////////////

// Function pointer for one of our arithmetic kernel functions.
using ArithmeticFunction = void (*)(const int a[], const int b[], int c[]);

// Used to hold the command line args passed in by the user.
struct Args
{
    int blockSize;
    int totalThreads;

    template<typename ElementType> 
    size_t arraySize() const
    {
        return this->totalThreads * sizeof(ElementType);
    }

    int numBlocks() const
    { 
        return this->totalThreads / this->blockSize; 
    }
};

// Used to hold the arrays used in the program.
struct Arrays
{
    int* h_inputA;
    int* h_inputB;
    int* h_output;

    int* d_inputA;
    int* d_inputB;
    int* d_output;
};

///////////////////////////////////////////////////////////////////////////////
// Kernels
///////////////////////////////////////////////////////////////////////////////

__global__
void add(const int a[], const int b[], int c[])
{
    unsigned index = threadIndex();
    c[index] = a[index] + b[index];
}

__global__
void subtract(const int a[], const int b[], int c[])
{
    unsigned index = threadIndex();
    c[index] = a[index] - b[index];
}

__global__
void multiply(const int a[], const int b[], int c[])
{
    unsigned index = threadIndex();
    c[index] = a[index] * b[index];
}

__global__
void modulo(const int a[], const int b[], int c[])
{
    unsigned index = threadIndex();
    c[index] = a[index] % b[index];
}

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

void printHeader(const char header[])
{
    printf("%12s ", header);
}

void printTime(long long time)
{
    printf("%12lld ", time);
}

void timeSynchronous(const Args& args, Arrays& arrays, ArithmeticFunction function)
{
    // Clear memory.
    std::fill_n(arrays.h_output, args.totalThreads, 0);
    HANDLE_ERROR(cudaMemset(arrays.d_inputA, 0, args.arraySize<int>()));
    HANDLE_ERROR(cudaMemset(arrays.d_inputB, 0, args.arraySize<int>()));
    HANDLE_ERROR(cudaMemset(arrays.d_output, 0, args.arraySize<int>()));

    HANDLE_ERROR(cudaDeviceSynchronize());
    auto time = timeIt([&]
    {
        // Copy input data to global memory.
        HANDLE_ERROR(cudaMemcpy(arrays.d_inputA, arrays.h_inputA, args.arraySize<int>(), cudaMemcpyDefault));
        HANDLE_ERROR(cudaMemcpy(arrays.d_inputB, arrays.h_inputB, args.arraySize<int>(), cudaMemcpyDefault));

        // Perform kernel call using global memory as inputs.
        function<<<args.numBlocks(), args.blockSize>>>(arrays.d_inputA, arrays.d_inputB, arrays.d_output);

        // Copy output from device global memory to pinned memory.
        HANDLE_ERROR(cudaMemcpy(arrays.h_output, arrays.d_output, args.arraySize<int>(), cudaMemcpyDefault));
    });

    printTime((long long)time.count());
}

void timeStreamed(const Args& args, Arrays& arrays, ArithmeticFunction function)
{
    // Clear memory.
    std::fill_n(arrays.h_output, args.totalThreads, 0);
    HANDLE_ERROR(cudaMemset(arrays.d_inputA, 0, args.arraySize<int>()));
    HANDLE_ERROR(cudaMemset(arrays.d_inputB, 0, args.arraySize<int>()));
    HANDLE_ERROR(cudaMemset(arrays.d_output, 0, args.arraySize<int>()));

    HANDLE_ERROR(cudaDeviceSynchronize());
    auto time = timeIt([&]
    {
        // Create stream
        cudaStream_t stream;
        HANDLE_ERROR(cudaStreamCreate(&stream));

        // Copy input data to global memory.
        HANDLE_ERROR(cudaMemcpyAsync(arrays.d_inputA, arrays.h_inputA, args.arraySize<int>(), cudaMemcpyDefault, stream));
        HANDLE_ERROR(cudaMemcpyAsync(arrays.d_inputB, arrays.h_inputB, args.arraySize<int>(), cudaMemcpyDefault, stream));

        // Perform kernel call using global memory as inputs.
        function<<<args.numBlocks(), args.blockSize, 0, stream>>>(arrays.d_inputA, arrays.d_inputB, arrays.d_output);

        // Copy output from device global memory to pinned memory.
        HANDLE_ERROR(cudaMemcpyAsync(arrays.h_output, arrays.d_output, args.arraySize<int>(), cudaMemcpyDefault, stream));

        // Wait for stream to complete.
        HANDLE_ERROR(cudaStreamSynchronize(stream));

        // Destroy stream
        HANDLE_ERROR(cudaStreamDestroy(stream));
    });

    printTime((long long)time.count());
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

Args parseArgs(int argc, char** argv)
{
    Args args;
    args.blockSize = 256;
    args.totalThreads = (1 << 20);
    
    if (argc >= 2) 
    {
        args.totalThreads = atoi(argv[1]);
    }
    if (argc >= 3) 
    {
        args.blockSize = atoi(argv[2]);
    }

    if (args.totalThreads % args.blockSize != 0)
    {
        args.totalThreads = (args.numBlocks() + 1) * args.blockSize;
        
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", args.totalThreads);
    }

    return args;
}

void fillHostInputArrays(const Args& args, Arrays& arrays)
{
    std::vector<int> inputA(args.totalThreads);
    std::vector<int> inputB(args.totalThreads);

    srand(time(NULL));
    std::iota(inputA.begin(), inputA.end(), 0);
    std::generate(inputB.begin(), inputB.end(), []{ return randRange(RANDOM_MIN, RANDOM_MAX + 1); });

    HANDLE_ERROR(cudaMemcpy(arrays.h_inputA, inputA.data(), args.arraySize<int>(), cudaMemcpyDefault));
    HANDLE_ERROR(cudaMemcpy(arrays.h_inputA, inputA.data(), args.arraySize<int>(), cudaMemcpyDefault));
}

int main(int argc, char** argv) 
{
    Arrays arrays = {};
    Args args = parseArgs(argc, argv);

    printf("With %d blocks of size %d (%d total threads)...\n", 
           args.numBlocks(), 
           args.blockSize, 
           args.totalThreads);

    // Allocate Pinned Memory
    HANDLE_ERROR(cudaMallocHost((void**)&arrays.h_inputA, args.arraySize<int>()));
    HANDLE_ERROR(cudaMallocHost((void**)&arrays.h_inputB, args.arraySize<int>()));
    HANDLE_ERROR(cudaMallocHost((void**)&arrays.h_output, args.arraySize<int>()));

    // Allocate Device Global Memory
    HANDLE_ERROR(cudaMalloc((void**)&arrays.d_inputA, args.arraySize<int>()));
    HANDLE_ERROR(cudaMalloc((void**)&arrays.d_inputB, args.arraySize<int>()));
    HANDLE_ERROR(cudaMalloc((void**)&arrays.d_output, args.arraySize<int>()));

    // Fill host input arrays
    fillHostInputArrays(args, arrays);

    // Print headers.
    printf("Timings (in microseconds)\n");
    printHeader("");
    printHeader("Add");
    printHeader("Subtract");
    printHeader("Multiply");
    printHeader("Modulo");
    printf("\n");

    // Measure times using synchronous calls.
    printHeader("Synchronous");
    timeSynchronous(args, arrays, add);
    timeSynchronous(args, arrays, subtract);
    timeSynchronous(args, arrays, multiply);
    timeSynchronous(args, arrays, modulo);
    printf("\n");

    // Measure times using streaming calls.
    printHeader("Streamed");
    timeStreamed(args, arrays, add);
    timeStreamed(args, arrays, subtract);
    timeStreamed(args, arrays, multiply);
    timeStreamed(args, arrays, modulo);
    printf("\n");

    printf("\n");

    HANDLE_ERROR(cudaGetLastError());

    // Free Pinned Memory
    HANDLE_ERROR(cudaFreeHost(arrays.h_inputA));
    HANDLE_ERROR(cudaFreeHost(arrays.h_inputB));
    HANDLE_ERROR(cudaFreeHost(arrays.h_output));

    // Free Device Global Memory
    HANDLE_ERROR(cudaFree(arrays.d_inputA));
    HANDLE_ERROR(cudaFree(arrays.d_inputB));
    HANDLE_ERROR(cudaFree(arrays.d_output));
}
