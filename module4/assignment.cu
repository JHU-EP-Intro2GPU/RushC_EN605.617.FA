//Based on the work of Andrew Krepps

// C
#include <stddef.h>
#include <stdio.h>

// C++
#include <chrono>
#include <functional>
#include <initializer_list>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////

// The range of the random values to generate for the second input array.
constexpr int RANDOM_MIN = 0;
constexpr int RANDOM_MAX = 3; 

///////////////////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////////////////

// A function pointer for a kernel function that performs an arithmetic operation. 
using ArithmeticKernelFunction = void (*)(const int*, const int*, int*);

// A function pointer for a CUDA malloc function (e.g. cudaMalloc(), cudaMallocHost())
using CudaMallocFunction = cudaError_t (*)(void**, size_t);

// A function pointer for a CUDA free function (e.g. cudaFree(), cudaFreeHost())
using CudaFreeFunction = cudaError_t (*)(void*);

// A structure to hold all of the command line args.
struct Args
{
    int numBlocks;
    int blockSize;
    int totalThreads;
    int offset;
};

// A structure to hold the GPU arrays used by the arithmetic kernels.
struct ArithmeticArrays
{
    int* inputA;
    int* inputB;
    int* output;
};

// A structure to hold the GPU arrays used by the Caeser cipher kernel.
struct CaeserArrays
{
    char *input;
    char *output;
};

///////////////////////////////////////////////////////////////////////////////
// Utilities
///////////////////////////////////////////////////////////////////////////////

/**
 * Calculate the index of the current GPU thread.
 *
 * This is intended to be called by a kernel function.
 */
__device__
unsigned threadIndex()
{
    return (blockIdx.x * blockDim.x) + threadIdx.x;
}

/**
 * Generate a random value in the range [min, max).
 */
int randRange(int min, int max)
{
    return (rand() % max) + min;
}

/**
 * Generate a random byte.
 */
 char randByte()
 {
     return (char)randRange(0x00, 0xFF);
 }

/**
 * Calculate the time it takes to perform a function.
 */
std::chrono::nanoseconds timeIt(std::function<void()> func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

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

__global__
void caeser(const char plaintext[], char ciphertext[], int offset)
{
    unsigned index = threadIndex();
    ciphertext[index] = plaintext[index] + offset;
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

/**
 * Parse the command line args.
 *
 * The body of this function was supplied in the starter file. The parsing of
 * the offset was added by me.
 */
Args parseArgs(int argc, char** argv)
{
    // read command line arguments
    Args args;
    args.blockSize = 256;
    args.totalThreads = (1 << 20);
    args.offset = 9;
    
    if (argc >= 2) 
    {
        args.totalThreads = atoi(argv[1]);
    }
    if (argc >= 3) 
    {
        args.blockSize = atoi(argv[2]);
    }
    if (argc >= 4) 
    {
        args.offset = atoi(argv[3]);
    }

    args.numBlocks = args.totalThreads / args.blockSize;

    // validate command line arguments
    if (args.totalThreads % args.blockSize != 0)
     {
        ++args.numBlocks;
        args.totalThreads = args.numBlocks * args.blockSize;
        
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", args.totalThreads);
    }

    return args;
}

ArithmeticArrays allocateArithmeticArrays(Args args, CudaMallocFunction deviceMalloc)
{
    int arraySize = args.totalThreads * sizeof(int);

    ArithmeticArrays arrays;
    deviceMalloc((void**)&arrays.inputA, arraySize);
    deviceMalloc((void**)&arrays.inputB, arraySize);
    deviceMalloc((void**)&arrays.output, arraySize);
    return arrays;
}

CaeserArrays allocateCaeserArrays(Args args, CudaMallocFunction deviceMalloc)
{
    CaeserArrays arrays;
    deviceMalloc((void**)&arrays.input,  args.totalThreads);
    deviceMalloc((void**)&arrays.output, args.totalThreads);
    return arrays;
}

void fillArithmeticArrays(ArithmeticArrays arrays, Args args)
{
    int arraySize = args.totalThreads * sizeof(int);

    std::vector<int> cpuA(args.totalThreads);
    std::vector<int> cpuB(args.totalThreads);

    for (int i = 0; i < args.totalThreads; i++)
    {
        cpuA[i] = i;
        cpuB[i] = randRange(RANDOM_MIN, RANDOM_MAX + 1);
    }

    cudaMemcpy(arrays.inputA, cpuA.data(), arraySize, cudaMemcpyDefault);
    cudaMemcpy(arrays.inputB, cpuB.data(), arraySize, cudaMemcpyDefault);
}

void fillCaeserArrays(CaeserArrays arrays, Args args)
{
    std::vector<char> cpu(args.totalThreads);

    for (int i = 0; i < args.totalThreads; i++)
    {
        cpu[i] = randByte();
    }

    cudaMemcpy(arrays.input, cpu.data(), args.totalThreads, cudaMemcpyDefault);
}

/**
 * Free the memory allocated for the specified arithmetic arrays.
 */
void freeArithmeticArrays(ArithmeticArrays arrays, CudaFreeFunction deviceFreeFunction)
{
    deviceFreeFunction(arrays.inputA);
    deviceFreeFunction(arrays.inputB);
    deviceFreeFunction(arrays.output);
}

/**
 * Free the memory allocated for the specified Caeser arrays.
 */
void freeCaeserArrays(CaeserArrays arrays, CudaFreeFunction deviceFreeFunction)
{
    deviceFreeFunction(arrays.input);
    deviceFreeFunction(arrays.output);
}

/**
 * Print the headers for each row of timing metrics.
 */
void printHeaders()
{
    printf("%10s %10s %10s %10s %10s %10s\n",
           "", "Add", "Subtract", "Multiply", "Modulo", "Caeser");
}

/**
 * Time how long it takes to run each of the kernels using the specified arrays.
 */
void printTimingMetrics(const char label[],
                        Args args, 
                        CudaMallocFunction deviceMalloc, 
                        CudaFreeFunction deviceFree, 
                        bool copyRequired)
{
    std::vector<int> arithmeticOutput(args.totalThreads);
    std::vector<char> caeserOutput(args.totalThreads);

    printf("%10s", label);

    // Time arithmetic kernels
    ArithmeticKernelFunction arithmeticOperations[] = { add, subtract, multiply, modulo, };
    for (auto func : arithmeticOperations)
    {
        ArithmeticArrays arrays = allocateArithmeticArrays(args, deviceMalloc);
        cudaDeviceSynchronize();
        auto time = timeIt([&]
        {
            fillArithmeticArrays(arrays, args);
            func<<<args.numBlocks, args.blockSize>>>(arrays.inputA, arrays.inputB, arrays.output);
            if (copyRequired)
            {
                cudaMemcpy(arithmeticOutput.data(), arrays.output, args.totalThreads * sizeof(int), cudaMemcpyDefault);
            }
        });
        freeArithmeticArrays(arrays, deviceFree);

        printf(" %10lld", (long long)time.count());
    }

    // Time Caeser Kernel    
    cudaDeviceSynchronize();
    auto time = timeIt([&]
    {
        CaeserArrays arrays = allocateCaeserArrays(args, deviceMalloc);
        fillCaeserArrays(arrays, args);
        caeser<<<args.numBlocks, args.blockSize>>>(arrays.input, arrays.output, args.offset);
        if (copyRequired)
        {
            cudaMemcpy(caeserOutput.data(), arrays.output, args.totalThreads, cudaMemcpyDefault);
        }
        freeCaeserArrays(arrays, deviceFree);
    });

    printf(" %10lld\n", (long long)time.count());
}

int main(int argc, char** argv) 
{
    // Parse command line args.
    Args args = parseArgs(argc, argv);
    printf("With %d blocks of size %d (%d total threads) and an offset of %d...\n", 
           args.numBlocks, 
           args.blockSize, 
           args.totalThreads, 
           args.offset);

    // Seed random to fill the inputs with random values.
    srand(time(NULL));

    printHeaders();
    printTimingMetrics("Paged", args, cudaMalloc, cudaFree, true);
    printTimingMetrics("Pinned", args, cudaMallocHost, cudaFreeHost, false);
    printf("\n");

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("FAILURE: %s\n", cudaGetErrorString(error));
    }
}
