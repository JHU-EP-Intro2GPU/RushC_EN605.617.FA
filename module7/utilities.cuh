#ifndef UTILITIES_CUH
#define UTILITIES_CUH

// C
#include <stdlib.h>

// C++
#include <chrono>
#include <functional>
#include <iostream>

/**
 * Print the error message and exit the program if a CUDA error occurs.
 */
#define HANDLE_ERROR(expression) (handleError((expression), __FILE__, __LINE__))

#define TRACE printf("[TRACE] %s: line %d\n", __FILE__, __LINE__);

/**
 * Print the error message and exit the program if a CUDA error occurs.
 */
inline void handleError(cudaError_t error, const char file[], int line)
{
    if (error != cudaSuccess)
    {
        std::cerr << std::endl
                  << cudaGetErrorString(error) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * Generate a random value in the range [min, max).
 */
int randRange(int min, int max)
{
    return (rand() % max) + min;
}

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
 * Calculate the time it takes to perform a function.
 */
template<typename DurationType = std::chrono::microseconds>
DurationType timeIt(std::function<void()> func)
{
    cudaEvent_t start;
    cudaEvent_t stop;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start));
    func();
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    auto elapsedDuration = std::chrono::microseconds((long)(elapsedTime * 1000));
    return std::chrono::duration_cast<DurationType>(elapsedDuration);
}

#endif // UTILITIES_CUH