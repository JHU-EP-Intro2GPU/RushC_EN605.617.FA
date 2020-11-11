#ifndef UTILITIES_CUH
#define UTILITIES_CUH

// C
#include <stdlib.h>

// C++
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>

/**
 * Print the error message and exit the program if a CUDA error occurs.
 */
#define HANDLE_ERROR(expression) (handleError((expression), __FILE__, __LINE__))

/**
 * Print the current file and line number to stdout.
 */
#define TRACE printf("[TRACE] %s: line %d\n", __FILE__, __LINE__);

/**
 * Print a value aligned to a column.
 */
template<typename T>
void printColumn(T t, int width = 8)
{
    std::cout << std::setw(width) << t;
}

/**
 * Generate a random value in the range [min, max).
 */
int randRange(int min, int max)
{
    return (rand() % max) + min;
}

/**
 * Calculate the time it takes to perform a function.
 */
template<typename DurationType = std::chrono::microseconds>
DurationType timeIt(std::function<void()> func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<DurationType>(end - start);
}

#endif // UTILITIES_CUH