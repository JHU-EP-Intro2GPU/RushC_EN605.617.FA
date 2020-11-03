// C
#include <stdlib.h>

// C++
#include <algorithm>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

// Custom headers
#include "utilities.cuh"

///////////////////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////////////////

// The range of random values to generate for the input arrays.
constexpr int RANDOM_MIN = 1;
constexpr int RANDOM_MAX = 256;

///////////////////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////////////////

// Holds the command line arguments for the program.
struct Args
{
    int numValues;
};

///////////////////////////////////////////////////////////////////////////////
// Operations
///////////////////////////////////////////////////////////////////////////////

template<typename Operator, typename VectorType>
void calculate(const VectorType& a, const VectorType& b, VectorType& c)
{
    thrust::transform(a.begin(), a.end(), b.begin(), c.begin(), Operator());
}

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

/**
 * Parse the command line arguments.
 */
Args parseArgs(int argc, const char** argv)
{
    Args args = {};
    args.numValues = 512;

    if (argc > 1)
    {
        args.numValues = atoi(argv[1]);
    }

    return args;
}

/**
 * Time how long it takes to perform a single operation.
 */
template<typename Operator, typename VectorType>
long long timeOperation(const VectorType& a, const VectorType& b, VectorType& c)
{
    // Clear output vector
    thrust::fill(c.begin(), c.end(), 0);

    auto time = timeIt([&]
    {
        calculate<Operator>(a, b, c);
    });

    return (long long)time.count();
}

/**
 * Time how long it takes to perform each of the operations.
 */
template<typename VectorType>
void timeAllOperations(const Args& args)
{
    // Generate input values
    thrust::host_vector<int> h_a(args.numValues);
    thrust::host_vector<int> h_b(args.numValues);

    srand(time(nullptr));
    std::generate(h_a.begin(), h_a.end(), []{ return randRange(RANDOM_MIN, RANDOM_MAX); });
    std::generate(h_b.begin(), h_b.end(), []{ return randRange(RANDOM_MIN, RANDOM_MAX); });

    // Create vectors
    VectorType a = h_a;
    VectorType b = h_b;
    VectorType c(args.numValues);

    // Time each operation
    printColumn(timeOperation<thrust::plus<int>>(a, b, c));
    printColumn(timeOperation<thrust::minus<int>>(a, b, c));
    printColumn(timeOperation<thrust::multiplies<int>>(a, b, c));
    printColumn(timeOperation<thrust::modulus<int>>(a, b, c));
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

/**
 * Main entry point for the program.
 */
int main(int argc, const char** argv)
{
    Args args = parseArgs(argc, argv);

    printf("With vectors of size %d...\n", args.numValues);
    printColumn("");
    printColumn("Add");
    printColumn("Subtract");
    printColumn("Multiply");
    printColumn("Modulo");
    printf("\n");

    printColumn("Device");
    timeAllOperations<thrust::device_vector<int>>(args);
    printf("\n");
}