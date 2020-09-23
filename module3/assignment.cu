//Based on the work of Andrew Krepps

// C
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

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

/**
 * Parse the command line args.
 *
 * The body of this function was supplied in the starter file. It was not written by me.
 */
void parseArgs(int argc, char** argv, 
	int& totalThreads, 
	int& blockSize, 
	int& numBlocks) 
{
	// read command line arguments
	totalThreads = (1 << 20);
	blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
}

int main(int argc, char** argv) 
{
	int totalThreads = 0;
	int blockSize = 0;
	int numBlocks = 0;
	parseArgs(argc, argv, totalThreads, blockSize, numBlocks);

	printf("With %d blocks of size %d (%d total threads)...\n", numBlocks, blockSize, totalThreads);

	// Create GPU arrays.
	int* gpuInputA;
	int* gpuInputB;
	int* gpuOutput;

	int arraySize = totalThreads * sizeof(int);
	cudaMalloc((void**)&gpuInputA, arraySize);
	cudaMalloc((void**)&gpuInputB, arraySize);
	cudaMalloc((void**)&gpuOutput, arraySize);

	// Fill input arrays.
	std::vector<int> cpuA(totalThreads);
	std::vector<int> cpuB(totalThreads);

	srand(time(NULL));
	for (int i = 0; i < totalThreads; i++)
	{
		cpuA[i] = i;
		cpuB[i] = randRange(RANDOM_MIN, RANDOM_MAX + 1);
	}

	cudaMemcpy(gpuInputA, cpuA.data(), arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuInputB, cpuB.data(), arraySize, cudaMemcpyHostToDevice);

	// Time how long it takes to perform each of the arithmetic operations.
	using OperationFunc = void (*)(const int*, const int*, int*);
	std::initializer_list<std::pair<const char*, OperationFunc>> operations = 
	{ 
		{ "Add     ", add      },
		{ "Subtract", subtract },
		{ "Multiply", multiply },
		{ "Modulo  ", modulo   }
	};

	for (auto operation : operations)
	{
		auto name = operation.first;
		auto func = operation.second;

		// Clear the output array.  
		cudaMemset((void*)gpuOutput, 0, arraySize);

		// Time the operation.
		cudaDeviceSynchronize();
		auto time = timeIt([&]
		{ 
			func<<<numBlocks, blockSize>>>(gpuInputA, gpuInputB, gpuOutput);
			cudaDeviceSynchronize();
		});

		printf("%s   %7lldns\n", name, (long long)time.count());
	}

	printf("\n");

	// Free GPU arrays.
	cudaFree(gpuInputA);
	cudaFree(gpuInputB);
	cudaFree(gpuOutput);
}
