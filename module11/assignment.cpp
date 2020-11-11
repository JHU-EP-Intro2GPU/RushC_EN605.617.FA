//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.
//
// -----------------------------------------------------------------------------
//
//    This file was modified by Caleb Rush to make the following changes:
//    - Change the size of the input signal to 49x49
//    - Change the size of the mask to 7x7
//    - Randomize the input signal
//    - Measure and display timing of convolutions
//    - Allow the user to specify how many convolutions to perform
//    - ALlow the user to specify whether or not the output should be displayed

// C
#include <stdlib.h>
#include <time.h>

// C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// Custom header
#include "utilities.hpp"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Struct containing the command line arguments.
struct Args
{
    int numConvolutions = 1;
    bool displayOutput = true;
};

// Constants
constexpr int SIGNAL_MIN = 0;
constexpr int SIGNAL_MAX = 9;
constexpr int NUM_CONVOLUTIONS = 3;

const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;

cl_uint inputSignal[inputSignalHeight][inputSignalWidth] =
{
	// Randomly generated
};

const unsigned int outputSignalWidth  = 46;
const unsigned int outputSignalHeight = 46;

cl_uint outputSignal[outputSignalHeight][outputSignalWidth];

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

cl_uint mask[maskHeight][maskWidth] =
{
	{1, 1, 1, 1, 1, 1, 1},
    {1, 2, 2, 2, 2, 2, 1},
	{1, 2, 3, 3, 3, 2, 1},
	{1, 2, 3, 4, 3, 2, 1},
	{1, 2, 3, 3, 3, 2, 1},
	{1, 2, 2, 2, 2, 2, 1},
	{1, 1, 1, 1, 1, 1, 1},
};

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

/**
 * Generate new values for the input signal.
 */
void generateInputSignal()
{
    srand(time(nullptr));

    for (int y = 0; y < inputSignalHeight; y++)
    {
        for (int x = 0; x < inputSignalWidth; x++)
        {
            inputSignal[y][x] = (cl_uint)(randRange(SIGNAL_MIN, SIGNAL_MAX+1));
        }
    }
}

/**
 * Time how long it takes to perform a convolution on the current input signal.
 */
void timeConvolution(cl_context context, cl_command_queue queue, cl_kernel kernel)
{
    cl_int errNum;
    cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

    // Randomize input signal.
    generateInputSignal();

    // Allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
		static_cast<void *>(inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * maskHeight * maskWidth,
		static_cast<void *>(mask),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

    // Set kernel args.
    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
	checkErr(errNum, "clSetKernelArg");

    // Finish all the tasks in the command queue so that we only measure the
    // time it takes to finish the kernel.
    errNum = clFinish(queue);
    checkErr(errNum, "clFinish");

    auto time = timeIt([&]
    {
        const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
        const size_t localWorkSize[2]  = { 1, 1 };

        // Queue the kernel up for execution across the array
        errNum = clEnqueueNDRangeKernel(
            queue, 
            kernel, 
            2,
            NULL,
            globalWorkSize,
            localWorkSize,
            0, 
            NULL, 
            NULL);
        checkErr(errNum, "clEnqueueNDRangeKernel");
        
        errNum = clEnqueueReadBuffer(
            queue, 
            outputSignalBuffer, 
            CL_TRUE,
            0, 
            sizeof(cl_uint) * outputSignalHeight * outputSignalHeight, 
            outputSignal,
            0, 
            NULL, 
            NULL);
        checkErr(errNum, "clEnqueueReadBuffer");
    });

    // Output time.
    std::cout << "Performed convolution in " << time.count() << " microseconds" << std::endl;
}

Args parseArgs(int argc, char** argv)
{
    Args args = {};

    if (argc > 1)
    {
        std::istringstream(argv[1]) >> args.numConvolutions;
    }
    if (argc > 2)
    {
        std::istringstream(argv[2]) >> std::boolalpha >> args.displayOutput;
    }

    return args;
}

///
//	main() for Convoloution example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;

    // Parse command line arguments.
    Args args = parseArgs(argc, argv);

    // First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
			buildLog, 
			NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

    // Perform and time multiple convolutions.
    for (int i = 0; i < args.numConvolutions; i++)
    {
        timeConvolution(context, queue, kernel);

        if (args.displayOutput)
        {
            // Output the result buffer
            for (int y = 0; y < outputSignalHeight; y++)
            {
                for (int x = 0; x < outputSignalWidth; x++)
                {
                    std::cout << outputSignal[y][x] << " ";
                }
                std::cout << std::endl;
            }

            std::cout << std::endl;
        }
    }

    std::cout << "Executed program succesfully." << std::endl;

	return 0;
}
