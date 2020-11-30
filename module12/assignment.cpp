/**
 * @file
 * @author Caleb Rush
 * 
 * @brief A program that uses OpenCL sub-buffers to calculate the average of
 *        various regions of a buffer and outputs timing metrics.
 */

// Enable OpenCL exceptions for simpler error handling.
#define __CL_ENABLE_EXCEPTIONS

//==============================================================================
// Includes
//==============================================================================

// C++
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

// OpenCL
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

//==============================================================================
// Constants
//==============================================================================

// The size of the sub buffers.
constexpr size_t SUB_BUFFER_SIZE = 2 * 2;

//==============================================================================
// Utilities
//==============================================================================

/**
 * Debugging macro: Print the current line to stdout.
 */
#define TRACE std::cout << __FILE__ << ": line " << __LINE__ << std::endl; 

/**
 * Read the contents of the file at a specified path.
 * 
 * @param path The path of the file to read the contents of.
 * 
 * @return The file contents. 
 */
std::string loadFile(const std::string& path)
{
    std::ifstream file(path);
    if (!file)
    {
        throw std::invalid_argument("Cannot open file: " + path);
    }

    std::string contents;
    file.seekg(std::ios::end);
    contents.reserve(file.tellg());
    file.seekg(0);
    contents.assign(std::istreambuf_iterator<char>(file),
                    std::istreambuf_iterator<char>());

    return contents;
}

/**
 * Print out an array of bytes.
 * 
 * @param[in] data     The array of bytes to print.
 * @param[in] dataSize The number of elements in the @c data array.
 */
void printData(const uint8_t data[], size_t dataSize)
{
    std::cout << std::hex;
    for (size_t i = 0; i < dataSize; i++)
    {
        std::cout << std::setw(2) << std::setfill('0') << (unsigned)data[i] << " ";
    }
    std::cout << std::dec;
}

/**
 * Calculate the size of a buffer based on a number of elements of a specific
 * type.
 * 
 * @tparam T The type of the elements stored in the buffer.
 * @param numElements The number of elements stored in the buffer.
 * 
 * @return the size of the buffer in bytes.
 */
template<typename T>
constexpr size_t size(size_t numElements)
{
    return numElements * sizeof(T);
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

//==============================================================================
// Helpers
//==============================================================================

/**
 * Create an OpenCL context using the first device on the default platform.
 * 
 * @return the created context.
 * 
 * @throw cl::Error          if an OpenCL error occurs.
 * @throw std::runtime_error if a miscellaneous error occurs.
 */
cl::Context createContext()
{
    // Use the default platform.
    auto platform = cl::Platform::getDefault();

    // Get a list of devices available on the platform.
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty())
    {
        throw std::runtime_error("No devices available for default platform");
    }

    // Create a context with the first device.
    return cl::Context(devices[0]);
}

/**
 * Compile an OpenCL kernel.
 * 
 * @param[in] programPath The path to the OpenCL program containing the kernel.
 * @param[in] kernelName  The name of the kernel to load from the program.
 * 
 * @return The loaded kernel.
 * 
 * @throw cl::Error             if an OpenCL error occurs.
 * @throw std::invalid_argument if @c programPath does not name a file,
 *                              or if the program fails to build,
 *                              or if @c kernelName does not name a valid kernel
 */
cl::Kernel loadKernel(const cl::Context& context, 
                      const std::string& programPath, 
                      const std::string& kernelName)
{
    // Build the OpenCL program
    cl::Program program(context, loadFile(programPath));
    try
    {
        program.build();
    }
    catch (const cl::Error& e)
    {
        cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

        std::stringstream error;
        error << "Build Errors for " << programPath << std::endl << buildLog;

        throw std::runtime_error(error.str());
    }

    // Load all of the kernels from the program.
    std::vector<cl::Kernel> kernels;
    program.createKernels(&kernels);

    // Find the kernel with the specified name.
    for (const cl::Kernel& kernel : kernels)
    {
        if (kernelName.compare(kernel.getInfo<CL_KERNEL_FUNCTION_NAME>()))
        {
            return kernel;
        }
    }

    throw std::invalid_argument("The program " 
                                + programPath + 
                                " does not contain a kernel named " 
                                + kernelName);
}

/**
 * Create sub buffers from a specified buffer.
 * 
 * @param[in] buffer        The buffer to create the sub buffers from.
 * @param[in] bufferSize    The size (in bytes) of @c buffer.
 * @param[in] subBufferSize The size (in bytes) of each sub buffer.
 * 
 * @return The created sub buffers.
 */
std::vector<cl::Buffer> createSubBuffers(cl::Buffer& buffer, 
                                         size_t bufferSize, 
                                         size_t subBufferSize)
{
    std::vector<cl::Buffer> subBuffers;

    for (size_t origin = 0; origin < bufferSize; origin += subBufferSize)
    {
        size_t size = std::min(subBufferSize, bufferSize - origin);
        cl_buffer_region region = { origin, size };
        subBuffers.push_back(buffer.createSubBuffer(
                CL_MEM_READ_WRITE, 
                CL_BUFFER_CREATE_TYPE_REGION, 
                &region));
    }

    return subBuffers;
}

/**
 * Time how long it takes to run a kernel on a set of data.
 * 
 * @param[in] kernel        The kernel to run.
 * @param[in] bufferSize    The size of the input/output data.
 * @param[in] subBufferSize The size of each sub-buffer to divide the main buffer into.
 * 
 * @return The time (in microseconds) it takes to run the kernel.
 */
long long timeKernel(const cl::Context& context,
                     cl::Kernel& kernel, 
                     size_t bufferSize, 
                     size_t subBufferSize)
{
    std::vector<uint8_t> input(bufferSize);
    std::vector<uint8_t> output(bufferSize);

    // Create main buffer.
    cl::Buffer buffer(context, CL_MEM_READ_WRITE, size<int>(bufferSize));

    // Create sub-buffers.
    std::vector<cl::Buffer> subBuffers = createSubBuffers(
                buffer, 
                bufferSize, 
                subBufferSize);
    
    // Generate input data.
    std::iota(input.begin(), input.end(), 0);

    // Create command queue.
    cl::CommandQueue commandQueue(context);

    auto time = timeIt([&]
    {
        // Write input data to the main buffer.
        commandQueue.enqueueWriteBuffer(buffer, true, 0, input.size(), input.data());

        // Call the kernel on each sub-buffer.
        for (cl::Buffer& subBuffer : subBuffers)
        {
            kernel.setArg(0, subBuffer);
            kernel.setArg<int>(1, subBufferSize);
            commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, 1);
        }

        // Read output data from the main buffer.
        commandQueue.enqueueReadBuffer(buffer, true, 0, output.size(), output.data());
    });

    // Print data.
    std::cout << "Input:  ";
    printData(input.data(), input.size());
    std::cout << std::endl;

    std::cout << "Output: ";
    printData(output.data(), output.size());
    std::cout << std::endl;

    return time.count();
}

//==============================================================================
// Main
//==============================================================================

/**
 * The entry point into the program.
 */
int main()
{
    try
    {
        cl::Context context = createContext();
        cl::Kernel kernel = loadKernel(context, "assignment.cl", "average");

        for (size_t numSubBuffers : { 4, 8, 16, 32 })
        {
            size_t bufferSize = numSubBuffers * SUB_BUFFER_SIZE;
            std::cout << "With a buffer of size " << bufferSize << "..." << std::endl;

            auto time = timeKernel(context, kernel, bufferSize, SUB_BUFFER_SIZE);
            std::cout << "Finished in " << time << " microseconds." << std::endl;

            std::cout << std::endl;
        }
    }
    catch (const cl::Error& e)
    {
        std::cerr << "OpenCL Error (" << e.err() << "): " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}
