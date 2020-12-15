/**
 * @file
 * @author Caleb Rush
 * 
 * @brief A program that takes in a list of kernel names from command line and
 *        runs each kernel.
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
constexpr size_t BUFFER_SIZE = 100000;

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

//==============================================================================
// Main
//==============================================================================

/**
 * Parse the command line arguments for the program.
 * 
 * @param[in] argc The number of command line arguments.
 * @param[in] argv The values of the command line arguments.
 * 
 * @return the list of commands entered by the user. 
 */
std::vector<std::string> parseArgs(int argc, char** argv)
{
    if (argc <= 1)
    {
        throw std::invalid_argument("Must specify at least one command!");
    }

    std::vector<std::string> commands(argc - 1);
    
    for (size_t i = 1; i < argc; i++)
    {
        commands.push_back(argv[i]);
    }

    return commands;
}

/**
 * The entry point into the program.
 */
int main(int argc, char** argv)
{
    try
    {
        // Read commands from the command line.
        auto commands = parseArgs(argc, argv);

        // Create an OpenCL context.
        cl::Context context = createContext();

        // Load the kernels specified in the command line.
        std::vector<cl::Kernel> kernels(commands.size());
        for (size_t i = 0; i < commands.size(); i++)
        {
            kernels[i] = loadKernel(context, "assignment.cl", commands[i]);
        }

        // Generate data to manipulate.
        std::array<uint8_t, BUFFER_SIZE> hostBuffer = {};
        cl::Buffer deviceBuffer(context, CL_MEM_READ_WRITE, BUFFER_SIZE);
        
        std::iota(hostBuffer.begin(), hostBuffer.end(), 0);

        // Run each kernel.
        auto time = timeIt([&]
        {
            // Create command queue.
            cl::CommandQueue commandQueue(context);

            // Copy data from host to device.
            commandQueue.enqueueWriteBuffer(deviceBuffer, 
                                            true, 
                                            0, 
                                            hostBuffer.size(), 
                                            hostBuffer.data());
            
            // Enqueue each kernel.
            for (cl::Kernel& kernel : kernels)
            {
                kernel.setArg(0, deviceBuffer);
                commandQueue.enqueueNDRangeKernel(kernel, 
                                                  cl::NullRange, 
                                                  BUFFER_SIZE, 
                                                  cl::NullRange);
            }

            // Copy data from device to host.
            commandQueue.enqueueReadBuffer(deviceBuffer, 
                                           true, 
                                           0, 
                                           hostBuffer.size(), 
                                           hostBuffer.data());
        });

        std::cout << "Finished in " << time.count() << " microseconds." << std::endl;
        std::cout << std::endl;
    }
    catch (const std::invalid_argument& e)
    {
        std::cerr << "Invalid args: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const cl::Error& e)
    {
        std::cerr << "OpenCL Error (" << e.err() << "): " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
