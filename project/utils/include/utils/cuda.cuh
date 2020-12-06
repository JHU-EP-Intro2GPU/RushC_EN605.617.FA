#ifndef UTILS_CUDA_CUH
#define UTILS_CUDA_CUH

// C
#include <stddef.h>

// C++
#include <functional>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>

namespace utils
{
    //--------------------------------------------------------------------------
    // Device Helpers
    //--------------------------------------------------------------------------

    /// GPU implementation of min().
    template<typename T>
    __device__
    inline T min(T a, T b)
    {
        return a < b ? a : b;
    }

    /**
     * Calculate the ID of the current thread.
     * 
     * @return the ID of the current thread.
     */
    __device__
    inline unsigned thread_id()
    {
        return (blockIdx.x * blockDim.x) + threadIdx.x;
    }

    //--------------------------------------------------------------------------
    // Error Handling
    //--------------------------------------------------------------------------

    /**
     * An exception that occurs due to a failed CUDA call.
     */
    class CudaException : public std::runtime_error
    {
    public:
        /**
         * Construct a CudaException.
         * 
         * @param[in] error The error that occurred.
         * @param[in] file  The file containing the line that caused the error.
         * @param[in] line  The line that caused the error.
         */
        CudaException(cudaError_t error, const char file[], int line) :
            std::runtime_error(message(error, file, line))
        {
            // Empty   
        }

    private:
        /**
         * Generate the @c what() message for the exception.
         * 
         * @param[in] error The error that occurred.
         * @param[in] file  The file containing the line that caused the error.
         * @param[in] line  The line that caused the error.
         * 
         * @return the generated message.
         */
        static std::string message(cudaError_t error, const char file[], int line)
        {
            std::stringstream string;
            string << "in " << file << " on line #" << line << ": "
                   << "(" << error << "): " << cudaGetErrorString(error);
            return string.str();
        }
    };

    /**
     * Call a CUDA function and handle any errors by throwing a CudaException.
     */
    #define CUDA_CALL(expression)                                  \
    do                                                             \
    {                                                              \
        cudaError_t error = (expression);                          \
        if (error != cudaSuccess)                                  \
        {                                                          \
            throw utils::CudaException(error, __FILE__, __LINE__); \
        }                                                          \
    } while (false)

    //--------------------------------------------------------------------------
    // Kernels
    //--------------------------------------------------------------------------

    /**
     * Launch a kernel using the most efficient grid sizes and block sizes as
     * reported by the hardware.
     * 
     * @tparam KernelFunction The type of the kernel function.
     * @tparam Args           The types of the arguments to pass to the kernel.
     * 
     * @param[in] kernel         The kernel to be launched.
     * @param[in] threads_needed The number of threads that need to be launched.
     * @param[in] args           The arguments to pass to the kernel.
     * 
     * @c threads_needed does not need to be rounded; it can just be the size of
     * the data. This function will perform any appropriate rounding/padding.
     * However, note that @c kernel will need to perform checks to make sure 
     * that it does not perform an out-of-bounds memory access in case more 
     * threads are launched than are needed.
     * 
     * @throws CudaException if a CUDA error occurs.
     */
    template<typename KernelFunction, typename... Args>
    void launch_kernel(KernelFunction kernel, 
                       size_t threads_needed, 
                       Args&&... args)
    {
        // Determine the ideal grid and block size.
        int min_grid_size;
        int block_size;
        CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, 
                                                     &block_size, 
                                                     kernel, 
                                                     0, 
                                                     0));
        
        // Calculate the actual grid size needed to launch each of the threads.
        auto grid_size = (threads_needed + block_size - 1) / block_size;

        // Call the kernel.
        kernel<<<grid_size, block_size>>>(args...);
    }

    //--------------------------------------------------------------------------
    // Memory Management
    //--------------------------------------------------------------------------

    /// Type of a function pointer for a CUDA malloc function.
    using CudaMalloc = cudaError_t (*)(void** ptr, size_t size);

    /// Type of a function pointer for a CUDA free function.
    using CudaFree = cudaError_t (*)(void* ptr);

    /**
     * Memory managed wrapper for a memory region allocated via CUDA.
     * 
     * @tparam T      The type of element held in the memory region.
     * @tparam MALLOC The CUDA function used to allocate the memory region.
     * @tparam FREE   The CUDA function used to clear the memory region.
     */
    template<typename T, CudaMalloc MALLOC, CudaFree FREE>
    class CudaBuffer
    {
    public:
        /**
         * Construct a CudaBuffer with the specified number of elements.
         * 
         * @param num_elements The number of elements to allocate.
         *
         * @throw std::bad_alloc If the memory allocation fails.
         */
        explicit CudaBuffer(size_t num_elements) :
            num_elements(num_elements),
            pointer(allocate(size()), FREE)
        {
            // Empty
        }

        /**
         * Copy memory from a host array to this buffer.
         * 
         * @param[in] source The host array to copy into this buffer.
         */
        void copy_from(const T* source)
        {
            CUDA_CALL(cudaMemcpy( this->ptr(), source, this->size(), cudaMemcpyDefault));
        }

        /**
         * Copy memory from this buffer to a host array.
         * 
         * @param[out] destination The host array to copy this buffer into.
         */
        void copy_to(T* destination) const
        {
            CUDA_CALL(cudaMemcpy(destination, this->ptr(), this->size(), cudaMemcpyDefault));
        }

        /// Get the number of elements the memory buffer can hold.
        size_t count() const { return this->num_elements; }
        
        /// Get the underlying pointer to the memory buffer.
        T* ptr() { return this->pointer.get(); }
        const T* ptr() const { return this->pointer.get(); }

        /// Get the size (in bytes) of the memory buffer.
        size_t size() const { return this->num_elements * sizeof(T); }

    private:
        /**
         * Allocate a buffer using the CUDA malloc function.
         * 
         * @param num_elements The number of elements in the buffer.
         *
         * @return the pointer to the allocated buffer.
         *
         * @throw utils::CudaException If the memory allocation fails.
         */
        static T* allocate(size_t num_elements)
        {
            T* ptr;
            CUDA_CALL(MALLOC((void**)&ptr, num_elements * sizeof(T)));
            return ptr;
        }

        size_t num_elements;
        std::unique_ptr<T, CudaFree> pointer;
    };

    /// A @c CudaBuffer that lives in paged memory.
    template<typename T>
    using PagedCudaBuffer = CudaBuffer<T, cudaMalloc, cudaFree>;

    /// A @c CudaBuffer that lives in pinned memory.
    template<typename T>
    using PinnedCudaBuffer = CudaBuffer<T, cudaMallocHost, cudaFreeHost>;

} // namespace utils

#endif // UTILS_CUDA_CUH