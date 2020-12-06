#ifndef UTILS_TIMING_HPP
#define UTILS_TIMING_HPP

// C++
#include <chrono>
#include <functional>

namespace utils
{
    /**
     * Calculate the amount of time it takes to perform a function.
     * 
     * @tparam DurationType The granularity of the timing measurement.
     * 
     * @param[in] function The function to time.
     * 
     * @return The duration that elapsed while running the function. 
     */
    template<typename DurationType = std::chrono::microseconds>
    DurationType time_it(const std::function<void()>& function)
    {
        auto start = std::chrono::high_resolution_clock::now();
        function();
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<DurationType>(end - start);
    }

} // namespace utils

#endif // UTILS_TIMING_HPP