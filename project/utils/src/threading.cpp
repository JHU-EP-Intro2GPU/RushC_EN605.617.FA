// C++
#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

// utils
#include <utils/threading.hpp>

void utils::run_jobs(size_t num_jobs, const std::function<void(size_t index)>& perform) 
{
    // Determine the number of threads based on the platform's recommendedations.
    size_t num_threads = std::min<size_t>(num_jobs, std::thread::hardware_concurrency());

    // Assign each thread a list of jobs using an interleaved approach.
    //
    // Example:
    //     num_jobs = 18
    //     num_threads = 4
    //
    //     Thread 1 --> Job 0, Job 4, Job  8, Job 12, Job 16,
    //     Thread 2 --> Job 1, Job 5, Job  9, Job 13, Job 17
    //     Thread 3 --> Job 2, Job 6, Job 10, Job 14
    //     Thread 4 --> Job 3, Job 7, Job 11, Job 15
    std::vector<std::thread> threads(num_threads);
    for (size_t thread_index = 0; thread_index < num_threads; thread_index++)
    {
        threads[thread_index] = std::thread([=]
        {
            for (size_t job_index = thread_index; 
                 job_index < num_jobs; 
                 job_index += num_threads)
            {
                perform(job_index);
            }
        });
    }

    // Wait until each thread completes.
    for (std::thread& thread : threads)
    {
        thread.join();
    }
}