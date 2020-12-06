#ifndef THREADING_HPP_
#define THREADING_HPP_

// C
#include <stddef.h>

// C++
#include <functional>
#include <thread>

namespace utils
{
    /**
     * Perform a number of jobs in parallel using various threads.
     * 
     * @param[in] num_jobs The number of jobs to perform.
     * @param[in] perform  The function to perform in each job.
     * 
     * @c perform will be passed in the index of the job being performed. How
     * many threads are used and how the jobs are divided among the threads is
     * decided automatically.
     * 
     * This function will block until all jobs are completed.
     */
    void run_jobs(size_t num_jobs, const std::function<void(size_t index)>& perform);

} // namespace utils

#endif // THREADING_HPP_