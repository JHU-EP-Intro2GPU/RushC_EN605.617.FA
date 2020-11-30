/**
 * @file
 * @author Caleb Rush
 * 
 * @brief Defines an OpenCL that filters a buffer.
 */

/**
 * Calculate the average of all of the elements in the sub buffer each of the
 * elements with the average value.
 */
__kernel void average(__global uchar* buffer, int size)
{
	size_t id = get_global_id(0);
    
    size_t start = id * size;
    size_t end = start + size;

    int average = 0;
    for (size_t i = start; i < end; i++)
    {
        average += buffer[i];
    }
    average /= size;

    for (size_t i = start; i < end; i++)
    {
        buffer[i] = average;
    }
}