/**
 * @file
 * @author Caleb Rush
 * 
 * @brief Defines various OpenCL kernels that modify values in a buffer.
 */

/**
 * Square all of the values in a buffer.
 */
__kernel void square(__global int* buffer)
{
    size_t id = get_global_id(0);
    buffer[id] = buffer[id] * buffer[id];
}

/**
 * Cube all of the values in a buffer.
 */
__kernel void cube(__global int* buffer)
{
    size_t id = get_global_id(0);
    buffer[id] = buffer[id] * buffer[id] * buffer[id];
}

/**
 * Perform an arithmetic negation (multiply by -1) on each value in a buffer.
 */
__kernel void negate_arithmetic(__global int* buffer)
{
    size_t id = get_global_id(0);
    buffer[id] *= -1;
}

/**
 * Perform a bitwise negation on each value in a buffer.
 */
__kernel void negate_bitwise(__global * buffer)
{
    size_t id = get_global_id(0);
    buffer[id] = ~buffer[id];
}