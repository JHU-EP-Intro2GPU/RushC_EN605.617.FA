
// For reference only. This shouldn't actually be used.
__kernel void hello_kernel(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] + b[gid];
}

/**
 * Add the corresponding elements of two arrays.
 */
__kernel void add_kernel(__global const float *a,
						 __global const float *b,
						 __global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] + b[gid];
}

/**
 * Subtract the corresponding elements of two arrays.
 */
__kernel void subtract_kernel(__global const float *a,
						      __global const float *b,
						      __global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] - b[gid];
}

/**
 * Multiply the corresponding elements of two arrays.
 */
__kernel void multiply_kernel(__global const float *a,
						      __global const float *b,
						      __global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] * b[gid];
}

/**
 * Divide the corresponding elements of two arrays.
 */
__kernel void divide_kernel(__global const float *a,
						    __global const float *b,
						    __global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] / b[gid];
}

/**
 * Raise each element of the first array to the power of the corresponding 
 * element of the second array.
 */
__kernel void power_kernel(__global const float *a,
						   __global const float *b,
						   __global float *result)
{
    int gid = get_global_id(0);

    result[gid] = pow(a[gid], b[gid]);
}

