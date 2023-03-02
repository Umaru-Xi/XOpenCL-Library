//
//  cpu_kernel.cl
//
//  Created by UmaruAya on 2/13/23.
//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void float_kernel(__global float *a, __global float *b, __global float *c){
    size_t id = get_global_id(0);
    c[id] = a[id] * b[id];
}
