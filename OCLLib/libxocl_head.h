//
//  libxocl_head.h
//
//  Created by UmaruAya on 2/16/23.
//

#ifndef libxocl_head_h
#define libxocl_head_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef __APPLE__
    #include <OpenCL/OpenCL.h>
#elifdef __linux__
    #include <CL/cl.h>
#else
    #warning "This operating system has not been considered."
    #include <CL/cl.h>
#endif

//    __sun           Defined on Solaris
//    __FreeBSD__     Defined on FreeBSD
//    __NetBSD__      Defined on NetBSD
//    __OpenBSD__     Defined on OpenBSD
//    __hpux          Defined on HP-UX
//    __osf__         Defined on Tru64 UNIX (formerly DEC OSF1)
//    __sgi           Defined on Irix
//    _AIX            Defined on AIX
//    _WIN32          Defined on Windows

enum xocl_return_flag{XOCL_RETURN_SUCCESS = 0, XOCL_RETURN_FAILD, XOCL_RETURN_WARNING};

enum xocl_device_type{XOCL_DEVICE_TYPE_CPU = 0, XOCL_DEVICE_TYPE_GPU, XOCL_DEVICE_TYPE_ALL};
enum xocl_program_type{XOCL_PROGRAM_TYPE_SOURCE = 0, XOCL_PROGRAM_TYPE_BINARY};

struct xocl_program{
    enum xocl_program_type type;
    cl_program program;
    char file[256];
};

struct xocl_context{
    cl_context context;
    struct xocl_platform *platform_ptr;
    struct xocl_program program;
    unsigned int num_devices;
    struct xocl_device **devices;
    struct xocl_context* forward_context;
    struct xocl_context* next_context;
};

struct xocl_device{
    cl_device_type type;
    bool half_support;
    bool double_support;
    char name[100];
    char version[100];
    char vendor[100];
    char profile[100];
    cl_uint max_units;
    cl_uint max_work_dim;
    size_t max_work_group[3];
    size_t max_work_item[3];
    cl_ulong max_freq;
    cl_ulong global_mem_size;
    cl_ulong global_cacheline_size;
    cl_device_id device_id;
    cl_command_queue queue;
};

struct xocl_platform{
    char name[100];
    char vendor[100];
    char version[100];
    char profile[100];
    cl_platform_id platform_id;
    unsigned int num_devices;
    struct xocl_device *devices;
};

struct xocl_host{
    unsigned int num_platforms;
    struct xocl_platform *platforms;
    unsigned int num_contexts;
    struct xocl_context *first_contexts;
    struct xocl_context *last_contexts;
};

struct xocl_simple{
    struct xocl_host host;
    unsigned int platform_index;

    unsigned int device_index;
    
    unsigned int context_index;
    struct xocl_context *context;
    
    char *kernel_file;
    char *kernel_name;
    cl_kernel kernel;
    
    unsigned int calc_dims;
    size_t global_size[3];
    size_t local_size[3];
};

#endif /* libxocl_head_h */
