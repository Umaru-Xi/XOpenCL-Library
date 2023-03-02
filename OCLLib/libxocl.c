//
//  libxocl.c
//
//  Created by UmaruAya on 2/16/23.
//

#include "libxocl.h"

const char build_options[] = " ";
const char unknow_str[] = "UNKNOW";
bool xocl_verbose_flag = true;

void xocl_verbose(bool isVerbose){
    xocl_verbose_flag = isVerbose;
}

void xocl_set_str(char **dst, const char *src){
    *dst = (char *)malloc(sizeof(char) * strlen(src));
    strcpy(*dst, src);
}

cl_int xocl_simple_read(struct xocl_simple *setting, void *var, cl_mem *buff, size_t size){
    if(clEnqueueReadBuffer(setting->host.platforms[setting->platform_index].devices[setting->device_index].queue, *buff, CL_TRUE, 0, size, var, 0, NULL, NULL) != CL_SUCCESS) return XOCL_RETURN_FAILD;
    return XOCL_RETURN_SUCCESS;
}

cl_int xocl_simple_exec(struct xocl_simple *setting){
    if(xocl_enqueue(&setting->host, setting->platform_index, setting->device_index, setting->context_index, setting->kernel, 1, (size_t *)setting->global_size, (size_t *)setting->local_size) == XOCL_RETURN_FAILD) return XOCL_RETURN_FAILD;
    while(clFinish(setting->host.platforms[setting->platform_index].devices[setting->device_index].queue) != CL_SUCCESS);
    return XOCL_RETURN_SUCCESS;
}

cl_int xocl_simple_buff(struct xocl_simple *setting, void *var, size_t size, cl_mem *buff, bool isWrite, unsigned int var_index){
    if(xocl_set_buff(var, size, buff, setting->context, setting->kernel, isWrite) == XOCL_RETURN_FAILD) return XOCL_RETURN_FAILD;
    if(clSetKernelArg(setting->kernel, var_index, sizeof(cl_mem), buff) != CL_SUCCESS) return XOCL_RETURN_FAILD;
    return XOCL_RETURN_SUCCESS;
}

cl_int xocl_simple_init(struct xocl_simple *setting){
    if(xocl_init_host(&setting->host) == XOCL_RETURN_FAILD) return XOCL_RETURN_FAILD;
    if(xocl_init_platforms(&setting->host) == XOCL_RETURN_FAILD) return XOCL_RETURN_FAILD;
    if(xocl_init_devices(&setting->host) == XOCL_RETURN_FAILD) return XOCL_RETURN_FAILD;

    if(xocl_create_context_bydevices(&setting->host, setting->platform_index, &setting->device_index, 1, XOCL_PROGRAM_TYPE_SOURCE, setting->kernel_file, &setting->context_index) == XOCL_RETURN_FAILD) return XOCL_RETURN_FAILD;
    setting->context = xocl_get_context_byindex(&setting->host, setting->context_index);
    if(xocl_create_kernel(&setting->host, setting->context_index, &setting->kernel, setting->kernel_name) == XOCL_RETURN_FAILD) return XOCL_RETURN_FAILD;
    return XOCL_RETURN_SUCCESS;
}

cl_int xocl_set_buff(void *var, size_t size, cl_mem *buf, struct xocl_context *context, cl_kernel kernel, bool is_write){
    cl_int err = 0;
    unsigned int tabs = 0;
    
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Trying to set kernel variable(s)...\n");
    
    tabs = 1;
    if(!is_write){
        *buf = clCreateBuffer(context->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, var, &err);
    }else{
        *buf = clCreateBuffer(context->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, var, &err);
    }
    if(err != CL_SUCCESS){
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("Can not set buffer variable(s), error %d.\n", err);
        return XOCL_RETURN_FAILD;
    }
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Buffer variable(s) set.\n");
    
    return XOCL_RETURN_SUCCESS;
}

cl_int xocl_enqueue(struct xocl_host *host, unsigned int platform_index, unsigned int device_index, unsigned int context_index, cl_kernel kernel, unsigned int work_dim, size_t *global_group_size, size_t *max_work_item_size){
    unsigned int tabs = 0;
    
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Trying to enqueue kernel from context %d to device %d of platform %d...\n", context_index, device_index, platform_index);
    
    tabs = 1;
    struct xocl_context *context = xocl_get_context_byindex(host, context_index);
    bool checked = false;
    for(unsigned int i = 0; i < context->num_devices; ++i){
        if(context->devices[i]->device_id == host->platforms[platform_index].devices[device_index].device_id){
            checked = true;
            break;
        }
    }
    if(checked == false){
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("This device does not in the context.\n");
        return XOCL_RETURN_FAILD;
    }
    
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Check max size...\n");
    
    tabs = 2;
    size_t global_work_size[3] = {0, 0, 0};
    size_t local_work_size[3] = {0, 0, 0};

    for(unsigned int i = 0; i < work_dim; ++i){
        global_work_size[i] = global_group_size[i];
        local_work_size[i] = max_work_item_size[i];
        size_t max_item = host->platforms[platform_index].devices[device_index].max_work_item[i];
        if(host->platforms[platform_index].devices[device_index].type == CL_DEVICE_TYPE_CPU){
            unsigned int compute_units = host->platforms[platform_index].devices[device_index].max_units;
            max_item /= compute_units;
        }
        if(local_work_size[i] > max_item || local_work_size[i] == 0) local_work_size[i] = max_item;
        while(global_work_size[i] % local_work_size[i] != 0){
            --local_work_size[i];
        }
    }
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Using global work group size: (");
    for(unsigned int i = 0; i < work_dim; ++i){
        if(xocl_verbose_flag)printf("%ld", global_work_size[i]);
        if(i != work_dim - 1) if(xocl_verbose_flag)printf(", ");
    }
    if(xocl_verbose_flag)printf(").\n");
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Using local work item size: (");
    for(unsigned int i = 0; i < work_dim; ++i){
        if(xocl_verbose_flag)printf("%ld", local_work_size[i]);
        if(i != work_dim - 1) if(xocl_verbose_flag)printf(", ");
    }
    if(xocl_verbose_flag)printf(").\n");
    
    tabs = 1;
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Enqueue...\n");
    
    cl_int return_code = clEnqueueNDRangeKernel(host->platforms[platform_index].devices[device_index].queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    if(return_code != CL_SUCCESS){
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("Can not enqueue, error %d.\n", return_code);
        return XOCL_RETURN_FAILD;
    }
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Enqueued.\n");
    
    return XOCL_RETURN_SUCCESS;
}

cl_int xocl_create_kernel(struct xocl_host *host, unsigned int context_index, cl_kernel *kernel, char *funcname){
    unsigned int tabs = 0;
    cl_int return_code = CL_SUCCESS;
    
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Trying to create kernel from the %d context.\n", context_index);
    
    tabs = 1;
    struct xocl_context *context = xocl_get_context_byindex(host, context_index);
    
    *kernel = clCreateKernel(context->program.program, funcname, &return_code);
    if(return_code != CL_SUCCESS){
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("Can not create kernel of function %s, error %d.\n", funcname, return_code);
        return XOCL_RETURN_FAILD;
    }
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Create kernel of function %s.\n", funcname);
    
    return XOCL_RETURN_SUCCESS;
}

struct xocl_context *xocl_get_context_byindex(struct xocl_host *host, unsigned int context_index){
    struct xocl_context *return_ptr = host->last_contexts;
    for(unsigned int i = 0; i < context_index; ++i){
        return_ptr = return_ptr->next_context;
    }
    return return_ptr;
}

cl_int xocl_create_context_bytype(struct xocl_host *host, unsigned int platform_index, cl_device_type type, enum xocl_program_type filetype, const char *filename, unsigned int *context_index){
    unsigned int tabs = 0;
    char type_str[25];
    
    xocl_type_string(type, type_str);
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Trying to collect device(s) with %s type on platform %d...\n", type_str, platform_index);
    
    struct __device{
        struct __device *forward;
        unsigned int device_indexs;
        struct __device *next;
    };
    struct __device *devices_chain_end = NULL;
    struct __device *tmp_dev_ptr = NULL;
    
    tabs = 1;
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Checking device(s) type...\n");
    
    tabs = 2;
    unsigned int num_devices = 0;
    for(unsigned int i = 0; i < host->platforms[platform_index].num_devices; ++i){
        if(host->platforms[platform_index].devices[i].type == type){
            devices_chain_end = (struct __device*)malloc(sizeof(struct __device));
            devices_chain_end->forward = NULL;
            devices_chain_end->next = tmp_dev_ptr;
            
            devices_chain_end->device_indexs = i;
            if(devices_chain_end->next != NULL)
                devices_chain_end->next->forward = devices_chain_end;
            tmp_dev_ptr = devices_chain_end;
            
            ++num_devices;
            xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
            if(xocl_verbose_flag)printf("Found %d devices.\n", num_devices);
        }
    }
    
    tabs = 1;
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Process device(s), and free temporary memory...\n");
    tmp_dev_ptr = devices_chain_end;
    unsigned int *device_indexs = (unsigned int *)malloc(sizeof(unsigned int) * num_devices);
    tabs = 2;
    for(unsigned int i = 0; i < num_devices; ++i){
        device_indexs[i] = tmp_dev_ptr->device_indexs;
        tmp_dev_ptr = devices_chain_end->forward;
        free(devices_chain_end);
        devices_chain_end = tmp_dev_ptr;
    }
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Collected %d devices.\n", num_devices);
    
    unsigned int return_code = xocl_create_context_bydevices(host, platform_index, device_indexs, num_devices, filetype, filename, context_index);
    
    free(device_indexs);
    
    return return_code;
}

cl_int xocl_create_context_bydevices(struct xocl_host *host, unsigned int platform_index, unsigned int *device_indexs, unsigned int num_devices, enum xocl_program_type filetype, const char *filename, unsigned int *context_index){
    unsigned int warn = 0;
    unsigned int tabs = 0;
    cl_device_id *devices = NULL;
    
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Trying to create context on platform %d...\n", platform_index);
    
    tabs = 1;
    if(host->num_platforms < 0){
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("There is no platform on host.\n");
        return XOCL_RETURN_FAILD;
    }
    
    
    if(host->platforms[platform_index].num_devices == 0){
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("There is no device on platform.\n");
        return XOCL_RETURN_FAILD;
    }
    
    devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
    for(unsigned int i = 0; i < num_devices; ++i){
        devices[i] = host->platforms[platform_index].devices[device_indexs[i]].device_id;
    }
    
    cl_context_properties properites[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)host->platforms[platform_index].platform_id, (cl_context_properties)0};

    struct xocl_context *tmp_ptr = host->last_contexts;
    host->last_contexts = (struct xocl_context *)malloc(sizeof(struct xocl_context));
    
    cl_int return_code = CL_SUCCESS;
    host->last_contexts->context = clCreateContext(properites, num_devices, devices, NULL, NULL, &return_code);
    
    if(return_code != CL_SUCCESS){
        host->last_contexts = tmp_ptr;
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("Can not create context by %d device(s).\n", num_devices);
        free(devices);
        return XOCL_RETURN_FAILD;
    }
    
    host->last_contexts->forward_context = tmp_ptr;
    host->last_contexts->next_context = NULL;
    host->last_contexts->num_devices = num_devices;
    
    cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
    
    host->last_contexts->devices = (struct xocl_device **)malloc(sizeof(struct xocl_device *) * num_devices);
    for(unsigned int i = 0; i < num_devices; ++i){
        host->last_contexts->devices[i] = &host->platforms[platform_index].devices[device_indexs[i]];
        host->last_contexts->devices[i]->queue = clCreateCommandQueue(host->last_contexts->context, host->last_contexts->devices[i]->device_id, props, &return_code);
        if(return_code != CL_SUCCESS){
            xocl_printhead(XOCL_RETURN_WARNING, tabs);
            if(xocl_verbose_flag)printf("Can not create queue on device %d, error %d.\n", i, return_code);
            free(devices);
            return XOCL_RETURN_FAILD;
        }
    }
    
    ++host->num_contexts;
    *context_index = host->num_contexts - 1;
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Context created by %d devices.\n", num_devices);
    
    tabs = 0;
    cl_int binary_status = 0;
    FILE *program_handle = NULL;
    
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Trying to create program on last context...\n");
    
    tabs = 1;
    program_handle = fopen(filename, "r");
    if(program_handle == NULL){
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("Can not read file %s.\n", filename);
        free(devices);
        return XOCL_RETURN_FAILD;
    }
    
    fseek(program_handle, 0, SEEK_END);
    unsigned long program_size = ftell(program_handle);
    rewind(program_handle);
    char *program_buffer = (char *)malloc(program_size + 1);
    fread(program_buffer, sizeof(char), program_size, program_handle);
    program_buffer[program_size] = '\0';
    fclose(program_handle);
    
    if(filetype == XOCL_PROGRAM_TYPE_BINARY){
        host->last_contexts->program.program = clCreateProgramWithBinary(host->last_contexts->context, 1, devices, &program_size, (const unsigned char **)program_buffer, &binary_status, &return_code);
    }else{
        host->last_contexts->program.program = clCreateProgramWithSource(host->last_contexts->context, 1, (const char **)&program_buffer, &program_size, &return_code);
    }
    host->last_contexts->program.type = filetype;
    strcpy(host->last_contexts->program.file, filename);
    
    if(return_code != CL_SUCCESS){
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("Can not create program.\n");
        free(devices);
        return XOCL_RETURN_FAILD;
    }
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Program from %s by %s created.\n", filename, (filetype == XOCL_PROGRAM_TYPE_BINARY)? "binary" : "source");
    
    tabs = 1;
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Trying to build program...\n");
    
    tabs = 2;
    
    return_code = clBuildProgram(host->last_contexts->program.program, host->last_contexts->num_devices, devices, build_options, NULL, NULL);
    
    if(return_code != CL_SUCCESS){
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("Can not build program for %d device(s), error code: %d.\n", host->last_contexts->num_devices, return_code);
        char *log;
        size_t log_size = 0;
        tabs = 3;
        if(clGetProgramBuildInfo(host->last_contexts->program.program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) != CL_SUCCESS){
            if(xocl_verbose_flag)printf("Can not get build log.\n");
            free(devices);
            return XOCL_RETURN_FAILD;
        }
        log = (char *)malloc(log_size);
        clGetProgramBuildInfo(host->last_contexts->program.program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
        if(xocl_verbose_flag)printf("The following is the build log:\n\t\t\t%s\n", log);
        free(log);
        free(devices);
        return XOCL_RETURN_FAILD;
    }
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Build program for %d device(s).\n", host->last_contexts->num_devices);
    free(devices);
    
    return (warn == 0)? XOCL_RETURN_SUCCESS : XOCL_RETURN_WARNING;
}

void xocl_release_host(struct xocl_host *host){
    unsigned int tabs =0;
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Trying to release all memory allocated on host.\n");
    
    tabs = 1;
    if(host->num_platforms == 0){
        xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
        if(xocl_verbose_flag)printf("There is not platform on host.\n");
    }else{
        tabs = 1;
        xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
        if(xocl_verbose_flag)printf("Release all platform(s)...\n");
        
        for(unsigned i = 0; i < host->num_platforms; ++i){
            tabs = 2;
            free(host->platforms[i].devices);
            xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
            if(xocl_verbose_flag)printf("Device(s) on platform %d released.\n", i);
        }
        free(host->platforms);
    }
    tabs = 2;
    host->num_platforms = 0;
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("All platform(s) released.\n");
    
    tabs = 1;
    if(host->num_contexts == 0){
        xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
        if(xocl_verbose_flag)printf("There is not context on host.\n");
    }else{
        tabs = 1;
        xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
        if(xocl_verbose_flag)printf("Release context(s) chain...\n");
        
        struct xocl_context *now_ptr = host->last_contexts;
        unsigned int i = 0;
        while(now_ptr != NULL){
            tabs = 2;
            xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
            if(xocl_verbose_flag)printf("Working on context(s) chain index %d ...\n", i);
            
            tabs = 3;
            free(now_ptr->devices);
            now_ptr->num_devices = 0;
            xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
            if(xocl_verbose_flag)printf("All device(s) released.\n");
            
            ++i;
            free(now_ptr);
            --host->num_contexts;
            if(now_ptr->forward_context == NULL || host->num_contexts == 0){
                host->first_contexts = NULL;
                host->last_contexts = NULL;
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Context(s) chain clear.\n");
                break;
            }else{
                host->last_contexts = now_ptr->forward_context;
                now_ptr = host->last_contexts;
            }
        }
        
    }
    tabs = 2;
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("All context(s) released.\n");
    
    tabs = 1;
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("All memory released.\n");
}

cl_int xocl_init_devices(struct xocl_host *host){
    unsigned int tabs = 0;
    unsigned int warn = 0;
    size_t size = 0;
    char unit = '\0';
    double human_num = 0;
    cl_device_id *device_ids = NULL;
    
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Trying to get devices...\n");
    
    for(unsigned int i = 0; i < host->num_platforms; ++i){
        tabs = 1;
        xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
        if(xocl_verbose_flag)printf("Get device(s) on platform %d...\n", i);
        
        if(host->platforms[i].num_devices == 0){
            xocl_printhead(XOCL_RETURN_WARNING, tabs);
            if(xocl_verbose_flag)printf("There is no device on platform %d, skip.\n", i);
            continue;
        }
        
        device_ids = (cl_device_id *)malloc(sizeof(cl_device_id) * host->platforms[i].num_devices);
        if(clGetDeviceIDs(host->platforms[i].platform_id, CL_DEVICE_TYPE_ALL, host->platforms[i].num_devices, device_ids, NULL) != CL_SUCCESS){
            free(device_ids);
            host->platforms[i].num_devices = 0;
            free(host->platforms[i].devices);
            xocl_printhead(XOCL_RETURN_WARNING, tabs);
            if(xocl_verbose_flag)printf("Can not get ID of device(s) on platform %d, the number of device(s) will be set to 0.\n", i);
            ++warn;
            continue;
        }
        
        host->platforms[i].devices = (struct xocl_device *)malloc(sizeof(struct xocl_device) * host->platforms[i].num_devices);
        
        for(unsigned int j = 0; j < host->platforms[i].num_devices; ++j){
            tabs = 2;
            
            host->platforms[i].devices[j].device_id = device_ids[j];
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_NAME, 0, NULL, &size) != CL_SUCCESS){
                strcpy(host->platforms[i].devices[j].name, unknow_str);
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get name of device %d on platform %d, it will be UNKNOW.\n", j, i);
                ++warn;
            }else{
                clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_NAME, size, host->platforms[i].devices[j].name, NULL);
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Device %d name: %s\n", j, host->platforms[i].devices[j].name);
            }
            
            tabs = 3;
            char type_str[32];
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &host->platforms[i].devices[j].type, NULL) != CL_SUCCESS){
                host->platforms[i].devices[j].type = CL_DEVICE_TYPE_CUSTOM;
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get type of device %d on platform %d, it will be CUSTOM.\n", j, i);
                ++warn;
            }else{
                xocl_type_string(host->platforms[i].devices[j].type, type_str);
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Type: %s\n", type_str);
            }
            
            cl_uint half_support = 0;
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, sizeof(cl_uint), &half_support, NULL) != CL_SUCCESS){
                host->platforms[i].devices[j].half_support = false;
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get half support flag of device %d on platform %d, it will be false.\n", j, i);
                ++warn;
            }else{
                host->platforms[i].devices[j].half_support = (half_support == 0)? false : true;
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Half support: %s\n", host->platforms[i].devices[j].half_support? "TRUE" : "FALSE");
            }
            
            cl_uint double_support = 0;
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &double_support, NULL) != CL_SUCCESS){
                host->platforms[i].devices[j].double_support = false;
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get double support flag of device %d on platform %d, it will be false.\n", j, i);
                ++warn;
            }else{
                host->platforms[i].devices[j].double_support = (double_support == 0)? false : true;
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Double support: %s\n", host->platforms[i].devices[j].double_support? "TRUE" : "FALSE");
            }
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_VERSION, 0, NULL, &size) != CL_SUCCESS){
                strcpy(host->platforms[i].devices[j].version, unknow_str);
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get version of device %d on platform %d, it will be UNKNOW.\n", j, i);
                ++warn;
            }else{
                clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_VERSION, size, host->platforms[i].devices[j].version, NULL);
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Version: %s\n", host->platforms[i].devices[j].version);
            }
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_VENDOR, 0, NULL, &size) != CL_SUCCESS){
                strcpy(host->platforms[i].devices[j].vendor, unknow_str);
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get vendor of device %d on platform %d, it will be UNKNOW.\n", j, i);
                ++warn;
            }else{
                clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_VENDOR, size, host->platforms[i].devices[j].vendor, NULL);
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Vendor: %s\n", host->platforms[i].devices[j].vendor);
            }
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_PROFILE, 0, NULL, &size) != CL_SUCCESS){
                strcpy(host->platforms[i].devices[j].profile, unknow_str);
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get profile of device %d on platform %d, it will be UNKNOW.\n", j, i);
                ++warn;
            }else{
                clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_PROFILE, size, host->platforms[i].devices[j].profile, NULL);
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Profile: %s\n", host->platforms[i].devices[j].profile);
            }
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &host->platforms[i].devices[j].max_units, NULL) != CL_SUCCESS){
                host->platforms[i].devices[j].max_units = 0;
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get max compute units of device %d on platform %d, it will be 0.\n", j, i);
                ++warn;
            }else{
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Max compute units: %d\n", host->platforms[i].devices[j].max_units);
            }
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &host->platforms[i].devices[j].max_freq, NULL) != CL_SUCCESS){
                host->platforms[i].devices[j].max_freq = 0;
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get max frequency of device %d on platform %d, it will be 0.\n", j, i);
                ++warn;
            }else{
                xocl_freq_calc(host->platforms[i].devices[j].max_freq * 1000 * 1000, &human_num, &unit);
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Max frequency: %0.2f %cHz\n", human_num, unit);
            }
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &host->platforms[i].devices[j].global_mem_size, NULL) != CL_SUCCESS){
                host->platforms[i].devices[j].global_mem_size = 0;
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get global memory size of device %d on platform %d, it will be 0.\n", j, i);
                ++warn;
            }else{
                xocl_memsize_calc(host->platforms[i].devices[j].global_mem_size, &human_num, &unit);
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Global memory size: %0.3f %cB\n", human_num, unit);
            }
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_ulong), &host->platforms[i].devices[j].global_cacheline_size, NULL) != CL_SUCCESS){
                host->platforms[i].devices[j].global_cacheline_size = 0;
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get global cache line size of device %d on platform %d, it will be 0.\n", j, i);
                ++warn;
            }else{
                xocl_memsize_calc(host->platforms[i].devices[j].global_cacheline_size, &human_num, &unit);
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Global cache line size: %0.3f %c\n", human_num, unit);
            }
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &host->platforms[i].devices[j].max_work_dim, NULL) != CL_SUCCESS){
                host->platforms[i].devices[j].max_work_dim = 0;
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get max work item dimension(s) of device %d on platform %d, it will be 0.\n", j, i);
                ++warn;
            }else{
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Max work item dimension(s): %d\n", host->platforms[i].devices[j].max_work_dim);
            }
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t) * host->platforms[i].devices[j].max_work_dim, &host->platforms[i].devices[j].max_work_group, NULL) != CL_SUCCESS){
                host->platforms[i].devices[j].max_work_group[0] = 0;
                host->platforms[i].devices[j].max_work_group[1] = 0;
                host->platforms[i].devices[j].max_work_group[2] = 0;
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get max work group size of device %d on platform %d, it will be (0, 0, 0).\n", j, i);
                ++warn;
            }else{
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Max work group size: (%ld, %ld, %ld)\n", host->platforms[i].devices[j].max_work_group[0], host->platforms[i].devices[j].max_work_group[1], host->platforms[i].devices[j].max_work_group[2]);
            }
            
            if(clGetDeviceInfo(host->platforms[i].devices[j].device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * host->platforms[i].devices[j].max_work_dim, &host->platforms[i].devices[j].max_work_item, NULL) != CL_SUCCESS){
                host->platforms[i].devices[j].max_work_item[0] = 0;
                host->platforms[i].devices[j].max_work_item[1] = 0;
                host->platforms[i].devices[j].max_work_item[2] = 0;
                xocl_printhead(XOCL_RETURN_WARNING, tabs);
                if(xocl_verbose_flag)printf("Can not get max work items size of device %d on platform %d, it will be (0, 0, 0).\n", j, i);
                ++warn;
            }else{
                xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
                if(xocl_verbose_flag)printf("Max work items size: (%ld, %ld, %ld)\n", host->platforms[i].devices[j].max_work_item[0], host->platforms[i].devices[j].max_work_item[1], host->platforms[i].devices[j].max_work_item[2]);
            }
               
        }
        free(device_ids);
    }
    
    return (warn == 0)? XOCL_RETURN_SUCCESS : XOCL_RETURN_WARNING;
}

void xocl_type_string(cl_device_type type, char *str){
    switch(type){
        case CL_DEVICE_TYPE_CPU: strcpy(str, "CPU"); break;
        case CL_DEVICE_TYPE_GPU: strcpy(str, "GPU"); break;
        case CL_DEVICE_TYPE_ACCELERATOR: strcpy(str, "ACCELERATOR"); break;
        case CL_DEVICE_TYPE_CUSTOM: strcpy(str, "CUSTOM"); break;
        case CL_DEVICE_TYPE_DEFAULT: strcpy(str, "DEFAULT"); break;
        case CL_DEVICE_TYPE_ALL: strcpy(str, "ALL"); break;
        default: strcpy(str, "UNKNOW"); break;
    }
}

void xocl_memsize_calc(unsigned long mem_size, double *mem_size_f, char *unit){
    *unit = '\0';
    *mem_size_f = (double)mem_size;
    if(mem_size < 1024){
        *unit = '\0';
    }
    else if(mem_size < (1024 * 1024)){
        *mem_size_f /= 1024.0;
        *unit = 'k';
    }
    else if(mem_size < (1024 * 1024 * 1024)){
        *mem_size_f /= (1024.0 * 1024.0);
        *unit = 'M';
    }
    else if(mem_size < ((unsigned long)1024 * 1024 * 1024 * 1024)){
        *mem_size_f /=(1024.0 * 1024.0 * 1024.0);
        *unit = 'G';
    }
    else{
        *mem_size_f /=(1024.0 * 1024.0 * 1024.0 * 1024.0);
        *unit = 'T';
    }
}

void xocl_freq_calc(unsigned long freq, double *freq_f, char *unit){
    *unit = '\0';
    *freq_f = (double)freq;
    if(freq < 1000){
        *unit = '\0';
    }
    else if(freq < (1000 * 1000)){
        *freq_f /= 1000.0;
        *unit = 'k';
    }
    else if(freq < (1000 * 1000 * 1000)){
        *freq_f /= (1000.0 * 1000.0);
        *unit = 'M';
    }
    else if(freq < ((unsigned long)1000 * 1000 * 1000 * 1000)){
        *freq_f /=(1000.0 * 1000.0 * 1000.0);
        *unit = 'G';
    }
    else{
        *freq_f /=(1000.0 * 1000.0 * 1000.0 * 1000.0);
        *unit = 'T';
    }
}

void xocl_num_calc(double num, double *num_f, char *unit){
    *unit = '\0';
    bool neg_flag = (num < 0.0)? true : false;
    *num_f = (num < 0.0)? -num : num;
    if(num < 1000.0){
        *unit = '\0';
    }
    else if(num < (1000.0 * 1000.0)){
        *num_f /= 1000.0;
        *unit = 'k';
    }
    else if(num < (1000.0 * 1000.0 * 1000.0)){
        *num_f /= (1000.0 * 1000.0);
        *unit = 'M';
    }
    else if(num < (1000.0 * 1000.0 * 1000.0 * 1000.0)){
        *num_f /= (1000.0 * 1000.0 * 1000.0);
        *unit = 'G';
    }
    else{
        *num_f /=(1000.0 * 1000.0 * 1000.0 * 1000.0);
        *unit = 'T';
    }
    *num_f = (neg_flag)? -(*num_f) : *num_f;
}

cl_int xocl_init_platforms(struct xocl_host *host){
    unsigned int tabs = 0;
    unsigned int warn = 0;
    size_t size = 0;
    cl_platform_id *platform_ids = NULL;
    
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Trying to get platforms...\n");
    
    if(host->num_platforms == 0){
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("There is no platform on this host.\n");
        return XOCL_RETURN_FAILD;
    }
    
    platform_ids = (cl_platform_id *)malloc(sizeof(cl_platform_id) * host->num_platforms);
    if(clGetPlatformIDs(host->num_platforms, platform_ids, NULL) != CL_SUCCESS){
        xocl_printhead(XOCL_RETURN_WARNING, tabs);
        if(xocl_verbose_flag)printf("Can not get ID of platform(s).\n");
        free(platform_ids);
        return XOCL_RETURN_FAILD;
    }
    
    host->platforms = (struct xocl_platform *)malloc(sizeof(struct xocl_platform) * host->num_platforms);
    for(unsigned int i = 0; i < host->num_platforms; ++i){
        tabs = 1;
        host->platforms[i].platform_id = platform_ids[i];
        
        if(clGetPlatformInfo(host->platforms[i].platform_id, CL_PLATFORM_NAME, 0, NULL, &size) != CL_SUCCESS){
            strcpy(host->platforms[i].name, unknow_str);
            xocl_printhead(XOCL_RETURN_WARNING, tabs);
            if(xocl_verbose_flag)printf("Can not get name of platform %d, it will be UNKNOW.\n", i);
            ++warn;
        }else{
            clGetPlatformInfo(host->platforms[i].platform_id, CL_PLATFORM_NAME, size, host->platforms[i].name, NULL);
            xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
            if(xocl_verbose_flag)printf("Name: %s\n", host->platforms[i].name);
        }
        
        tabs = 2;
        if(clGetPlatformInfo(host->platforms[i].platform_id, CL_PLATFORM_VERSION, 0, NULL, &size) != CL_SUCCESS){
            strcpy(host->platforms[i].version, unknow_str);
            xocl_printhead(XOCL_RETURN_WARNING, tabs);
            if(xocl_verbose_flag)printf("Can not get version of platform %d, it will be UNKNOW.\n", i);
            ++warn;
        }else{
            clGetPlatformInfo(host->platforms[i].platform_id, CL_PLATFORM_VERSION, size, host->platforms[i].version, NULL);
            xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
            if(xocl_verbose_flag)printf("Version: %s\n", host->platforms[i].version);
        }
        
        if(clGetPlatformInfo(host->platforms[i].platform_id, CL_PLATFORM_VENDOR, 0, NULL, &size) != CL_SUCCESS){
            strcpy(host->platforms[i].vendor, unknow_str);
            xocl_printhead(XOCL_RETURN_WARNING, tabs);
            if(xocl_verbose_flag)printf("Can not get vendor of platform %d, it will be UNKNOW.\n", i);
            ++warn;
        }else{
            clGetPlatformInfo(host->platforms[i].platform_id, CL_PLATFORM_VENDOR, size, host->platforms[i].vendor, NULL);
            xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
            if(xocl_verbose_flag)printf("Vendor: %s\n", host->platforms[i].vendor);
        }
        
        if(clGetPlatformInfo(host->platforms[i].platform_id, CL_PLATFORM_PROFILE, 0, NULL, &size) != CL_SUCCESS){
            strcpy(host->platforms[i].profile, unknow_str);
            xocl_printhead(XOCL_RETURN_WARNING, tabs);
            if(xocl_verbose_flag)printf("Can not get profile of platform %d, it will be UNKNOW.\n", i);
            ++warn;
        }else{
            clGetPlatformInfo(host->platforms[i].platform_id, CL_PLATFORM_PROFILE, size, host->platforms[i].profile, NULL);
            xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
            if(xocl_verbose_flag)printf("Profile: %s\n", host->platforms[i].profile);
        }
        
        if(clGetDeviceIDs(host->platforms[i].platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &host->platforms[i].num_devices) != CL_SUCCESS){
            host->platforms[i].num_devices = 0;
            xocl_printhead(XOCL_RETURN_WARNING, tabs);
            if(xocl_verbose_flag)printf("Can not get device(s) number of platform %d, it will be 0.\n", i);
            ++warn;
        }else{
            xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
            if(xocl_verbose_flag)printf("Device(s): %d\n", host->platforms[i].num_devices);
        }
    }
    
    free(platform_ids);
    return (warn == 0)? XOCL_RETURN_SUCCESS : XOCL_RETURN_WARNING;
}

cl_int xocl_init_host(struct xocl_host *host){
    unsigned int tabs = 0;
    host->num_platforms = 0;
    host->num_contexts = 0;
    host->first_contexts = NULL;
    host->last_contexts = NULL;
    if(clGetPlatformIDs(0, NULL, &host->num_platforms) != CL_SUCCESS){
        host->num_platforms = 0;
        xocl_printhead(XOCL_RETURN_FAILD, tabs);
        if(xocl_verbose_flag)printf("Can not find any OpenCL platform, it will be 0.\n");
        return XOCL_RETURN_FAILD;
    }
    xocl_printhead(XOCL_RETURN_SUCCESS, tabs);
    if(xocl_verbose_flag)printf("Found %d platform(s).\n", host->num_platforms);
    return XOCL_RETURN_SUCCESS;
}

void xocl_printhead(enum xocl_return_flag flag, unsigned int tabs_name){
    for(unsigned int i = 0; i < tabs_name; ++i) if(xocl_verbose_flag)printf("\t");
    if(xocl_verbose_flag)printf("#%s: ", (flag == XOCL_RETURN_SUCCESS)? "INFO" : (flag == XOCL_RETURN_FAILD)? "ERRO" : "WARN");
}
