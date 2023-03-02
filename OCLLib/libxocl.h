//
//  libxocl.h
//
//  Created by UmaruAya on 2/16/23.
//

#ifndef libxocl_h
#define libxocl_h

#include "libxocl_head.h"

void xocl_verbose(bool isVerbose);
void xocl_set_str(char **dst, const char *src);
cl_int xocl_simple_read(struct xocl_simple *setting, void *var, cl_mem *buff, size_t size);
cl_int xocl_simple_exec(struct xocl_simple *setting);
cl_int xocl_simple_buff(struct xocl_simple *setting, void *var, size_t size, cl_mem *buff, bool isWrite, unsigned int var_index);
cl_int xocl_simple_init(struct xocl_simple *setting);

cl_int xocl_set_buff(void *var, size_t size_vars, cl_mem *buf, struct xocl_context *context, cl_kernel kernel, bool is_write);
cl_int xocl_enqueue(struct xocl_host *host, unsigned int platform_index, unsigned int device_index, unsigned int context_index, cl_kernel kernel, unsigned int work_dim, size_t *max_global_group_size, size_t *max_work_item_size);
cl_int xocl_create_kernel(struct xocl_host *host, unsigned int context_index, cl_kernel *kernel, char *funcname);
struct xocl_context *xocl_get_context_byindex(struct xocl_host *host, unsigned int context_index);
cl_int xocl_create_context_bytype(struct xocl_host *host, unsigned int platform_index, cl_device_type type, enum xocl_program_type filetype, const char *filename, unsigned int *context_index);
cl_int xocl_create_context_bydevices(struct xocl_host *host, unsigned int platform_index, unsigned int *device_indexs, unsigned int num_devices, enum xocl_program_type filetype, const char *filename, unsigned int *context_index);
void xocl_release_host(struct xocl_host *host);
cl_int xocl_init_devices(struct xocl_host *host);
void xocl_type_string(cl_device_type type, char *str);
void xocl_memsize_calc(unsigned long mem_size, double *mem_size_f, char *unit);
void xocl_freq_calc(unsigned long freq, double *freq_f, char *unit);
void xocl_num_calc(double num, double *num_f, char *unit);
cl_int xocl_init_platforms(struct xocl_host *host);
cl_int xocl_init_host(struct xocl_host *host);
void xocl_printhead(enum xocl_return_flag flag, unsigned int tabs_name);

#endif /* libxocl_h */
