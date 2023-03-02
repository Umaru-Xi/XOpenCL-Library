//
//  main.c
//  OCLLib
//
//  Created by UmaruAya on 3/2/23.
//

#include "libxocl.h"
#include <time.h>

#define TEST_LOOPS 100000000

int main(int argc, const char * argv[]) {
    
    xocl_verbose(true);
    
    struct xocl_simple setting;
    float *var[3] = {NULL, NULL, NULL};
    cl_mem buf[3];

    setting.platform_index = 0;
    setting.device_index = 0;
    setting.calc_dims = 1;
    setting.global_size[0] = TEST_LOOPS;
    setting.local_size[0] = 0;
    xocl_set_str(&setting.kernel_file, "kernel.cl");
    xocl_set_str(&setting.kernel_name, "float_kernel");
    
    if(xocl_simple_init(&setting) != XOCL_RETURN_SUCCESS) return 1;

    size_t size = sizeof(float) * TEST_LOOPS;
    for(unsigned int i = 0; i < 3; ++i){
        var[i] = (float *)malloc(size);
        if(i != 2){
            for(unsigned int j = 0; j < TEST_LOOPS; ++j) var[i][j] = (float)(j % 32768);
            if(xocl_simple_buff(&setting, var[i], size, &buf[i], false, i) != XOCL_RETURN_SUCCESS) return 1;
        }else{
            if(xocl_simple_buff(&setting, var[i], size, &buf[i], true, i) != XOCL_RETURN_SUCCESS) return 1;
        }
    }

    clock_t start_clk = clock();
    if(xocl_simple_exec(&setting) != XOCL_RETURN_SUCCESS) return 1;
    clock_t end_clk = clock();
    if(xocl_simple_read(&setting, var[2], &buf[2], size) != XOCL_RETURN_SUCCESS) return 1;

    xocl_release_host(&setting.host);
    
    bool pass = true;
    unsigned int ei = 0;
    for(unsigned int i = 0; i < TEST_LOOPS; ++i){
        float x = (float)(i % 32768);
        float py = x * x;
        if(var[2][i] != py){
            pass = false;
            ei = i;
            break;
        }
    }

    double time_spend = (end_clk - start_clk) / CLOCKS_PER_SEC;
    char unit = '\0';
    double floops = 0;
    xocl_num_calc((double)TEST_LOOPS / time_spend, &floops, &unit);
    printf("\n(== _ ==) : Finished in %lfs, about %lf %cLOOP/S.\n", time_spend, floops, unit);
    if(pass) printf("\t#INFO: All data passed.\n");
    else    printf("\t#ERRO: Error occured on %d, which should be %d * %d = %d but %f\n", ei, ei % 32768, ei % 32768, (ei % 32768) * (ei % 32768), var[2][ei]);
    
    return 0;
}
