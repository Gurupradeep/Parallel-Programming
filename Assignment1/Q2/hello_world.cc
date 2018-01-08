/*
 * Hello World Program - Version 2.
 * Run the program as follows:
 * (Compilation) g++ hello_world.cc -fopenmp
 * (Execution) ./a.out -t THREAD_COUNT 
 * Arguments:
 * 1) -t THREAD_COUNT. (Optional) Number of threads to spawn. By default uses the number of cores available. Note: Number of threads spawned might be different than THREAD_COUNT, uses omp_set_num_threads(). 
 */

#include "omp.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#define ASSERT(condition) { if(!(condition)) { std::cerr << "ASSERT FAILED: " << #condition << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; exit(1); } }

namespace {

void print_hello(int thread_id) {
    printf("Hello World. Thread id: %d\n", thread_id);
    return;
}

double get_parallel_execution_time(int num_of_threads) {
    if (num_of_threads > 0) {
        omp_set_num_threads(num_of_threads);
    }
    double omp_start_time = omp_get_wtime();
    #pragma omp parallel
    {
        num_of_threads = omp_get_num_threads();
        print_hello(omp_get_thread_num());
    }
    printf("Executed using %d threads.\n", num_of_threads);
    double omp_end_time = omp_get_wtime();
    return omp_end_time - omp_start_time;
}

double get_serial_execution_time(int num_of_threads) {
    if (num_of_threads > 0) {
        omp_set_num_threads(num_of_threads);
    }
    double omp_start_time = omp_get_wtime();
    #pragma omp parallel
    {
        num_of_threads = omp_get_num_threads();
    }
    for (int i = 0; i < num_of_threads; i++) {
        printf("Hello World.\n");
    }
    printf("Executed serially.\n");
    double omp_end_time = omp_get_wtime();
    return omp_end_time - omp_start_time;
}

}

int main(int argc, char *argv[]) {
    int num_of_threads = -1;
    // num_of_threads passed as command line argument.
    ASSERT(argc <= 3);
    if (argc == 3) {
        ASSERT(std::string(argv[1]) == "-t");
        ASSERT(atoi(argv[2]) > 0);
        num_of_threads = atoi(argv[2]);
    }
    printf("\n** Serial Execution **\n");
    double execution_time_serial = get_serial_execution_time(num_of_threads);
    printf("Execution time: %lf seconds.\n", execution_time_serial);
    printf("\n** Parallel Execution **\n");
    double execution_time_parallel = get_parallel_execution_time(num_of_threads);
    printf("Execution time: %lf seconds.\n", execution_time_parallel);
    return 0;
}
