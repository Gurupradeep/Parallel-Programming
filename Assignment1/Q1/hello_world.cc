/*
 * Basic Hello world program
 * Run the program as follows
 * (Compilation) g++ -fopenmp hello_world.cc 
 * (Execution) ./a.out THREAD_COUNT
 * Arguments
 * 1) THREAD_COUNT. (Optional Parameter) Number of threads to be created. By default uses number of cores avaliable. Note that the number of threads spawned might be different than THREAD_COUNT depending upon the avaliability.
 */
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <stdlib.h>
#include "string.h"
namespace {

/*
 * printf by each thread might cause an I/O overhead. 
 * To avoid that, we can concatenate 'hello world' to a shared variable.
 * Print it at the end. The shared variable needs to be protected using 'critical' keywords. Note: Atomic won't work, as std::string is not a native type.
 */
double hello_world_concatenation_critical(int no_of_threads) {
    if (no_of_threads > 0) {
        omp_set_num_threads(no_of_threads);
    }
    std::string output_str = "";
    double begin = omp_get_wtime();
    #pragma omp parallel 
    {
        no_of_threads = omp_get_num_threads();
        #pragma omp critical
        output_str += "Hello World.\n";
    }
    printf("%s", output_str.c_str());
    double end = omp_get_wtime();
    printf("Number of threads created are: %d.\n", no_of_threads);
    return end - begin;
}

double hello_world_default(int no_of_threads) {
	if(no_of_threads > 0){ 
		omp_set_num_threads(no_of_threads);
	}
	// Starting timer
	double begin = omp_get_wtime();

	#pragma omp parallel 
	{
		printf("Hello World!!\n");
		no_of_threads = omp_get_num_threads();
	}

	// Stopping timer
	double end = omp_get_wtime();

	printf("No of threads created are %d\n",no_of_threads);
	return end - begin;
}

}
int main( int argc, char* argv[]) {

	double time_taken_for_default;
	// No arguments provided, will use default no of threads
	if(argc == 1)
		time_taken_for_default = hello_world_default(0);

	// Can Explicitly set number of threads using command line arguments
	else
		time_taken_for_default = hello_world_default(atoi(argv[1]));
	printf("Time taken is %lf seconds for default hello world program\n",time_taken_for_default);
	double time_taken_for_critical;
    if (argc == 1) {
        time_taken_for_critical = hello_world_concatenation_critical(0);
    } else {
        time_taken_for_critical = hello_world_concatenation_critical(atoi(argv[1]));
    }
    printf("Time taken is %lf seconds for critical.\n", time_taken_for_critical);
    return 0;
}
