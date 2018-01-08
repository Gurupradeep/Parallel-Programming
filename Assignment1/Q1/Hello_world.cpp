/*
 * Basic Hello world program
 * Run the program as follows
 * (Compilation) g++ -fopenmp Hello_world.cpp 
 * (Execution) ./a.out THREAD_COUNT
 * Arguments
 * 1) THREAD_COUNT. (Optional Parameter) Number of threads to be created. By default uses 
 	number of cores avaliable. Note that the number of threads spawned might be different
 	than THREAD_COUNT depending upon the avaliability.
 */
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <stdlib.h>

namespace {

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

	printf("Time taken is %lf milliseconds for default hello world program\n",time_taken_for_default*1000);
	return 0;
}