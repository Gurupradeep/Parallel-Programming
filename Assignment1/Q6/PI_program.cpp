/*
 * PI Calculation using Worksharing and Reduction
 * Run the program as follows
 * (Compilation) g++ -fopenmp PI_program.cpp 
 * (Execution) ./a.out THREAD_COUNT NO_OF_STEPS
 * Arguments
 * 1) THREAD_COUNT. (Optional Parameter) Number of threads to be created. By default uses 
 	number of cores avaliable. Note that the number of threads spawned might be different
 	than THREAD_COUNT depending upon the avaliability.
 * 2) NO_OF_STEPS. (Optional Parameter) No of steps to be taken for the calculation of PI,
 	more the number of steps the smaller the step size will be.
 */

#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

// Defining Default parameters
#define NO_OF_STEPS 10000
#define NO_OF_THREADS 5


namespace {

double pi_worksharing_and_reduction_parallel(int no_of_threads, int no_of_steps) {

	//Setting number of threads based on the passed parameter
	if(no_of_threads > 0)
		omp_set_num_threads(no_of_threads);
	else
	{
		no_of_threads = NO_OF_THREADS;
		omp_set_num_threads(NO_OF_THREADS);
	}
	
	int step_count;

	if(no_of_steps > 0)
		step_count = no_of_steps;
	else
		step_count = NO_OF_STEPS;
	double step, sum = 0.0, pi, x;

	//Calculating the step size, we are integrating from 0 to 1
	step = 1.0/(double)step_count;

	// Starting timer
	double begin = omp_get_wtime();

	//Parallel step
	#pragma omp parallel for private(x) reduction(+:sum)
		for(int i=0;i<step_count;i++)
		{
			x = (i + 0.5)*step;
			sum = sum + 4.0/(1.0 + x*x);	
		}

	//Area calculation
	pi = sum*step;

	// Stopping timer
	double end = omp_get_wtime();

	printf("PI value is %lf, Calculated using %d number of threads and %d number of steps\n",pi,no_of_threads,step_count);

	return end - begin;
}

}

int main(int argc, char* argv[]) {

	double time_taken_parallel;
	double time_taken_serial;	
	if(argc == 1) {
		time_taken_parallel = pi_worksharing_and_reduction_parallel(0,0);
	}
	else if(argc == 2) {
		time_taken_parallel = pi_worksharing_and_reduction_parallel(atoi(argv[1]),0);
	}
	else if(argc == 3) {
		time_taken_parallel = pi_worksharing_and_reduction_parallel(atoi(argv[1]),atoi(argv[2]));

	}
	else {
		printf("ERROR: Provide Correct Number of arguments\n");
		exit(0);
	}

	printf("Time taken for parallel version is : %lf\n", time_taken_parallel);
}