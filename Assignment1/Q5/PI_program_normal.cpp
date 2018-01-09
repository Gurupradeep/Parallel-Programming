/*
 * Calculating value of PI parallely
 * Run the program as follows
 * (Compilation) g++ -fopenmp PI_program_normal.cpp 
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

double pi_parallel(int no_of_threads, int no_of_steps, int flag) {

	
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
	double step, pi = 0.0;

	//Calculating the step size, we are integrating from 0 to 1
	step = 1.0/(double)step_count;

	// Starting timer
	double begin = omp_get_wtime();

	//Parallel step
	#pragma omp parallel
	{
		double sum = 0.0, x;
		int id;
		id = omp_get_thread_num();
		int n_threads;
		n_threads = omp_get_num_threads();
		int i;

		//Parallel task part
		for(i=id;i<step_count;i = i + n_threads) {
			x = (i + 0.5)*step;
			sum = sum + 4.0/(1.0 + x*x);	
		}

		//Sum is private to each thread, multiplying sum with step to reduce the no of operations in critical section
		sum = sum * step;
		
		if(flag == 0) {
			//Only one operation in critical section
			#pragma omp critical
				pi += sum;
		}
		else {
			//Only one operation in critical section
			#pragma omp atomic
				pi += sum;

		}

	}

	printf("PI value is %lf, Calculated using %d number of threads and %d number of steps\n",pi,no_of_threads,step_count);
	// Stopping timer
	double end = omp_get_wtime();

	return end - begin;
}


}

int main(int argc, char* argv[]) {

	double time_taken_critical;
	double time_taken_atomic;	
	if(argc == 1) {
		time_taken_critical = pi_parallel(0,0,0);
		time_taken_atomic = pi_parallel(0,0,1);
	}
	else if(argc == 2) {
		time_taken_critical = pi_parallel(atoi(argv[1]),0,0);
		time_taken_atomic = pi_parallel(atoi(argv[1]),0,1);
	}
	else if(argc == 3) {
		time_taken_critical = pi_parallel(atoi(argv[1]),atoi(argv[2]),0);
		time_taken_atomic = pi_parallel(atoi(argv[1]),atoi(argv[2]),1);

	}
	else {
		printf("ERROR: Provide Correct Number of arguments\n");
		exit(0);
	}

	printf("Time taken for parallel version is using critical construct is : %lf Seconds\n", time_taken_critical);
	printf("Time taken for parallel version is using atomic construct is : %lf Seconds\n", time_taken_atomic);
	
}
