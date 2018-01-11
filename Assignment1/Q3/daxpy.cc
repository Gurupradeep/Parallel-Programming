/*
 * DAXPI program.
 * Run the program as follows.
 * Compilation:
 * g++ daxpy.cc -fopenmp
 * Run:
 * ./a.out
 */
#include <stdio.h>
#include <omp.h>
#include <iostream>

#define SIZE 1 << 16

namespace {

std::pair<double, int> get_parallel_execution_time(int X[], int Y[], int a, int num_of_threads) {
	omp_set_num_threads(num_of_threads);
	double start_time = omp_get_wtime();
	int thread_id;
	#pragma omp parallel shared(X, Y)
	{
		thread_id = omp_get_thread_num();
		if (thread_id == 0) {
			num_of_threads = omp_get_num_threads();
		}
		#pragma omp for 
		for (int i = 0; i < SIZE; i++)
			X[i] = a*X[i] + Y[i];
	}
	double end_time = omp_get_wtime();
	return std::make_pair(end_time - start_time, num_of_threads);
}

/*
 * Unroll the loop to maximum the work done in a single iteration. This will save few thread operations (like computing the range to operate on etc) and compiler work.
 *
 */
std::pair<double, int> get_parallel_execution_time_2(int X[], int Y[], int a, int num_of_threads) {
	omp_set_num_threads(num_of_threads);
	double start_time = omp_get_wtime();
	int thread_id;
	#pragma omp parallel shared(X, Y) private(thread_id)
	{
		thread_id = omp_get_thread_num();
		if (thread_id == 0) {
			num_of_threads = omp_get_num_threads();
		}
		#pragma omp for
		for (int i = 0; i < SIZE; i += 2) {
			X[i] = a * X[i] + Y[i];
			X[i+1] = a * X[i+1] + Y[i + 1];
		}
	}
	double end_time = omp_get_wtime();
	return std::make_pair(end_time - start_time, num_of_threads);
}

void reset_dataset(int X[], int Y[], int size) {
	for (int i = 0; i < SIZE; i++) {
		X[i] = 1;
		Y[i] = 2;
	}
}

}

int main() {
	int X[SIZE];
	int Y[SIZE];
	int a = 6;
	reset_dataset(X, Y, SIZE);
	double single_thread_time = get_parallel_execution_time(X, Y, a, 1).first;
	printf("Time taken for execution using one thread is %lf seconds.\n", single_thread_time);
	double time_taken;
	for(int thread = 2; thread <= 15; thread++) {
		reset_dataset(X, Y, SIZE);
		std::pair<double, int> result = get_parallel_execution_time(X, Y, a, thread);
		double speed_up  = result.first / single_thread_time;
		printf("Approach 1: For number of threads: %d speedup is: %lf.\n", result.second, speed_up);
		reset_dataset(X, Y, SIZE);
		std::pair<double, int> result_2 = get_parallel_execution_time_2(X, Y, a, thread);
		speed_up = result_2.first / single_thread_time;
		printf("Approach 2: For number of threads: %d speedup is: %lf.\n\n", result_2.second, speed_up);
	}
	return 0;
}
