#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <stdlib.h>


namespace {
	double get_parallel_execution_time(int X[], int Y[], int a, int no_of_threads) {
		omp_set_num_threads(no_of_threads);
		double begin = omp_get_wtime();
		#pragma omp parallel
		{
			no_of_threads = omp_get_num_threads();
			#pragma omp for 
			for (int i = 0;i < (1<<16); i++)
				X[i] = a*X[i] + Y[i];
		}
		double end = omp_get_wtime();
		printf("No of threads used are %d\n",no_of_threads);
		return end - begin;
	}
}
int main() {
	int X[1<<16] = {1};
	int Y[1<<16] = {2};
	int a = 6;
	
	double	single_thread_time = get_parallel_execution_time(X,Y,a,1);
	printf("Time taken for single threaded execution is %lf Seconds\n", single_thread_time);


	double time_taken ;
	double speed_up ;
	for(int i = 2;i<=10;i++) {
		time_taken = get_parallel_execution_time(X,Y,a,i);
		speed_up = single_thread_time/time_taken;
		// printf("Time taken : %lf Seconds ",time_taken);
		printf("Speedup is : %lf\n",speed_up);
	}

}