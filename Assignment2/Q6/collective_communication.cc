/*
 * Collective Communication - Scatter - Gather
 * Run the program as follows
 * (Compilation) mpicxx collective_communication.cc 
 * (Execution) mpirun -n NO_OF_PROCESSES ./a.out NO_OF_ELEMENTS_PER_PROCESS
 * Arguments
 * 1) NO_OF_PROCESES. (Optional Parameter) : no of processes to be created.
 * 2) NO_OF_ELEMENTS_PER_PROCESS (Compulsary parameter) : No of elements that should
 	be allocated to each process.
 */
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <bits/stdc++.h>
namespace {
/*
* Returns a random array of Numbers between 0 and 1
*/
float *create_rand_nums(int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  for (int i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float)RAND_MAX);
  }
  return rand_nums;
}

/*
*	Returns an array by calculating square root of every element in the array
*/
float *compute_square_root(float *a, int num_of_elements) {
	float *square_root = (float *)malloc(sizeof(float) * num_of_elements);
	for(int i = 0; i < num_of_elements; i++) {
		square_root[i] =  (float)sqrt(a[i]);
		
	}
	return square_root;
}

}
int main(int argc, char* argv[]) {
	if(argc != 2) {
		printf("Incorrect No of arguments, should give no of elements per process as argument\n");
		exit(1);
	}
	int number_of_elements_per_process = atoi(argv[1]);

	//Initialsiing the MPI environment
	MPI_Init(NULL, NULL);

	int rank, size ;

	//To get no of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//To get process id
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	float *rand_nums = NULL;

	//Creating an array of random numbers
	if(rank == 0) {
		rand_nums = create_rand_nums(number_of_elements_per_process * size);
		for(int i=0; i<(number_of_elements_per_process * size); i++) {
			printf("element at postion %d is %f\n",i,rand_nums[i]);
		}
	}

	//Creating a buffer for each process to hold the subset of the entire array
	float *sub_rand_nums = (float*) malloc(sizeof(float) * number_of_elements_per_process);

	//Scatter the random numbers from the root process to all the processes in the 
	// MPI world.
	MPI_Scatter(rand_nums, number_of_elements_per_process, MPI_FLOAT, sub_rand_nums, 
		number_of_elements_per_process, MPI_FLOAT,0,MPI_COMM_WORLD);

	//Computes the square root of the subset.
	float *square_root_partial = compute_square_root(sub_rand_nums, number_of_elements_per_process);
	
	//Gather all the square roots at the root
	float *square_root_complete = NULL;
	if(rank == 0) {
		square_root_complete = (float*)malloc(sizeof(float)* number_of_elements_per_process* size);
	} 
	MPI_Gather(square_root_partial, number_of_elements_per_process, MPI_FLOAT,square_root_complete,
	 number_of_elements_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//Displaying the final results
	if(rank == 0) {
		for(int i=0; i<(number_of_elements_per_process * size); i++) {
			printf("Square root of element at postion %d is %f\n",i,square_root_complete[i]);
		}
		//Final clean up of memory specific to root process
		free(rand_nums);
		free(square_root_complete);
	}
	//Final clean up of memory specific to all processes.	
	free(square_root_partial);
	free(sub_rand_nums);

	MPI_Finalize();

	return 0;
}
