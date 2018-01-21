/*
 * Collective Communication - Scatter - Gather - Non unifrom version
 * Run the program as follows
 * (Compilation) mpicxx collective_communication_v2.cc 
 * (Execution) mpirun -n NO_OF_PROCESSES ./a.out 
 * Arguments
 * 1) NO_OF_PROCESES. (Optional Parameter) : no of processes to be created.
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
		// printf("%f \n",square_root[i]);
	}
	return square_root;
}

}
int main(int argc, char* argv[]) {
	
	//Initialsiing the MPI environment
	MPI_Init(NULL, NULL);

	int rank, size ;

	//To get no of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//To get process id
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	float *rand_nums = NULL;

	int total_count = (size * (size + 1))/2;
	//Creating an array of random numbers
	if(rank == 0) {
		
		printf("size is %d\n",size);
		rand_nums = create_rand_nums(total_count);
		for(int i=0; i<(total_count); i++) {
			printf("element at postion %d is %f\n",i,rand_nums[i]);
		}
	}

	//Creating a buffer for each process to hold the subset of the entire array
	float *sub_rand_nums = (float*) malloc(sizeof(float) * (rank + 1));

	//Creating send counts and displacement arrays which are required to 
	// split the array among the processes.
	int *sendcounts = (int*) malloc(sizeof(int) * size);
	int *displs = (int*) malloc(sizeof(int) * size);
	for(int i=0; i < size; i++) {
		sendcounts[i] = i + 1;
		displs[i] = (i*(i+1))/2;
	}
	
	//Scatter the random numbers from the root process to all the processes in the 
	// MPI world.
	MPI_Scatterv(rand_nums, sendcounts,displs, MPI_FLOAT, sub_rand_nums, 
		rank + 1, MPI_FLOAT,0,MPI_COMM_WORLD);

	//Computes the square root of the subset.
	float *square_root_partial = compute_square_root(sub_rand_nums, rank + 1);
	
	//Gathering the partial results at the root process.
	float *square_root_complete = NULL;
	if(rank == 0) {
		square_root_complete = (float*)malloc(sizeof(float)* total_count);
	} 

	MPI_Gatherv(square_root_partial, rank + 1, MPI_FLOAT,square_root_complete,
	 sendcounts,displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//Displaying the final results
	if(rank == 0) {
		for(int i=0; i<(total_count); i++) {
			printf("Square root of element at postion %d is %f\n",i,square_root_complete[i]);
		}
		//Final clean up of memory specific to root process
		free(rand_nums);
		free(square_root_complete);
	}	
	//Final clean up of memory specific to all processes.
	free(square_root_partial);
	free(sub_rand_nums);
	free(displs);
	free(sendcounts);

	MPI_Finalize();

	return 0;
}