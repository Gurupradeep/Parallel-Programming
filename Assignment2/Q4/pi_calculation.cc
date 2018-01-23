/*
 * Basic Hello world program
 * Run the program as follows
 * (Compilation) mpicxx Pi_calulation.cc
 * (Execution) mpirun -n NO_OF_PROCESSES ./a.out
 * Arguments
 * 1) NO_OF_PROCESES. (Optional Parameter) : no of processes to be created.
 */
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

namespace {
/*
 * Computes the partial value of PI 
*/
double compute_partial_sum(int rank, int num_of_steps, int no_of_processes) {
	double sum = 0.0, x, pi, step;
	step = 1.0/(double)num_of_steps;
	for(int i = rank; i < num_of_steps; i = i + no_of_processes ) {
		x = (i+0.5)*step;
		sum = sum + 4.0/(1.0+x*x);
	}
	pi = step*sum;
	return pi;
}
}

int main(int argc, char* argv[]) {

	int rank, size, namelen;

	//For name of the processor
	char name[100];

	//Initialsiing the MPI environment
	MPI_Init(NULL, NULL);

	//To get no of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//To get process id
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//To get processor name 
	MPI_Get_processor_name(name, &namelen);
	
	int num_of_steps;
	if(rank == 0) {
		num_of_steps = 100000;
	}
	//Broadcasting number of steps
	MPI_Bcast(&num_of_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	double partial_pi = compute_partial_sum(rank, num_of_steps,size);

	double PI;
	//Combining the partial sums to get the value of PI
	MPI_Reduce(&partial_pi, &PI,1,MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	
	if(rank == 0) {
		printf("Value of PI is %lf\n",PI);
	}
	MPI_Finalize();

	return 0;
}
