/*
 * Basic Hello world program
 * Run the program as follows
 * (Compilation) mpicxx hello_world.cc 
 * (Execution) mpirun -n NO_OF_PROCESSES ./a.out
 * Arguments
 * 1) NO_OF_PROCESES. (Optional Parameter) : no of processes to be created.
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

	//Declaring variables for process_identifier, no of processes and length of the processor name
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

	printf ("Hello World. Rank %d out of %d running on %s!\n", rank, size, name);

	//Terminate MPI environment
	MPI_Finalize();

	return 0;
}