/*
 * Basic Hello world program
 * Run the program as follows
 * (Compilation) mpicxx hello_world_version2.cc 
 * (Execution) mpirun -n NO_OF_PROCESSES ./a.out
 * Arguments
 * 1) NO_OF_PROCESES. (Optional Parameter) : no of processes to be created.
 */
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <bits/stdc++.h>
using namespace std;
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
	
	//Message to be sent to master process.
	char message[100] = "Hello world";

	if(rank == 0) {
		for(int i = 1; i < size; i++) {
			MPI_Status status;
			int count ;
			//Probing the sender to get the size of the message
			MPI_Probe(i,0, MPI_COMM_WORLD, &status);
			MPI_Get_count(&status, MPI_CHAR, &count);
			
			//Receiving the message
			MPI_Recv(message,count, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("Recieved %s from process %d\n",message,i);
		}
	}
	else {
		
		//Sending the message to the master process.		
		MPI_Send(message,strlen(message), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	}
	MPI_Finalize();

	return 0;
}