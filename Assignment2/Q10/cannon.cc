#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <bits/stdc++.h>

namespace {

}
typedef struct {
	int N;
	int size;
	int row;
	int col;
	int my_rank;
	MPI_Comm comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;
	int A; 
	int B;
	int C;
}grid_info;

void setup_mesh(grid_info *info, int num_of_processes, int rank) {
	info->N = sqrt(num_of_processes);
	info->size = num_of_processes;
	int dims[2] = {info->N, info->N} ;
	int periods[2] = {1,1};
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &info->comm);
	MPI_Comm_rank(info->comm, &(info->my_rank));
	int coords[2];
	MPI_Cart_coords(info->comm, info->my_rank, 2, coords);
	info->row = coords[0];
	info->col = coords[1];

	int keepdims[2];
	keepdims[0] = 1;
	keepdims[1] = 0;
	MPI_Cart_sub(info->comm, keepdims, &info->row_comm);
	keepdims[0] = 0;
	keepdims[1] = 1;
	MPI_Cart_sub(info->comm, keepdims, &info->col_comm);

}
void ring_shift(int *data, int count, MPI_Comm ring, int disp) {
	int src, dst;
	MPI_Status status;

	int *temp;

	//printf("Before ring shift\n");
	MPI_Cart_shift(ring, 0, disp, &src, &dst);
	//printf("After ring shift\n");
	MPI_Sendrecv(data, count, MPI_INT, dst, 0, temp, count, MPI_INT, src, 0, ring, &status);
	//printf("After Sendrecv\n");
	*data = *temp;

}

void cannon(grid_info *info ) {

	//printf("Before cannon\n");
	ring_shift(&info->A, 1, info->row_comm, -(info->row));
	ring_shift(&info->B, 1, info->col_comm, -(info->col));

	for(int i=0;i<info->N; i++) {
		info->C += (info->A)*(info->B);
		printf("C value is %d\n",info->C);
		ring_shift(&info->A, 1, info->row_comm, -1);
		ring_shift(&info->B, 1, info->col_comm, -1);
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

	grid_info *info;

	setup_mesh(info, size, rank);

	printf("X- %d, Y - %d, Rank - %d\n",info->row, info->col, info->my_rank);
	info->A = info->my_rank;
	info->B = info->my_rank;
	info->C = 0;
	
	//printf("C value is %d\n",info->C);
	cannon(info);
	//printf("C value is %d\n",info->C);
	MPI_Finalize();

	return 0;
}