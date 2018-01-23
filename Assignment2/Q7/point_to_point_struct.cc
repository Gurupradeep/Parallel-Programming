/*
 * Building a custom datatype (struct in this case) in a mpi program 
 * and sending it via point to point communication.
 *
 * compile: mpicxx point_to_point_struct.cc
 * run: mpirun -n 4 ./a.out
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

struct dd {
    char c;
    int i[2];
    float f[4];
};

namespace {
}

int main () {
    struct dd struct_data;
    MPI_Datatype dd_type;
    MPI_Datatype types[] = {MPI_CHAR, MPI_INT, MPI_FLOAT};
    int block_length[] = {1, 2, 4};
    MPI_Aint displacements[3];
    MPI_Init(NULL, NULL);
    MPI_Status status;

    //Getting the byte displacement of each block
    MPI_Aint block_address, block_address1, block_address2;
    MPI_Get_address(&struct_data.c, &block_address);
    displacements[0] = block_address - block_address;
    MPI_Get_address(&struct_data.i, &block_address1);
    displacements[1] = block_address1 - block_address;
    MPI_Get_address(&struct_data.f, &block_address2);
    displacements[2] = block_address2 - block_address;

    // Creates a struct and returns the handle to the derived type in dd_type
    MPI_Type_create_struct(3, block_length, displacements, types, &dd_type);
    //Commits the new type
    MPI_Type_commit(&dd_type);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_rank == 0) {
        struct_data.c = 'b';
        struct_data.i[0] = struct_data.i[1] = 2;
        struct_data.f[0] = struct_data.f[1] = struct_data.f[2] = struct_data.f[3] = 4.0;
    }
    if (world_rank == 0) {
    	for (int i = 0; i < world_size; i++) {
    		if (i != world_rank) {
    			MPI_Send(&struct_data, 1, dd_type, i, 0, MPI_COMM_WORLD);
    		}
    	}
    } else {
    	MPI_Recv(&struct_data, 1, dd_type, 0, 0, MPI_COMM_WORLD, &status);
    	printf("Data received via point to point. Rank: %d.\n", world_rank);
      printf("c: %c. i[0]: %d i[1]: %d. f[0]: %f f[1]: %f f[2]: %f f[3]: %f.\n", struct_data.c, struct_data.i[0], struct_data.i[1], struct_data.f[0], struct_data.f[1], struct_data.f[2], struct_data.f[3]);
    }
    MPI_Finalize();
    return 0;   
}
