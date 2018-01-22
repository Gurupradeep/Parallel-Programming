#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

namespace {

}

int main () {
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int half = world_size;
    int value = world_rank + 1, recv_value;
    MPI_Status status;
    while (half > 1 && world_rank < half) {
        // Odd number of processors or parent nodes.
        if (half & 1 && world_rank == half - 1) {
            // send to 0.
            MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else if (world_rank >= half / 2) {
            // send to world_rank - half/2.
            MPI_Send(&value, 1, MPI_INT, world_rank - half / 2, 0, MPI_COMM_WORLD);
        } else {
            // recv from world_rank + half/2.
            MPI_Recv(&recv_value, 1, MPI_INT, world_rank + half / 2, 0, MPI_COMM_WORLD, &status);
            value += recv_value;
            if (world_rank == 0 && half & 1) {
                // recv from half - 1.
                MPI_Recv(&recv_value, 1, MPI_INT, half - 1, 0, MPI_COMM_WORLD, &status);
                value += recv_value;
            }
        }
        half = half / 2;
    }
    if (world_rank == 0) {
        printf("Numer of processors: %d.\nTotal sum value: %d.\nValue ((proc)*(proc + 1)/2): %d.\n",
             world_size, value, ((world_size)*(world_size + 1)/2));    
    }
    MPI_Finalize();
    return 0;
}
