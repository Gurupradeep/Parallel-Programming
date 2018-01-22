#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main () {
    MPI_Init(NULL, NULL);
    int world_rank, world_size;
    char c;
    int i[2];
    float f[4];
    char interm_buffer[100];
    int position = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0) {
        c = 'b';
        i[0] = i[1] = 5;
        f[0] = f[1] = f[2] = f[3] = 2.0;
        MPI_Pack(&c, 1, MPI_CHAR, interm_buffer, 100, &position, MPI_COMM_WORLD);
        MPI_Pack(i, 2, MPI_INT, interm_buffer, 100, &position, MPI_COMM_WORLD);
        MPI_Pack(f, 4, MPI_FLOAT, interm_buffer, 100, &position, MPI_COMM_WORLD);
        for(int i=1; i < world_size; i++) {
             MPI_Send(interm_buffer, position, MPI_PACKED, i, 100, MPI_COMM_WORLD);
        }
    } else if (world_rank != 0) {
        MPI_Recv(interm_buffer, 100, MPI_PACKED, 0, 100, MPI_COMM_WORLD, &status);
        MPI_Unpack(interm_buffer, 100, &position, &c, 1, MPI_CHAR, MPI_COMM_WORLD);
        MPI_Unpack(interm_buffer, 100, &position, i, 2, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(interm_buffer, 100, &position, f, 4, MPI_FLOAT, MPI_COMM_WORLD);
        printf("Received by rank: %d, values:\nc: %c, i[0]: %d i[1]: %d, f[0]: %f f[1]: %f f[2]: %f f[3]: %f\n", world_rank, c, i[0], i[1], f[0], f[1], f[2], f[3]);
    }
    MPI_Finalize();
    return 0;
}
