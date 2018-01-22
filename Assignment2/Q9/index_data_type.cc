#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 5

int main () {
    int displacement[N], block_lengths[N];
    int matrix[N*N];
    MPI_Datatype matrix_indexed_datatype;
    MPI_Status status;
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    for (int i = 0; i < N; i++) {
        block_lengths[i] = N - i;
        displacement[i] = i * N + i; // 1 D array.
    }
    // Creates a new derived datatype and returns the handle to the derived datatye in
    // matrix_indexed_datatype
    MPI_Type_indexed(N, block_lengths, displacement, MPI_INT, &matrix_indexed_datatype);
    //Commiting the new data type
    MPI_Type_commit(&matrix_indexed_datatype);

    if (world_rank == 0) {
        for (int i = 0; i < N * N; i++) {
            matrix[i] = i + 1; // matrix[0][0] = 0 might confuse output.
        }
        MPI_Send(matrix, 1, matrix_indexed_datatype, 1, 0, MPI_COMM_WORLD);
    } else if (world_rank == 1) {
        for (int i = 0; i < N * N; i++) {
            matrix[i] = 0;
        }
        MPI_Recv(matrix, 1, matrix_indexed_datatype, 0, 0, MPI_COMM_WORLD, &status);
        int elements = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", matrix[elements]);
                elements++;
            }
            printf("\n");
        }
    } else {
    
    }
    MPI_Finalize();
    return 0;
}
