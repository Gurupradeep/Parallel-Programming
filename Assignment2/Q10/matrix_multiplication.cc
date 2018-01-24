#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#define N 2

struct grid_info {
    int row, col, rank, A, B, C;
    int num_of_processes, proc_per_dim;
    MPI_Comm comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
};

namespace {

void init_matrix(int A[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i * i + j * j;
        }
    }
}

}

int main () {
    int A[N][N], B[N][N], C[N][N];
    MPI_Init(NULL, NULL);
    grid_info grid;
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    grid.rank = world_rank;
    grid.num_of_processes = world_size;
    grid.proc_per_dim = sqrt(world_size);
    if (world_rank == 0) {
        init_matrix(A);
        init_matrix(B);
    }
    int elem_a, elem_b;
    MPI_Scatter(A, 1, MPI_INT, &elem_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, 1, MPI_INT, &elem_b, 1, MPI_INT, 0, MPI_COMM_WORLD);
    grid.A = elem_a;
    grid.B = elem_b;
    // N processes per dimension.
    int dimensions[2] = {N, N};
    // Period grid. 
    int periods[2] = {1, 1};
    MPI_Comm communicator; // new communication for the grid.
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 0, &communicator);
    grid.comm = communicator;
    // Subgroups for each row and column to be handled by a process.
    int sub_group[2][2] = {{1, 0}, {0, 1}};
    MPI_Cart_sub(communicator, sub_group[0], &grid.row_comm);
    MPI_Cart_sub(communicator, sub_group[1], &grid.col_comm);

    MPI_Comm_rank(grid.col_comm, &grid.row);
    MPI_Comm_rank(grid.row_comm, &grid.col);

    grid.C = 0;
    // wrap around case.
    MPI_Sendrecv_replace(&grid.A, 1, MPI_INT, (grid.col - grid.row + grid.proc_per_dim) % grid.proc_per_dim , 0, (grid.col + grid.row) % grid.proc_per_dim, 0, grid.row_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(&grid.B, 1, MPI_INT, (grid.row - grid.col + grid.proc_per_dim) % grid.proc_per_dim , 0, (grid.row + grid.col) % grid.proc_per_dim, 0, grid.col_comm, MPI_STATUS_IGNORE);

    grid.C += grid.A * grid.B;
    
    for (int i = 0; i < grid.proc_per_dim - 1; i++) {
        // left shift.
        MPI_Sendrecv_replace(&grid.A, 1, MPI_INT, (grid.col - 1 + grid.proc_per_dim) % grid.proc_per_dim, 0, (grid.col + 1) % grid.proc_per_dim, 0, grid.row_comm, MPI_STATUS_IGNORE);
        // upper shift.
        MPI_Sendrecv_replace(&grid.B, 1, MPI_INT, (grid.row - 1 + grid.proc_per_dim) % grid.proc_per_dim, 0, (grid.row + 1) % grid.proc_per_dim, 0, grid.col_comm, MPI_STATUS_IGNORE);
        grid.C += grid.A * grid.B;
    }

    MPI_Gather(&grid.C, 1, MPI_INT, C, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", C[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
