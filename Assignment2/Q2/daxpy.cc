#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

const int size = 1 << 16;
const int a_const = 4;

namespace {
//Initialising the arrays of given size
void initialize_arrays(int X[], int Y[], int size) {
    for (int index = 0; index < size; index++) {
        X[index] = 4;
        Y[index] = 8;
    }
}

//Computes the new values of for first sub array 
void compute_subarray(int subarray_x[], int subarray_y[], int number_elements) {
    for (int index = 0; index < number_elements; index++) {
        subarray_x[index] = subarray_x[index] * a_const + subarray_y[index];
    }
    return;
}

}


int main () {
    int *X = NULL, *Y = NULL;
    double start_time, end_time;
    X = (int *) malloc (sizeof(int) * size);
    Y = (int *) malloc (sizeof(int) * size);
    initialize_arrays(X, Y, size);

    //Serial Execution : Start
    start_time = MPI_Wtime();
    for (int i = 0; i < size; i++) {
        X[i] = a_const * X[i] + Y[i];
    }
    end_time = MPI_Wtime();
    free(X), free(Y);
    double serial_time = end_time - start_time;
    //Serial Execution : End

    //Initialising mpi environment
    MPI_Init (NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_rank == 0) {
        X = (int *) malloc (sizeof(int) * size);
        Y = (int *) malloc (sizeof(int) * size);
        initialize_arrays(X, Y, size);
        start_time = MPI_Wtime();
    }
    int number_elements_per_proc = size / world_size;
    //Creating a buffer for each process to hold the subset of the entire array
    int *subarray_x = NULL, *subarray_y = NULL;
    subarray_x = (int *) malloc(sizeof(int) * number_elements_per_proc);
    subarray_y = (int *) malloc(sizeof(int) * number_elements_per_proc);

    //Scattering arrays across all the processes.
    MPI_Scatter(X, number_elements_per_proc, MPI_INT, subarray_x, number_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y, number_elements_per_proc, MPI_INT, subarray_y, number_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    compute_subarray(subarray_x, subarray_y, number_elements_per_proc);

    //Gathering the final results
    MPI_Gather(subarray_x, number_elements_per_proc, MPI_INT, X, number_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        end_time = MPI_Wtime();
        printf("Parallel execution time: %lf seconds.\nSerial execution time: %lf seconds.\n", end_time - start_time, serial_time);
    }
    MPI_Finalize();
    return 0;
}
