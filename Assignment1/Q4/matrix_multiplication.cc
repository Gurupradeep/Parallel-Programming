/*
 * Matrix Multiplication
 * Run the program as follows:
 * Compilation:
 * g++ matrix_multiplication.cc -fopenmp
 * Execution:
 * ./a.out
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "omp.h"
#include <time.h>

#define SIZE 1000

int result[SIZE][SIZE];

namespace {

void initialise_matrix(int mat_x[SIZE][SIZE], int mat_y[SIZE][SIZE], int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mat_x[i][j] = rand() % 10 + 2;
        }
    }
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            mat_y[i][j] = rand() % 10 + 2;
        }
    }
}

double calculate_linearly(int mat_x[][SIZE], int mat_y[][SIZE], int size) {
    double start_time = omp_get_wtime();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int val_ij = 0;
            for (int k = 0; k < size; k++) {
                val_ij += mat_x[i][k] * mat_y[k][j];
            }
            result[i][j] = val_ij;
        }
    }
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

std::pair<double, int> calculate_parallelly(int mat_x[SIZE][SIZE], int mat_y[SIZE][SIZE], int size, int num_of_threads) {
    omp_set_num_threads(num_of_threads);
    // int result[SIZE][SIZE];
    double start_time = omp_get_wtime();
    int thread_id;
    int row_x, col_y, k, temp;
    #pragma omp parallel shared(mat_x, mat_y, result, size) private(row_x, col_y, k, temp) 
    {
        thread_id = omp_get_thread_num();
        if (thread_id == 0) {
            num_of_threads = omp_get_num_threads();
        }
        #pragma omp for
        for (row_x = 0; row_x < size; row_x++) {
            for (col_y = 0; col_y < size; col_y++) {
                temp = 0;
                for (k = 0; k < size; k++) {
                    temp += mat_x[row_x][k] * mat_y[k][col_y];
                }
                result[row_x][col_y] = temp;
            }
        }
    }
    double end_time = omp_get_wtime();
    return std::make_pair(end_time - start_time, num_of_threads);
}

/*
 * Unroll the outer loop such that 4 rows are covered in each iteration. Also caching of variables like mat_y[k][col_y] would result in 'hits' with very high probability.
 */
std::pair<double, int> calculate_parallelly_2(int mat_x[SIZE][SIZE], int mat_y[SIZE][SIZE], int size, int num_of_threads) {
    omp_set_num_threads(num_of_threads);
    // int result[SIZE][SIZE];
    double start_time = omp_get_wtime();
    int thread_id;
    int row_x, col_y, k, temp[4];
    #pragma omp parallel shared(mat_x, mat_y, result, size) private(row_x, col_y, k, temp) 
    {
        thread_id = omp_get_thread_num();
        if (thread_id == 0) {
            num_of_threads = omp_get_num_threads();
        }
        #pragma omp for
        for (row_x = 0; row_x < size; row_x += 4) {
            for (col_y = 0; col_y < size; col_y++) {
                temp[0] = temp[1] = temp[2] = temp[3] = 0;
                for (k = 0; k < size; k++) {
                    temp[0] += mat_x[row_x][k] * mat_y[k][col_y];
                    temp[1] += mat_x[row_x + 1][k] * mat_y[k][col_y];
                    temp[2] += mat_x[row_x + 2][k] * mat_y[k][col_y];
                    temp[3] += mat_x[row_x + 3][k] * mat_y[k][col_y];
                }
                result[row_x][col_y] = temp[0];
                result[row_x + 1][col_y] = temp[1];
                result[row_x + 2][col_y] = temp[2];
                result[row_x + 3][col_y] = temp[3];
            }
        }
    }
    double end_time = omp_get_wtime();
    return std::make_pair(end_time - start_time, num_of_threads);
}

}

int main() {
    int mat_x[SIZE][SIZE], mat_y[SIZE][SIZE];
    initialise_matrix(mat_x, mat_y, SIZE);
    double serial_time = calculate_linearly(mat_x, mat_y, SIZE);
    printf("Linear execution run time: %lf seconds.\n", serial_time);
    for (int thread = 2; thread <= 15; thread++) {
        std::pair<double, int> result_1 = calculate_parallelly(mat_x, mat_y, SIZE, thread);
        std::pair<double, int> result_2 = calculate_parallelly_2(mat_x, mat_y, SIZE, thread);
        printf("Approach 1 with %d threads took %lf seconds. Approach 2 with %d threads took %lf seconds.\n", result_1.second, result_1.first, result_2.second, result_2.first);
        printf("Speed up: Approach 1: %lf. Approach 2: %lf.\n", result_1.first / serial_time, result_2.first / serial_time);
    }
	return 0;
}
