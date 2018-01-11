/*
 * Producer Consumer problem.
 * Compilation: producer_consumer.cc -fopenmp
 * Execution: ./a.out
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <time.h>

#define N 1000
#define MAX 100
#define MIN 10

namespace {

double get_random() {
    double before = rand() % (int) MAX + (int) MIN;
    double after = (double) rand() / RAND_MAX;
    double result = before + after;
    return result;
}

double get_sum(double arr[], int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

void reset(double arr[], int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = get_random();
    }
    return;
}

double get_linear_execution_time(double arr[], int size) {
    double start_time = omp_get_wtime();
    reset(arr, size); // producer.
    double sum = get_sum(arr, size); // consumer.
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

double get_parallel_execution_time(double arr[], int size) {
    int flag = 0;
    double sum = 0.0;
    double start_time = omp_get_wtime();
    #pragma omp sections
    {
        #pragma omp section
        {
            // producer.
            #pragma omp parallel for shared(arr)
            for (int i = 0; i < size; i++) {
                arr[i] = get_random();
            }
            #pragma atomic write
                flag = 1;
        }
        #pragma omp section
        {
            // consumer.
            int flag_read = 0;
            while (1) {
                #pragma omp flush(flag)
                #pragma omp atomic read
                    flag_read = flag;
                if (flag_read == 1)
                    break;
            }
            #pragma omp parallel for shared(sum, arr)
            for (int i = 0; i < size; i++) {
                #pragma omp atomic
                sum += arr[i];
            }
        }
    }
    double end_time = omp_get_wtime();
    // works fine! tested!
    // printf("Parallel sum: %lf. Serial sum: %lf.\n", sum, get_sum(arr, size));
    return end_time - start_time;
}

}

int main(int argc, char *argv[]) {
    int size = N;
    srand(time(NULL));
    if (argc > 1) {
        if (std::string(argv[1]) == "-n") {
            size = atoi(argv[2]);
            if (size <= 0) {
                printf("Size of the input cannot be wrong.\n");
                exit(1);
            }
        } else {
            printf("Wrong command line argument.\n");
            exit(1);
        }
    }
    double arr[size];
    double linear_time = get_linear_execution_time(arr, size);
    printf("Linear producer-consumer execution time: %lf seconds.\n", linear_time);
    double parallel_time = get_parallel_execution_time(arr, size);
    printf("Parallel producer-consumer execution time: %lf seconds.\n", parallel_time);
    return 0;
}
