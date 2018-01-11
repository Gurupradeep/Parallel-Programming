/*
 * Monte Carlo simulation. Calculation of PI.
 * compilation: g++ monte_carlo.cc -fopenmp
 * Execution: ./a.out
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "omp.h"

#define mod 2147483647
#define add 12345
#define mul 1103515245
#define ITER 10000000

long long int rand_last_serial, rand_next_serial;
long long int rand_last_parallel, rand_next_parallel;

#pragma omp threadprivate(rand_last_parallel, rand_next_parallel)

long long int frog_leap_add, frog_leap_mul;
namespace {

void initialize_seeds() {
    rand_last_serial = mul / mod;
}

inline bool is_inside_circle(double x, double y) {
    return (x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5) <= (double) 0.25 ? true : false;
}

double get_rand() {
    rand_next_serial = (((mul % mod) * (rand_last_serial % mod)) % mod + add % mod) % mod;
    double rand_value = 1.0 * rand_last_serial / mod;
    rand_last_serial = rand_next_serial;
    return rand_value;
}

double get_pi_value_serial() {
    double rand_last = 0;
    int inside = 0;
    for (int i = 0; i < ITER; i++) {
        double x = get_rand();
        double y = get_rand();
        if (is_inside_circle(x, y)) {
            inside++;
        }
    }
    return (double) 4.0 * (1.0 * inside / ITER * 1.0);
}
// parallel execution code and helper.

long long int modular_exponentiation(long long int base, long long int expo) {
    long long int ans = 1;
    while (expo > 0) {
        if (expo & 1) {
            ans = (ans * base) % mod;
        }
        base = (base * base) % mod;
        expo = expo >> 1;
    }
    return ans;
}

double get_rand_parallel() {
    rand_next_parallel = ((frog_leap_mul * rand_last_parallel) % mod + frog_leap_add) % mod;
    double rand = rand_last_parallel * 1.0 / mod;
    rand_last_parallel = rand_next_parallel;
    return rand;
}

double get_pi_value_parallel() {
    long long int *seeds;
    int inside_circle = 0;
    #pragma omp parallel shared(seeds)
    {
        #pragma omp single 
        {
            int threads = omp_get_num_threads();
            seeds = (long long int *) malloc (threads * sizeof(long long int));
            seeds[0] = mul / mod;
            for (int i = 1; i < threads; i++) {
                seeds[i] = (((mul % mod) * (seeds[i-1] % mod)) % mod + add)%mod;
            }
            frog_leap_mul = modular_exponentiation(mul, threads);
            frog_leap_add = ((((modular_exponentiation(mul, threads) - 1 + mod) % mod) * (modular_exponentiation(mul - 1, mod - 2) % mod) % mod) * add) % mod;
        }
        int thread_id = omp_get_thread_num();
        rand_last_parallel = seeds[thread_id];
        #pragma omp for reduction(+:inside_circle)
        for (int i = 0; i < ITER; i++) {
            double x = get_rand_parallel();
            double y = get_rand_parallel();
            if (is_inside_circle(x, y)) {
                inside_circle++;
            }
        }
    }
    return 4.0 * inside_circle / ITER;
}

}

int main () {
    initialize_seeds();
    double serial_pi_value = get_pi_value_serial();
    printf("Linear pi value: %lf.\n", serial_pi_value);
    double parallel_pi_value = get_pi_value_parallel();
    printf("Parallel pi value: %lf.\n", parallel_pi_value);
    return 0;
}
