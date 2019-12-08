#include <iostream>
#include <fstream>
#include <time.h>
#include "distribution_function.h"
#include "metropolis_hastings.h"

using namespace std;

int main(int argc, char* argv[]) {
    const int dimension = 200;
    const int num_samples = argc>1 ? atoi(argv[1]) : 100;
    clock_t start_time = clock();
    double** samples = (double**)malloc(num_samples * sizeof(double*));
    for(int i=0; i<num_samples; i++) {
        samples[i] = (double*)malloc(dimension * sizeof(double));
    }
    clock_t malloc_end_time = clock();
    clock_t memory_allocation_time = (malloc_end_time - start_time)/(CLOCKS_PER_SEC/1000000);
    metropolis_hastings(distribution_function, num_samples, dimension, samples);
    clock_t algo_end_time = clock();
    clock_t algo_time = (algo_end_time - malloc_end_time)/(CLOCKS_PER_SEC/1000000);

    ofstream output_file;
    output_file.open("samples.csv");
    for(int i=0; i<num_samples; i++) {
        for(int j=0; j<dimension-1; j++) {
            output_file<<samples[i][j]<<",";
        }
        output_file<<samples[i][dimension-1]<<"\n";
    }
    output_file.close();

    cout<<"Memory Allocation Time: "<<memory_allocation_time<<" microseconds"<<endl;
    cout<<"Algorithm Running Time: "<<algo_time<<" microseconds"<<endl;

    return 0;
}
