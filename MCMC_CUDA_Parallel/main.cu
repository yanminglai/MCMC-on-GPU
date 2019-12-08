#include <iostream>
#include <fstream>
#include <time.h>
#include "metropolis_hastings.h"

using namespace std;

int main(int argc, char* argv[]) {
    const int dimension = 20;
    const int num_samples = argc>1 ? atoi(argv[1]) : 100;
    const int num_blocks = argc>3 ? atoi(argv[2]) : 1;
    const int num_threads = argc>3 ? atoi(argv[3]) : 1;
    // Memory allocation
    clock_t start_time = clock();
    float** samples;
    cudaMallocManaged(&samples, 10000*sizeof(float*));
    
    for(int i=0; i<num_samples; i++) {
        cudaMallocManaged(&samples[i], dimension*sizeof(float));
    }
    clock_t malloc_end_time = clock();
    clock_t memory_allocation_time = (malloc_end_time - start_time)/(CLOCKS_PER_SEC / 1000000);

    cout<<"start computation"<<endl;

    metropolis_hastings<<<num_blocks,num_threads>>>(num_samples, dimension, samples);
    cudaDeviceSynchronize();
    
    clock_t algo_end_time = clock();
    clock_t algo_time = (algo_end_time - malloc_end_time)/(CLOCKS_PER_SEC / 1000000);

    ofstream output_file;
    output_file.open("samples.csv");
    for(int i=0; i<num_samples; i++) {
        for(int j=0; j<dimension-1; j++) {
            output_file<<samples[i][j]<<",";
        }
        output_file<<samples[i][dimension-1]<<"\n";
    }
    output_file.close();

  //  cout<<"Memory Allocation Time: "<<memory_allocation_time<<" microseconds"<<endl;
 //   cout<<"Algorithm Running Time: "<<algo_time<<" microseconds"<<endl;
    cout<<algo_time<<endl;
    return 0;
}
