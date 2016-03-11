// Copyright (C) 2016 Peter Zaspel
//
// This file is part of MPLA.
//
// MPLA is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// MPLA is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with MPLA.  If not, see <http://www.gnu.org/licenses/>.

#include "mpla.h"
#include "mpi.h"
#include <curand.h>

int idx(int i, int j, int m, int n)
{
	return j*m + i;
}


void checkCUDAError(const char* msg) {
cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[])
{
	int m,n;
	m=atoi(argv[1]);
	n=atoi(argv[2]);

	int trials = 100;	
	if (argc==4)
		trials = atoi(argv[3]);

	
	MPI_Init( &argc, &argv );

	// init MPLA
	struct mpla_instance instance;
	mpla_init_instance(&instance, MPI_COMM_WORLD);

	// init matrix and vectors
	struct mpla_matrix A;
	mpla_init_matrix(&A, &instance, m, n);

	struct mpla_vector x;
	mpla_init_vector(&x, &instance, n);

	struct mpla_vector Ax;
	mpla_init_vector(&Ax, &instance, m);

	// set up random number generator
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	// fill vector by random numbers
	curandGenerateUniformDouble(gen, x.data, x.cur_proc_row_count);

	// fill matrix by random data
	curandGenerateUniformDouble(gen, A.data, A.cur_proc_row_count*A.cur_proc_col_count);

	// setup of time measurements
	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// compute kernel
	cudaEventRecord(start, 0);
	for (int n = 1; n<trials; n++) 
	{
		// calculate MVP
		mpla_dgemv(&Ax, &A, &x, &instance);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// evaluate time
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("Time spent: %.10f\n", gpu_time/1000);

	mpla_free_matrix(&A, &instance);
	mpla_free_vector(&Ax, &instance);
	mpla_free_vector(&x, &instance);

	// curand cleanup
	curandDestroyGenerator(gen);

	// event cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop); 

	MPI_Finalize();

}

