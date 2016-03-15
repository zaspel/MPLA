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
#include <cuda_runtime.h>
#include <curand.h>
#include <stdlib.h>

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

void mpla_generic_dgemv_core_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance)
{
	double one = 1;
	double zero = 0;
	cublasDgemv((instance->cublas_handle), CUBLAS_OP_N, A->cur_proc_row_count, A->cur_proc_col_count, &one, (double*)(A->data), A->cur_proc_row_count, x_redist->data, 1, &zero, b->data, 1);
}

int main(int argc, char* argv[])
{
	int n;
	n=atoi(argv[1]);
	
	int trials = 100;
	if (argc==3)
		trials = atoi(argv[2]);

	void (*mpla_dgemv_core)(struct mpla_vector*, struct mpla_generic_matrix*, struct mpla_vector*, struct mpla_instance*);
	mpla_dgemv_core = &mpla_generic_dgemv_core_cublas;

	MPI_Init( &argc, &argv );

	// init MPLA
	struct mpla_instance instance;
	mpla_init_instance(&instance, MPI_COMM_WORLD);

	// init matrix and vectors
	struct mpla_matrix A;
	mpla_init_matrix(&A, &instance, n, n);

	struct mpla_vector x;
	mpla_init_vector(&x, &instance, n);

	struct mpla_vector x_tmp;
	mpla_init_vector(&x_tmp, &instance, n);

	struct mpla_vector b;
	mpla_init_vector(&b, &instance, n);

        // set up GPU random number generator
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1235ULL);

	// generate (non-symetric (actually wrong)) system matrix ( here acceptable for benchmark purposes... )
	curandGenerateUniformDouble(gen, A.data, A.cur_proc_row_count*A.cur_proc_col_count);

	// generate manufactured RHS
	curandGenerateUniformDouble(gen, x.data, x.cur_proc_row_count);
	mpla_generic_dgemv(&b, (struct mpla_generic_matrix*)&A, &x, mpla_dgemv_core, &instance);

	// fill initial solution vector by random numbers
	curandGenerateUniformDouble(gen, x.data, x.cur_proc_row_count);

	// store initial solution vector in x_tmp
	cudaMemcpy(x_tmp.data, x.data, sizeof(double)*x.cur_proc_row_count, cudaMemcpyDeviceToDevice);

	// setup of time measurements
	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// solve system
	cudaEventRecord(start, 0);
	for (int t=0; t<trials; t++)
	{
		mpla_generic_conjugate_gradient(&b, (struct mpla_generic_matrix*)&A, &x, 50, 1.0e-15, mpla_dgemv_core, &instance);
		cudaMemcpy(x.data, x_tmp.data, sizeof(double)*x.cur_proc_row_count, cudaMemcpyDeviceToDevice);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// evaluate time
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("Time spent: %.10f\n", gpu_time/1000);

	checkCUDAError("bla");

	mpla_free_matrix(&A, &instance);
	mpla_free_vector(&x, &instance);
	mpla_free_vector(&b, &instance);

	MPI_Finalize();

}

