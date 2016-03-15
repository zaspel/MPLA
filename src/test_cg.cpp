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
	
	int iter_max = 1000;
	if (argc==3)
		iter_max = atoi(argv[2]);

	void (*mpla_dgemv_core)(struct mpla_vector*, struct mpla_generic_matrix*, struct mpla_vector*, struct mpla_instance*);
	mpla_dgemv_core = &mpla_generic_dgemv_core_cublas;

	double* A_h = new double[n*n];
	for (int i=0; i<n; i++)
	{
		for (int k=0; k<n; k++)
		{
			if (i==k)
				A_h[idx(i,k,n,n)] = 2;
			if (i==k-1)
				A_h[idx(i,k,n,n)] = -1;
			if (i==k+1)
				A_h[idx(i,k,n,n)] = -1;
		}
	}

	MPI_Init( &argc, &argv );

	// init MPLA
	struct mpla_instance instance;
	mpla_init_instance(&instance, MPI_COMM_WORLD);

	// init matrix and vectors
	struct mpla_matrix A;
	mpla_init_matrix(&A, &instance, n, n);

	struct mpla_vector x;
	mpla_init_vector(&x, &instance, n);

	struct mpla_vector b;
	mpla_init_vector(&b, &instance, n);

        // set up GPU random number generator
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1235ULL);

	// fill matrix
	double* Atmp = new double[A.cur_proc_row_count*A.cur_proc_col_count];
	for (int i=A.cur_proc_row_offset; i<A.cur_proc_row_offset+A.cur_proc_row_count; i++)
	{
		for (int j=A.cur_proc_col_offset; j<A.cur_proc_col_offset+A.cur_proc_col_count; j++)
		{
			Atmp[ idx(i-A.cur_proc_row_offset,j-A.cur_proc_col_offset,A.cur_proc_row_count,A.cur_proc_col_count) ] = A_h[ idx(i,j,n,n) ];
		}
	}
	cudaMemcpy(A.data, Atmp, sizeof(double)*A.cur_proc_row_count*A.cur_proc_col_count, cudaMemcpyHostToDevice);
	delete [] Atmp;
	delete [] A_h;

	// generate manufactured RHS
	curandGenerateUniformDouble(gen, x.data, x.cur_proc_row_count);
	mpla_generic_dgemv(&b, (struct mpla_generic_matrix*)&A, &x, mpla_dgemv_core, &instance);

	// fill initial solution vector by random numbers
	curandGenerateUniformDouble(gen, x.data, x.cur_proc_row_count);
	
	// solve system
	mpla_generic_conjugate_gradient(&b, (struct mpla_generic_matrix*)&A, &x, iter_max, 1.0e-10, mpla_dgemv_core, &instance);

	struct mpla_vector Ax;
	mpla_init_vector(&Ax, &instance, n);
	
	// calculate residual
	mpla_generic_dgemv(&Ax, (struct mpla_generic_matrix*)&A, &x, mpla_dgemv_core, &instance);
	mpla_daxpy(&b, -1, &Ax, &instance);
	
	// calculate norm of residual
	double norm;
	mpla_ddot(&norm, &b, &b, &instance);
	norm = sqrt(norm/(double)n);
		
	printf("Residual norm is %e\n", norm);

	mpla_free_matrix(&A, &instance);
	mpla_free_vector(&Ax, &instance);
	mpla_free_vector(&x, &instance);
	mpla_free_vector(&b, &instance);

	MPI_Finalize();

}

