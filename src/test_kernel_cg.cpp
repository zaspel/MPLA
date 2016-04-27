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

#include "kernel.h"  // for now a dirty hack!!!

int idx(int i, int j, int m, int n)
{
	return j*m + i;
}

int calculate_max_row_count(int rows, int cols)
{
	size_t free_bytes;
	size_t total_bytes;
	cudaMemGetInfo( &free_bytes, &total_bytes );
	size_t free_double_values = free_bytes / 8;

	return (double)(free_double_values / cols) * 0.95;
}

int main(int argc, char* argv[])
{
	int n;
	n=atoi(argv[1]);
	
	int iter_max = 1000;
	if (argc==3)
		iter_max = atoi(argv[2]);

	void (*mpla_dgemv_core)(struct mpla_vector*, struct mpla_generic_matrix*, struct mpla_vector*, struct mpla_instance*);
	mpla_dgemv_core = &mpla_dgemv_core_kernel_streamed_cublas;

/*	double* A_h = new double[n*n];
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
*/
	MPI_Init( &argc, &argv );

	// init MPLA
	struct mpla_instance instance;
	mpla_init_instance(&instance, MPI_COMM_WORLD);

	// init matrix and vectors
	struct mpla_generic_matrix A;
	mpla_init_generic_matrix(&A, &instance, n, n);

	struct mpla_vector x;
	mpla_init_vector(&x, &instance, n);

	struct mpla_vector b;
	mpla_init_vector(&b, &instance, n);

        // set up GPU random number generator
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1235ULL);

/*	// fill matrix
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
*/
	double** points_ptr_on_host;
	points_ptr_on_host = new double*[3];
	for (int d=0; d<3; d++)
	{
		cudaMalloc((void**)&(points_ptr_on_host[d]), sizeof(double)*n);
		curandGenerateUniformDouble(gen, points_ptr_on_host[d], n); // warning: generating pseudo-random numbers this is why they are all identical  on all machines; however this is an implicit assumption !!!!
	}
	double** points;
	cudaMalloc((void**)&points, sizeof(double*)*3);
	cudaMemcpy(points, points_ptr_on_host, sizeof(double*)*3, cudaMemcpyHostToDevice);

	struct kernel_matrix_data matrix_data;
	matrix_data.points = points;
	matrix_data.max_row_count_per_dgemv = calculate_max_row_count( A.cur_proc_row_count, A.cur_proc_col_count);

	A.data = (void*)&matrix_data;

	// generate manufactured RHS
	curandGenerateUniformDouble(gen, x.data, x.cur_proc_row_count);
	mpla_generic_dgemv(&b, &A, &x, mpla_dgemv_core, &instance);

	// fill initial solution vector by random numbers
	curandGenerateUniformDouble(gen, x.data, x.cur_proc_row_count);
	
	// solve system
	mpla_generic_conjugate_gradient(&b, &A, &x, iter_max, 1.0e-10, mpla_dgemv_core, &instance);

	struct mpla_vector Ax;
	mpla_init_vector(&Ax, &instance, n);
	
	// calculate residual
	mpla_generic_dgemv(&Ax, &A, &x, mpla_dgemv_core, &instance);
	mpla_daxpy(&b, -1, &Ax, &instance);
	
	// calculate norm of residual
	double norm;
	mpla_ddot(&norm, &b, &b, &instance);
	norm = sqrt(norm/(double)n);
		
	printf("Residual norm is %e\n", norm);

	mpla_free_generic_matrix(&A, &instance);
	mpla_free_vector(&Ax, &instance);
	mpla_free_vector(&x, &instance);
	mpla_free_vector(&b, &instance);

	for (int d=0; d<3; d++)
	{
		cudaFree(points_ptr_on_host[d]);
	}
	cudaFree(points);
	delete [] points_ptr_on_host;

	MPI_Finalize();

}

