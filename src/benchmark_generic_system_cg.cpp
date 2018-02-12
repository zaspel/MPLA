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

//#include "kernel.h"  // for now a dirty hack!!!

#include "generic_system_adapter.h"
#include "kernel_system_assembly.h"

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

	int trials = 100;
	
	if (argc==3)
		trials = atoi(argv[2]);

	void (*mpla_dgemv_core)(struct mpla_vector*, struct mpla_generic_matrix*, struct mpla_vector*, struct mpla_instance*);
//	mpla_dgemv_core = &mpla_dgemv_core_generic_system_matrix_cublas;
	mpla_dgemv_core = &mpla_dgemv_core_generic_system_matrix_streamed_cublas;

	MPI_Init( &argc, &argv );

	// init MPLA
	struct mpla_instance instance;
	mpla_init_instance(&instance, MPI_COMM_WORLD);

	// init matrix and vectors
	struct mpla_generic_matrix A;
	mpla_init_generic_matrix(&A, &instance, n, n);

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

	// generate random points	
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

	// setup data structure with kernel matrix data
	struct gaussian_kernel_system_assembly assem;
	assem.points = points;
	assem.max_row_count_per_dgemv = calculate_max_row_count( A.cur_proc_row_count, A.cur_proc_col_count);
	assem.dim = 3;
	struct gaussian_kernel_system_assembly** assem_d_p;
	cudaMalloc((void***)&assem_d_p, sizeof(struct gaussian_kernel_system_assembly*));
//	cudaMemcpy(assem_d, &assem, sizeof(struct gaussian_kernel_system_assembly), cudaMemcpyHostToDevice);
	
	create_gaussian_kernel_system_assembly_object(assem_d_p, assem.points, assem.max_row_count_per_dgemv, assem.dim);

	struct gaussian_kernel_system_assembly* assem_d;
	cudaMemcpy(&assem_d, assem_d_p, sizeof(struct gaussian_kernel_system_assembly*), cudaMemcpyDeviceToHost);

	A.data = (void*)assem_d;

	// generate manufactured RHS
	curandGenerateUniformDouble(gen, x.data, x.cur_proc_row_count);

	mpla_generic_dgemv(&b, &A, &x, mpla_dgemv_core, &instance);

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
		mpla_generic_conjugate_gradient(&b, &A, &x, 50, 1.0e-15, mpla_dgemv_core, &instance);
		cudaMemcpy(x.data, x_tmp.data, sizeof(double)*x.cur_proc_row_count, cudaMemcpyDeviceToDevice);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// evaluate time
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("Time spent: %.10f\n", gpu_time/1000);

	checkCUDAError("bla");

	destroy_gaussian_kernel_system_assembly_object(assem_d_p);
	cudaThreadSynchronize();
	checkCUDAError("destroy_gaussian_kernel_system_assembly_object");

	cudaFree(assem_d_p);

//	cudaFree(assem_d);	

	mpla_free_generic_matrix(&A, &instance);
	mpla_free_vector(&x, &instance);
	mpla_free_vector(&x_tmp, &instance);
	mpla_free_vector(&b, &instance);

	for (int d=0; d<3; d++)
	{
		cudaFree(points_ptr_on_host[d]);
	}
	cudaFree(points);
	delete [] points_ptr_on_host;

	MPI_Finalize();

}

