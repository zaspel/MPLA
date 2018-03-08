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
#include <gsl/gsl_qrng.h>

#include <hmglib.h>
#include "hmglib_adapter.h"
#include <kernel_system_assembler.h>

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

cudaEvent_t test_start, test_stop;
float test_milliseconds;
#define TIME_test_start  {cudaEventCreate(&test_start); cudaEventCreate(&test_stop); cudaEventRecord(test_start);}
#define TIME_test_stop(a) {cudaEventRecord(test_stop); cudaEventSynchronize(test_stop); cudaEventElapsedTime(&test_milliseconds, test_start, test_stop); printf("%s: Elapsed time: %lf ms\n", a, test_milliseconds); }


int main(int argc, char* argv[])
{
	MPI_Init( &argc, &argv );

	int dim = 2;

	int n = 1000;

	int restart = 0;

	double tol = 1.0e-7;

	int iterations_until_checkpoint = 100;	

	int c_leaf=512;
	
	int k=20;
	
	double epsilon;
	
	int max_iter = 1000;

	double eta = 1.5;

	int max_batched_dense_size = (int) pow(2,24);
	int max_batched_aca_size = (int) pow(2,24);


	if (argc==1)
	{
		printf("%s <n> <dim> <tolerance> <restart> <checkpoint_iter> <k> <c_leaf> <max_iter> <eta> <max_batched_dense_size> <max_batched_aca_size>\n",argv[0]);
		exit(0);
	}
	
	n=atoi(argv[1]);
	
	if (argc>2)
		dim=atoi(argv[2]);

	if (argc>3)
		tol=atof(argv[3]);	

	if (argc>4)
		restart=atoi(argv[4]);

	if (argc>5)
		iterations_until_checkpoint = atoi(argv[5]);
	
	if (argc>6)
		k = atoi(argv[6]);

	if (argc>7)
		c_leaf = atoi(argv[7]);

	if (argc>8)
		max_iter = atoi(argv[8]);

	if (argc>9)
		eta = atof(argv[9]);

	if (argc>10)
		max_batched_dense_size = atoi(argv[10]);

	if (argc>11)
		max_batched_aca_size = atoi(argv[11]);

	int bits = (dim==2) ? 32 : 20;	


	void (*mpla_dgemv_core_hmatrix)(struct mpla_vector*, struct mpla_generic_matrix*, struct mpla_vector*, struct mpla_instance*);
	void (*mpla_dgemv_core_full)(struct mpla_vector*, struct mpla_generic_matrix*, struct mpla_vector*, struct mpla_instance*);
	mpla_dgemv_core_hmatrix = &mpla_dgemv_core_hmglib_h_matrix;
	mpla_dgemv_core_full = &mpla_dgemv_core_hmglib_full_matrix;


	// init MPLA
	struct mpla_instance instance;
	mpla_init_instance_block_rows(&instance, MPI_COMM_WORLD);

	if (instance.is_parent)
		printf("Running on %d processors.\n",instance.proc_count);

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
	double** points_host;
	points_host = new double*[dim];
	for (int d=0; d<dim; d++)
	{
		points_host[d] = new double[n];
	}


	gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_halton, dim);
	
	double* rand_pt = new double[dim];

	for (int i=0; i<n; i++)
	{
		gsl_qrng_get (q, rand_pt);
		for (int d=0; d<dim; d++)
			points_host[d][i] = rand_pt[d];
	}

	gsl_qrng_free (q);
	delete [] rand_pt;

	
	double** points_ptr_on_host[2];
	double** points[2];
	for (int i=0; i<2; i++)
	{
		points_ptr_on_host[i] = new double*[dim];
		for (int d=0; d<dim; d++)
		{
			cudaMalloc((void**)&(points_ptr_on_host[i][d]), sizeof(double)*n);
			cudaMemcpy(points_ptr_on_host[i][d], points_host[d], sizeof(double)*n, cudaMemcpyHostToDevice);
		}
		cudaMalloc((void**)&(points[i]), sizeof(double*)*dim);
		cudaMemcpy(points[i], points_ptr_on_host[i], sizeof(double*)*dim, cudaMemcpyHostToDevice);
	}

	unsigned int* point_ids_h[2];
	for (int i=0; i<2; i++)
	{
		point_ids_h[i] = new unsigned int[n];
		for (int j=0; j<n; j++)
			point_ids_h[i][j] = j;
	}

	unsigned int* point_ids_d_ptr_on_host[2];
	for (int i=0; i<2; i++)
	{
		cudaMalloc((void**)&(point_ids_d_ptr_on_host[i]), sizeof(unsigned int)*n);
		cudaMemcpy(point_ids_d_ptr_on_host[i], point_ids_h[i], sizeof(unsigned int)*n, cudaMemcpyHostToDevice);	
	}

	delete [] point_ids_h[0];
	delete [] point_ids_h[1];


	// setup data structure with hmglib data
	struct h_matrix_data data;
	A.data = (void*)&data;
	double** global_coords[2];
	global_coords[0] = points_ptr_on_host[0];
	global_coords[1] = points_ptr_on_host[1];
	int global_point_count[2];
	global_point_count[0] = n;
	global_point_count[1] = n;

        // setup kernel matrix assembler
        double regularization = 0.0;
        struct gaussian_kernel_system_assembler assem;
        struct gaussian_kernel_system_assembler** assem_d_p;
        cudaMalloc((void***)&assem_d_p, sizeof(struct gaussian_kernel_system_assembler*));
        create_gaussian_kernel_system_assembler_object(assem_d_p, regularization);
        struct gaussian_kernel_system_assembler* assem_d;
        cudaMemcpy(&assem_d, assem_d_p, sizeof(struct gaussian_kernel_system_assembler*), cudaMemcpyDeviceToHost);

	// setup H matrix
	if (instance.is_parent)
		TIME_test_start;
	mpla_init_hmglib(&A, global_point_count, global_coords, point_ids_d_ptr_on_host, assem_d, eta, dim, bits, c_leaf, k, max_batched_dense_size, max_batched_aca_size, &instance);
	MPI_Barrier(instance.comm);
	if (instance.is_parent)
		TIME_test_stop("Setup time");
 
	// generate RHS
	set_gaussian_kernel_rhs(b.data, &data);

	// fill initial solution vector by random numbers
	curandGenerateUniformDouble(gen, x.data, x.cur_proc_row_count);
	
	// setup of time measurements
	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int d=0; d<dim; d++)
		delete [] points_host[d];
	delete [] points_host;


//	// solve system
//	cudaEventRecord(start, 0);

	if (instance.is_parent)
		TIME_test_start;
	mpla_generic_conjugate_gradient_with_checkpoint_restart(&b, &A, &x, max_iter, tol, mpla_dgemv_core_hmatrix, iterations_until_checkpoint, restart, &instance);

	MPI_Barrier(instance.comm);
	if (instance.is_parent)
		TIME_test_stop("Solve time");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

//	// evaluate time
//	cudaEventElapsedTime(&gpu_time, start, stop);
//	printf("Time spent: %.10f\n", gpu_time/1000);

	char filename[100];
	sprintf(filename, "weights.dat");

	mpla_save_vector(&x, filename, &instance);
	

	checkCUDAError("bla");

        // cleanup of assembler
        destroy_gaussian_kernel_system_assembler_object(assem_d_p);;
        cudaFree(assem_d_p);


	mpla_destroy_hmglib(&A, &instance);
	mpla_free_generic_matrix(&A, &instance);
	mpla_free_vector(&x, &instance);
	mpla_free_vector(&x_tmp, &instance);
	mpla_free_vector(&b, &instance);

	for (int i=0; i<2; i++)
	{
		for (int d=0; d<dim; d++)
		{
			cudaFree(points_ptr_on_host[i][d]);
		}
		cudaFree(points[i]);
		delete [] points_ptr_on_host[i];

		cudaFree(point_ids_d_ptr_on_host[i]);
	}

	mpla_destroy_instance(&instance);
	
	MPI_Finalize();

	curandDestroyGenerator(gen);


	cudaDeviceReset();
}

