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

int main(int argc, char* argv[])
{
	int m,n;
	m=atoi(argv[1]);
	n=atoi(argv[2]);
	
	double* A_h = new double[m*n];
	for (int i=0; i<m*n; i++)
		A_h[i] = i;

	double* x_h = new double[n];
	for (int i=0; i<n; i++)
		x_h[i] = i;

	double* Ax_h = new double[m];
	for (int i=0; i<m; i++)
	{
		Ax_h[i]=0;
		for (int j=0; j<n; j++)
		{
			Ax_h[i] += A_h[ idx(i,j,m,n) ] * x_h[j];
		}		
	}

	
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

	// fill vector
	cudaMemcpy(x.data, &(x_h[x.cur_proc_row_offset]), sizeof(double)*x.cur_proc_row_count, cudaMemcpyHostToDevice);
	checkCUDAError("memcpy");
	
	
	// fill matrix
	double* Atmp = new double[A.cur_proc_row_count*A.cur_proc_col_count];
	for (int i=A.cur_proc_row_offset; i<A.cur_proc_row_offset+A.cur_proc_row_count; i++)
	{
		for (int j=A.cur_proc_col_offset; j<A.cur_proc_col_offset+A.cur_proc_col_count; j++)
		{
			Atmp[ idx(i-A.cur_proc_row_offset,j-A.cur_proc_col_offset,A.cur_proc_row_count,A.cur_proc_col_count) ] = A_h[ idx(i,j,m,n) ];
		}
	}
	cudaMemcpy(A.data, Atmp, sizeof(double)*A.cur_proc_row_count*A.cur_proc_col_count, cudaMemcpyHostToDevice);
	delete [] Atmp;

	int trials = 100;	
	if (argc==4)
		trials = atoi(argv[3]);

	for (int n = 1; n<trials; n++) 
	{
		// calculate MVP
		mpla_dgemv(&Ax, &A, &x, &instance);
	}

	// retrieve data from GPU
	double* Ax_cur_proc_from_GPU = new double[Ax.cur_proc_row_count];
	cudaMemcpy(Ax_cur_proc_from_GPU, Ax.data, sizeof(double)*Ax.cur_proc_row_count, cudaMemcpyDeviceToHost);
	
	for (int i=0; i<Ax.cur_proc_row_count; i++)
	{
		if ((fabs(Ax_h[Ax.cur_proc_row_offset+i] - Ax_cur_proc_from_GPU[i]))/fabs(Ax_h[Ax.cur_proc_row_offset+i])>1.0e-12)
		{
			printf("Results do not match!\n"); fflush(stdout);
			exit(1);
		}
	}
	printf("Results match\n");
	
	
	delete [] Ax_cur_proc_from_GPU;
	

	mpla_free_matrix(&A, &instance);
	mpla_free_vector(&Ax, &instance);
	mpla_free_vector(&x, &instance);

	delete [] A_h;
	delete [] x_h;
	delete [] Ax_h;

	MPI_Finalize();

}

