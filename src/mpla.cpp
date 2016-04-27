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
#include <stdio.h> 
#include "cublas_v2.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mpi.h>

void info()
{
	printf("Hello world\n");
}



void mpla_init_instance(struct mpla_instance* instance, MPI_Comm comm)
{
	instance->comm = comm;	

	// get number of process
	MPI_Comm_size(comm, &(instance->proc_count));

	// find number of current process
	MPI_Comm_rank(comm, &(instance->cur_proc_rank));
	if (instance->cur_proc_rank==0)
		instance->is_parent = true;
	else
		instance->is_parent = false;

	// compute the process grid
	int dims[2]; 
	dims[0]=dims[1]=0;
	MPI_Dims_create(instance->proc_count, 2, dims);
	instance->proc_rows = dims[0];
	instance->proc_cols = dims[1];

	// create cartesian communicator and retrieve cartesian coordinates
	int periods[2];
	periods[0]=periods[1]=0;
	MPI_Cart_create(comm, 2, dims, periods, 0, &(instance->comm));
	int cur_proc_coord[2];
	MPI_Cart_get(instance->comm, 2, dims, periods, cur_proc_coord);
	instance->cur_proc_row = cur_proc_coord[0];
	instance->cur_proc_col = cur_proc_coord[1];

	cublasCreate(&(instance->cublas_handle));
	
}

void mpla_init_matrix(struct mpla_matrix* matrix, struct mpla_instance* instance, int mat_row_count, int mat_col_count)
{
	// setting global matrix dimensions
	matrix->mat_row_count = mat_row_count;
	matrix->mat_col_count = mat_col_count;

	// allocating memory for process-wise matrix information
	matrix->proc_row_count = new int*[instance->proc_rows];
	matrix->proc_col_count = new int*[instance->proc_rows];
	matrix->proc_row_offset = new int*[instance->proc_rows];
	matrix->proc_col_offset = new int*[instance->proc_rows];
	for (int i=0; i<instance->proc_rows; i++)
	{
		matrix->proc_row_count[i] = new int[instance->proc_cols];
		matrix->proc_col_count[i] = new int[instance->proc_cols];
		matrix->proc_row_offset[i] = new int[instance->proc_cols];
		matrix->proc_col_offset[i] = new int[instance->proc_cols];
	}

/*
	// computing general row block sizes
	int filled_row_block_size = ceil((float)mat_row_count / (float)(instance->proc_rows));
//	int filled_row_block_count = mat_row_count / filled_row_block_size;
	int last_row_block_size =  mat_row_count % filled_row_block_size;

	// computing general column block sizes
	int filled_col_block_size = ceil((float)mat_col_count / (float)(instance->proc_cols));
//	int filled_col_block_count = mat_col_count / filled_col_block_size;
	int last_col_block_size =  mat_col_count % filled_col_block_size;


	// computing process-wise block row / column counts
	for (int i=0; i < instance->proc_rows; i++)
	{
		for (int j=0; j < instance->proc_cols; j++)
		{
			if ((i==(instance->proc_rows-1)) && (last_row_block_size>0)) // handling last row block which is only partially filled
				matrix->proc_row_count[i][j] = last_row_block_size;
			else
				matrix->proc_row_count[i][j] = filled_row_block_size;
	
			if ((j==(instance->proc_cols-1)) && (last_col_block_size>0)) // handling last column block which is only partially filled
				matrix->proc_col_count[i][j] = last_col_block_size;
			else
				matrix->proc_col_count[i][j] = filled_col_block_size;
		}
	}
*/
	
	// computing general row block sizes
	int almost_filled_row_block_size = mat_row_count / instance->proc_rows;
	int remaining_rows = mat_row_count % instance->proc_rows;
	if (almost_filled_row_block_size == 0)
	{
		printf("MPLA: There are more process block rows than matrix rows. Exiting...\n");
		exit(1);
	}


	// computing general column block sizes
	int almost_filled_col_block_size = mat_col_count / instance->proc_cols;
	int remaining_cols = mat_col_count % instance->proc_cols;

	if (almost_filled_row_block_size == 0)
	{
		printf("MPLA: There are more process block columns than matrix columns. Exiting...\n");
		exit(1);
	}


	// computing process-wise block row / column counts
	for (int i=0; i< instance->proc_rows; i++)
	{
		for (int j=0; j<instance->proc_cols; j++)
		{
			matrix->proc_row_count[i][j] = almost_filled_row_block_size + ( (i<remaining_rows) ? 1 : 0 );
			matrix->proc_col_count[i][j] = almost_filled_col_block_size + ( (j<remaining_cols) ? 1 : 0 );
		}
	}
	



	// computing process-wise block row / column offsets
	matrix->proc_row_offset[0][0] = 0;
	matrix->proc_col_offset[0][0] = 0;
	for (int i=1; i<instance->proc_rows; i++)
		matrix->proc_col_offset[i][0] = 0;
	for (int j=1; j<instance->proc_cols; j++)
		matrix->proc_row_offset[0][j] = 0;
	for (int i=1; i < instance->proc_rows; i++)
		for (int j=0; j < instance->proc_cols; j++)
			matrix->proc_row_offset[i][j] = matrix->proc_row_offset[i-1][j] + matrix->proc_row_count[i-1][j];
	for (int j=1; j < instance->proc_cols; j++)
		for (int i=0; i < instance->proc_rows; i++)
			matrix->proc_col_offset[i][j] = matrix->proc_col_offset[i][j-1] + matrix->proc_col_count[i][j-1];
		
	// retrieving local data for the current process
	matrix->cur_proc_row_count = matrix->proc_row_count[instance->cur_proc_row][instance->cur_proc_col];
	matrix->cur_proc_col_count = matrix->proc_col_count[instance->cur_proc_row][instance->cur_proc_col];
	matrix->cur_proc_row_offset = matrix->proc_row_offset[instance->cur_proc_row][instance->cur_proc_col];
	matrix->cur_proc_col_offset = matrix->proc_col_offset[instance->cur_proc_row][instance->cur_proc_col];

	// allocating matrix data storage
	cudaMalloc((void**)&(matrix->data), sizeof(double)*matrix->cur_proc_row_count*matrix->cur_proc_col_count);
	cudaThreadSynchronize();
	checkCUDAError("cudaMalloc");
}

void mpla_init_generic_matrix(struct mpla_generic_matrix* matrix, struct mpla_instance* instance, int mat_row_count, int mat_col_count)
{
	// setting global matrix dimensions
	matrix->mat_row_count = mat_row_count;
	matrix->mat_col_count = mat_col_count;

	// allocating memory for process-wise matrix information
	matrix->proc_row_count = new int*[instance->proc_rows];
	matrix->proc_col_count = new int*[instance->proc_rows];
	matrix->proc_row_offset = new int*[instance->proc_rows];
	matrix->proc_col_offset = new int*[instance->proc_rows];
	for (int i=0; i<instance->proc_rows; i++)
	{
		matrix->proc_row_count[i] = new int[instance->proc_cols];
		matrix->proc_col_count[i] = new int[instance->proc_cols];
		matrix->proc_row_offset[i] = new int[instance->proc_cols];
		matrix->proc_col_offset[i] = new int[instance->proc_cols];
	}

/*
	// computing general row block sizes
	int filled_row_block_size = ceil((float)mat_row_count / (float)(instance->proc_rows));
//	int filled_row_block_count = mat_row_count / filled_row_block_size;
	int last_row_block_size =  mat_row_count % filled_row_block_size;

	// computing general column block sizes
	int filled_col_block_size = ceil((float)mat_col_count / (float)(instance->proc_cols));
//	int filled_col_block_count = mat_col_count / filled_col_block_size;
	int last_col_block_size =  mat_col_count % filled_col_block_size;


	// computing process-wise block row / column counts
	for (int i=0; i < instance->proc_rows; i++)
	{
		for (int j=0; j < instance->proc_cols; j++)
		{
			if ((i==(instance->proc_rows-1)) && (last_row_block_size>0)) // handling last row block which is only partially filled
				matrix->proc_row_count[i][j] = last_row_block_size;
			else
				matrix->proc_row_count[i][j] = filled_row_block_size;
	
			if ((j==(instance->proc_cols-1)) && (last_col_block_size>0)) // handling last column block which is only partially filled
				matrix->proc_col_count[i][j] = last_col_block_size;
			else
				matrix->proc_col_count[i][j] = filled_col_block_size;
		}
	}
*/
	
	// computing general row block sizes
	int almost_filled_row_block_size = mat_row_count / instance->proc_rows;
	int remaining_rows = mat_row_count % instance->proc_rows;
	if (almost_filled_row_block_size == 0)
	{
		printf("MPLA: There are more process block rows than matrix rows. Exiting...\n");
		exit(1);
	}


	// computing general column block sizes
	int almost_filled_col_block_size = mat_col_count / instance->proc_cols;
	int remaining_cols = mat_col_count % instance->proc_cols;

	if (almost_filled_row_block_size == 0)
	{
		printf("MPLA: There are more process block columns than matrix columns. Exiting...\n");
		exit(1);
	}


	// computing process-wise block row / column counts
	for (int i=0; i< instance->proc_rows; i++)
	{
		for (int j=0; j<instance->proc_cols; j++)
		{
			matrix->proc_row_count[i][j] = almost_filled_row_block_size + ( (i<remaining_rows) ? 1 : 0 );
			matrix->proc_col_count[i][j] = almost_filled_col_block_size + ( (j<remaining_cols) ? 1 : 0 );
		}
	}
	



	// computing process-wise block row / column offsets
	matrix->proc_row_offset[0][0] = 0;
	matrix->proc_col_offset[0][0] = 0;
	for (int i=1; i<instance->proc_rows; i++)
		matrix->proc_col_offset[i][0] = 0;
	for (int j=1; j<instance->proc_cols; j++)
		matrix->proc_row_offset[0][j] = 0;
	for (int i=1; i < instance->proc_rows; i++)
		for (int j=0; j < instance->proc_cols; j++)
			matrix->proc_row_offset[i][j] = matrix->proc_row_offset[i-1][j] + matrix->proc_row_count[i-1][j];
	for (int j=1; j < instance->proc_cols; j++)
		for (int i=0; i < instance->proc_rows; i++)
			matrix->proc_col_offset[i][j] = matrix->proc_col_offset[i][j-1] + matrix->proc_col_count[i][j-1];
		
	// retrieving local data for the current process
	matrix->cur_proc_row_count = matrix->proc_row_count[instance->cur_proc_row][instance->cur_proc_col];
	matrix->cur_proc_col_count = matrix->proc_col_count[instance->cur_proc_row][instance->cur_proc_col];
	matrix->cur_proc_row_offset = matrix->proc_row_offset[instance->cur_proc_row][instance->cur_proc_col];
	matrix->cur_proc_col_offset = matrix->proc_col_offset[instance->cur_proc_row][instance->cur_proc_col];

	// no matrix data storage allocation, since the user controls the content of the custom data
}

void mpla_init_vector(struct mpla_vector* vector, struct mpla_instance* instance, int vec_row_count)
{
	// setting global vector size
	vector->vec_row_count = vec_row_count;

	// allocating memory for process-wise vector information
	vector->proc_row_count = new int*[instance->proc_rows];
	vector->proc_row_offset = new int*[instance->proc_rows];
	for (int i=0; i<instance->proc_rows; i++)
	{
		vector->proc_row_count[i] = new int[instance->proc_cols];
		vector->proc_row_offset[i] = new int[instance->proc_cols];
	}

	// computing general row block sizes
	int almost_filled_row_block_size = vec_row_count / instance->proc_rows;
	int remaining_rows = vec_row_count % instance->proc_rows;

	if (almost_filled_row_block_size == 0)
	{
		printf("MPLA: There are more process block rows than vector rows. Exiting...\n");
		exit(1);
	}


	// computing process-wise block row / column counts
	for (int i=0; i< instance->proc_rows; i++)
	{
		for (int j=0; j<instance->proc_cols; j++)
		{
			vector->proc_row_count[i][j] = almost_filled_row_block_size + ( (i<remaining_rows) ? 1 : 0 );
		}
	}

	// computing process-wise block row / column offsets
	vector->proc_row_offset[0][0] = 0;
	for (int j=1; j < instance->proc_cols; j++)
		vector->proc_row_offset[0][j] = 0;	
	for (int i=1; i < instance->proc_rows; i++)
		for (int j=0; j < instance->proc_cols; j++)
			vector->proc_row_offset[i][j] = vector->proc_row_offset[i-1][j] + vector->proc_row_count[i-1][j];
		
	// retrieving local data for the current process
	vector->cur_proc_row_count = vector->proc_row_count[instance->cur_proc_row][instance->cur_proc_col];
	vector->cur_proc_row_offset = vector->proc_row_offset[instance->cur_proc_row][instance->cur_proc_col];

	// allocating matrix data storage
	cudaMalloc((void**)&(vector->data), sizeof(double)*vector->cur_proc_row_count);
	cudaThreadSynchronize();
	checkCUDAError("cudaMalloc");
}

void mpla_init_vector_for_block_rows(struct mpla_vector* vector, struct mpla_instance* instance, int vec_row_count)
{
	// setting global vector size
	vector->vec_row_count = vec_row_count;

	// allocating memory for process-wise vector information
	vector->proc_row_count = new int*[instance->proc_rows];
	vector->proc_row_offset = new int*[instance->proc_rows];
	for (int i=0; i<instance->proc_rows; i++)
	{
		vector->proc_row_count[i] = new int[instance->proc_cols];
		vector->proc_row_offset[i] = new int[instance->proc_cols];
	}

	// computing general row block sizes
	int almost_filled_row_block_size = vec_row_count / instance->proc_cols;
	int remaining_rows = vec_row_count % instance->proc_cols;

	if (almost_filled_row_block_size == 0)
	{
		printf("MPLA: There are more process block columns than matrix columns. Exiting...\n");
		exit(1);
	}


	// computing process-wise block row / column counts
	for (int i=0; i< instance->proc_rows; i++)
	{
		for (int j=0; j<instance->proc_cols; j++)
		{
			vector->proc_row_count[i][j] = almost_filled_row_block_size + ( (j<remaining_rows) ? 1 : 0 );
		}
	}
	

/*

	// computing general row block sizes
	int filled_row_block_size = ceil((float)vec_row_count / (float)(instance->proc_cols));
//	int filled_row_block_count = vec_row_count / filled_row_block_size;
	int last_row_block_size =  vec_row_count % filled_row_block_size;

	// computing process-wise block row / column counts
	for (int i=0; i < instance->proc_rows; i++)
	{
		for (int j=0; j < instance->proc_cols; j++)
		{
			if ((j==(instance->proc_cols-1)) && (last_row_block_size>0)) // handling last row block which is only partially filled
				vector->proc_row_count[i][j] = last_row_block_size;
			else
				vector->proc_row_count[i][j] = filled_row_block_size;
		}
	}
*/
	// computing process-wise block row / column offsets
	vector->proc_row_offset[0][0] = 0;
	for (int i=1; i < instance->proc_rows; i++)
		vector->proc_row_offset[i][0] = 0;
	for (int j=1; j < instance->proc_cols; j++)
		for (int i=0; i < instance->proc_rows; i++)
			vector->proc_row_offset[i][j] = vector->proc_row_offset[i][j-1] + vector->proc_row_count[i][j-1];
	
	// retrieving local data for the current process
	vector->cur_proc_row_count = vector->proc_row_count[instance->cur_proc_row][instance->cur_proc_col];
	vector->cur_proc_row_offset = vector->proc_row_offset[instance->cur_proc_row][instance->cur_proc_col];

	// allocating matrix data storage
	cudaMalloc((void**)&(vector->data), sizeof(double)*vector->cur_proc_row_count);
	cudaThreadSynchronize();
	checkCUDAError("cudaMalloc");
}



void mpla_redistribute_vector_for_dgesv(struct mpla_vector* b_redist, struct mpla_vector* b, struct mpla_matrix* A, struct mpla_instance* instance)
{
	// attention: this code does no correctness check for the input data



//	b_redist->vec_row_count = b->vec_row_count;
//
//	// allocating memory for process-wise vector information
//	vector->proc_row_count = new int*[instance->proc_rows];
//	vector->proc_row_offset = new int*[instance->proc_rows];
//	for (int i=0; i<instance->proc_rows; i++)
//	{
//		b_redist->proc_row_count[i] = new int[instance->proc_cols];
//		b_redist->proc_row_offset[i] = new int[instance->proc_cols];
//	}
//
//	// set sizes of 
//	for (int i=0; i<instance->proc_rows; i++)
//	{
//		for (int j=0; j<instance->proc_cols; j++)
//		{
//			b_redist->proc_row_count[i][j] = A->proc_col_count[i][j];
//			b_redist->proc_row_offset[i][j] = A->proc_col_offset[i][j];
//		}
//	}
//
//	// retrieving local data for current process
//	b_redist->cur_proc_row_count = A->cur_proc_col_count;
//	b_redist->cur_proc_row_offset = A->cur_proc_col_offset;
//
//	// allocating temporary vector storage
//	cudaMalloc((void*)&(b_redist->data), sizeof(double)*b_redist->cur_proc_row_count);

	// WARNING: The following code is not efficient for a strong parallelization !!!!!


	// create sub-communicator for each process column
	int remain_dims[2];
	remain_dims[0]=1;
	remain_dims[1]=0;
	MPI_Comm column_comm;
	MPI_Cart_sub(instance->comm, remain_dims, &column_comm);
	int column_rank;
	MPI_Comm_rank(column_comm, &column_rank);
	
	// columnwise creation of the full vector
	double* full_vector;
	int* recvcounts = new int[instance->proc_rows];
	int* displs = new int[instance->proc_rows];
	for (int i=0; i<instance->proc_rows; i++)
	{
		recvcounts[i] = b->proc_row_count[i][instance->cur_proc_col];
		displs[i] = b->proc_row_offset[i][instance->cur_proc_col];
	}
	cudaMalloc((void**)&full_vector, sizeof(double)*b->vec_row_count);
	cudaThreadSynchronize();
	checkCUDAError("cudaMalloc");
	MPI_Allgatherv(b->data, b->cur_proc_row_count, MPI_DOUBLE, full_vector, recvcounts, displs, MPI_DOUBLE, column_comm);

	// extract column-wise local part of full vector
	cudaMemcpy(b_redist->data, &(full_vector[b_redist->cur_proc_row_offset]), sizeof(double)*b_redist->cur_proc_row_count, cudaMemcpyDeviceToDevice);

	// memory cleanup
	cudaFree(full_vector);
}

void mpla_redistribute_vector_for_generic_dgesv(struct mpla_vector* b_redist, struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_instance* instance)
{
	// attention: this code does no correctness check for the input data


	// WARNING: The following code is not efficient for a strong parallelization !!!!!


	// create sub-communicator for each process column
	int remain_dims[2];
	remain_dims[0]=1;
	remain_dims[1]=0;
	MPI_Comm column_comm;
	MPI_Cart_sub(instance->comm, remain_dims, &column_comm);
	int column_rank;
	MPI_Comm_rank(column_comm, &column_rank);
	
	// columnwise creation of the full vector
	double* full_vector;
	int* recvcounts = new int[instance->proc_rows];
	int* displs = new int[instance->proc_rows];
	for (int i=0; i<instance->proc_rows; i++)
	{
		recvcounts[i] = b->proc_row_count[i][instance->cur_proc_col];
		displs[i] = b->proc_row_offset[i][instance->cur_proc_col];
	}
	cudaMalloc((void**)&full_vector, sizeof(double)*b->vec_row_count);
	cudaThreadSynchronize();
	checkCUDAError("cudaMalloc");
	MPI_Allgatherv(b->data, b->cur_proc_row_count, MPI_DOUBLE, full_vector, recvcounts, displs, MPI_DOUBLE, column_comm);

	// extract column-wise local part of full vector
	cudaMemcpy(b_redist->data, &(full_vector[b_redist->cur_proc_row_offset]), sizeof(double)*b_redist->cur_proc_row_count, cudaMemcpyDeviceToDevice);

	// memory cleanup
	cudaFree(full_vector);
}




void mpla_free_vector(struct mpla_vector* x, struct mpla_instance* instance)
{
	cudaFree(x->data);
	for (int i=0; i<instance->proc_rows; i++)
	{
		delete [] x->proc_row_count[i];
		delete [] x->proc_row_offset[i];
	}
	delete [] x->proc_row_count;
	delete [] x->proc_row_offset;
}

void mpla_free_matrix(struct mpla_matrix* A, struct mpla_instance* instance)
{
	cudaFree(A->data);
	for (int i=0; i<instance->proc_rows; i++)
	{
		delete [] A->proc_row_count[i];
		delete [] A->proc_row_offset[i];
		delete [] A->proc_col_count[i];
		delete [] A->proc_col_offset[i];
	}
	delete [] A->proc_row_count;
	delete [] A->proc_row_offset;
	delete [] A->proc_col_count;
	delete [] A->proc_col_offset;
}

void mpla_free_generic_matrix(struct mpla_generic_matrix* A, struct mpla_instance* instance)
{
	for (int i=0; i<instance->proc_rows; i++)
	{
		delete [] A->proc_row_count[i];
		delete [] A->proc_row_offset[i];
		delete [] A->proc_col_count[i];
		delete [] A->proc_col_offset[i];
	}
	delete [] A->proc_row_count;
	delete [] A->proc_row_offset;
	delete [] A->proc_col_count;
	delete [] A->proc_col_offset;
}

void mpla_dgemv(struct mpla_vector* b, struct mpla_matrix* A, struct mpla_vector* x, struct mpla_instance* instance)
{
	double one = 1;
	double zero = 0;

	// allocate redistributed vector
	struct mpla_vector x_redist;
	mpla_init_vector_for_block_rows(&x_redist, instance, x->vec_row_count);
	
	// redistribute input vector with row-block parallel distribution to column-block parallel distribution
	mpla_redistribute_vector_for_dgesv(&x_redist, x, A, instance);
		
	// computation core: matrix-vector product
	cublasDgemv((instance->cublas_handle), CUBLAS_OP_N, A->cur_proc_row_count, A->cur_proc_col_count, &one, A->data, A->cur_proc_row_count, x_redist.data, 1, &zero, b->data, 1);

	// create sub-communicator for each process row
	int remain_dims[2];
	remain_dims[0]=0;
	remain_dims[1]=1;
	MPI_Comm row_comm;
	MPI_Cart_sub(instance->comm, remain_dims, &row_comm);

	// summation of block row results
	double* sum;
	cudaMalloc((void**)&sum, sizeof(double)*b->cur_proc_row_count);
	cudaThreadSynchronize();
	checkCUDAError("cudaMalloc");
	MPI_Allreduce(b->data, sum, b->cur_proc_row_count, MPI_DOUBLE, MPI_SUM, row_comm);
	cudaMemcpy(b->data, sum, sizeof(double)*b->cur_proc_row_count, cudaMemcpyDeviceToDevice);

	// cleanup
	cudaFree(sum);
	mpla_free_vector(&x_redist, instance);
}

void mpla_ddot(double* xy, struct mpla_vector* x, struct mpla_vector* y, struct mpla_instance* instance)
{
	// compute process-wise dot product
	double xy_tmp;
	cublasDdot(instance->cublas_handle, x->cur_proc_row_count, x->data, 1, y->data, 1, &xy_tmp);

	// create sub-communicator for each process column
	int remain_dims[2];
	remain_dims[0]=1;
	remain_dims[1]=0;
	MPI_Comm column_comm;
	MPI_Cart_sub(instance->comm, remain_dims, &column_comm);

	// parallel summation and communication
	MPI_Allreduce(&xy_tmp, xy, 1, MPI_DOUBLE, MPI_SUM, column_comm);
}

void mpla_generic_dgemv(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x, void (*mpla_dgemv_core)(struct mpla_vector*, struct mpla_generic_matrix*, struct mpla_vector*, struct mpla_instance*), struct mpla_instance* instance)
{
	// allocate redistributed vector
	struct mpla_vector x_redist;
	mpla_init_vector_for_block_rows(&x_redist, instance, x->vec_row_count);
	
	// redistribute input vector with row-block parallel distribution to column-block parallel distribution
	mpla_redistribute_vector_for_generic_dgesv(&x_redist, x, A, instance);
		
	// generic computation core: matrix-vector product
	mpla_dgemv_core(b, A, &x_redist, instance);

	// create sub-communicator for each process row
	int remain_dims[2];
	remain_dims[0]=0;
	remain_dims[1]=1;
	MPI_Comm row_comm;
	MPI_Cart_sub(instance->comm, remain_dims, &row_comm);

	// summation of block row results
	double* sum;
	cudaMalloc((void**)&sum, sizeof(double)*b->cur_proc_row_count);
	cudaThreadSynchronize();
	checkCUDAError("cudaMalloc");
	MPI_Allreduce(b->data, sum, b->cur_proc_row_count, MPI_DOUBLE, MPI_SUM, row_comm);
	cudaMemcpy(b->data, sum, sizeof(double)*b->cur_proc_row_count, cudaMemcpyDeviceToDevice);

	// cleanup
	cudaFree(sum);
	mpla_free_vector(&x_redist, instance);
}

void mpla_daxpy(struct mpla_vector* y, double alpha, struct mpla_vector* x, struct mpla_instance* instance)
{
	// compute process-wise axpy
	cublasDaxpy(instance->cublas_handle, x->cur_proc_row_count, &alpha, x->data, 1, y->data, 1);
}

void mpla_vector_set_zero(struct mpla_vector* x, struct mpla_instance* instance)
{
	cudaMemset(x->data, 0, sizeof(double)*x->cur_proc_row_count);
}

void mpla_generic_conjugate_gradient(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x, int iter_max, double epsilon, void (*mpla_dgemv_core)(struct mpla_vector*, struct mpla_generic_matrix*, struct mpla_vector*, struct mpla_instance*), struct mpla_instance* instance)
{
	// init some vectors
	struct mpla_vector r;
	struct mpla_vector d;
	struct mpla_vector z;
	mpla_init_vector(&r, instance, x->vec_row_count);
	mpla_init_vector(&d, instance, x->vec_row_count);
	mpla_init_vector(&z, instance, x->vec_row_count);

	double alpha,beta;

	// r_0 = b - A * x_0
	mpla_generic_dgemv(&z, A, x, mpla_dgemv_core, instance);
	mpla_vector_set_zero(&r, instance);
	mpla_daxpy(&r, 1, b, instance);
	mpla_daxpy(&r, -1, &z, instance);

	// d_0 = r_0
	mpla_vector_set_zero(&d, instance);
	mpla_daxpy(&d, 1, &r, instance);

	double res;
	mpla_ddot(&res, &r, &r, instance);
//	printf("%d: %e\n", 0, sqrt(res/(double)(x->vec_row_count)));

	if (sqrt(res/(double)(x->vec_row_count))>=epsilon)
	for (int k=1; k<iter_max; k++)
	{
		// z = A * d_k
		mpla_generic_dgemv(&z, A, &d, mpla_dgemv_core, instance);
		
		// alpha_k = <r_k,r_k>/<d_k, z>
		double t1,t2;
		mpla_ddot(&t1, &r, &r, instance);
		mpla_ddot(&t2, &d, &z, instance);
		alpha = t1 / t2;

		// x_{k+1} = x_k + alpha_k d_k
		mpla_daxpy(x, alpha, &d, instance);
		
		// r_{k+1} = r_k - alpha_k z
		mpla_daxpy(&r, -alpha, &z, instance);

		// beta_k = <r_{k+1}, r_{k+1}> / <r_k, r_k>
		mpla_ddot(&t2, &r, &r, instance);
		beta = t2 / t1;

//		printf("%d: %e\n", k, sqrt(t2/(double)(x->vec_row_count)));

		if (sqrt(t2/(double)(x->vec_row_count))<epsilon)
		{
			break;
		}
		
		// d_{k+1} = r_{k+1} + beta_k d_k
		mpla_vector_set_zero(&z, instance);
		mpla_daxpy(&z, beta, &d, instance);
		mpla_daxpy(&z, 1, &r, instance);
		mpla_vector_set_zero(&d, instance);
		mpla_daxpy(&d, 1, &z, instance); 
	}

	// memory cleanup
	mpla_free_vector(&r, instance);
	mpla_free_vector(&d, instance);
	mpla_free_vector(&z, instance);
}

void checkCUDAError(const char* msg) {
cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


